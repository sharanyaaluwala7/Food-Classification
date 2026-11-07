#!/usr/bin/env python
"""Web app. Serves predictions ONLY from genuinely trained checkpoints.

A registry entry is accepted when all of these hold:
  * weights and metrics files exist and the weights' sha256 matches the registry;
  * it was trained on the Hub dataset (data_source == "hub") with the expected dataset id;
  * it has real metrics.json produced by train.py.
If no entry is valid, /api/predict answers 503 and the page shows how to train.
Set FOODCLF_ALLOW_UNVERIFIED=1 only for automated tests with synthetic data.

    python app.py            # http://127.0.0.1:5000
"""
from __future__ import annotations

import io
import json
import os
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError

from foodclf import config
from foodclf import modelzoo as Z

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 40_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
UNCERTAIN_BELOW = float(os.environ.get("FOODCLF_UNCERTAIN_BELOW", "0.5"))
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ModelStore:
    def __init__(self, models_dir: Path, allow_unverified: bool = False):
        self.dir = Path(models_dir)
        self.allow_unverified = allow_unverified
        self.valid: dict[str, dict] = {}
        self.rejected: list[dict] = []
        self._loaded: dict[str, tuple] = {}
        self._metrics: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.refresh()

    def refresh(self):
        self.valid, self.rejected = {}, []
        for e in Z.load_registry(self.dir / "registry.json").get("models", []):
            reason = self._check(e)
            if reason:
                self.rejected.append({"id": e.get("id"), "reason": reason})
            else:
                self.valid[e["id"]] = e

    def _check(self, e):
        w, m = self.dir / e.get("weights", ""), self.dir / e.get("metrics", "")
        if not w.is_file() or not m.is_file():
            return "weights or metrics file missing"
        if Z.sha256_file(w) != e.get("sha256"):
            return "checksum mismatch"
        if not self.allow_unverified:
            if e.get("dataset_id") != config.DATASET_ID or e.get("data_source") != "hub":
                return "not trained on the Hugging Face dataset"
            if e.get("partial_training_subset"):
                return "trained on a partial subset (debug run)"
            if e.get("pretrained") is False:
                return "trained without pretrained weights (test run)"
        return None

    def metrics(self, mid):
        if mid not in self._metrics:
            self._metrics[mid] = json.loads((self.dir / self.valid[mid]["metrics"]).read_text())
        return self._metrics[mid]

    def get(self, mid):
        with self._lock:
            if mid not in self._loaded:
                self._loaded[mid] = Z.load_checkpoint(self.dir / self.valid[mid]["weights"], "cpu")
            return self._loaded[mid]


def preprocess(img: Image.Image, meta):
    import torch
    from torchvision import transforms as T

    size = meta["img_size"]
    tf = T.Compose([T.Resize(int(round(size * 1.14))), T.CenterCrop(size), T.ToTensor(),
                    T.Normalize(meta["mean"], meta["std"])])
    return tf(img).unsqueeze(0)


def evidence_for(metrics: dict, cls: str):
    """What was actually measured for this class."""
    test = metrics["test"]["open_set_all_classes"]["per_class"]
    if cls in test:
        pc = test[cls]
        return {"basis": "official_test_split", "f1": pc["f1"], "precision": pc["precision"],
                "recall": pc["recall"], "n_images": pc["support"], "low_support": pc["low_support"]}
    pc = metrics["internal_holdout"]["open_set_all_classes"]["per_class"].get(cls)
    return {"basis": "internal_holdout_only", "f1": pc["f1"], "precision": pc["precision"],
            "recall": pc["recall"], "n_images": pc["support"], "low_support": pc["low_support"]}


def create_app(models_dir: Path | str | None = None, allow_unverified: bool | None = None):
    if allow_unverified is None:
        allow_unverified = os.environ.get("FOODCLF_ALLOW_UNVERIFIED") == "1"
    store = ModelStore(models_dir or config.MODELS_DIR, allow_unverified)
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["STORE"] = store

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/status")
    def status():
        store.refresh()
        return jsonify({"ready": bool(store.valid), "n_models": len(store.valid), "rejected": store.rejected,
                        "uncertain_below": UNCERTAIN_BELOW})

    @app.get("/api/models")
    def models():
        store.refresh()
        out = []
        for mid, e in store.valid.items():
            m = store.metrics(mid)
            out.append({"id": mid, "arch": e["arch"], "mode": e["mode"], "created_at": e["created_at"],
                        "headline": e["headline"], "training": m["training"],
                        "class_coverage": m["class_coverage"]})
        return jsonify({"models": out})

    @app.get("/api/metrics/<mid>")
    def metrics(mid):
        if mid not in store.valid:
            return jsonify({"error": "unknown model"}), 404
        return jsonify(store.metrics(mid))

    @app.get("/api/dataset")
    def dataset():
        if config.DATASET_REPORT_PATH.exists():
            rep, source = json.loads(config.DATASET_REPORT_PATH.read_text()), "generated_by_inspect_dataset"
        elif config.DATASET_SNAPSHOT_PATH.exists():
            rep, source = json.loads(config.DATASET_SNAPSHOT_PATH.read_text()), "reference_snapshot_2026-10-11"
        else:
            rep, source = {}, "none"
        return jsonify({"id": config.DATASET_ID, "url": config.DATASET_URL, "license": config.DATASET_LICENSE,
                        "attribution": config.ATTRIBUTION, "source": source, "report": rep})

    @app.post("/api/predict")
    def predict():
        store.refresh()
        if not store.valid:
            return jsonify({"error": "no_trained_models",
                            "message": "No trained model is installed, so no prediction can be made. "
                                       "Run train.py first (see README).",
                            "rejected": store.rejected}), 503
        f = request.files.get("image")
        if f is None:
            return jsonify({"error": "missing_image", "message": "Attach an image in the 'image' field."}), 400
        raw = f.read()
        try:
            probe = Image.open(io.BytesIO(raw))
            fmt = probe.format
            probe.verify()
            if fmt not in ALLOWED_FORMATS:
                return jsonify({"error": "unsupported_format", "message": "Use a JPEG, PNG or WebP image."}), 415
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            return jsonify({"error": "invalid_image", "message": "That file could not be read as an image."}), 400

        wanted = request.form.get("model", "")
        ids = list(store.valid) if wanted == "all" else [wanted or next(iter(store.valid))]
        if any(i not in store.valid for i in ids):
            return jsonify({"error": "unknown_model", "message": "Choose one of the listed models."}), 400
        try:
            top_k = max(1, min(10, int(request.form.get("top_k", 5))))
        except ValueError:
            return jsonify({"error": "invalid_top_k", "message": "top_k must be a whole number."}), 400

        import torch

        results = []
        for mid in ids:
            model, meta = store.get(mid)
            with torch.inference_mode():
                probs = torch.softmax(model(preprocess(img, meta)), dim=1)[0]
            vals, idx = probs.topk(min(top_k, probs.numel()))
            m = store.metrics(mid)
            preds = [{"class": meta["class_names"][int(i)], "probability": float(v),
                      "evidence": evidence_for(m, meta["class_names"][int(i)])} for v, i in zip(vals, idx)]
            results.append({"model": mid, "arch": meta["arch"], "predictions": preds,
                            "uncertain": preds[0]["probability"] < UNCERTAIN_BELOW})
        return jsonify({"results": results, "uncertain_below": UNCERTAIN_BELOW,
                        "note": "The model only knows the dishes it was trained on and will always name one of them, "
                                "even for a photo that is not food."})

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"error": "too_large", "message": "Images must be 8 MB or smaller."}), 413

    return app


if __name__ == "__main__":
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")), debug=False)
