
#!/usr/bin/env python
"""Indian Food Classifier Flask application."""

from __future__ import annotations

import gc
import io
import json
import logging
import os
import threading
from pathlib import Path

# Limit CPU thread usage before importing PyTorch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError

from foodclf import config
from foodclf import modelzoo as Z

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 8_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

UNCERTAIN_BELOW = float(
    os.environ.get("FOODCLF_UNCERTAIN_BELOW", "0.5")
)

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ModelStore:

    def __init__(self, models_dir: Path, allow_unverified=False):
        self.dir = Path(models_dir)
        self.allow_unverified = allow_unverified

        self.valid = {}
        self.rejected = []

        # Keep only one model in memory.
        self._loaded = None

        self._metrics = {}

        # Protect model loading.
        self._model_lock = threading.Lock()

        # Prevent concurrent predictions from loading multiple models.
        self._predict_lock = threading.Lock()

        self.refresh()

    def refresh(self):
        self.valid = {}
        self.rejected = []

        registry = Z.load_registry(self.dir / "registry.json")

        for entry in registry.get("models", []):
            reason = self._check(entry)

            if reason:
                self.rejected.append({
                    "id": entry.get("id"),
                    "reason": reason
                })
            else:
                self.valid[entry["id"]] = entry

        # Clear cached model if it is no longer valid.
        if self._loaded is not None:
            if self._loaded[0] not in self.valid:
                self._loaded = None
                gc.collect()

    def _check(self, entry):
        weights = self.dir / entry.get("weights", "")
        metrics = self.dir / entry.get("metrics", "")

        if not weights.is_file() or not metrics.is_file():
            return "weights or metrics file missing"

        if Z.sha256_file(weights) != entry.get("sha256"):
            return "checksum mismatch"

        if not self.allow_unverified:
            if (
                entry.get("dataset_id") != config.DATASET_ID
                or entry.get("data_source") != "hub"
            ):
                return "not trained on the Hugging Face dataset"

            if entry.get("partial_training_subset"):
                return "trained on a partial subset"

            if entry.get("pretrained") is False:
                return "trained without pretrained weights"

        return None

    def metrics(self, mid):
        if mid not in self._metrics:
            path = self.dir / self.valid[mid]["metrics"]
            self._metrics[mid] = json.loads(path.read_text())

        return self._metrics[mid]

    def get(self, mid):
        with self._model_lock:

            # Return the cached model if already loaded.
            if self._loaded is not None:
                if self._loaded[0] == mid:
                    return self._loaded[1], self._loaded[2]

            # Release the previous model before loading another.
            self._loaded = None
            gc.collect()

            logger.info("Loading model: %s", mid)

            model, meta = Z.load_checkpoint(
                self.dir / self.valid[mid]["weights"],
                "cpu"
            )

            self._loaded = (mid, model, meta)

            logger.info("Model loaded successfully: %s", mid)

            return model, meta


def preprocess(img: Image.Image, meta):
    import torch
    from torchvision import transforms as T

    size = meta["img_size"]

    transform = T.Compose([
        T.Resize(int(round(size * 1.14))),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(meta["mean"], meta["std"])
    ])

    tensor = transform(img)

    return tensor.unsqueeze(0)


def evidence_for(metrics: dict, cls: str):
    test = metrics["test"]["open_set_all_classes"]["per_class"]

    if cls in test:
        pc = test[cls]

        return {
            "basis": "official_test_split",
            "f1": pc["f1"],
            "precision": pc["precision"],
            "recall": pc["recall"],
            "n_images": pc["support"],
            "low_support": pc["low_support"]
        }

    pc = metrics["internal_holdout"]["open_set_all_classes"][
        "per_class"
    ][cls]

    return {
        "basis": "internal_holdout_only",
        "f1": pc["f1"],
        "precision": pc["precision"],
        "recall": pc["recall"],
        "n_images": pc["support"],
        "low_support": pc["low_support"]
    }


def create_app(models_dir=None, allow_unverified=None):

    if allow_unverified is None:
        allow_unverified = (
            os.environ.get("FOODCLF_ALLOW_UNVERIFIED") == "1"
        )

    store = ModelStore(
        models_dir or config.MODELS_DIR,
        allow_unverified
    )

    app = Flask(
        __name__,
        static_folder="static",
        static_url_path="/static"
    )

    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["STORE"] = store

    @app.get("/")
    def index():
        return send_from_directory(
            app.static_folder,
            "index.html"
        )

    @app.get("/api/status")
    def status():
        return jsonify({
            "ready": bool(store.valid),
            "n_models": len(store.valid),
            "rejected": store.rejected,
            "uncertain_below": UNCERTAIN_BELOW
        })

    @app.get("/api/models")
    def models():
        output = []

        for mid, entry in store.valid.items():
            metrics = store.metrics(mid)

            output.append({
                "id": mid,
                "arch": entry["arch"],
                "mode": entry["mode"],
                "created_at": entry["created_at"],
                "headline": entry["headline"],
                "training": metrics["training"],
                "class_coverage": metrics["class_coverage"]
            })

        return jsonify({"models": output})

    @app.get("/api/metrics/<mid>")
    def metrics(mid):

        if mid not in store.valid:
            return jsonify({"error": "unknown model"}), 404

        return jsonify(store.metrics(mid))

    @app.get("/api/dataset")
    def dataset():

        if config.DATASET_REPORT_PATH.exists():
            report = json.loads(
                config.DATASET_REPORT_PATH.read_text()
            )
            source = "generated_by_inspect_dataset"

        elif config.DATASET_SNAPSHOT_PATH.exists():
            report = json.loads(
                config.DATASET_SNAPSHOT_PATH.read_text()
            )
            source = "reference_snapshot_2026-10-11"

        else:
            report = {}
            source = "none"

        return jsonify({
            "id": config.DATASET_ID,
            "url": config.DATASET_URL,
            "license": config.DATASET_LICENSE,
            "attribution": config.ATTRIBUTION,
            "source": source,
            "report": report
        })

    @app.post("/api/predict")
    def predict():

        if not store.valid:
            return jsonify({
                "error": "no_trained_models",
                "message": "No valid trained model is installed.",
                "rejected": store.rejected
            }), 503

        uploaded = request.files.get("image")

        if uploaded is None:
            return jsonify({
                "error": "missing_image",
                "message": "Please upload an image."
            }), 400

        raw = uploaded.read()

        if len(raw) > MAX_UPLOAD_BYTES:
            return jsonify({
                "error": "too_large",
                "message": "Image must be 8 MB or smaller."
            }), 413

        try:
            probe = Image.open(io.BytesIO(raw))

            image_format = probe.format
            width, height = probe.size

            if image_format not in ALLOWED_FORMATS:
                probe.close()

                return jsonify({
                    "error": "unsupported_format",
                    "message": "Use JPEG, PNG or WebP."
                }), 415

            if width * height > MAX_PIXELS:
                probe.close()

                return jsonify({
                    "error": "image_dimensions_too_large",
                    "message": "Image dimensions are too large. "
                               "Please upload a smaller image."
                }), 413

            probe.verify()
            probe.close()

            img = Image.open(
                io.BytesIO(raw)
            ).convert("RGB")

        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError
        ):
            return jsonify({
                "error": "invalid_image",
                "message": "The uploaded image could not be read."
            }), 400

        wanted = request.form.get("model", "")

        # Never load all models simultaneously.
        if wanted == "all":
            img.close()

            return jsonify({
                "error": "multiple_models_disabled",
                "message": "Please select one model at a time."
            }), 400

        mid = wanted or next(iter(store.valid))

        if mid not in store.valid:
            img.close()

            return jsonify({
                "error": "unknown_model",
                "message": "Choose a valid model."
            }), 400

        try:
            top_k = max(
                1,
                min(10, int(request.form.get("top_k", 5)))
            )

        except ValueError:
            img.close()

            return jsonify({
                "error": "invalid_top_k",
                "message": "top_k must be a whole number."
            }), 400

        try:
            import torch

            torch.set_num_threads(1)

            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

            results = []

            # Serialize inference to avoid duplicate model memory.
            with store._predict_lock:

                model, meta = store.get(mid)

                tensor = preprocess(img, meta)

                with torch.inference_mode():
                    logits = model(tensor)
                    probabilities = torch.softmax(
                        logits,
                        dim=1
                    )[0]

                    values, indices = probabilities.topk(
                        min(top_k, probabilities.numel())
                    )

                metrics_data = store.metrics(mid)

                predictions = []

                for value, index in zip(values, indices):

                    class_name = meta["class_names"][int(index)]

                    predictions.append({
                        "class": class_name,
                        "probability": float(value),
                        "evidence": evidence_for(
                            metrics_data,
                            class_name
                        )
                    })

                results.append({
                    "model": mid,
                    "arch": meta["arch"],
                    "predictions": predictions,
                    "uncertain": (
                        predictions[0]["probability"]
                        < UNCERTAIN_BELOW
                    )
                })

                del tensor
                del logits
                del probabilities
                del values
                del indices

            img.close()
            gc.collect()

            return jsonify({
                "results": results,
                "uncertain_below": UNCERTAIN_BELOW,
                "note": (
                    "The model only knows the dishes it was "
                    "trained on and will always name one of them, "
                    "even for a photo that is not food."
                )
            })

        except Exception:
            img.close()
            logger.exception("Prediction failed")

            gc.collect()

            return jsonify({
                "error": "prediction_failed",
                "message": (
                    "The model could not process this image. "
                    "Please check the server logs."
                )
            }), 500

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({
            "error": "too_large",
            "message": "Images must be 8 MB or smaller."
        }), 413

    return app


if __name__ == "__main__":

    application = create_app()

    application.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
        threaded=False
    )
