
"""Model construction, checkpoint saving/loading and registry."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Dict

from . import config


ARCHS = {
    "mobilenetv3_large_100": {
        "img_size": 224,
        "note": "small, fast on CPU"
    },
    "efficientnet_b0": {
        "img_size": 224,
        "note": "balanced"
    },
    "convnext_tiny": {
        "img_size": 224,
        "note": "larger, usually most accurate"
    }
}


def build_model(
    arch: str,
    num_classes: int,
    pretrained: bool
):
    import timm

    model = timm.create_model(
        arch,
        pretrained=pretrained,
        num_classes=num_classes
    )

    return model


def sha256_file(path: Path) -> str:

    hasher = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def load_registry(path: Path = config.REGISTRY_PATH) -> Dict:

    if not path.exists():
        return {"models": []}

    return json.loads(path.read_text())


def add_to_registry(
    entry: Dict,
    path: Path = config.REGISTRY_PATH
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    registry = load_registry(path)

    registry["models"] = [
        model
        for model in registry["models"]
        if model["id"] != entry["id"]
    ] + [entry]

    path.write_text(
        json.dumps(registry, indent=2)
    )


def save_checkpoint(
    path: Path,
    state_dict,
    meta: Dict
) -> None:

    import torch

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "state_dict": state_dict,
            "meta": meta
        },
        path
    )


def load_checkpoint(path: Path, device="cpu"):

    import torch

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}"
        )

    # Load weights safely and use memory mapping where supported.
    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
            mmap=True
        )

    except (TypeError, RuntimeError):
        # Compatibility fallback for older PyTorch versions.
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True
        )

    if "state_dict" not in checkpoint:
        raise ValueError(
            "Invalid checkpoint: state_dict is missing."
        )

    if "meta" not in checkpoint:
        raise ValueError(
            "Invalid checkpoint: metadata is missing."
        )

    meta = checkpoint["meta"]

    required = [
        "arch",
        "class_names",
        "img_size",
        "mean",
        "std"
    ]

    for key in required:
        if key not in meta:
            raise ValueError(
                f"Checkpoint metadata is missing: {key}"
            )

    # Construct architecture without downloading pretrained weights.
    model = build_model(
        meta["arch"],
        len(meta["class_names"]),
        pretrained=False
    )

    # Assign checkpoint tensors directly where supported.
    # This can reduce temporary memory duplication.
    try:
        model.load_state_dict(
            checkpoint["state_dict"],
            assign=True
        )

    except TypeError:
        model.load_state_dict(
            checkpoint["state_dict"]
        )

    model = model.to(device)
    model.eval()

    del checkpoint
    gc.collect()

    return model, meta
