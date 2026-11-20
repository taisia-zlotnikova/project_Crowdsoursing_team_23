"""Load one pinned butterfly detection model and its processor."""

import argparse
from typing import NamedTuple

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoProcessor,
    GroundingDinoForObjectDetection,
    Owlv2ForObjectDetection,
    Qwen3VLForConditionalGeneration,
)


class ModelSpec(NamedTuple):
    model_id: str
    revision: str
    model_class: type


MODELS = {
    "grounding-dino": ModelSpec(
        "IDEA-Research/grounding-dino-tiny",
        "a2bb814dd30d776dcf7e30523b00659f4f141c71",
        GroundingDinoForObjectDetection,
    ),
    "owlv2": ModelSpec(
        "google/owlv2-base-patch16-ensemble",
        "cfd3195ba4ea9592eec887ded089f4c08eff231d",
        Owlv2ForObjectDetection,
    ),
    "qwen3-vl": ModelSpec(
        "Qwen/Qwen3-VL-2B-Instruct",
        "89644892e4d85e24eaac8bacfd4f463576704203",
        Qwen3VLForConditionalGeneration,
    ),
}


def load_model(name: str, device: str = "cpu", *, local_files_only: bool = False):
    """Return a model in eval mode and the processor from the same revision."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(MODELS)}.")

    spec = MODELS[name]
    device = torch.device(device)
    snapshot = snapshot_download(
        spec.model_id,
        revision=spec.revision,
        allow_patterns=["model.safetensors", "*.json", "*.txt", "*.jinja"],
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(
        snapshot, local_files_only=True, use_fast=True
    )
    model = spec.model_class.from_pretrained(
        snapshot,
        dtype="auto",
        use_safetensors=True,
        local_files_only=True,
    )
    model.to(device).eval()
    return model, processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or mps")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    model, processor = load_model(
        args.model, args.device, local_files_only=args.local_files_only
    )
    spec = MODELS[args.model]
    print(f"Loaded {spec.model_id}@{spec.revision} on {model.device} ({model.dtype})")
    print(f"Processor: {type(processor).__name__}")


if __name__ == "__main__":
    main()
