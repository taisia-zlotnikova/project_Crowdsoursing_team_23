"""Run pinned local models sequentially and persist one prediction per photo."""

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
from time import perf_counter

from inference.data import ROOT, load_image, read_dataset, sha256, write_json, write_jsonl
from inference.metrics import read_jsonl, validate_box


MODEL_NAMES = ("grounding-dino", "owlv2", "qwen3-vl")


def validate_config(config, names):
    if type(config.get("seed")) is not int:
        raise ValueError("config.seed must be an integer")
    for name in names:
        settings = config["models"][name]
        if name == "qwen3-vl":
            if not isinstance(settings.get("prompt"), str) or not settings["prompt"].strip():
                raise ValueError("Qwen requires a nonempty prompt")
            for key in ("min_pixels", "max_pixels", "max_new_tokens"):
                if type(settings.get(key)) is not int or settings[key] <= 0:
                    raise ValueError(f"{name}.{key} must be a positive integer")
            if settings["min_pixels"] > settings["max_pixels"]:
                raise ValueError("min_pixels exceeds max_pixels")
        else:
            keys = ["confidence", "nms_iou"] + (["text_threshold"] if name == "grounding-dino" else [])
            for key in keys:
                value = settings.get(key)
                if key == "nms_iou" and value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{name}.{key} must be between 0 and 1")
            prompt = settings.get("prompt") if name == "grounding-dino" else settings.get("queries")
            if name == "grounding-dino":
                valid = isinstance(prompt, str) and bool(prompt.strip())
            else:
                valid = isinstance(prompt, list) and bool(prompt) and all(isinstance(v, str) and v.strip() for v in prompt)
            if not valid:
                raise ValueError(f"{name} requires fixed text queries")


def synchronize(torch, device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _existing(path):
    return {r["photo_id"]: r for r in read_jsonl(path)} if path.exists() else {}


def run_model(name, records, settings, output, args, torch):
    from inference.adapters import get_predictor
    from inference.loader import load_model

    folder = output / name
    folder.mkdir(exist_ok=True)
    predictions = _existing(folder / "predictions.jsonl")
    diagnostics = _existing(folder / "diagnostics.jsonl")
    successful = {key for key, value in predictions.items()
                  if value.get("error") is None and key in diagnostics and not diagnostics[key].get("error")}
    if len(successful) == len(records):
        print(f"{name}: all {len(records)} predictions already saved", flush=True)
        return True
    summary_path = folder / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"sessions": []}
    session = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "running"}
    summary["sessions"].append(session)
    write_json(summary_path, summary)
    started = perf_counter()
    model = processor = None
    try:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        loading = perf_counter()
        model, processor = load_model(name, args.device, local_files_only=args.local_files_only)
        synchronize(torch, args.device)
        session.update(load_seconds=perf_counter() - loading, dtype=str(model.dtype), device=str(model.device))
        predictor = get_predictor(name)
        for position, record in enumerate(records, 1):
            photo_id = record["photo_id"]
            if photo_id in successful:
                continue
            previous = diagnostics.get(photo_id, {})
            detail = {"photo_id": photo_id, "attempts": list(previous.get("attempts", []))}
            for attempt_number in range(args.retries + 1):
                attempt = {"stage": "image", "started_at": datetime.now(timezone.utc).isoformat()}
                attempt_started = perf_counter()
                inference_started = None
                try:
                    image, image_info = load_image(record, args.cache_dir, offline=args.offline_images)
                    attempt.update(image_info)
                    attempt["stage"] = "inference"
                    synchronize(torch, args.device)
                    inference_started = perf_counter()
                    result = predictor(image, model, processor, settings)
                    synchronize(torch, args.device)
                    attempt["inference_seconds"] = perf_counter() - inference_started
                    if not isinstance(result.get("boxes"), list):
                        raise ValueError("Adapter must return a boxes list")
                    for box in result["boxes"]:
                        validate_box(box)
                    attempt.update(result["diagnostics"], error=None)
                    prediction = {"photo_id": photo_id, "boxes": result["boxes"], "error": None}
                except Exception as error:
                    if inference_started is not None and "inference_seconds" not in attempt:
                        synchronize(torch, args.device)
                        attempt["inference_seconds"] = perf_counter() - inference_started
                    message = f"{type(error).__name__}: {error}"
                    attempt.update(getattr(error, "diagnostics", {}), error=message)
                    prediction = {"photo_id": photo_id, "boxes": None, "error": message}
                attempt["attempt_seconds"] = perf_counter() - attempt_started
                detail["attempts"].append(attempt)
                if prediction["error"] is None:
                    break
            detail.update(error=prediction["error"], inference_seconds=attempt.get("inference_seconds"),
                          image_seconds=attempt.get("image_seconds"))
            predictions[photo_id], diagnostics[photo_id] = prediction, detail
            # A resume only trusts a success with both files present.
            write_jsonl(folder / "diagnostics.jsonl", diagnostics.values())
            write_jsonl(folder / "predictions.jsonl", predictions.values())
            status = prediction["error"] or f"{len(prediction['boxes'])} boxes"
            print(f"{name} [{position}/{len(records)}] {photo_id}: {status}", flush=True)
        session["status"] = "complete" if all(p.get("error") is None for p in predictions.values()) else "incomplete"
    except BaseException as error:
        session.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        session["wall_seconds"] = perf_counter() - started
        write_json(summary_path, summary)
        del model, processor
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
        elif args.device == "cuda":
            torch.cuda.empty_cache()
    return session["status"] == "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/annotations.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "inference/configs/baseline.json")
    parser.add_argument("--models", nargs="+", choices=(*MODEL_NAMES, "all"), default=["all"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/images")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="Smoke test only; saved explicitly as a partial dataset")
    selection.add_argument("--photo-ids", type=int, nargs="+", help="Evaluate selected photos; recorded as a subset")
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline-images", action="store_true")
    args = parser.parse_args()
    try:
        import torch
        from inference.loader import MODELS

        if args.retries < 0 or (args.limit is not None and args.limit <= 0):
            raise ValueError("retries must be nonnegative and limit must be positive")
        if args.device == "auto":
            args.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable in this process; choose --device cpu")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this process")
        names = list(MODEL_NAMES) if "all" in args.models else list(dict.fromkeys(args.models))
        config = json.loads(args.config.read_text(encoding="utf-8"))
        validate_config(config, names)
        args.seed = config["seed"]
        records = read_dataset(args.dataset)
        dataset_total = len(records)
        if args.limit:
            records = records[:args.limit]
        if args.photo_ids:
            requested = set(args.photo_ids)
            unknown = requested - {r["photo_id"] for r in records}
            if unknown:
                raise ValueError(f"Unknown photo IDs: {sorted(unknown)}")
            records = [r for r in records if r["photo_id"] in requested]
        identity = {
            "dataset_sha256": sha256(args.dataset), "config": config, "models": names,
            "photo_ids": [r["photo_id"] for r in records], "device": args.device,
            "versions": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "transformers", "Pillow", "huggingface-hub")},
            "weights": {name: {"model_id": MODELS[name].model_id, "revision": MODELS[name].revision} for name in names},
            "implementation_sha256": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in sorted((ROOT / "inference").rglob("*.py"))
                if path.name not in ("evaluate.py", "metrics.py")
            },
        }
        manifest_path = args.output / "manifest.json"
        if manifest_path.exists():
            if not args.resume:
                raise ValueError("Output exists; use --resume or a new output folder")
            previous = json.loads(manifest_path.read_text())
            if previous["identity"] != identity:
                raise ValueError("Cannot resume: dataset, selection, settings, code, weights, versions or device changed")
        else:
            if args.output.exists() and any(args.output.iterdir()):
                raise ValueError("Output folder is not empty and has no manifest")
            write_json(manifest_path, {
                "identity": identity, "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset_path": str(args.dataset.resolve()), "cache_dir": str(args.cache_dir.resolve()),
                "dataset_total": dataset_total, "selected_images": len(records),
                "scope": "full" if len(records) == dataset_total else "subset",
                "config_path": str(args.config.resolve()), "platform": platform.platform(),
                "postprocessing": "clip detector boxes to image; class-agnostic configured NMS; strict Qwen JSON and 0..1000 bounds",
                "image_transform": "source pixels converted to RGB without EXIF rotation; model processor resize",
            })
        completed = []
        for name in names:
            completed.append(run_model(name, records, config["models"][name], args.output, args, torch))
        print(f"Saved predictions to {args.output.resolve()}", flush=True)
        if not all(completed):
            raise SystemExit(1)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
