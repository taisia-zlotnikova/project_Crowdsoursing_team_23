"""Evaluate saved predictions against the exact dataset selection in a run manifest."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

from inference.data import read_dataset, sha256, write_json
from inference.metrics import evaluate, read_jsonl


def summarize_timing(diagnostics, summary):
    attempts = [attempt for row in diagnostics for attempt in row.get("attempts", [])]
    successful = [row["inference_seconds"] for row in diagnostics
                  if row.get("error") is None and row.get("inference_seconds") is not None]
    durations = sorted(successful)
    p95 = durations[max(0, math.ceil(len(durations) * .95) - 1)] if durations else None
    sessions = summary.get("sessions", [])
    return {
        "successful_images_timed": len(successful),
        "successful_inference_seconds": sum(successful) if successful else None,
        "mean_inference_seconds": mean(successful) if successful else None,
        "median_inference_seconds": median(successful) if successful else None,
        "p95_inference_seconds": p95,
        "all_attempt_inference_seconds": sum(a.get("inference_seconds", 0) for a in attempts),
        "image_io_seconds": sum(a.get("image_seconds", 0) for a in attempts),
        "all_attempt_seconds": sum(a.get("attempt_seconds", 0) for a in attempts),
        "load_seconds": sum(s.get("load_seconds", 0) for s in sessions),
        "wall_seconds": sum(s.get("wall_seconds", 0) for s in sessions),
        "attempts": len(attempts),
    }


def comparison_row(name, report):
    row = {"model": name, "scope": report["scope"], "status": report["status"],
           "images": report["images_total"], "succeeded": report["images_succeeded"],
           "technical_errors": report["technical_errors"]}
    quality = report["quality"]
    for threshold in ("0.5", "0.75"):
        for metric in ("f1", "precision", "recall", "tp", "fp", "fn"):
            row[f"{metric}@{threshold}"] = quality["by_iou"][threshold][metric] if quality else None
        row[f"detection_errors@{threshold}"] = (
            quality["by_iou"][threshold]["fp"] + quality["by_iou"][threshold]["fn"] if quality else None
        )
    for key in ("negative_images", "negative_images_with_false_positives",
                "false_positive_boxes_on_negative_images", "negative_image_false_positive_rate"):
        row[key] = quality[key] if quality else None
    row.update(report["timing"])
    return row


def evaluate_run(run_dir, *, dataset_path=None):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset_path = Path(dataset_path or manifest["dataset_path"])
    if sha256(dataset_path) != manifest["identity"]["dataset_sha256"]:
        raise ValueError("Dataset changed since inference; use the original annotation file")
    all_records = read_dataset(dataset_path, require_annotations=True)
    indexed = {row["photo_id"]: row for row in all_records}
    ids = manifest["identity"]["photo_ids"]
    references = [indexed[photo_id] for photo_id in ids]
    reports, table = {}, []
    for name in manifest["identity"]["models"]:
        folder = run_dir / name
        prediction_path = folder / "predictions.jsonl"
        predictions = read_jsonl(prediction_path) if prediction_path.exists() else []
        seen = {row["photo_id"] for row in predictions}
        # Interrupted runs remain visible, with missing photos counted as errors.
        missing = [photo_id for photo_id in ids if photo_id not in seen]
        predictions.extend({"photo_id": photo_id, "boxes": None, "error": "Prediction not saved"} for photo_id in missing)
        diagnostic_path = folder / "diagnostics.jsonl"
        diagnostics = read_jsonl(diagnostic_path) if diagnostic_path.exists() else []
        summary_path = folder / "run_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        timing = summarize_timing(diagnostics, summary)
        elapsed = timing["successful_inference_seconds"] if len(diagnostics) == len(ids) and timing["successful_images_timed"] == len(ids) else None
        report = evaluate(predictions, references, elapsed_seconds=elapsed)
        report.update(model=name, scope=manifest["scope"], missing_predictions=len(missing),
                      dataset_sha256=manifest["identity"]["dataset_sha256"], timing=timing)
        report["groups"] = {}
        for label, predicate in (("empty", lambda n: n == 0), ("one_object", lambda n: n == 1),
                                 ("multiple_objects", lambda n: n > 1)):
            group = [r for r in references if predicate(len(r["boxes"]))]
            if group:
                group_ids = {r["photo_id"] for r in group}
                result = evaluate([p for p in predictions if p["photo_id"] in group_ids], group)
                report["groups"][label] = {k: result[k] for k in ("status", "images_total", "technical_errors", "quality")}
        write_json(folder / "metrics.json", report)
        reports[name] = report
        table.append(comparison_row(name, report))
    path = run_dir / "comparison.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    temporary.replace(path)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="Location override for the same annotation file")
    args = parser.parse_args()
    try:
        reports = evaluate_run(args.run, dataset_path=args.dataset)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    for name, report in reports.items():
        quality = report["quality"]
        f1 = f"{quality['by_iou']['0.5']['f1']:.4f}" if quality else "unavailable"
        print(f"{name}: {report['status']}, scope={report['scope']}, F1@0.5={f1}, errors={report['technical_errors']}")
    print(f"Saved reports and comparison.csv in {args.run.resolve()}")


if __name__ == "__main__":
    main()
