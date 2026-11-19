"""Evaluate normalized detection JSONL against a reviewed golden set."""

import argparse
from collections import deque
import json
import math
from pathlib import Path


IOU_THRESHOLDS = (0.5, 0.75)
BOX_FIELDS = ("left", "top", "width", "height")


def _finite_number(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number.")
    return value


def validate_box(box, label="box"):
    if not isinstance(box, dict) or not all(key in box for key in BOX_FIELDS):
        raise ValueError(f"{label} must contain {', '.join(BOX_FIELDS)}.")
    left, top, width, height = (
        _finite_number(box[key], f"{label}.{key}") for key in BOX_FIELDS
    )
    if not (0 <= left <= 1 and 0 <= top <= 1 and 0 < width <= 1 and 0 < height <= 1):
        raise ValueError(f"{label} must have normalized coordinates and positive dimensions.")
    if left + width > 1 + 1e-9 or top + height > 1 + 1e-9:
        raise ValueError(f"{label} extends outside the image.")


def box_iou(first: dict, second: dict) -> float:
    """Calculate intersection over union for two valid normalized boxes."""
    left = max(first["left"], second["left"])
    top = max(first["top"], second["top"])
    right = min(
        first["left"] + first["width"], second["left"] + second["width"]
    )
    bottom = min(
        first["top"] + first["height"], second["top"] + second["height"]
    )
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (
        first["width"] * first["height"]
        + second["width"] * second["height"]
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def _maximum_matches(overlaps: list[list[float]], threshold: float) -> int:
    """Find maximum-cardinality matching using augmenting paths."""
    if not overlaps:
        return 0
    neighbors = [
        [index for index, overlap in enumerate(row) if overlap >= threshold]
        for row in overlaps
    ]
    matched_prediction = [-1] * len(overlaps)
    matched_reference = [-1] * len(overlaps[0])

    for start in range(len(overlaps)):
        queue = deque([start])
        visited_predictions = {start}
        parent_reference = {}
        free_reference = -1

        while queue and free_reference == -1:
            prediction = queue.popleft()
            for reference in neighbors[prediction]:
                if reference in parent_reference:
                    continue
                parent_reference[reference] = prediction
                partner = matched_reference[reference]
                if partner == -1:
                    free_reference = reference
                    break
                if partner not in visited_predictions:
                    visited_predictions.add(partner)
                    queue.append(partner)

        while free_reference != -1:
            prediction = parent_reference[free_reference]
            previous_reference = matched_prediction[prediction]
            matched_prediction[prediction] = free_reference
            matched_reference[free_reference] = prediction
            free_reference = previous_reference

    return sum(partner != -1 for partner in matched_prediction)


def _scores(tp: int, fp: int, fn: int) -> dict:
    # Use zero when the relevant denominator is zero, including an all-empty set.
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
    }


def _index_records(records, *, ground_truth: bool) -> dict:
    kind = "golden set" if ground_truth else "predictions"
    indexed = {}
    for position, record in enumerate(records, start=1):
        label = f"{kind}, record {position}"
        if not isinstance(record, dict):
            raise ValueError(f"{label} must be an object.")
        photo_id = record.get("photo_id")
        if type(photo_id) is not int:
            raise ValueError(f"{label}.photo_id must be an integer.")
        if photo_id in indexed:
            raise ValueError(f"Duplicate photo_id {photo_id} in {kind}.")
        error = record.get("error")
        if error is not None:
            if ground_truth:
                raise ValueError(f"Golden set photo {photo_id} has an unresolved error.")
            if not isinstance(error, str) or not error.strip():
                raise ValueError(f"{label}.error must be null or a nonempty string.")
            if "boxes" not in record or record["boxes"] not in (None, []):
                raise ValueError(f"Failed prediction {photo_id} must have null or empty boxes.")
        else:
            boxes = record.get("boxes")
            if not isinstance(boxes, list):
                raise ValueError(f"{label}.boxes must be a list.")
            for index, box in enumerate(boxes):
                validate_box(box, f"{label}.boxes[{index}]")
        indexed[photo_id] = record
    if not indexed:
        raise ValueError(f"The {kind} is empty.")
    return indexed


def read_jsonl(path: str | Path) -> list[dict]:
    """Read JSON objects and report malformed lines with their location."""
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except ValueError as error:
                raise ValueError(f"{path}, line {line_number}: {error}") from error
    return records


def evaluate(
    predictions: list[dict],
    references: list[dict],
    *,
    elapsed_seconds: float | None = None,
) -> dict:
    """Evaluate identical photo sets. Full-set scores require successful predictions.

    Both inputs use photo_id and normalized boxes. A reference can additionally
    carry image_url, which is copied to the per-image report. elapsed_seconds is
    the measured inference-run duration, supplied by the caller.
    """
    predicted = _index_records(predictions, ground_truth=False)
    expected = _index_records(references, ground_truth=True)
    missing = expected.keys() - predicted.keys()
    extra = predicted.keys() - expected.keys()
    if missing or extra:
        raise ValueError(
            f"Photo IDs differ. Missing predictions: {sorted(missing)}. "
            f"Unexpected predictions: {sorted(extra)}."
        )
    if elapsed_seconds is not None:
        _finite_number(elapsed_seconds, "elapsed_seconds")
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be nonnegative.")

    totals = {str(threshold): [0, 0, 0] for threshold in IOU_THRESHOLDS}
    per_image = []
    errors = []
    negative_images = 0
    negative_images_with_fp = 0
    false_boxes_on_negatives = 0

    for photo_id, reference in expected.items():
        prediction = predicted[photo_id]
        result = {"photo_id": photo_id}
        if "image_url" in reference:
            result["image_url"] = reference["image_url"]
        if prediction.get("error") is not None:
            errors.append({"photo_id": photo_id, "error": prediction["error"]})
            result.update(error=prediction["error"], metrics=None)
            per_image.append(result)
            continue

        predicted_boxes = prediction["boxes"]
        reference_boxes = reference["boxes"]
        overlaps = [
            [box_iou(box, truth) for truth in reference_boxes]
            for box in predicted_boxes
        ]
        image_metrics = {}
        for threshold in IOU_THRESHOLDS:
            key = str(threshold)
            tp = _maximum_matches(overlaps, threshold)
            fp = len(predicted_boxes) - tp
            fn = len(reference_boxes) - tp
            image_metrics[key] = _scores(tp, fp, fn)
            totals[key] = [a + b for a, b in zip(totals[key], (tp, fp, fn))]
        if not reference_boxes:
            negative_images += 1
            negative_images_with_fp += bool(predicted_boxes)
            false_boxes_on_negatives += len(predicted_boxes)
        result.update(error=None, metrics=image_metrics)
        per_image.append(result)

    # A failed image never becomes an empty successful prediction or disappears
    # silently from the full-set result. Per-image results remain diagnostic.
    quality = None
    if not errors:
        quality = {
            "by_iou": {key: _scores(*counts) for key, counts in totals.items()},
            "negative_images": negative_images,
            "negative_images_with_false_positives": negative_images_with_fp,
            "false_positive_boxes_on_negative_images": false_boxes_on_negatives,
            "negative_image_false_positive_rate": (
                negative_images_with_fp / negative_images if negative_images else None
            ),
        }
    return {
        "status": "incomplete" if errors else "complete",
        "images_total": len(expected),
        "images_succeeded": len(expected) - len(errors),
        "technical_errors": len(errors),
        "errors": errors,
        "elapsed_seconds": elapsed_seconds,
        "mean_seconds_per_image": (
            elapsed_seconds / len(expected) if elapsed_seconds is not None else None
        ),
        "quality": quality,
        "per_image": per_image,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--elapsed-seconds", type=float)
    args = parser.parse_args()
    try:
        report = evaluate(
            read_jsonl(args.predictions),
            read_jsonl(args.references),
            elapsed_seconds=args.elapsed_seconds,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
