"""Coordinate conversion and class-agnostic NMS for the single target class."""

import math

from inference.metrics import box_iou, validate_box


def normalized_box(values, width, height, *, clip=False):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("Expected [xmin, ymin, xmax, ymax]")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("Box coordinates must be finite numbers")
    x1, y1, x2, y2 = [v / scale for v, scale in zip(values, (width, height, width, height))]
    if clip:
        x1, y1, x2, y2 = [min(1.0, max(0.0, v)) for v in (x1, y1, x2, y2)]
    box = {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}
    validate_box(box)
    return box


def suppress(boxes, scores, threshold):
    if threshold is None:
        return list(range(len(boxes)))
    remaining = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    kept = []
    while remaining:
        current, *remaining = remaining
        kept.append(current)
        remaining = [i for i in remaining if box_iou(boxes[current], boxes[i]) <= threshold]
    return kept


def detector_result(result, image, settings):
    raw_boxes = result["boxes"].detach().cpu().tolist()
    raw_scores = result["scores"].detach().cpu().tolist()
    labels = result.get("text_labels", result.get("labels", []))
    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    boxes, scores, dropped = [], [], []
    for index, (coordinates, score) in enumerate(zip(raw_boxes, raw_scores)):
        try:
            box = normalized_box(coordinates, image.width, image.height, clip=True)
        except ValueError as error:
            dropped.append({"index": index, "reason": str(error)})
            continue
        boxes.append(box)
        scores.append(score)
    kept = suppress(boxes, scores, settings.get("nms_iou"))
    return {"boxes": [boxes[i] for i in kept], "diagnostics": {
        "raw_boxes_xyxy": raw_boxes, "raw_scores": raw_scores, "raw_labels": labels,
        "dropped_boxes": dropped, "kept_indices_after_validation": kept,
        "scores": [scores[i] for i in kept],
    }}
