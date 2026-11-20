"""Save a random sample of cached photos with reference and prediction boxes."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont

from inference.data import ROOT, load_image, read_dataset, sha256, write_json
from inference.metrics import read_jsonl, validate_box


GREEN = "#22ee77"
RED = "#ff5555"
HEADER_HEIGHT = 80


def font(size=18):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow 10.0 has a fixed-size default font.
        return ImageFont.load_default()


def render(image, reference, prediction, model, *, max_size=1200, predictions_only=False):
    """Draw normalized boxes after resizing, then add a legend outside the photo."""
    image = image.copy()
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    # A wider GT stroke remains visible beneath a coincident prediction stroke.
    reference_boxes = None if predictions_only else reference["boxes"]
    for boxes, color, prefix, width in (
        (reference_boxes or [], GREEN, "GT", 6),
        (prediction.get("boxes") or [], RED, "P", 2),
    ):
        for number, box in enumerate(boxes, 1):
            left = box["left"] * image.width
            top = box["top"] * image.height
            right = (box["left"] + box["width"]) * image.width
            bottom = (box["top"] + box["height"]) * image.height
            draw.rectangle((left, top, right, bottom), outline=color, width=width)
            label_y = top + 4 if prefix == "GT" else max(top, bottom - 24)
            draw.text((left + 4, label_y), f"{prefix}{number}", fill=color,
                      font=font(), stroke_width=1, stroke_fill="black")
    canvas = Image.new("RGB", (max(480, image.width), image.height + HEADER_HEIGHT), "#151820")
    canvas.paste(image, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"photo_id={reference['photo_id']} | {model}", fill="white", font=font())
    reference_count = "not provided" if reference_boxes is None else str(len(reference_boxes))
    draw.text((12, 32), f"GT: {reference_count}", fill=GREEN, font=font())
    predicted_count = "ERROR" if prediction.get("error") else str(len(prediction["boxes"]))
    draw.text((220, 32), f"Prediction: {predicted_count}", fill=RED, font=font())
    if prediction.get("error"):
        draw.text((12, 56), prediction["error"][:100], fill=RED, font=font(14))
    return canvas


def visualize(run, model, *, n=5, seed=None, output=None, dataset=None, cache_dir=None, max_size=1200,
              predictions_only=False):
    if type(n) is not int or n <= 0 or type(max_size) is not int or max_size <= 0:
        raise ValueError("n and max_size must be positive integers")
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if model not in manifest["identity"]["models"]:
        raise ValueError(f"Model {model!r} is absent from this run")
    dataset = Path(dataset or manifest["dataset_path"])
    if sha256(dataset) != manifest["identity"]["dataset_sha256"]:
        raise ValueError("Annotations differ from the dataset used for inference")
    references = {r["photo_id"]: r for r in read_dataset(dataset, require_annotations=not predictions_only)}
    ids = manifest["identity"]["photo_ids"]
    if n > len(ids):
        raise ValueError(f"Requested {n} images, but the run contains {len(ids)}")
    if len(set(ids)) != len(ids) or any(photo_id not in references for photo_id in ids):
        raise ValueError("Run IDs must be unique and present in the reference dataset")
    predictions = {}
    for record in read_jsonl(run / model / "predictions.jsonl"):
        photo_id = record.get("photo_id")
        if type(photo_id) is not int or photo_id in predictions or photo_id not in ids:
            raise ValueError(f"Invalid, duplicate or unexpected prediction ID: {photo_id}")
        error = record.get("error")
        if error is not None:
            if not isinstance(error, str) or not error.strip() or record.get("boxes") not in (None, []):
                raise ValueError(f"Invalid failed prediction: {photo_id}")
        else:
            if not isinstance(record.get("boxes"), list):
                raise ValueError(f"Prediction {photo_id} must contain a boxes list")
            for box in record["boxes"]:
                validate_box(box)
        predictions[photo_id] = record
    selected = random.Random(seed).sample(ids, n)
    cache_dir = Path(cache_dir or manifest["cache_dir"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output) if output else run / model / "visualizations" / stamp
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output folder is not empty; choose another --output")
    # Check every selected image before creating any output files.
    for photo_id in selected:
        load_image(references[photo_id], cache_dir, offline=True)
    output.mkdir(parents=True, exist_ok=True)
    samples, cards = [], []
    for photo_id in selected:
        reference = references[photo_id]
        prediction = predictions.get(photo_id, {"photo_id": photo_id, "boxes": None, "error": "Prediction not saved"})
        image, _ = load_image(reference, cache_dir, offline=True)
        annotated = render(image, reference, prediction, model, max_size=max_size, predictions_only=predictions_only)
        filename = f"{photo_id}.png"
        annotated.save(output / filename)
        samples.append({"photo_id": photo_id, "file": filename, "image_url": reference["image_url"],
                        "reference_boxes": None if predictions_only else reference["boxes"], "prediction": prediction})
        annotated.thumbnail((480, 400), Image.Resampling.LANCZOS)
        card = Image.new("RGB", (480, 430), "#151820")
        card.paste(annotated, (0, 30))
        ImageDraw.Draw(card).text((10, 5), f"photo_id={photo_id}", fill="white", font=font())
        cards.append(card)
    columns = min(2, n)
    sheet = Image.new("RGB", (columns * 480, math.ceil(n / columns) * 430 + 36), "#151820")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), "GT: not provided" if predictions_only else "GT: green", fill=GREEN, font=font())
    draw.text((230, 8), f"Prediction: red | {model}", fill=RED, font=font())
    for index, card in enumerate(cards):
        sheet.paste(card, (index % columns * 480, index // columns * 430 + 36))
    sheet.save(output / "contact_sheet.png")
    write_json(output / "sample.json", {"run": str(run.resolve()), "model": model, "seed": seed,
               "predictions_only": predictions_only,
               "dataset_sha256": manifest["identity"]["dataset_sha256"], "samples": samples})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/baseline")
    parser.add_argument("--model", required=True, help="grounding-dino, owlv2 or qwen3-vl")
    parser.add_argument("--n", "-n", type=int, default=5)
    parser.add_argument("--seed", type=int, help="Repeat the same random sample")
    parser.add_argument("--output", type=Path, help="New/empty output folder; default: under the selected run/model")
    parser.add_argument("--dataset", type=Path, help="Location override for the same annotation file")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-size", type=int, default=1200, help="Maximum photo edge in output pixels")
    parser.add_argument("--predictions-only", action="store_true", help="Allow unlabeled data and draw predictions without references")
    args = parser.parse_args()
    try:
        output = visualize(args.run, args.model, n=args.n, seed=args.seed, output=args.output,
                           dataset=args.dataset, cache_dir=args.cache_dir, max_size=args.max_size,
                           predictions_only=args.predictions_only)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Saved {args.n} images and contact_sheet.png to {output.resolve()}")


if __name__ == "__main__":
    main()
