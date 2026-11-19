"""Read dataset manifests and cache original image bytes without changing geometry."""

import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image

from inference.metrics import read_jsonl, validate_box


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_dataset(path, *, require_annotations=False):
    """Accept annotated CSV or image-only CSV/JSONL with image_url or photo_url."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = read_jsonl(path)
    records, seen = [], set()
    for number, row in enumerate(rows, 2 if path.suffix.lower() == ".csv" else 1):
        try:
            raw_id = row["photo_id"]
            if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
                raise ValueError("photo_id must be an integer")
            photo_id = int(raw_id)
            if photo_id < 0 or photo_id in seen:
                raise ValueError(f"Invalid or duplicate photo_id: {photo_id}")
            if row.get("image_url") and row.get("photo_url") and row["image_url"] != row["photo_url"]:
                raise ValueError("image_url and photo_url refer to different images")
            url = row.get("image_url") or row.get("photo_url")
            if not isinstance(url, str) or urlparse(url).scheme not in ("https", "http") or not urlparse(url).netloc:
                raise ValueError("image_url must be an HTTP(S) URL")
            record = dict(row, photo_id=photo_id, image_url=url)
            for key in ("width", "height"):
                if row.get(key) not in (None, ""):
                    record[key] = int(row[key])
                    if record[key] <= 0:
                        raise ValueError(f"{key} must be positive")
            if "boxes_normalized_xywh" in row:
                coordinates = json.loads(row["boxes_normalized_xywh"])
                if not isinstance(coordinates, list):
                    raise ValueError("boxes_normalized_xywh must be a list")
                boxes = []
                for values in coordinates:
                    if not isinstance(values, list) or len(values) != 4:
                        raise ValueError("Each box must contain four coordinates")
                    boxes.append(dict(zip(("left", "top", "width", "height"), values)))
                record["boxes"] = boxes
            if "boxes" in record:
                if not isinstance(record["boxes"], list):
                    raise ValueError("boxes must be a list")
                for box in record["boxes"]:
                    validate_box(box)
                status = record.get("status")
                if status and (status not in ("HAS_BUTTERFLY", "NO_BUTTERFLY") or
                               (status == "HAS_BUTTERFLY") != bool(record["boxes"])):
                    raise ValueError("status contradicts the annotation")
            elif require_annotations:
                raise ValueError("Missing reference boxes")
            seen.add(photo_id)
            records.append(record)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}, row {number}: {error}") from error
    if not records:
        raise ValueError(f"Empty dataset: {path}")
    return records


def image_path(cache_dir, photo_id):
    return Path(cache_dir) / f"{int(photo_id)}.jpg"


def _decode(content):
    # Do not EXIF-rotate: annotations use the source image's coordinate system.
    with Image.open(BytesIO(content)) as source:
        source.load()
        return source.convert("RGB")


def load_image(record, cache_dir, *, offline=False):
    """Return RGB image plus cache/download details; validate declared dimensions."""
    started = perf_counter()
    path = image_path(cache_dir, record["photo_id"])
    metadata = path.with_suffix(".json")
    cached = False
    if path.exists() and metadata.exists():
        info = json.loads(metadata.read_text(encoding="utf-8"))
        cached = info.get("image_url") == record["image_url"] and info.get("sha256") == sha256(path)
    if cached:
        content = path.read_bytes()
        image = _decode(content)
    else:
        if offline:
            raise FileNotFoundError(f"No valid cached image for {record['photo_id']}")
        request = Request(record["image_url"], headers={"User-Agent": "Team23-evaluation/1.0"})
        with urlopen(request, timeout=30) as response:
            content = response.read(50 * 1024 * 1024 + 1)
        if len(content) > 50 * 1024 * 1024:
            raise ValueError("Image exceeds 50 MiB")
        image = _decode(content)
    for key, actual in (("width", image.width), ("height", image.height)):
        if record.get(key) and record[key] != actual:
            raise ValueError(f"Image {record['photo_id']}: {key}={actual}, annotation says {record[key]}")
    digest = hashlib.sha256(content).hexdigest()
    if not cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        temporary.write_bytes(content)
        temporary.replace(path)
        write_json(metadata, {"image_url": record["image_url"], "sha256": digest})
    return image, {"cached": cached, "image_seconds": perf_counter() - started,
                   "image_sha256": digest, "width": image.width, "height": image.height}
