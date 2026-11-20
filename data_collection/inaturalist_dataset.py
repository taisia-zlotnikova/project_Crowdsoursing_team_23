#!/usr/bin/env python3
"""Collect and validate a CC0 iNaturalist image dataset.

The script deliberately keeps collection, manual curation, validation, and
freezing of the final manifest as separate steps.  It never treats a failed
download as a photograph without a butterfly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from PIL import Image
except ImportError:  # A clear message is emitted only by the validation command.
    Image = None  # type: ignore[assignment]


API_URL = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "mipt-butterfly-dataset/1.0 (educational dataset collection)"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def api_get(params: dict[str, Any], timeout: int) -> tuple[dict[str, Any], str]:
    query_url = f"{API_URL}?{urlencode(params)}"
    request = Request(query_url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response), query_url


def original_url(square_url: str) -> str:
    """Turn iNaturalist's square rendition URL into its original rendition URL."""
    marker = "/square."
    if marker not in square_url:
        raise ValueError(f"Expected a square rendition URL, got {square_url!r}")
    return square_url.replace(marker, "/original.", 1)


def medium_url(square_url: str) -> str:
    """Make a reasonably small preview URL for the manual gallery."""
    marker = "/square."
    if marker not in square_url:
        raise ValueError(f"Expected a square rendition URL, got {square_url!r}")
    return square_url.replace(marker, "/medium.", 1)


def command_fetch(args: argparse.Namespace) -> None:
    if not 1 <= args.per_page <= 200:
        raise ValueError("--per-page must be from 1 to 200")

    records: list[dict[str, Any]] = []
    seen_photo_ids: set[int] = set()
    seen_queries: list[str] = []
    for page in range(args.page, args.page + args.pages):
        params: dict[str, Any] = {
            "taxon_id": args.taxon_id,
            "photos": "true",
            "photo_license": "cc0",
            "per_page": args.per_page,
            "page": page,
            "order_by": args.order_by,
            "order": args.order,
        }
        if args.d1:
            params["d1"] = args.d1
        if args.d2:
            params["d2"] = args.d2
        if args.without_taxon_id:
            params["without_taxon_id"] = args.without_taxon_id
        payload, query_url = api_get(
            params,
            args.timeout,
        )
        seen_queries.append(query_url)
        results = payload.get("results", [])
        for observation in results:
            photos_from_observation = 0
            for photo in observation.get("photos", []):
                # The server-side filter is helpful but this client-side check is mandatory.
                if str(photo.get("license_code", "")).casefold() != "cc0":
                    continue
                photo_id = photo.get("id")
                source_url = photo.get("url")
                if not isinstance(photo_id, int) or not isinstance(source_url, str):
                    continue
                if photo_id in seen_photo_ids:
                    continue
                seen_photo_ids.add(photo_id)
                try:
                    image_url = original_url(source_url)
                except ValueError:
                    # Keep malformed candidates out; they cannot be passed to annotation.
                    continue
                taxon = observation.get("taxon") or {}
                user = observation.get("user") or {}
                records.append(
                    {
                        "photo_id": photo_id,
                        "observation_id": observation.get("id"),
                        "observation_url": observation.get("uri"),
                        "image_url": image_url,
                        "source_rendition_url": source_url,
                        "photo_license": photo.get("license_code"),
                        "photo_attribution": photo.get("attribution"),
                        "photo_attribution_name": photo.get("attribution_name"),
                        "observation_user_login": user.get("login"),
                        "taxon_id": taxon.get("id"),
                        "taxon_name": taxon.get("name"),
                        "taxon_rank": taxon.get("rank"),
                        "species_guess": observation.get("species_guess"),
                        "observed_on": observation.get("observed_on"),
                        "sampling_group": args.sampling_group,
                        "api_dimensions": photo.get("original_dimensions"),
                        "api_query": query_url,
                        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                photos_from_observation += 1
                if args.max_candidates and len(records) >= args.max_candidates:
                    break
                if photos_from_observation >= args.max_photos_per_observation:
                    break
            if args.max_candidates and len(records) >= args.max_candidates:
                break
        if args.max_candidates and len(records) >= args.max_candidates:
            break
        if not results:
            break
    write_jsonl(args.output, records)
    print(f"Saved {len(records)} CC0 photo candidates to {args.output}")
    print(f"API pages requested: {len(seen_queries)}")


def command_gallery(args: argparse.Namespace) -> None:
    records = read_jsonl(args.input)
    cards: list[str] = []
    for item in records:
        photo_id = html.escape(str(item["photo_id"]))
        image_url = html.escape(str(item["image_url"]), quote=True)
        preview_url = html.escape(medium_url(str(item["source_rendition_url"])), quote=True)
        observation_url = html.escape(str(item.get("observation_url") or ""), quote=True)
        label = html.escape(
            f"{item.get('sampling_group', 'candidate')} · photo {item['photo_id']} · "
            f"observation {item.get('observation_id')} · "
            f"{item.get('taxon_name') or item.get('species_guess') or 'unknown taxon'}"
        )
        cards.append(
            f'<article><a href="{observation_url}" target="_blank" rel="noreferrer">{label}</a>'
            f'<br><a href="{image_url}" target="_blank" rel="noreferrer">'
            f'<img loading="lazy" src="{preview_url}" alt="{photo_id}"></a></article>'
        )
    document = """<!doctype html><meta charset="utf-8"><title>iNaturalist candidates</title>
<style>body{font-family:system-ui;margin:16px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}article{border:1px solid #ddd;padding:8px;overflow-wrap:anywhere}img{width:100%;max-height:300px;object-fit:contain;background:#f4f4f4}</style>
<h1>iNaturalist candidates</h1><p>Decide photo IDs manually, then put one selected ID per line in <code>selected_photo_ids.txt</code>.</p><main>""" + "\n".join(cards) + "</main>"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(f"Saved gallery with {len(records)} candidates to {args.output}")


def inspect_image(record: dict[str, Any], timeout: int) -> dict[str, Any]:
    result = dict(record)
    result.update({"download_ok": False, "duplicate_of_photo_id": None, "sha256": None})
    try:
        request = Request(record["image_url"], headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get_content_type()
            image_bytes = response.read()
        if not content_type.startswith("image/"):
            raise ValueError(f"unexpected Content-Type {content_type!r}")
        if not image_bytes:
            raise ValueError("empty response body")
        if Image is None:
            raise RuntimeError("Pillow is required: pip install -r requirements.txt")
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            result["actual_dimensions"] = {"width": image.width, "height": image.height}
            result["image_format"] = image.format
        result.update(
            {
                "download_ok": True,
                "http_content_type": content_type,
                "bytes": len(image_bytes),
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "validation_error": None,
            }
        )
    except (HTTPError, URLError, OSError, RuntimeError, ValueError) as error:
        result["validation_error"] = f"{type(error).__name__}: {error}"
    return result


def command_validate(args: argparse.Namespace) -> None:
    if Image is None:
        raise RuntimeError("Pillow is required for image decoding: pip install -r requirements.txt")
    records = read_jsonl(args.input)
    if args.selected_ids:
        selected_ids = {
            int(line.strip())
            for line in args.selected_ids.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        available_ids = {record["photo_id"] for record in records}
        missing_ids = selected_ids - available_ids
        if missing_ids:
            raise ValueError(f"Photo IDs are absent from input: {sorted(missing_ids)}")
        records = [record for record in records if record["photo_id"] in selected_ids]
    print(f"Checking {len(records)} original images with {args.workers} workers...", flush=True)
    checked: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(inspect_image, record, args.timeout): index for index, record in enumerate(records)}
        ordered: list[dict[str, Any] | None] = [None] * len(records)
        for completed, future in enumerate(as_completed(futures), 1):
            ordered[futures[future]] = future.result()
            if completed == len(records) or completed % 20 == 0:
                print(f"Checked {completed}/{len(records)}", flush=True)
    checked = [record for record in ordered if record is not None]
    hashes: dict[str, int] = {}
    for record in checked:
        digest = record.get("sha256")
        if record["download_ok"] and digest:
            if digest in hashes:
                record["duplicate_of_photo_id"] = hashes[digest]
            else:
                hashes[digest] = record["photo_id"]
    write_jsonl(args.output, checked)
    failed = sum(not record["download_ok"] for record in checked)
    duplicates = sum(record["duplicate_of_photo_id"] is not None for record in checked)
    report = {"checked": len(checked), "download_or_decode_failed": failed, "byte_duplicates": duplicates}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


def command_validate_all(args: argparse.Namespace) -> None:
    """Validate every record in the input manifest."""
    args.selected_ids = None
    command_validate(args)


def command_finalize(args: argparse.Namespace) -> None:
    wanted_ids = {int(line.strip()) for line in args.selected_ids.read_text(encoding="utf-8").splitlines() if line.strip()}
    records = read_jsonl(args.input)
    chosen = [
        record for record in records
        if record["photo_id"] in wanted_ids and record.get("download_ok") and not record.get("duplicate_of_photo_id")
    ]
    selected_ids = {record["photo_id"] for record in chosen}
    missing = wanted_ids - selected_ids
    if missing:
        raise ValueError(f"Selected IDs are invalid, failed validation, or duplicates: {sorted(missing)}")
    if args.expected_count is not None and len(chosen) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} final photos, got {len(chosen)}")
    chosen.sort(key=lambda record: record["photo_id"])
    write_jsonl(args.output, chosen)
    with args.csv_output.open("w", encoding="utf-8", newline="") as target:
        fields = ["photo_id", "observation_id", "observation_url", "image_url", "photo_license", "photo_attribution", "taxon_id", "taxon_name", "actual_dimensions", "sha256"]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record.get(field) for field in fields} for record in chosen)
    print(f"Frozen {len(chosen)} validated, non-duplicate photos in {args.output}")


def command_collect_plan(args: argparse.Namespace) -> None:
    """Collect the three fixed candidate pools used by this project."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pools = (
        ("butterflies", 47224, None, 500, "butterfly_candidate"),
        # Lepidoptera with Papilionoidea excluded = moth candidates, not butterflies.
        ("moths_and_cocoons", 47157, 47224, 100, "moth_or_cocoon_candidate"),
        ("flowers", 47125, None, 100, "flower_candidate"),
    )
    all_records: list[dict[str, Any]] = []
    seen_photo_ids: set[int] = set()
    for filename, taxon_id, without_taxon_id, count, sampling_group in pools:
        output = args.output_dir / f"{filename}.jsonl"
        fetch_args = argparse.Namespace(
            taxon_id=taxon_id,
            without_taxon_id=without_taxon_id,
            page=1,
            pages=3,
            per_page=200,
            max_candidates=count,
            max_photos_per_observation=1,
            sampling_group=sampling_group,
            d1=args.d1,
            d2=args.d2,
            order_by="random",
            order="desc",
            timeout=args.timeout,
            output=output,
        )
        command_fetch(fetch_args)
        pool_records = read_jsonl(output)
        if len(pool_records) != count:
            raise RuntimeError(
                f"{sampling_group}: requested {count} candidates, received {len(pool_records)}. "
                "Widen the --d1/--d2 date range and run collect-plan again."
            )
        for record in pool_records:
            if record["photo_id"] not in seen_photo_ids:
                seen_photo_ids.add(record["photo_id"])
                all_records.append(record)
    combined = args.output_dir / "all_candidates.jsonl"
    write_jsonl(combined, all_records)
    print(f"Saved {len(all_records)} total candidates to {combined}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="request CC0 observation photos from iNaturalist")
    fetch.add_argument("--taxon-id", type=int, default=47224)
    fetch.add_argument("--without-taxon-id", type=int, help="exclude this taxon and its descendants")
    fetch.add_argument("--page", type=int, default=1)
    fetch.add_argument("--pages", type=int, default=3)
    fetch.add_argument("--per-page", type=int, default=200)
    fetch.add_argument("--max-candidates", type=int)
    fetch.add_argument("--max-photos-per-observation", type=int, default=1)
    fetch.add_argument("--sampling-group", default="butterfly_candidate")
    fetch.add_argument("--d1", help="first iNaturalist observation date, YYYY-MM-DD")
    fetch.add_argument("--d2", help="last iNaturalist observation date, YYYY-MM-DD")
    fetch.add_argument("--order-by", default="created_at")
    fetch.add_argument("--order", choices=("asc", "desc"), default="desc")
    fetch.add_argument("--timeout", type=int, default=30)
    fetch.add_argument("--output", type=Path, default=Path("data/candidates.jsonl"))
    fetch.set_defaults(func=command_fetch)
    plan = commands.add_parser("collect-plan", help="collect 500 butterflies, 100 moth/cocoon, and 100 flower candidates")
    plan.add_argument("--output-dir", type=Path, default=Path("data/raw_candidates"))
    plan.add_argument("--d1", help="first iNaturalist observation date, YYYY-MM-DD")
    plan.add_argument("--d2", help="last iNaturalist observation date, YYYY-MM-DD")
    plan.add_argument("--timeout", type=int, default=30)
    plan.set_defaults(func=command_collect_plan)
    gallery = commands.add_parser("gallery", help="make a local HTML page for manual review")
    gallery.add_argument("--input", type=Path, default=Path("data/candidates.jsonl"))
    gallery.add_argument("--output", type=Path, default=Path("data/gallery.html"))
    gallery.set_defaults(func=command_gallery)
    validate = commands.add_parser("validate", help="download, decode, and deduplicate input photos")
    validate.add_argument("--input", type=Path, default=Path("data/candidates.jsonl"))
    validate.add_argument("--output", type=Path, default=Path("data/validated.jsonl"))
    validate.add_argument("--report", type=Path, default=Path("data/validation_report.json"))
    validate.add_argument("--workers", type=int, default=8)
    validate.add_argument("--timeout", type=int, default=30)
    validate.add_argument("--selected-ids", type=Path, help="validate only IDs listed in this file")
    validate.set_defaults(func=command_validate)
    validate_all = commands.add_parser("validate-all", help="download, decode, and deduplicate every photo in the input")
    validate_all.add_argument("--input", type=Path, default=Path("data/candidates.jsonl"))
    validate_all.add_argument("--output", type=Path, default=Path("data/all_validated.jsonl"))
    validate_all.add_argument("--report", type=Path, default=Path("data/all_validation_report.json"))
    validate_all.add_argument("--workers", type=int, default=8)
    validate_all.add_argument("--timeout", type=int, default=30)
    validate_all.set_defaults(func=command_validate_all)
    finalize = commands.add_parser("finalize", help="freeze the manually selected, validated dataset")
    finalize.add_argument("--input", type=Path, default=Path("data/validated.jsonl"))
    finalize.add_argument("--selected-ids", type=Path, default=Path("data/selected_photo_ids.txt"))
    finalize.add_argument("--output", type=Path, default=Path("data/final_dataset.jsonl"))
    finalize.add_argument("--csv-output", type=Path, default=Path("data/final_dataset.csv"))
    finalize.add_argument("--expected-count", type=int, default=400)
    finalize.set_defaults(func=command_finalize)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
