#!/usr/bin/env python3
"""Recheck CC0 licenses of the selected photos and build the standard gallery."""

import argparse
from pathlib import Path

from inaturalist_dataset import api_get, command_gallery, read_jsonl


def main():
    root = Path(__file__).resolve().parents[1] / "data_special"
    records = read_jsonl(root / "all_validated.jsonl")
    ids = sorted({r["observation_id"] for r in records})
    payload, _ = api_get({"id": ",".join(map(str, ids)), "per_page": 200}, timeout=60)
    observations = {o["id"]: o for o in payload["results"]}
    for record in records:
        observation = observations.get(record["observation_id"], {})
        photo = next((p for p in observation.get("photos", [])
                      if p["id"] == record["photo_id"]), {})
        if photo.get("license_code") != "cc0":
            raise RuntimeError(f"Photo {record['photo_id']} is missing or not CC0")
    print(f"Confirmed CC0 licenses: {len(records)}/{len(records)}")
    command_gallery(argparse.Namespace(input=root / "all_validated.jsonl",
                                       output=root / "validated_gallery.html"))


if __name__ == "__main__":
    main()
