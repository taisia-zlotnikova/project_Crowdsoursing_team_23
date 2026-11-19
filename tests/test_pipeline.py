"""Deterministic checks for matching, dataset integrity, failures and resuming."""

import argparse
import csv
from io import BytesIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from inference.adapters.common import normalized_box, suppress
from inference.adapters.qwen3_vl import parse_boxes
from inference.data import load_image, read_dataset, sha256, write_json, write_jsonl
from inference.evaluate import evaluate_run
from inference.metrics import evaluate, read_jsonl
from inference.run import run_model, validate_config


BOX = {"left": .1, "top": .2, "width": .3, "height": .4}


class MetricsTests(unittest.TestCase):
    def test_augmenting_path_beats_greedy_and_counts_empty_images(self):
        reference = [{"photo_id": 1, "boxes": [
            {"left": 0, "top": 0, "width": .5, "height": .5},
            {"left": .25, "top": 0, "width": .5, "height": .5}]},
            {"photo_id": 2, "boxes": []}, {"photo_id": 3, "boxes": [BOX]},
            {"photo_id": 4, "boxes": []}]
        prediction = [{"photo_id": 1, "boxes": [
            {"left": .125, "top": 0, "width": .5, "height": .5},
            {"left": 0, "top": 0, "width": .375, "height": .5}]},
            {"photo_id": 2, "boxes": [BOX]}, {"photo_id": 3, "boxes": []},
            {"photo_id": 4, "boxes": []}]
        quality = evaluate(prediction, reference)["quality"]
        self.assertEqual(quality["by_iou"]["0.5"]["tp"], 2)
        self.assertAlmostEqual(quality["by_iou"]["0.5"]["f1"], 2 / 3)
        self.assertAlmostEqual(quality["by_iou"]["0.75"]["f1"], 1 / 3)
        self.assertEqual(quality["negative_image_false_positive_rate"], .5)

    def test_failure_is_not_an_empty_detection(self):
        report = evaluate([{"photo_id": 1, "boxes": None, "error": "timeout"}],
                          [{"photo_id": 1, "boxes": []}])
        self.assertEqual(report["technical_errors"], 1)
        self.assertIsNone(report["quality"])

    def test_duplicate_missing_and_nonfinite_rejected(self):
        good = {"photo_id": 1, "boxes": [BOX]}
        for predictions in ([good, good], [{"photo_id": 2, "boxes": []}],
                            [{"photo_id": 1, "boxes": [dict(BOX, left=float("nan"))]}]):
            with self.assertRaises(ValueError):
                evaluate(predictions, [good])


class AdapterTests(unittest.TestCase):
    def test_qwen_coordinates_json_and_empty_result(self):
        result = parse_boxes('```json\n[{"bbox_2d":[100,200,400,600]}]\n```')
        self.assertEqual(len(result), 1)
        for key, value in BOX.items():
            self.assertAlmostEqual(result[0][key], value)
        self.assertEqual(len(parse_boxes('[{"bbox_2d":[0,0,100,100],"label":"butterfly"}]')), 1)
        self.assertEqual(parse_boxes("[]"), [])
        for text in ('Here are boxes: []', '{"boxes": []}', '[{"bbox_2d":[0,0,1001,1000]}]',
                     '[{"bbox_2d":[0,0,0,10]}]', '[{"bbox_2d":[true,0,10,10]}]',
                     '[{"bbox_2d":[NaN,0,10,10]}]', '[{"bbox_2d":[0,0,10,10],"score":1}]'):
            with self.assertRaises(ValueError, msg=text):
                parse_boxes(text)

    def test_clip_and_class_agnostic_nms(self):
        clipped = normalized_box([-10, -5, 200, 100], 200, 100, clip=True)
        self.assertEqual(clipped, {"left": 0., "top": 0., "width": 1., "height": 1.})
        self.assertEqual(suppress([BOX, BOX], [.4, .9], .5), [1])
        self.assertEqual(suppress([BOX, BOX], [.4, .9], None), [0, 1])

    def test_config_rejects_invalid_threshold(self):
        with self.assertRaises(ValueError):
            validate_config({"seed": 42, "models": {"owlv2": {"queries": ["moth"], "confidence": float("nan"), "nms_iou": .5}}}, ["owlv2"])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rows = [{"photo_id": 1, "image_url": "https://example.org/1.jpg", "boxes": [BOX]},
                     {"photo_id": 2, "image_url": "https://example.org/2.jpg", "boxes": []}]

    def tearDown(self):
        self.temporary.cleanup()

    def test_csv_bom_status_and_duplicates(self):
        path = self.root / "annotations.csv"
        columns = ["photo_id", "image_url", "boxes_normalized_xywh", "status"]
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerow({"photo_id": 1, "image_url": "https://example.org/1.jpg", "boxes_normalized_xywh": "[]", "status": "NO_BUTTERFLY"})
        self.assertEqual(read_dataset(path, require_annotations=True)[0]["boxes"], [])
        with path.open("a") as stream:
            stream.write('1,https://example.org/2.jpg,[],NO_BUTTERFLY\n')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_dataset(path)

    def test_unlabeled_photo_url_is_normalized_without_fake_reference(self):
        path = self.root / "final.jsonl"
        row = {"photo_id": 18821, "photo_url": "https://example.org/18821.jpg"}
        write_jsonl(path, [row])
        result = read_dataset(path)[0]
        self.assertEqual(result["image_url"], row["photo_url"])
        self.assertNotIn("boxes", result)
        with self.assertRaisesRegex(ValueError, "Missing reference boxes"):
            read_dataset(path, require_annotations=True)
        write_jsonl(path, [dict(row, image_url="https://example.org/other.jpg")])
        with self.assertRaisesRegex(ValueError, "different images"):
            read_dataset(path)

    def test_cache_geometry_hash_and_offline(self):
        content = BytesIO()
        Image.new("RGB", (20, 10)).save(content, format="PNG")
        row = dict(self.rows[0], width=20, height=10)
        with patch("inference.data.urlopen", return_value=BytesIO(content.getvalue())) as request:
            image, info = load_image(row, self.root / "cache")
            self.assertFalse(info["cached"])
            self.assertEqual(image.size, (20, 10))
            _, cached = load_image(row, self.root / "cache", offline=True)
            self.assertTrue(cached["cached"])
            self.assertEqual(request.call_count, 1)
        with self.assertRaisesRegex(ValueError, "annotation says"):
            load_image(dict(row, width=21), self.root / "cache", offline=True)
        (self.root / "cache/1.jpg").write_bytes(b"broken")
        with self.assertRaises(FileNotFoundError):
            load_image(row, self.root / "cache", offline=True)

    def create_run(self):
        dataset = self.root / "annotations.jsonl"
        write_jsonl(dataset, self.rows)
        run = self.root / "runs/example"
        write_json(run / "manifest.json", {
            "identity": {"dataset_sha256": sha256(dataset), "photo_ids": [1, 2], "models": ["owlv2"],
                         "device": "cpu", "config": {}},
            "dataset_path": str(dataset), "cache_dir": str(self.root / "cache"),
            "scope": "full", "selected_images": 2, "dataset_total": 2,
        })
        return dataset, run

    def test_evaluation_missing_recovery_and_dataset_identity(self):
        dataset, run = self.create_run()
        write_jsonl(run / "owlv2/predictions.jsonl", [{"photo_id": 1, "boxes": [BOX], "error": None}])
        report = evaluate_run(run)["owlv2"]
        self.assertEqual(report["missing_predictions"], 1)
        self.assertIsNone(report["quality"])
        self.assertEqual(report["groups"]["one_object"]["quality"]["by_iou"]["0.5"]["f1"], 1.)
        write_jsonl(run / "owlv2/predictions.jsonl", [dict(r, error=None) for r in self.rows])
        report = evaluate_run(run)["owlv2"]
        self.assertEqual(report["quality"]["by_iou"]["0.5"]["f1"], 1.)
        self.assertTrue((run / "comparison.csv").exists())
        dataset.write_text(dataset.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Dataset changed"):
            evaluate_run(run)

    def test_resume_keeps_success_and_retries_failed_photo(self):
        output = self.root / "output"
        output.mkdir()
        args = argparse.Namespace(seed=42, device="cpu", local_files_only=True, retries=0,
                                  cache_dir=self.root / "cache", offline_images=True)
        torch = SimpleNamespace(manual_seed=lambda seed: None)
        model = SimpleNamespace(dtype="float32", device="cpu")
        image = Image.new("RGB", (20, 10))
        image_info = {"cached": True, "image_seconds": .001, "width": 20, "height": 10}
        with patch("inference.loader.load_model", return_value=(model, None)), \
             patch("inference.run.load_image", return_value=(image, image_info)), \
             patch("inference.adapters.get_predictor") as get_predictor:
            first = get_predictor.return_value
            first.side_effect = [{"boxes": [BOX], "diagnostics": {}}, ValueError("bad JSON")]
            run_model("owlv2", self.rows, {}, output, args, torch)
            records = read_jsonl(output / "owlv2/predictions.jsonl")
            self.assertIsNone(records[0]["error"])
            self.assertIn("bad JSON", records[1]["error"])
            first.reset_mock(side_effect=True)
            first.return_value = {"boxes": [], "diagnostics": {}}
            run_model("owlv2", self.rows, {}, output, args, torch)
            self.assertEqual(first.call_count, 1)
            self.assertEqual(len(read_jsonl(output / "owlv2/diagnostics.jsonl")[1]["attempts"]), 2)
            self.assertEqual(len(read_jsonl(output / "owlv2/predictions.jsonl")), 2)
            with patch("inference.loader.load_model") as load:
                run_model("owlv2", self.rows, {}, output, args, torch)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
