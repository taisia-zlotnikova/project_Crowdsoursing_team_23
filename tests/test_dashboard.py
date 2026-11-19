"""Exercise dashboard rendering, filters and experiment comparison without models."""

from io import BytesIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from inference.data import ROOT, load_image, sha256, write_json, write_jsonl
from inference.evaluate import evaluate_run


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.environment = patch.dict(os.environ, BUTTERFLY_RUNS_DIR=str(self.root / "runs"))
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def fixture(self, name="baseline"):
        box = {"left": .1, "top": .1, "width": .5, "height": .5}
        rows = [{"photo_id": 1, "image_url": "https://example.org/1.jpg", "boxes": [box]},
                {"photo_id": 2, "image_url": "https://example.org/2.jpg", "boxes": []}]
        dataset = self.root / "annotations.jsonl"
        write_jsonl(dataset, rows)
        folder = self.root / "runs" / name
        write_json(folder / "manifest.json", {
            "identity": {"dataset_sha256": sha256(dataset), "photo_ids": [1, 2], "models": ["owlv2"],
                         "device": "cpu", "config": {"experiment": name}},
            "scope": "full", "selected_images": 2, "dataset_total": 2,
            "dataset_path": str(dataset), "cache_dir": str(self.root / "cache"),
        })
        write_jsonl(folder / "owlv2/predictions.jsonl", [
            {"photo_id": row["photo_id"], "boxes": [box], "error": None} for row in rows])
        content = BytesIO()
        Image.new("RGB", (32, 32), color="white").save(content, format="PNG")
        for row in rows:
            with patch("inference.data.urlopen", return_value=BytesIO(content.getvalue())):
                load_image(row, self.root / "cache")
        evaluate_run(folder)
        return folder

    def app(self):
        return AppTest.from_file(str(ROOT / "dashboard/app.py")).run(timeout=30)

    def test_empty_dashboard(self):
        app = self.app()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("пока нет" in message.value for message in app.info))

    def test_filters_boxes_and_experiment_comparison(self):
        self.fixture()
        self.fixture("prompt-change")
        app = self.app()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 4)
        filter_widget = next(widget for widget in app.selectbox if widget.label == "Фильтр")
        filter_widget.select("Пустые фото с FP").run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        photo = next(widget for widget in app.selectbox if widget.label == "Фото")
        self.assertEqual(photo.value, 2)
        next(widget for widget in app.checkbox if widget.label == "Рамки на одном изображении").uncheck().run(timeout=30)
        self.assertEqual(len(app.exception), 0)

    def test_incomplete_run_shows_errors_without_winner(self):
        folder = self.fixture()
        write_jsonl(folder / "owlv2/predictions.jsonl", [
            {"photo_id": 1, "boxes": None, "error": "Invalid Qwen response"}])
        evaluate_run(folder)
        app = self.app()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.success), 0)
        self.assertTrue(any("Invalid Qwen" in error.value for error in app.error))

    def test_updated_predictions_hide_old_quality_until_reevaluation(self):
        folder = self.fixture()
        path = folder / "owlv2/predictions.jsonl"
        write_jsonl(path, [{"photo_id": 1, "boxes": None, "error": "changed result"}])
        # Explicit timestamps avoid filesystem timestamp granularity assumptions.
        report_time = (folder / "owlv2/metrics.json").stat().st_mtime
        os.utime(path, (report_time + 1, report_time + 1))
        app = self.app()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.success), 0)
        self.assertTrue(any("Пересчитайте" in warning.value for warning in app.warning))


if __name__ == "__main__":
    unittest.main()
