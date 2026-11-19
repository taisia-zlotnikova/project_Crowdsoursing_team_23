"""Read-only experiment dashboard: tables, error filters and bounding-box overlays."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from PIL import ImageDraw
import streamlit as st

from inference.data import load_image, read_dataset, sha256
from inference.evaluate import comparison_row
from inference.metrics import read_jsonl


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_reports(folder):
    manifest = read_json(folder / "manifest.json")
    reports = {}
    for name in manifest["identity"]["models"]:
        path = folder / name / "metrics.json"
        predictions = folder / name / "predictions.jsonl"
        if path.exists() and (not predictions.exists() or path.stat().st_mtime >= predictions.stat().st_mtime):
            reports[name] = read_json(path)
    return manifest, reports


def draw_boxes(image, boxes, color, prefix):
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(image.size) / 250))
    for number, box in enumerate(boxes or [], 1):
        left, top = box["left"] * image.width, box["top"] * image.height
        right = left + box["width"] * image.width
        bottom = top + box["height"] * image.height
        draw.rectangle((left, top, right, bottom), outline=color, width=line_width)
        draw.text((left + line_width, top + line_width), f"{prefix}{number}", fill=color,
                  stroke_width=1, stroke_fill="black")
    return image


def overview(folder, manifest, reports):
    st.subheader("Сравнение моделей")
    rows = [comparison_row(name, report) for name, report in reports.items()]
    if not rows:
        st.info("Метрики ещё не рассчитаны. После инференса запустите inference.evaluate.")
        return
    frame = pd.DataFrame(rows)
    columns = ["model", "status", "images", "f1@0.5", "precision@0.5", "recall@0.5", "f1@0.75",
               "tp@0.5", "fp@0.5", "fn@0.5", "technical_errors", "mean_inference_seconds"]
    st.dataframe(frame[columns], hide_index=True, width="stretch")
    complete = [row for row in rows if row["f1@0.5"] is not None]
    if complete:
        best = sorted(complete, key=lambda r: (-r["f1@0.5"], r["mean_inference_seconds"] if r["mean_inference_seconds"] is not None else float("inf")))[0]
        st.success(f"Лучший результат среди завершённых: {best['model']}, F1@0.5 = {best['f1@0.5']:.4f}")
        chart = pd.DataFrame(complete).set_index("model")[["f1@0.5", "precision@0.5", "recall@0.5", "f1@0.75"]]
        st.bar_chart(chart)
    st.caption("Метрики суммируются по объектам. При технических ошибках итоговое качество недоступно. Время — секунды на изображение без загрузки весов и скачивания фото.")
    st.download_button("Скачать все показатели CSV", data=frame.to_csv(index=False).encode("utf-8"),
                       file_name=f"{folder.name}-comparison.csv", mime="text/csv")
    with st.expander("Все показатели, включая IoU 0.75 и время"):
        st.dataframe(frame, hide_index=True, width="stretch")
    with st.expander("Настройки эксперимента"):
        st.json(manifest)


def photos(folder, manifest, reports):
    st.subheader("Фотографии и ошибки")
    name = st.selectbox("Модель", manifest["identity"]["models"], key="photo_model")
    prediction_path = folder / name / "predictions.jsonl"
    predictions = {row["photo_id"]: row for row in read_jsonl(prediction_path)} if prediction_path.exists() else {}
    default_dataset = manifest["dataset_path"]
    dataset_path = Path(st.text_input("Файл разметки", default_dataset))
    try:
        if sha256(dataset_path) != manifest["identity"]["dataset_sha256"]:
            st.error("Разметка отличается от использованной при запуске.")
            return
        references = {row["photo_id"]: row for row in read_dataset(dataset_path, require_annotations=True)}
    except (OSError, ValueError) as error:
        st.error(str(error))
        return
    threshold = st.selectbox("Порог IoU для фильтров", ["0.5", "0.75"])
    report = reports.get(name, {})
    per_image = {row["photo_id"]: row for row in report.get("per_image", [])}
    records = []
    for photo_id in manifest["identity"]["photo_ids"]:
        prediction = predictions.get(photo_id, {"error": "Предсказание не сохранено", "boxes": None})
        metric = (per_image.get(photo_id, {}).get("metrics") or {}).get(threshold, {})
        records.append({"photo_id": photo_id, "reference_boxes": len(references[photo_id]["boxes"]),
                        "predicted_boxes": len(prediction.get("boxes") or []),
                        "tp": metric.get("tp"), "fp": metric.get("fp"), "fn": metric.get("fn"),
                        "error": prediction.get("error")})
    filters = ["Все", "Пропуски (FN)", "Лишние рамки (FP)", "Пустые фото с FP", "Технические ошибки", "Без ошибок"]
    chosen = st.selectbox("Фильтр", filters)
    filtered = []
    for row in records:
        keep = (chosen == "Все" or
                chosen == "Пропуски (FN)" and (row["fn"] or 0) > 0 or
                chosen == "Лишние рамки (FP)" and (row["fp"] or 0) > 0 or
                chosen == "Пустые фото с FP" and row["reference_boxes"] == 0 and (row["fp"] or 0) > 0 or
                chosen == "Технические ошибки" and bool(row["error"]) or
                chosen == "Без ошибок" and not row["error"] and row["fp"] == 0 and row["fn"] == 0)
        if keep:
            filtered.append(row)
    st.dataframe(pd.DataFrame(filtered), hide_index=True, width="stretch")
    if not filtered:
        st.info("Нет фотографий, подходящих под фильтр.")
        return
    photo_id = st.selectbox("Фото", [r["photo_id"] for r in filtered])
    reference = references[photo_id]
    prediction = predictions.get(photo_id, {"error": "Предсказание не сохранено", "boxes": None})
    if prediction.get("error"):
        st.error(prediction["error"])
    cache_dir = Path(st.text_input("Кеш изображений", manifest["cache_dir"]))
    try:
        image, _ = load_image(reference, cache_dir, offline=True)
    except (OSError, ValueError) as error:
        st.warning(f"Изображение отсутствует в проверенном кеше: {error}")
    else:
        overlay = st.checkbox("Рамки на одном изображении", value=True)
        if overlay:
            image = draw_boxes(image, reference["boxes"], "#22ee77", "GT ")
            image = draw_boxes(image, prediction.get("boxes"), "#ff5555", "P ")
            st.image(image, caption=f"Фото {photo_id}: зелёные — эталон, красные — {name}", width="stretch")
        else:
            left, right = st.columns(2)
            left.image(draw_boxes(image.copy(), reference["boxes"], "#22ee77", "GT "), caption="Эталон", width="stretch")
            right.image(draw_boxes(image.copy(), prediction.get("boxes"), "#ff5555", "P "), caption=name, width="stretch")
    st.link_button("Открыть исходное изображение", reference["image_url"])
    diagnostic_path = folder / name / "diagnostics.jsonl"
    detail = next((row for row in read_jsonl(diagnostic_path) if row["photo_id"] == photo_id), {}) if diagnostic_path.exists() else {}
    with st.expander("Рамки, метрики и ответ модели"):
        st.json({"reference": reference["boxes"], "prediction": prediction,
                 "metrics": per_image.get(photo_id), "diagnostics": detail})


def groups(reports):
    st.subheader("Группы и ложные срабатывания")
    rows, negatives = [], []
    for name, report in reports.items():
        for group, result in report.get("groups", {}).items():
            quality = result["quality"]
            score = quality["by_iou"]["0.5"] if quality else {}
            rows.append(dict(model=name, group=group, images=result["images_total"], status=result["status"], **score))
        quality = report["quality"]
        if quality:
            negatives.append({"model": name, **{k: v for k, v in quality.items() if k != "by_iou"}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Группы: 0, 1 и несколько эталонных объектов. F1 пустой группы равен 0 по принятому соглашению; её качество оценивайте по ложным находкам ниже.")
    st.dataframe(pd.DataFrame(negatives), hide_index=True, width="stretch")


def experiments(folder, manifest, reports, folders):
    st.subheader("Сравнение экспериментов")
    others = [path for path in folders if path != folder]
    if not others:
        st.info("Для сравнения нужен ещё один сохранённый эксперимент.")
        return
    other = st.selectbox("Сравнить с", others, format_func=lambda p: p.name)
    other_manifest, other_reports = load_reports(other)
    keys = ("dataset_sha256", "photo_ids")
    if any(manifest["identity"][key] != other_manifest["identity"][key] for key in keys):
        st.warning("Наборы фотографий или разметка различаются. Разницу качества напрямую не сравниваем.")
        return
    rows = []
    for name in reports.keys() & other_reports.keys():
        current, previous = comparison_row(name, reports[name]), comparison_row(name, other_reports[name])
        for metric in ("f1@0.5", "precision@0.5", "recall@0.5", "f1@0.75", "mean_inference_seconds"):
            a, b = current[metric], previous[metric]
            rows.append({"model": name, "metric": metric, folder.name: a, other.name: b,
                         "delta_current_minus_other": a - b if a is not None and b is not None else None})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        st.write(folder.name)
        st.json(manifest["identity"]["config"])
    with right:
        st.write(other.name)
        st.json(other_manifest["identity"]["config"])


def main():
    st.set_page_config(page_title="Butterfly evaluation", layout="wide")
    st.title("Оценка детекции бабочек")
    root = Path(st.sidebar.text_input("Папка экспериментов", os.environ.get("BUTTERFLY_RUNS_DIR", str(ROOT / "runs"))))
    folders = sorted((p.parent for p in root.glob("*/manifest.json")),
                     key=lambda p: (p / "manifest.json").stat().st_mtime, reverse=True)
    if not folders:
        st.info("Сохранённых запусков пока нет. Выполните inference.run, затем inference.evaluate.")
        return
    folder = st.sidebar.selectbox("Эксперимент", folders, format_func=lambda p: p.name)
    st.sidebar.button("Обновить результаты")
    try:
        manifest, reports = load_reports(folder)
        st.caption(f"{folder.name} · {manifest['selected_images']} из {manifest['dataset_total']} изображений · {manifest['identity']['device']}")
        if manifest["scope"] != "full":
            st.warning("Пробный запуск на части датасета. Эти метрики не являются итоговой оценкой на 400 фото.")
        if any((folder / name / "predictions.jsonl").exists() and
               (not (folder / name / "metrics.json").exists() or
                (folder / name / "predictions.jsonl").stat().st_mtime > (folder / name / "metrics.json").stat().st_mtime)
               for name in manifest["identity"]["models"]):
            st.warning("Предсказания обновились. Пересчитайте отчёты командой inference.evaluate.")
        tabs = st.tabs(["Модели", "Фотографии", "Группы", "Эксперименты"])
        with tabs[0]:
            overview(folder, manifest, reports)
        with tabs[1]:
            photos(folder, manifest, reports)
        with tabs[2]:
            groups(reports)
        with tabs[3]:
            experiments(folder, manifest, reports, folders)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Не удалось прочитать эксперимент: {error}")


if __name__ == "__main__":
    main()
