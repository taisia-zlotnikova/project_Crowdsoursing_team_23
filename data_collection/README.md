# Сбор фотографий iNaturalist

`inaturalist_dataset.py` собирает ссылки на CC0-фотографии, создаёт HTML-галерею,
проверяет доступность и декодирование оригиналов и фиксирует выбранный набор.
Сбор кандидатов, визуальный отбор и итоговая разметка выполняются отдельно.

## Установка

Команды выполняются из корня репозитория в общем окружении проекта.
Для проверки изображений требуется Pillow из корневого `requirements.txt`.

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python data_collection/inaturalist_dataset.py --help
```

## Источник данных

Используется публичный метод
[`GET /v1/observations`](https://api.inaturalist.org/v1/docs/#!/Observations/get_observations)
без авторизации. Одно наблюдение может содержать несколько фотографий;
по умолчанию выбирается не более одной фотографии на наблюдение внутри запроса.

API-запрос включает `photos=true` и `photo_license=cc0`. Скрипт дополнительно
проверяет `photos[].license_code`, исключает повторяющиеся `photo_id` и строит
ссылку на оригинал из ссылки на вариант `square`.
Формат публичных ссылок описан в [iNaturalist Open Data](https://github.com/inaturalist/inaturalist-open-data).

Для фотографии сохраняются ID фотографии и наблюдения, ссылки, лицензия,
атрибуция, таксон, дата наблюдения, группа отбора и параметры API-запроса.
Проверка оригинала добавляет фактические размеры, формат, объём и SHA-256.

## Сбор кандидатов

План `collect-plan` формирует три группы:

| Группа | Количество | Параметры запроса |
| --- | ---: | --- |
| Papilionoidea | 500 | `taxon_id=47224` |
| Lepidoptera без Papilionoidea | 100 | `taxon_id=47157`, `without_taxon_id=47224` |
| Angiospermae | 100 | `taxon_id=47125` |

Группа отбора не является эталонной меткой наличия взрослой бабочки.
Итоговый статус фотографии определяется при разметке и независимом REVIEW.

```bash
.venv/bin/python data_collection/inaturalist_dataset.py collect-plan --d1 2018-01-01 --d2 2026-11-09 --output-dir data/raw_candidates_2018_2026
```

В папке вывода создаются файлы отдельных групп и общий `all_candidates.jsonl`.
Если какая-либо группа не набрала требуемое количество, команда завершается
с ошибкой. Диапазон дат задаётся через `--d1` и `--d2`.

Для отдельного запроса используется `fetch`:

```bash
.venv/bin/python data_collection/inaturalist_dataset.py fetch --taxon-id 47224 --max-candidates 100 --d1 2018-01-01 --d2 2026-11-09 --output data/butterfly_candidates.jsonl
```

`collect-plan` использует случайный порядок наблюдений. Для воспроизведения
зафиксированного набора используются сохранённые манифесты, а не новый сбор.

## HTML-галерея

```bash
.venv/bin/python data_collection/inaturalist_dataset.py gallery --input data/raw_candidates_2018_2026/all_candidates.jsonl --output data/candidates_gallery.html
```

Галерея показывает группу отбора, ID фотографии и наблюдения, таксон и превью.
Заголовок карточки открывает наблюдение, изображение — оригинал.
Визуальный отбор сохраняется отдельным списком ID, по одному на строку.

## Проверка оригиналов

```bash
.venv/bin/python data_collection/inaturalist_dataset.py validate-all --input data/raw_candidates_2018_2026/all_candidates.jsonl --output data/all_validated.jsonl --report data/all_validation_report.json
```

Команда скачивает каждый оригинал, проверяет `Content-Type: image/*`, непустое
тело ответа, структуру изображения через Pillow `verify()` и декодирование
пикселей через `load()`. SHA-256 используется для поиска побайтных дубликатов.
Число параллельных загрузок задаётся `--workers`, таймаут — `--timeout`.

В JSONL сохраняются результаты проверки каждой фотографии. JSON-отчёт содержит
число проверенных изображений, ошибок загрузки/декодирования и дубликатов.
Ошибки загрузки не считаются фотографиями без бабочек.

Для проверки только выбранных ID используется `validate`:

```bash
.venv/bin/python data_collection/inaturalist_dataset.py validate --input data/raw_candidates_2018_2026/all_candidates.jsonl --selected-ids data/selected_photo_ids.txt --output data/validated.jsonl --report data/validation_report.json
```

Галерея проверенных кандидатов:

```bash
.venv/bin/python data_collection/inaturalist_dataset.py gallery --input data/all_validated.jsonl --output data/validated_gallery.html
```

## Фиксация выбранного набора

После визуального отбора сохраните ID в `data/selected_photo_ids.txt`
и зафиксируйте набор:

```bash
.venv/bin/python data_collection/inaturalist_dataset.py finalize --input data/all_validated.jsonl --selected-ids data/selected_photo_ids.txt --output data/final_dataset.jsonl --csv-output data/final_dataset.csv --expected-count 400
```

`finalize` допускает только успешно проверенные фотографии без побайтных
дубликатов и проверяет ожидаемое количество. Команда сохраняет JSONL-манифест
и CSV с источниками, лицензиями, размерами и хешами. Этот этап фиксирует
изображения для разметки; рамки на нём ещё не создаются.

Процесс разметки и независимого REVIEW описан в
[clean_annotation/README.md](../clean_annotation/README.md).
Готовые аннотации опубликованы в
[exports/final_butterflies_v1/](../exports/final_butterflies_v1/).
