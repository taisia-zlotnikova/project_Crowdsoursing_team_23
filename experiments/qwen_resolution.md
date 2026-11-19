# Qwen3-VL: влияние разрешения входного изображения

## Гипотеза

Увеличение `max_pixels` с 262144 до 1048576 улучшит F1@IoU=0.75
за счёт более точных границ и увеличит время обработки.
Основная метрика выбора модели — F1@IoU=0.5.

Гипотеза зафиксирована до запуска эксперимента, 19 ноября 2026 года.

## Условия

| Параметр | Baseline | Эксперимент |
| --- | --- | --- |
| `max_pixels` | 262144 | 1048576 |
| `min_pixels` | 65536 | 65536 |
| `max_new_tokens` | 512 | 512 |
| Seed | 42 | 42 |
| Конфигурация | `inference/configs/baseline.json` | `inference/configs/qwen_maxpixels_1048576.json` |
| Результаты | `runs/baseline/qwen3-vl` | `runs/qwen-maxpixels-1048576/qwen3-vl` |

Промпт, веса и версии библиотек одинаковы. Оба варианта обрабатывают все
400 фотографий из `data/annotations.csv` в одном порядке на одном устройстве.
Финальный конкурсный датасет в эксперименте не используется.

SHA-256 аннотаций:
`efc87cc902b5492625821707e2f4062f1535e1190d6a0261a91088c1a250843a`.

## Запуск

Baseline:

```bash
.venv/bin/python -m inference.run --dataset data/annotations.csv --config inference/configs/baseline.json --models all --output runs/baseline --device auto
.venv/bin/python -m inference.evaluate --run runs/baseline
```

Увеличенное разрешение:

```bash
.venv/bin/python -m inference.run --dataset data/annotations.csv --config inference/configs/qwen_maxpixels_1048576.json --models qwen3-vl --output runs/qwen-maxpixels-1048576 --device auto --resume
```

После успешного завершения инференса:

```bash
.venv/bin/python -m inference.evaluate --run runs/qwen-maxpixels-1048576
```

`--resume` запускает обработку с нуля при отсутствии папки результатов либо
продолжает совместимый запуск, пропуская сохранённые успешные ответы.
Для независимого повтора указывается новая папка `--output`.

## Оценка

Сравниваются:

- F1, Precision, Recall и TP/FP/FN при IoU 0.5 и 0.75;
- число ложных находок и доля ошибок на пустых фотографиях;
- среднее, медиана и p95 времени обработки изображения.

Метрики сохраняются в `metrics.json`, сводная таблица — в `comparison.csv`
папки соответствующего запуска. В дашборде сравнение доступно во вкладке
«Эксперименты» при выборе `baseline` и `qwen-maxpixels-1048576`.

Для проверки гипотезы оцениваются изменение F1@0.75 и затраты времени.
Выбор итоговой конфигурации учитывает основную метрику F1@0.5.
