# Lenta Price Tag OCR Pipeline

Пайплайн распознавания розничных ценников и ценовых лент в выделенных видео-треков ценников. Проект ориентирован на следующий сценарий применения: камера получает поток с полки, detector/tracker выделяет crop'ы ценников, OCR-пайплайн извлекает цену, товар, скидку, служебные статусы, QR/barcode/DataMatrix и диагностические признаки качества.
Основной акцент текущей версии принятие решения по треку: несколько кадров одного ценника выравниваются, проходят OCR, голосование, visual/OCR consensus, denoise/fusion и проверку через каталог `goods.csv`.

---

### Поля товара и каталога

`item_id` id найденного товара из каталога `goods.csv`. Идентификатор отсутствует, если статусы `price_only_candidates`, `weak_match`, `ambiguous` или `not_found`.

Контракт:

```json
{
  "product_name": "Вино безалкогольное CARL JUNG Cuvee red ...",
  "main_price": 1199.99,
  "old_price": 1684.00,
  "discount_percent": 28,

  "item_id": null,
  "catalog_status": "price_only_candidates",
  "catalog_product_prior": {
    "item_id": "866240",
    "name": "Вино безалкогольное CARL JUNG ...",
    "score": 0.42
  },

  "qr_payload": null,
  "barcode_value": null,
  "qr_item_id": null
}
```

То есть:

- `item_id` — финально принятый catalog item;
- `catalog_product_prior` — подсказка из каталога, если совпадение не принято как финальное;
- `qr_payload` — сырой payload QR/DataMatrix;
- `barcode_value` — линейный barcode;
- `qr_item_id` — item id, извлечённый именно из QR, если известен формат payload.

Если на debug-карточке изображено `code_fb=0 attempts=70`, это означает, что QR/barcode fallback сделал 70 попыток, но код не был декодирован. В этом случае любой `item_id` пришёл не из QR, а из DB/CSV matching.

### Правила цены и скидки

Для промо-ценников, особенно `red_promo`, `main_price` должна выбираться из большой промо-цены в нижней цветной зоне. Маленькая цена в верхней/правой зоне обычно является старой или регулярной ценой.

Пример:

```text
большая цена: 1199.99  -> main_price
малая цена:   1684.00  -> old_price
пузырь:       -28%     -> discount_percent
```

`discount_percent` должен содержать только процент, например `28%`, а не склеенный OCR-фрагмент вида `28%1199`.

---

## Что извлекает пайплайн

- тип шаблона ценника: обычный, красный/жёлтый промо, progressive, service/stockout и другие эвристические типы;
- основную цену `main_price`;
- старую/регулярную цену `old_price`, если она отделима от промо-цены;
- процент скидки `discount_percent`;
- товарное название из OCR;
- единицу продажи / фасовку, если она выводится из OCR или каталога;
- `item_id` только при безопасном catalog match;
- catalog candidates / catalog priors для спорных случаев;
- QR/barcode/DataMatrix payload, если код реально декодирован;
- статус `товар закончился` для сервисных ценников;
- флаги `needs_review`, warnings и диагностику нестабильных треков.

---

## Основные режимы работы

### 1. Одиночные изображения / папка изображений / price rail

Entry point:

```bash
python run_price_tag_pipeline.py \
  --config configs/rail_track_csv_pipeline.yaml \
  --input_dir ./rails \
  --out_dir ./runs/rail_pipeline
```

Одиночный crop ценника:

```bash
python run_price_tag_pipeline.py \
  --config configs/rail_track_csv_pipeline.yaml \
  --image ./samples/tag_001.jpg \
  --input_mode single_tag \
  --out_dir ./runs/tag_001
```

Полочная лента с несколькими ценниками:

```bash
python run_price_tag_pipeline.py \
  --config configs/rail_track_csv_pipeline.yaml \
  --image ./samples/rail_001.jpg \
  --input_mode price_rail \
  --enable_rail_split \
  --out_dir ./runs/rail_001
```

Автоматический режим `input_structure.mode: auto` пытается отличить одиночный ценник от price rail без OCR, по геометрии, цветовым зонам и разделителям.

### 2. Датасет уже найденных треков

Entry point:

```bash
python run_detected_tracks_dataset.py \
  --config configs/detected_tracks_dataset.yaml \
  --root_dir d:/dev/datasets/toy-dataset-2 \
  --out_dir runs/detected_tracks_dataset
```

Ожидаемая структура:

```text
root_dir/
  sequence_001/
    7/
      frame_0001.jpg
      frame_0002.jpg
      ...
    210/
      frame_0001.jpg
      frame_0002.jpg
      ...
  sequence_002/
    33/
      frame_0001.jpg
      ...
```

debug-режим:

```bash
python run_detected_tracks_dataset.py \
  --config configs/detected_tracks_dataset.yaml \
  --root_dir d:/dev/datasets/toy-dataset-2 \
  --out_dir runs/detected_tracks_dataset_debug \
  --keep_items \
  --save_frame_debug \
  --write_frame_json \
  --write_debug_plots
```

Отключить split смешанных треков:

```bash
python run_detected_tracks_dataset.py \
  --config configs/detected_tracks_dataset.yaml \
  --root_dir d:/dev/datasets/toy-dataset-2 \
  --out_dir runs/no_split \
  --no_split_tracks
```

Проверить итоговый конфиг без OCR:

```bash
python run_detected_tracks_dataset.py \
  --config configs/detected_tracks_dataset.yaml \
  --print_config
```

---

## Структура проекта

```text
configs/
  detected_tracks_dataset.yaml        # основной конфиг для sequence/track_id/images
  rail_pipeline.yaml                  # общий rail/single-image конфиг

goods.csv                             # товарный каталог
w2c.npy                               # Color Names LUT

run_price_tag_pipeline.py             # одиночные изображения / папки / price rail
run_detected_tracks_dataset.py        # датасет готовых треков

price_tag_pipeline/
  cli.py                              # CLI, сборка backend'ов
  config.py                           # DEFAULT_CONFIG, YAML merge, CLI overrides
  pipeline.py                         # end-to-end обработка одного изображения
  detected_tracks_dataset.py          # обработка sequence/track_id/images
  track_aggregator.py                 # vote, best-frame, mixed-track split, catalog gate
  track_image_fusion.py               # visual/OCR consensus + fused image
  track_debug_plots.py                # timeline/vote/global plots
  ocr_backends.py                     # PaddleOCR adapter, batch API, NullOCR
  csv_corrector.py                    # deterministic matching с goods.csv
  product_card.py                     # финальная карточка, скидки, stockout, product filters
  spatial_parser.py                   # OCR spatial parsing: цены, товар, промо-зоны
  template_classifier.py              # шаблон ценника через Color Names
  color_names.py                      # Color Names descriptor + w2c LUT
  price_rail_splitter.py              # поиск и нарезка полочной ленты
  tilt_corrector.py                   # коррекция наклона
  preprocess_glare.py                 # подавление бликов/дымки
  code_decoder.py                     # QR/barcode/DataMatrix decoder
  debug_vis.py                        # отрисовка OCR/layout/final debug
  layout.py                           # layout extractor
  price_parser.py                     # парсинг цены из OCR-текста
  quality.py                          # blur/brightness checks
  image_ops.py, io_utils.py           # image I/O и базовые операции

scripts/
  convert_w2c_mat_to_npy.py           # конвертация Color Names LUT
  debug_paddleocr_raw.py              # сырой debug PaddleOCR
  download_paddlenlp_qwen.py          # загрузка локального PaddleNLP/Qwen
```

---

## End-to-end логика

### Single image / rail mode

1. Загрузка изображения и базовая проверка качества.
2. Опциональная коррекция наклона.
3. Определение структуры: одиночный ценник или price rail.
4. Если включён rail mode — поиск полочной ленты и разбиение на ячейки.
5. Классификация шаблона через Color Names descriptor.
6. Подавление бликов/дымки, если включено.
7. OCR через PaddleOCR или `NullOCR`.
8. Исключение QR/barcode/DataMatrix зон из OCR-зон.
9. Spatial parsing: главная цена, старая цена, товар, скидка, stockout/service text.
10. Декодирование QR/barcode/DataMatrix.
11. Формирование `product_card`.
12. CSV-коррекция по `goods.csv` как enrichment/prior, а не как безусловная истина.
13. Опциональная LLM-коррекция.
14. Запись JSON/TSV/debug-изображений.

### Detected-track mode

1. Обход `root_dir/sequence/track_id/images`.
2. OCR каждого кадра трека через обычный single-tag pipeline.
3. Извлечение наблюдений: цена, товар, OCR-текст, шаблон, качество, коды.
4. Split mixed tracks, если одна папка содержит несколько ценников.
5. Visual consensus по компактному image descriptor.
6. OCR consensus через char n-gram embedding + symmetric kNN + GZip/NCD.
7. Boost согласованных кадров и штраф outlier-кадров.
8. Голосование по цене, товару, скидке, stockout и catalog evidence.
9. Выбор best observation.
10. Fusion согласованных кадров: alignment + `fastNlMeansDenoisingColored` + mean/median.
11. QR/barcode fallback на fused image, если код не найден по отдельным кадрам.
12. Запись summary, frame table, compact best-debug и timeline/vote plots.

---

## OCR и batch inference

В `ocr_backends.py` есть batch API:

```yaml
ocr:
  backend: "paddle"
  batch_inference: true
  batch_size: 16
  min_batch_jobs: 2
  gpu: true
  full_tag_ocr: true
  paddle:
    lang: "ru"
    ocr_version: "PP-OCRv5"
    use_angle_cls: true
    text_detection_model_name: "PP-OCRv5_mobile_det"
    text_recognition_model_name: "eslav_PP-OCRv5_mobile_rec"
    input_mode: "ndarray"
    mkldnn: false
    pir_api: false
```

Пайплайн собирает OCR-задачи внутри одного tag crop и отправляет их пачкой, если текущая версия PaddleOCR поддерживает list-input. 

---

## Track fusion и OCR consensus

`track_image_fusion.py` решает две типовые проблемы видео-OCR:

- в одном track folder могут оказаться кадры соседнего ценника;
- лучший визуальный кадр не всегда содержит лучший OCR-текст.

Используются два уровня близости:

1. **Visual similarity** — компактный descriptor изображения.
2. **OCR similarity** — char n-gram embedding + kNN + GZip/NCD.

Фрагмент конфига:

```yaml
detected_tracks_dataset:
  track_fusion:
    enabled: true
    max_images: 9
    align: true
    denoise_h: 7.0
    denoise_h_color: 7.0

    score_consensus_enabled: true
    fusion_consensus_enabled: true
    consensus_max_candidates: 32
    consensus_feature_size: 64
    visual_similarity_threshold: 0.66
    evidence_similarity_threshold: 0.50
    min_cluster_size: 2

    ocr_similarity_enabled: true
    ocr_embedding_dim: 384
    ocr_knn_k: 4
    ocr_similarity_threshold: 0.56
    ocr_strong_similarity_threshold: 0.70
    ocr_gzip_weight: 0.55

    selected_score_boost: 0.22
    outlier_score_penalty: 0.32
    decode_codes_on_fused: true
    decode_only_when_missing: true
```

Практический смысл:

- согласованные кадры получают больший вес;
- одиночные выбросы с другой ценой/товаром штрафуются;
- fused image собирается из доминирующего кластера, а не из всего трека;
- QR/barcode повторно проверяется на fused crop.

---

## Split mixed tracks

Если tracker объединил несколько ценников в одну папку, `track_aggregator.py` может разделить её на несколько гипотез.

Ключевые параметры:

```yaml
track_aggregation:
  split_mixed_tracks: true
  split_min_segment_observations: 2
  split_min_price_support: 2
  split_min_reliable_score: 0.72
  split_price_gap_ratio: 0.16
  split_product_token_overlap: 0.34
```

Split нужен для случаев, где в одном треке чередуются:

- шумный ценник;
- реальный shelf label;
- соседний товар;
- poster/banner;
- service tag;
- QR-board.

Если split сработал, итоговые строки получают `split_index`, `split_count`, `split_reason`.

---

## CSV / DB matching

`csv_corrector.py` выполняет deterministic matching по `goods.csv`. Он использует OCR-текст, цену, fuzzy token coverage, family matching и защиту от price-only false positive.

Пример конфига:

```yaml
csv_corrector:
  enabled: true
  path: "goods.csv"
  top_k: 8
  min_text_score: 0.26
  min_accept_score: 0.58

  allow_price_correction: true
  allow_price_only_match: true
  allow_price_only_autofill: false
  force_catalog_price_when_matched: false
  max_price_conflict_ratio: 0.12
  max_price_only_candidates: 5

  allow_close_family_match: true
  family_match_min_name_overlap: 0.72
  family_match_min_price_score: 0.92
  min_text_score_with_price: 0.46
  min_price_score_for_soft_accept: 0.94

  reject_weak_query_text: true
  min_query_content_tokens: 2
  min_query_alpha_chars: 8
  min_product_text_score_for_autofill: 0.48
  retain_candidate_match: true
  strong_text_accept_score: 0.48
```

### Безопасные правила DB matching

- `allow_price_only_autofill: false` — товар нельзя автозаполнять только по цене.
- `force_catalog_price_when_matched: false` — цена с ценника не перетирается ценой из каталога.
- `reject_weak_query_text: true` — мусорный OCR не запускает полноценный product match.
- `retain_candidate_match: true` — слабый кандидат сохраняется как prior, но не становится финальным товаром.

Статусы каталога:

```text
matched                 # уверенный catalog match, item_id можно принять
soft_match              # мягкое совпадение, зависит от track gate
weak_match              # подсказка, не финальный item_id
ambiguous               # несколько похожих кандидатов, нужен review
price_only_candidates   # кандидаты найдены только/почти только по цене
not_found               # кандидаты не найдены
rejected_by_track_evidence
```

`price_only_candidates`, `weak_match`, `ambiguous` и `not_found` не должны попадать в финальный `item_id`.

### Track-level catalog gate

Даже если `csv_corrector` нашёл кандидата, агрегатор может отклонить его на уровне всего трека.

```yaml
track_aggregation:
  catalog_gate_enabled: true
  catalog_reject_on_review: true
  catalog_min_text_score_for_track_accept: 0.50
  catalog_min_price_consistency_for_track_accept: 0.62
  catalog_min_product_consistency_for_track_accept: 0.62
```

Причины отклонения:

- нестабильная цена;
- несколько уникальных цен в одном треке;
- неоднозначный product vote;
- низкая OCR-поддержка товара;
- false-tag признаки;
- сильный конфликт OCR-цены и catalog price;
- service/stockout tag.

---

## QR / barcode / DataMatrix

Декодер находится в `code_decoder.py`.

```yaml
code_decoder:
  enabled: true
  use_pyzbar: true
  try_opencv_qr: true
  qr_roi_scan: true
  preprocessing_variants: true
  keep_undecoded_qr: true
```

В track mode дополнительно:

```yaml
detected_tracks_dataset:
  track_fusion:
    decode_codes_on_fused: true
    decode_only_when_missing: true
```

Декодер пробует:

- полный crop;
- верхне-правые QR ROI;
- нижнюю barcode-зону;
- QR-like контуры;
- grayscale/CLAHE/sharpen/Otsu/adaptive/inverted варианты;
- OpenCV `detectAndDecode`, `detectAndDecodeMulti`, `detectAndDecodeCurved`;
- `pyzbar`, если доступен ZBar.

Debug-поля:

```text
code_fb=<decoded_count> attempts=<attempt_count>
```

Интерпретация:

- `code_fb=0 attempts=70` — было 70 попыток, ничего не декодировано;
- `code_fb=1` — хотя бы один код декодирован;
- `qr_payload` / `barcode_value` должны хранить реальный payload;
- `item_id` не должен считаться QR-результатом без `qr_payload` или `qr_item_id`.

---

## Spatial parser как OCR prior

`spatial_parser.py` извлекает кандидатные зоны цены, товара, скидки и служебных текстов.

```yaml
spatial_parser:
  enabled: true
  draw_semantic_boxes: true
  draw_full_ocr: true
  product_as_ocr_prior: true
```

При `product_as_ocr_prior: true` spatial product проходит через фильтры качества текста. Мусорные строки фильтруются:

```text
69 | 99 | ттоа
УпсйТодорзахончился
Coanuromi Ищьтое с Res10
```


---

## Статус «товар закончился»

Сервисный ценник распознаётся по явному OCR-тексту:

- `товар закончился`;
- плотные OCR-варианты без пробелов;
- фразы с `скоро привез...`;
- паттерны `упс ... товар ... закончился`.

Ожидаемый результат:

```json
{
  "stock_status": "out_of_stock",
  "stock_status_label": "товар закончился",
  "stock_status_text": "..."
}
```

Для stockout/service tag DB matching должен быть заблокирован или переведён в prior. Служебный ценник не должен получать случайный `item_id` из каталога по цене или QR-like шуму.

---

## Шаблоны и Color Names

Классификатор шаблонов использует Color Names descriptor и LUT `w2c.npy`.

```yaml
color_names:
  backend: "w2c_lut"
  lut_path: "w2c.npy"
  temperature: 950.0
  max_pixels: 80000
  w2c_index_order: "rgb_fast"
```

Это помогает различать:

- обычный белый ценник;
- красный/оранжевый промо;
- жёлтый промо;
- progressive;
- service-like tags;
- false-tag/banner/poster кандидаты.

Если LUT хранится в `.mat`, конвертация:

```bash
python scripts/convert_w2c_mat_to_npy.py \
  --input ./w2c.mat \
  --output ./w2c.npy
```

---

## Debug-артефакты

### Single image / rail mode

```text
runs/rail_pipeline/
  resolved_config.yaml
  summary.json
  summary.tsv
  items/
  tracks/
```

`summary.tsv` содержит:

- `image_path`;
- `status`;
- `mode`;
- `template_name`, `template_confidence`;
- `main_price`, `old_price`, `final_main_price`;
- `final_product_name`;
- `csv_status`;
- `llm_status`;
- `needs_review`;
- `codes_decoded`;
- `quality_status`;
- `rail_count`, `rail_cell_count`;
- `tilt_angle`;
- `glare_applied`.

### Detected-track mode

```text
runs/detected_tracks_dataset/
  resolved_config.yaml
  detected_tracks_results.json
  detected_tracks_summary.csv
  detected_tracks_summary.tsv
  detected_tracks_frames.tsv
  debug_images/
    sequence__track_id__best.jpg
  debug_plots/
    *_timeline.png
    *_votes.png
    global__*.png
```

Ключевые поля summary:

- `sequence_name`;
- `source_track_id`;
- `track_key`;
- `status`;
- `num_images`, `num_observations`;
- `best_image`, `best_debug_image`;
- `best_score`;
- `main_price`, `old_price`, `discount_percent`;
- `product_name`;
- `unit`;
- `item_id`;
- `catalog_status` / `csv_status`;
- `catalog_gate_rejected`;
- `catalog_reject_reasons`;
- `needs_review`;
- `split_index`, `split_count`, `split_reason`;
- `price_consistency`, `price_unique_count`;
- `product_consistency`;
- `best_fused_image`;
- `fused_code_fallback_decoded`;
- `stock_status`;
- `warnings`.

### Timeline plots

`track_debug_plots.py` строит:

- график score по кадрам;
- график price по кадрам;
- фон consensus cluster / selected region;
- строки evidence: `ocr`, `price`, `discount`, `stockout`, `cluster`, `ocr_knn`, `db`;
- vote diagnostics по цене, товару, stockout и catalog evidence.

Строка `db` должна показывать только значимые состояния: `matched`, `weak`, `ambiguous`, `price_only`, `rejected`. Постоянный `not_found` лучше скрывать, иначе график становится шумным.

---

## Установка

Минимально:

```bash
pip install numpy opencv-python pyyaml tqdm pillow matplotlib scipy
```

OCR:

```bash
pip install paddleocr paddlepaddle
```

Для GPU установите сборку PaddlePaddle под вашу CUDA/драйверную конфигурацию.

QR/barcode через `pyzbar`:

```bash
pip install pyzbar
```

На Windows для `pyzbar` может потребоваться установленный ZBar. Если ZBar недоступен, OpenCV QR fallback остаётся доступен, но он не заменяет полноценный barcode/DataMatrix decoder.

Опционально для LLM-коррекции:

```bash
pip install paddlenlp
```

По умолчанию LLM отключён:

```yaml
llm_corrector:
  backend: "none"
```

---

## PaddleOCR compatibility

`ocr_backends.py` поддерживает:

- PaddleOCR 3.x pipeline API: `PaddleOCR(...).predict(...)`;
- PaddleOCR 2.x legacy API: `PaddleOCR(...).ocr(...)`.

Backend пытается инициализировать современный API. Если часть kwargs не поддерживается, она удаляется, после чего выполняется fallback на legacy API.

Для снижения конфликтов на CPU/Windows в YAML используются явные флаги:

```yaml
paddle:
  mkldnn: false
  pir_api: false
```

При `pir_api: false` выставляется `FLAGS_enable_pir_api=0`. При `mkldnn: false` отключаются oneDNN/MKLDNN workspace-флаги.

---

## Практический workflow ревизии качества

1. Запустить `run_detected_tracks_dataset.py` на свежем наборе треков.
2. Открыть `detected_tracks_summary.csv`.
3. Отсортировать по:
   - `needs_review`;
   - `warnings`;
   - `price_consistency`;
   - `price_unique_count`;
   - `catalog_status` / `csv_status`;
   - `fused_code_fallback_decoded`.
4. Быстро просмотреть `debug_images/*__best.jpg`.
5. Для спорных кейсов открыть `debug_plots/*timeline.png` и `*votes.png`.
6. Если DB дал ложный товар — ужесточить CSV matching и catalog gate.
7. Если fused image собирается из соседнего ценника — поднять visual/evidence thresholds.
8. Если OCR-полезные кадры отбрасываются — ослабить OCR/visual thresholds.
9. Если QR/barcode не читается — проверить `best_fused_image`, `code_fb`, `attempts` и наличие ZBar.

---

## Настройки для типовых проблем

### Ложное DB-сопоставление

```yaml
csv_corrector:
  allow_price_only_autofill: false
  force_catalog_price_when_matched: false
  min_text_score_with_price: 0.50
  min_price_score_for_soft_accept: 0.95
  family_match_min_name_overlap: 0.80
  max_price_only_candidates: 3

track_aggregation:
  catalog_gate_enabled: true
  catalog_min_text_score_for_track_accept: 0.55
  catalog_min_price_consistency_for_track_accept: 0.66
  catalog_min_product_consistency_for_track_accept: 0.66
```

Полностью запретить price-only matching:

```yaml
csv_corrector:
  allow_price_only_match: false
```

### В одном треке несколько ценников

```yaml
track_aggregation:
  split_mixed_tracks: true
  split_min_segment_observations: 2
  split_min_price_support: 2
  split_price_gap_ratio: 0.16
  split_product_token_overlap: 0.34

detected_tracks_dataset:
  track_fusion:
    visual_similarity_threshold: 0.70
    evidence_similarity_threshold: 0.55
    min_cluster_size: 2
    outlier_score_penalty: 0.38
```

### OCR плохо читает товар, но цена стабильна

```yaml
detected_tracks_dataset:
  track_fusion:
    ocr_similarity_enabled: true
    ocr_similarity_threshold: 0.52
    ocr_gzip_weight: 0.60

track_aggregation:
  product_vote_boost: 1.10
  price_vote_boost: 1.35
```

### Промо-цена путается со старой ценой

Проверить, что spatial/parser учитывает шаблон и позицию цены:

```yaml
price_parser:
  allow_compact_in_main_price_zones: true
  allow_compact_in_full_tag: false

template_classifier:
  enabled: true

spatial_parser:
  enabled: true
  draw_semantic_boxes: true
```

Для `red_promo` большая цена в нижней цветной зоне должна иметь приоритет над маленькой верхней/правой ценой.

### Скидка склеивается с ценой

Ожидаемое правило парсинга: процент заканчивается на первом символе `%`.

```text
28%1199 -> discount_percent=28, не "28%1199"
-17%329 -> discount_percent=17, не "17%329"
```

### QR/barcode не читается

```yaml
code_decoder:
  enabled: true
  use_pyzbar: true
  try_opencv_qr: true
  qr_roi_scan: true
  preprocessing_variants: true

detected_tracks_dataset:
  track_fusion:
    decode_codes_on_fused: true
    decode_only_when_missing: true
```

Проверить в debug:

```text
code_fb=0 attempts=N     # декодирования нет
code_fb=1 attempts=N     # код декодирован
```

---

## Smoke-проверка

Печать конфига:

```bash
python run_price_tag_pipeline.py \
  --config configs/rail_track_csv_pipeline.yaml \
  --image ./samples/tag_001.jpg \
  --out_dir ./runs/smoke_single \
  --print_config
```

Реальный single-image OCR:

```bash
python run_price_tag_pipeline.py \
  --config configs/rail_track_csv_pipeline.yaml \
  --image ./samples/tag_001.jpg \
  --out_dir ./runs/smoke_single
```

Detected tracks:

```bash
python run_detected_tracks_dataset.py \
  --config configs/detected_tracks_dataset.yaml \
  --root_dir ./toy-dataset-2 \
  --out_dir ./runs/smoke_tracks
```

Проверить:

```text
runs/smoke_tracks/detected_tracks_summary.csv
runs/smoke_tracks/debug_images/
runs/smoke_tracks/debug_plots/
```

---

## Ограничения текущей реализации

- OCR остаётся главным источником ошибок на мелком, смазанном и бликующем тексте.
- Batch OCR зависит от версии PaddleOCR: если list-input не поддерживается, используется fallback.
- DB/CSV matching не должен рассматриваться как ground truth. При слабом OCR безопаснее получить `needs_review`, чем ложный `item_id`.
- QR fallback на fused image не гарантирует декодирование без корректно установленного `pyzbar`/ZBar.
- DataMatrix обычно требует более сильного внешнего decoder'а, чем стандартный OpenCV QR.
- Color Names classifier устойчив к типовым цветовым шаблонам, но нестандартные промо-макеты могут уходить в `unknown` или false-tag.
- Glare suppression может как улучшить OCR, так и ухудшить мелкий текст. Для спорных случаев сохраняйте glare debug.

---

## Последующие доработки

1. Жёстко разделить в коде поля `item_id`, `catalog_candidate_item_id`, `qr_item_id` и `barcode_value`.
2. Запретить протекание `price_only_candidates` в финальный `item_id`.
3. Добавить отдельные unit tests на:
   - `1199.99` vs `1684.00` на red promo;
   - `28%1199` как discount + price;
   - alcohol percent vs discount;
   - stockout service tag;
   - false DB match по одной цене;
   - QR fallback с `code_fb=0`;
   - `1.79` vs `179.00`;
   - `66` vs `99`.
4. Добавить калибратор confidence для catalog states: `catalog_suggested`, `catalog_soft_accept`, `catalog_confirmed`.
5. Разнести thresholds CSV matching по шаблонам: white tag, red promo, yellow promo, service tag.
6. Добавить RKNN/edge-профиль: latency по стадиям, CPU/NPU split, memory, FPS, температура.
7. Для production-робота добавить ROS2 node: camera crop input -> OCR track aggregation -> structured result topic/service.

---

## Целевой production-сценарий

1. Робот подъезжает к полке и стабилизирует камеру.
2. Детектор/трекер выделяет ценники в виде `sequence/track_id/images`.
3. OCR-пайплайн обрабатывает каждый track, строит consensus и fused crop.
4. На выходе формируется таблица с ценой, скидкой, товаром, catalog prior, item_id, QR/barcode и quality flags.
5. Строки с `needs_review`, `ambiguous`, `price_only_candidates` и warnings уходят в ручную проверку или цикл доразметки.
6. Подтверждённые ошибки используются для расширения датасета и настройки OCR/catalog gates.
