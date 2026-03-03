# ROIAlign-onnx-modules
Various implementations of DynamicRoIAlign. This provides an effective component for appropriately filtering the output of object detection models without going through NMS, and then merging various models directly into the subsequent stages.

## Variants

| Variant | Summary | Jump |
| --- | --- | --- |
| `01_dynamic_roi_align.py` | Basic DynamicRoIAlign export (`rois` is a direct input). | [Go to Section](#variant-01) |
| `02_dynamic_roi_align_yolo.py` | Integrated YOLO preprocessing + DynamicRoIAlign export. | [Go to Section](#variant-02) |
| `03_dynamic_roi_align_vit.py` | Integrated ViT output preprocessing + DynamicRoIAlign export. | [Go to Section](#variant-03) |

<a id="variant-01"></a>
## 01_dynamic_roi_align.py Usage

`01_dynamic_roi_align.py` exports a DynamicRoIAlign model to ONNX, runs `onnxsim`, and writes metadata into the exported model.

- Static `output_height`/`output_width`

  <img width="1300" height="1098" alt="image" src="https://github.com/user-attachments/assets/7b44fb0d-2d81-45d1-b1b3-7c6a9ae63a9b" />

- Full dynamic

  <img width="2011" height="1105" alt="image" src="https://github.com/user-attachments/assets/1e5411fb-b4d6-43cd-b815-ab5ce71e404f" />

### Basic run

```bash
python 01_dynamic_roi_align.py
```

Generated file:

- `dynamic_roi_align.onnx` (overwritten each run)

### CLI options

```
python 01_dynamic_roi_align.py \
  [--channels CHANNELS] \
  [--batch-size BATCH_SIZE] \
  [--input-hw-size H W] \
  [--spatial-scale SPATIAL_SCALE [SPATIAL_SCALE ...]] \
  [--output-height OUTPUT_HEIGHT] \
  [--output-width OUTPUT_WIDTH] \
  [--opset-version OPSET_VERSION] \
  [--onnx-output-path ONNX_OUTPUT_PATH] \
  [--aligned | --no-aligned]
```

Option summary:

- `--channels`: fix input/output channel dimension in ONNX. Omit to keep channels dynamic.
- `--batch-size`: fix batch dimension in ONNX input. Omit to keep batch size dynamic.
- `--input-hw-size`: fix `input_images_or_features` height/width (`H W`) in ONNX input.
- `--spatial-scale`: ROI coordinate scale. One value means shared H/W scale, two values mean `(scale_h, scale_w)`.
- `--output-height`: output height behavior. Omitted means dynamic scalar ONNX input (`output_height`), specified integer means fixed output height in graph.
- `--output-width`: output width behavior. Omitted means dynamic scalar ONNX input (`output_width`), specified integer means fixed output width in graph.
- `--opset-version`: ONNX opset version (`>= 16`).
- `--onnx-output-path`: export destination path.
- `--aligned` / `--no-aligned`: switch ROIAlign alignment behavior (`align_corners=True/False`). Default is `--no-aligned`.

`--spatial-scale` fallback behavior:

1. `--spatial-scale` specified: use the specified value.
2. `--spatial-scale` omitted and `--input-hw-size` specified: use `(H, W)` as fixed spatial scale.
3. both omitted: use runtime `input_images_or_features` `H/W` dynamically (`spatial_scale=None`).

Note:

- `--input-hw-size` requires exactly two integers: `H W`.

### Output size behavior (important)

The exported ONNX inputs change depending on `--output-height` / `--output-width`:

1. Both omitted: ONNX inputs include `output_height` and `output_width` (both dynamic scalar inputs), and output shape `H/W` are dynamic.
2. `--output-height` specified, `--output-width` omitted: ONNX input includes only `output_width`; output height is fixed and output width is dynamic.
3. `--output-height` omitted, `--output-width` specified: ONNX input includes only `output_height`; output height is dynamic and output width is fixed.
4. Both specified: ONNX inputs do not include output size scalars, and output `H/W` are fixed constants.

Note:

- If output size options are omitted, the script still uses local test values (`7x7`) only for the internal test forward pass before export.

### ONNX names

- Input name: `input_images_or_features`
- ROI input: `rois`
- Output name: `aligned_features`

### Metadata written to ONNX

The script adds descriptions to ONNX metadata for:

- `input_images_or_features`
- `rois`
- `aligned_features`
- `input_hw_size` (only when fixed input size is used)
- `output_height` (only when fixed output height is used)
- `output_width` (only when fixed output width is used)

### Examples

1. Fully dynamic output size:

```bash
python 01_dynamic_roi_align.py
```

2. Fixed output size (`5x9`):

```bash
python 01_dynamic_roi_align.py --output-height 5 --output-width 9
```

3. Fixed channels and batch, dynamic output size:

```bash
python 01_dynamic_roi_align.py --channels 256 --batch-size 1
```

4. Non-square spatial scale:

```bash
python 01_dynamic_roi_align.py --spatial-scale 480 640
```

5. Use opset 18:

```bash
python 01_dynamic_roi_align.py --opset-version 18
```

6. Enable aligned sampling:

```bash
python 01_dynamic_roi_align.py --aligned
```

7. Save to custom path:

```bash
python 01_dynamic_roi_align.py --onnx-output-path out/dynamic_roi_align.onnx
```

<a id="variant-02"></a>
## 02_dynamic_roi_align_yolo.py Usage

`02_dynamic_roi_align_yolo.py` exports an integrated ONNX model that includes:

1. YOLO output preprocessing (`[B, S, N] -> rois [B*N, 5]`)
2. DynamicRoIAlign inference
3. ONNX simplification (`onnxsim`)
4. Metadata annotation

<img width="1433" height="1240" alt="image" src="https://github.com/user-attachments/assets/8cdd5dfa-bc94-4ef6-97bd-bb56aabdd28b" />

### Basic run

```bash
python 02_dynamic_roi_align_yolo.py
```

Generated file:

- `dynamic_roi_align_yolo.onnx` (overwritten each run)

### CLI options

```
python 02_dynamic_roi_align_yolo.py \
  [--channels CHANNELS] \
  [--batch-size BATCH_SIZE] \
  [--input-hw-size H W] \
  [--spatial-scale SPATIAL_SCALE [SPATIAL_SCALE ...]] \
  [--output-height OUTPUT_HEIGHT] \
  [--output-width OUTPUT_WIDTH] \
  [--opset-version OPSET_VERSION] \
  [--yolo-batch-size YOLO_BATCH_SIZE] \
  [--yolo-output-channels YOLO_OUTPUT_CHANNELS] \
  [--yolo-num-candidates YOLO_NUM_CANDIDATES] \
  [--yolo-box-format {xywh,xyxy}] \
  [--onnx-output-path ONNX_OUTPUT_PATH] \
  [--use-score-threshold USE_SCORE_THRESHOLD] \
  [--score-threshold-as-input] \
  [--aligned | --no-aligned]
```

Option summary:

- `--channels`: fix feature map channel dimension in ONNX. Omit to keep channels dynamic.
- `--batch-size`: fix feature map batch dimension in ONNX. Omit to keep batch dynamic.
- `--input-hw-size`: fix `input_images_or_features` height/width (`H W`) in ONNX input.
- `--spatial-scale`: ROI coordinate scale. One value means shared H/W scale, two values mean `(scale_h, scale_w)`.
- `--output-height`: output height behavior. Omitted means dynamic scalar ONNX input (`output_height`), specified integer means fixed output height in graph.
- `--output-width`: output width behavior. Omitted means dynamic scalar ONNX input (`output_width`), specified integer means fixed output width in graph.
- `--opset-version`: ONNX opset version (`>= 16`).
- `--yolo-batch-size`: fix YOLO output axis-0 (`B`). Omit to keep dynamic.
- `--yolo-output-channels`: fix YOLO output axis-1 (`S`). Omit to keep dynamic.
- `--yolo-num-candidates`: fix YOLO output axis-2 (`N`). Omit to keep dynamic.
- `--yolo-box-format`: interpretation of first 4 YOLO channels (`xywh` or `xyxy`).
- `--onnx-output-path`: export destination path.
- `--use-score-threshold`: enable score filtering with a fixed threshold value in `[0.001, 1.000]`.
- `--score-threshold-as-input`: enable score filtering and expose `score_threshold` as a runtime scalar ONNX input.
- `--aligned` / `--no-aligned`: switch ROIAlign alignment behavior (`align_corners=True/False`). Default is `--no-aligned`.

`--spatial-scale` fallback behavior:

1. `--spatial-scale` specified: use the specified value.
2. `--spatial-scale` omitted and `--input-hw-size` specified: use `(H, W)` as fixed spatial scale.
3. both omitted: use runtime `input_images_or_features` `H/W` dynamically (`spatial_scale=None`).

Note:

- `--input-hw-size` requires exactly two integers: `H W`.

### YOLO output behavior (important)

The integrated model expects YOLO output as `[B, S, N]`:

1. `B`: batch size
2. `S`: channel count (first 4 channels are box coordinates)
3. `N`: number of candidates

Box conversion behavior:

1. First 4 channels are decoded by `--yolo-box-format`.
2. Coordinates are converted to normalized `[x1, y1, x2, y2]` in `[0, 1]`.
3. `rois` are generated as `[batch_idx, x1, y1, x2, y2]`.
4. ROI count is `B * N`.

When score filtering is enabled (`--use-score-threshold` or `--score-threshold-as-input`):

1. Score filtering is applied directly from YOLO class scores (`yolo_output[:, 4:, :]`).
2. Candidates with `max(class_scores) >= threshold` are kept.
3. Output is flattened as `[num_rois, C, output_H, output_W]` because candidate count becomes data-dependent.
4. `--use-score-threshold` and `--score-threshold-as-input` are mutually exclusive.

For typical YOLOv9 wholebody output (`[1, 29, 6300]`), use:

```bash
python 02_dynamic_roi_align_yolo.py \
--yolo-batch-size 1 \
--yolo-output-channels 29 \
--yolo-num-candidates 6300
```

### Output size behavior (important)

The exported ONNX inputs change depending on `--output-height` / `--output-width`:

1. Both omitted: ONNX inputs include `output_height` and `output_width` (both dynamic scalar inputs), and output shape `H/W` are dynamic.
2. `--output-height` specified, `--output-width` omitted: ONNX input includes only `output_width`; output height is fixed and output width is dynamic.
3. `--output-height` omitted, `--output-width` specified: ONNX input includes only `output_height`; output height is dynamic and output width is fixed.
4. Both specified: ONNX inputs do not include output size scalars, and output `H/W` are fixed constants.

Note:

- If output size options are omitted, the script still uses local test values (`7x7`) only for the internal test forward pass before export.

### ONNX names

- Input name: `input_images_or_features`
- Input name: `yolo_output`
- Optional input names: `output_height`, `output_width` (only when dynamic), `score_threshold` (when `--score-threshold-as-input` is used)
- Output name: `aligned_features`

### Metadata written to ONNX

The script adds descriptions to ONNX metadata for:

- `input_images_or_features`
- `yolo_output`
- `yolo_box_format`
- `use_score_threshold`, `score_threshold_as_input` (`score_threshold` is added when enabled)
- `aligned`
- `aligned_features`
- `input_hw_size` (only when fixed input size is used)
- `output_height` (only when fixed output height is used)
- `output_width` (only when fixed output width is used)

### Examples

1. Fully dynamic output sizes and dynamic YOLO axes:

```bash
python 02_dynamic_roi_align_yolo.py
```

2. Typical YOLOv9 wholebody output shape with fixed ROIAlign output size:

```bash
python 02_dynamic_roi_align_yolo.py \
--batch-size 1 \
--channels 3 \
--input-hw-size 480 640 \
--yolo-batch-size 1 \
--yolo-output-channels 29 \
--yolo-num-candidates 6300 \
--score-threshold-as-input \
--output-height 128 \
--output-width 64
```

3. Fix channels and batch, keep output sizes dynamic:

```bash
python 02_dynamic_roi_align_yolo.py \
--channels 256 \
--batch-size 1
```

4. Use `xyxy` YOLO box format:

```bash
python 02_dynamic_roi_align_yolo.py --yolo-box-format xyxy
```

5. Save to custom path:

```bash
python 02_dynamic_roi_align_yolo.py --onnx-output-path out/roi_align_yolo.onnx
```

6. Enable score threshold filtering:

```bash
python 02_dynamic_roi_align_yolo.py --use-score-threshold 0.250
```

7. Expose score threshold as runtime ONNX input:

```bash
python 02_dynamic_roi_align_yolo.py --score-threshold-as-input
```

<a id="variant-03"></a>
## 03_dynamic_roi_align_vit.py Usage

`03_dynamic_roi_align_vit.py` exports an integrated ONNX model that includes:

1. ViT output preprocessing (`[B, Q, F] -> rois [B*Q, 5]`)
2. DynamicRoIAlign inference
3. ONNX simplification (`onnxsim`)
4. Metadata annotation

<img width="1565" height="1287" alt="image" src="https://github.com/user-attachments/assets/7169bc93-4f27-46d6-95a7-eafdd4f84e07" />

### Basic run

```bash
python 03_dynamic_roi_align_vit.py
```

Generated file:

- `dynamic_roi_align_vit.onnx` (overwritten each run)

### CLI options

```
python 03_dynamic_roi_align_vit.py \
  [--channels CHANNELS] \
  [--batch-size BATCH_SIZE] \
  [--input-hw-size H W] \
  [--spatial-scale SPATIAL_SCALE [SPATIAL_SCALE ...]] \
  [--output-height OUTPUT_HEIGHT] \
  [--output-width OUTPUT_WIDTH] \
  [--opset-version OPSET_VERSION] \
  [--vit-batch-size VIT_BATCH_SIZE] \
  [--vit-num-queries VIT_NUM_QUERIES] \
  [--vit-output-fields VIT_OUTPUT_FIELDS] \
  [--vit-box-format {xyxy,xywh}] \
  [--onnx-output-path ONNX_OUTPUT_PATH] \
  [--use-score-threshold USE_SCORE_THRESHOLD] \
  [--score-threshold-as-input] \
  [--aligned | --no-aligned]
```

Option summary:

- `--channels`: fix feature map channel dimension in ONNX. Omit to keep channels dynamic.
- `--batch-size`: fix feature map batch dimension in ONNX. Omit to keep batch dynamic.
- `--input-hw-size`: fix `input_images_or_features` height/width (`H W`) in ONNX input.
- `--spatial-scale`: ROI coordinate scale. One value means shared H/W scale, two values mean `(scale_h, scale_w)`.
- `--output-height`: output height behavior. Omitted means dynamic scalar ONNX input (`output_height`), specified integer means fixed output height in graph.
- `--output-width`: output width behavior. Omitted means dynamic scalar ONNX input (`output_width`), specified integer means fixed output width in graph.
- `--opset-version`: ONNX opset version (`>= 16`).
- `--vit-batch-size`: fix ViT output axis-0 (`B`). Omit to keep dynamic.
- `--vit-num-queries`: fix ViT output axis-1 (`Q`). Omit to keep dynamic.
- `--vit-output-fields`: fix ViT output axis-2 (`F`). Omit to keep dynamic.
- `--vit-box-format`: interpretation of fields `1:5` in ViT output (`xyxy` or `xywh`).
- `--onnx-output-path`: export destination path.
- `--use-score-threshold`: enable score filtering with a fixed threshold value in `[0.001, 1.000]`.
- `--score-threshold-as-input`: enable score filtering and expose `score_threshold` as a runtime scalar ONNX input.
- `--aligned` / `--no-aligned`: switch ROIAlign alignment behavior (`align_corners=True/False`). Default is `--no-aligned`.

`--spatial-scale` fallback behavior:

1. `--spatial-scale` specified: use the specified value.
2. `--spatial-scale` omitted and `--input-hw-size` specified: use `(H, W)` as fixed spatial scale.
3. both omitted: use runtime `input_images_or_features` `H/W` dynamically (`spatial_scale=None`).

Note:

- `--input-hw-size` requires exactly two integers: `H W`.

### ViT output behavior (important)

The integrated model expects ViT output as `[B, Q, F]`:

1. `B`: batch size
2. `Q`: number of queries
3. `F`: number of fields per query

Expected common field layout:

1. `field[0]`: label id
2. `field[1:5]`: box coordinates
3. `field[5]`: score

Box conversion behavior:

1. Fields `1:5` are decoded by `--vit-box-format`.
2. Coordinates are converted to normalized `[x1, y1, x2, y2]` in `[0, 1]`.
3. `rois` are generated as `[batch_idx, x1, y1, x2, y2]`.
4. ROI count is `B * Q`.

When score filtering is enabled (`--use-score-threshold` or `--score-threshold-as-input`):

1. Filtering uses `field[5]` as score.
2. Queries with `score >= threshold` are kept.
3. Output is flattened as `[num_rois, C, output_H, output_W]` because ROI count becomes data-dependent.
4. `--use-score-threshold` and `--score-threshold-as-input` are mutually exclusive.

For typical DEIMv2 output (`[N, 680, 6]`), use:

```bash
python 03_dynamic_roi_align_vit.py \
--vit-num-queries 680 \
--vit-output-fields 6
```

### Output size behavior (important)

The exported ONNX inputs change depending on `--output-height` / `--output-width`:

1. Both omitted: ONNX inputs include `output_height` and `output_width` (both dynamic scalar inputs), and output shape `H/W` are dynamic.
2. `--output-height` specified, `--output-width` omitted: ONNX input includes only `output_width`; output height is fixed and output width is dynamic.
3. `--output-height` omitted, `--output-width` specified: ONNX input includes only `output_height`; output height is dynamic and output width is fixed.
4. Both specified: ONNX inputs do not include output size scalars, and output `H/W` are fixed constants.

Note:

- If output size options are omitted, the script still uses local test values (`7x7`) only for the internal test forward pass before export.

### ONNX names

- Input name: `input_images_or_features`
- Input name: `vit_output`
- Optional input names: `output_height`, `output_width` (only when dynamic), `score_threshold` (when `--score-threshold-as-input` is used)
- Output name: `aligned_features`

### Metadata written to ONNX

The script adds descriptions to ONNX metadata for:

- `input_images_or_features`
- `vit_output`
- `vit_box_format`
- `use_score_threshold`, `score_threshold_as_input` (`score_threshold` is added when enabled)
- `aligned`
- `aligned_features`
- `input_hw_size` (only when fixed input size is used)
- `output_height` (only when fixed output height is used)
- `output_width` (only when fixed output width is used)

### Examples

1. Fully dynamic output sizes and dynamic ViT axes:

```bash
python 03_dynamic_roi_align_vit.py
```

2. Typical DEIMv2 output shape with fixed ROIAlign output size:

```bash
python 03_dynamic_roi_align_vit.py \
--batch-size 1 \
--channels 3 \
--input-hw-size 480 640 \
--vit-batch-size 1 \
--vit-num-queries 680 \
--vit-output-fields 6 \
--score-threshold-as-input \
--output-height 128 \
--output-width 64
```

3. Fix channels and batch, keep output sizes dynamic:

```bash
python 03_dynamic_roi_align_vit.py \
--channels 256 \
--batch-size 1
```

4. Use `xywh` ViT box format:

```bash
python 03_dynamic_roi_align_vit.py --vit-box-format xywh
```

5. Save to custom path:

```bash
python 03_dynamic_roi_align_vit.py --onnx-output-path out/roi_align_vit.onnx
```

6. Enable score threshold filtering:

```bash
python 03_dynamic_roi_align_vit.py --use-score-threshold 0.250
```

7. Expose score threshold as runtime ONNX input:

```bash
python 03_dynamic_roi_align_vit.py --score-threshold-as-input
```
