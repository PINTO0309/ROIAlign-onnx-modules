# ROIAlign-onnx-modules
Various implementations of DynamicRoIAlign

## 01_dynamic_roi_align.py Usage

`01_dynamic_roi_align.py` exports a DynamicRoIAlign model to ONNX, runs `onnxsim`, and writes metadata into the exported model.

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
  [--spatial-scale SPATIAL_SCALE [SPATIAL_SCALE ...]] \
  [--output-height OUTPUT_HEIGHT] \
  [--output-width OUTPUT_WIDTH] \
  [--opset-version OPSET_VERSION] \
  [--aligned | --no-aligned]
```

Option summary:

- `--channels`: fix input/output channel dimension in ONNX. Omit to keep channels dynamic.
- `--batch-size`: fix batch dimension in ONNX input. Omit to keep batch size dynamic.
- `--spatial-scale`: ROI coordinate scale. One value means shared H/W scale, two values mean `(scale_h, scale_w)`, omitted means using input tensor `H/W` dynamically at runtime.
- `--output-height`: output height behavior. Omitted means dynamic scalar ONNX input (`output_height`), specified integer means fixed output height in graph.
- `--output-width`: output width behavior. Omitted means dynamic scalar ONNX input (`output_width`), specified integer means fixed output width in graph.
- `--opset-version`: ONNX opset version (`>= 16`).
- `--aligned` / `--no-aligned`: switch ROIAlign alignment behavior (`align_corners=True/False`). Default is `--no-aligned`.

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
- `output_height` (only when dynamic input is used)
- `output_width` (only when dynamic input is used)

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
