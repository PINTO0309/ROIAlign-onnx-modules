"""Dynamic ROI Align with integrated YOLO output preprocessing.

This module exports an ONNX model that:
1. receives YOLO output tensor [B, S, N],
2. converts it into ROIAlign rois [B*N, 5],
3. runs DynamicRoIAlign.
"""

import argparse
from typing import Optional, Tuple, cast

import onnx
import torch
from onnxsim import simplify


class DynamicRoIAlign(torch.nn.Module):
    """Dynamic ROI Align layer that supports variable output sizes."""

    def __init__(
        self,
        spatial_scale: Optional[Tuple[float, float] | float] = (640, 640),
        sampling_ratio: int = -1,
        aligned: bool = False,
    ):
        super().__init__()
        if spatial_scale is None:
            self.spatial_scale = None
            self.spatial_scale_h = None
            self.spatial_scale_w = None
        elif isinstance(spatial_scale, (list, tuple)):
            assert len(spatial_scale) == 2, "spatial_scale tuple must have 2 elements (height, width)"
            self.spatial_scale = spatial_scale
            self.spatial_scale_h = spatial_scale[0]
            self.spatial_scale_w = spatial_scale[1]
        else:
            self.spatial_scale = spatial_scale
            self.spatial_scale_h = spatial_scale
            self.spatial_scale_w = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(
        self,
        input_images_or_features: torch.Tensor,
        rois: torch.Tensor,
        output_height: Optional[torch.Tensor | int],
        output_width: Optional[torch.Tensor | int],
    ) -> torch.Tensor:
        batch_indices = rois[:, 0].long()

        if self.spatial_scale is None:
            spatial_scale_h = input_images_or_features.shape[2]
            spatial_scale_w = input_images_or_features.shape[3]
        else:
            assert self.spatial_scale_h is not None
            assert self.spatial_scale_w is not None
            spatial_scale_h = self.spatial_scale_h
            spatial_scale_w = self.spatial_scale_w

        x1 = rois[:, 1] * spatial_scale_w
        y1 = rois[:, 2] * spatial_scale_h
        x2 = rois[:, 3] * spatial_scale_w
        y2 = rois[:, 4] * spatial_scale_h

        roi_width = x2 - x1
        roi_height = y2 - y1

        if isinstance(output_width, (list, tuple)):
            output_width = output_width[0] if len(output_width) > 0 else output_width
        if isinstance(output_height, (list, tuple)):
            output_height = output_height[0] if len(output_height) > 0 else output_height
        if not torch.is_tensor(output_width):
            output_width = torch.tensor(output_width, dtype=torch.int64, device=rois.device)
        else:
            output_width = output_width.to(dtype=torch.int64, device=rois.device).reshape(())
        if not torch.is_tensor(output_height):
            output_height = torch.tensor(output_height, dtype=torch.int64, device=rois.device)
        else:
            output_height = output_height.to(dtype=torch.int64, device=rois.device).reshape(())

        output_width_f = output_width.to(dtype=rois.dtype)
        output_height_f = output_height.to(dtype=rois.dtype)

        x_indices = torch.arange(cast(int, output_width), device=rois.device, dtype=rois.dtype)
        y_indices = torch.arange(cast(int, output_height), device=rois.device, dtype=rois.dtype)

        x_denominator = torch.clamp(output_width_f - 1, min=1)
        y_denominator = torch.clamp(output_height_f - 1, min=1)
        x_coords_normalized = x_indices / x_denominator
        y_coords_normalized = y_indices / y_denominator

        grid_y, grid_x = torch.meshgrid(y_coords_normalized, x_coords_normalized, indexing="ij")

        fx = x1.unsqueeze(1).unsqueeze(2) + grid_x * roi_width.unsqueeze(1).unsqueeze(2)
        fy = y1.unsqueeze(1).unsqueeze(2) + grid_y * roi_height.unsqueeze(1).unsqueeze(2)

        h_feat, w_feat = input_images_or_features.shape[2], input_images_or_features.shape[3]
        if self.aligned:
            normalized_fx = (fx / (w_feat - 1)) * 2 - 1
            normalized_fy = (fy / (h_feat - 1)) * 2 - 1
        else:
            normalized_fx = (fx / w_feat) * 2 - 1
            normalized_fy = (fy / h_feat) * 2 - 1

        grids_tensor = torch.stack([normalized_fx, normalized_fy], dim=-1)
        selected_feature_maps = torch.index_select(input_images_or_features, 0, batch_indices)

        pooled_features = torch.nn.functional.grid_sample(
            selected_feature_maps,
            grids_tensor,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.aligned,
        )
        return pooled_features


class YoloOutputToRois(torch.nn.Module):
    """Convert YOLO output [B, S, N] to ROIAlign rois [B*N, 5]."""

    def __init__(self, box_format: str = "xywh", score_threshold: Optional[float] = None):
        super().__init__()
        if box_format not in ("xywh", "xyxy"):
            raise ValueError("box_format must be 'xywh' or 'xyxy'")
        self.box_format = box_format
        self.score_threshold = score_threshold

    def forward(self, yolo_output: torch.Tensor, input_images_or_features: torch.Tensor) -> torch.Tensor:
        # yolo_output: [B, S, N], use first 4 channels as box coordinates.
        boxes = yolo_output[:, :4, :]
        if self.box_format == "xywh":
            cx = boxes[:, 0, :]
            cy = boxes[:, 1, :]
            w = boxes[:, 2, :]
            h = boxes[:, 3, :]
            half_w = w * 0.5
            half_h = h * 0.5
            x1 = cx - half_w
            y1 = cy - half_h
            x2 = cx + half_w
            y2 = cy + half_h
        else:
            x1 = boxes[:, 0, :]
            y1 = boxes[:, 1, :]
            x2 = boxes[:, 2, :]
            y2 = boxes[:, 3, :]

        # Normalize coordinates to [0, 1] for DynamicRoIAlign's expected rois input.
        feat_h, feat_w = input_images_or_features.shape[2], input_images_or_features.shape[3]
        x1 = x1 / feat_w
        y1 = y1 / feat_h
        x2 = x2 / feat_w
        y2 = y2 / feat_h

        x_min = torch.clamp(torch.minimum(x1, x2), 0.0, 1.0)
        y_min = torch.clamp(torch.minimum(y1, y2), 0.0, 1.0)
        x_max = torch.clamp(torch.maximum(x1, x2), 0.0, 1.0)
        y_max = torch.clamp(torch.maximum(y1, y2), 0.0, 1.0)

        batch_size = yolo_output.shape[0]
        num_candidates = yolo_output.shape[2]
        batch_indices = (
            torch.arange(batch_size, device=yolo_output.device, dtype=yolo_output.dtype)
            .unsqueeze(1)
            .expand(batch_size, num_candidates)
        )

        rois = torch.stack([batch_indices, x_min, y_min, x_max, y_max], dim=2).reshape(-1, 5)

        if self.score_threshold is not None:
            # Class channels are expected at yolo_output[:, 4:, :].
            class_scores = yolo_output[:, 4:, :]
            max_scores = torch.amax(class_scores, dim=1)
            score_mask = max_scores >= self.score_threshold
            rois = rois[score_mask.reshape(-1)]

        return rois


class DynamicRoIAlignFromYolo(torch.nn.Module):
    """Integrated pipeline: YOLO output preprocessing + DynamicRoIAlign."""

    def __init__(
        self,
        roi_align_module: DynamicRoIAlign,
        yolo_box_format: str = "xywh",
        score_threshold: Optional[float] = None,
    ):
        super().__init__()
        self.roi_align = roi_align_module
        self.yolo_to_rois = YoloOutputToRois(
            box_format=yolo_box_format,
            score_threshold=score_threshold,
        )

    def forward(
        self,
        input_images_or_features: torch.Tensor,
        yolo_output: torch.Tensor,
        output_height: Optional[torch.Tensor | int],
        output_width: Optional[torch.Tensor | int],
    ) -> torch.Tensor:
        rois = self.yolo_to_rois(yolo_output, input_images_or_features)
        aligned_features = self.roi_align(input_images_or_features, rois, output_height, output_width)

        if self.yolo_to_rois.score_threshold is not None:
            # Filtering makes per-batch ROI counts data-dependent.
            # Keep flattened [num_rois, C, H, W] representation.
            return aligned_features

        # Preserve batch axis for the non-filtering case: [B, N, C, H, W].
        batch_size = yolo_output.shape[0]
        num_candidates = yolo_output.shape[2]
        channels = aligned_features.shape[1]
        out_h = aligned_features.shape[2]
        out_w = aligned_features.shape[3]
        return aligned_features.reshape(batch_size, num_candidates, channels, out_h, out_w)


def _set_input_dim_param(model: onnx.ModelProto, input_name: str, axis: int, dim_param: str) -> None:
    for graph_input in model.graph.input:
        if graph_input.name == input_name:
            dim = graph_input.type.tensor_type.shape.dim[axis]
            dim.ClearField("dim_value")
            dim.dim_param = dim_param
            return


def _set_output_dim_param(model: onnx.ModelProto, output_name: str, axis: int, dim_param: str) -> None:
    for graph_output in model.graph.output:
        if graph_output.name == output_name:
            dim = graph_output.type.tensor_type.shape.dim[axis]
            dim.ClearField("dim_value")
            dim.dim_param = dim_param
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channels",
        type=int,
        default=None,
        help="If specified, fix the feature-map channel dimension in ONNX export.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="If specified, fix the feature-map batch dimension in ONNX export.",
    )
    parser.add_argument(
        "--spatial-scale",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Spatial scale for ROI coordinates. Specify 1 value (shared) or 2 values "
            "(height width). If omitted, input_images_or_features H/W are used."
        ),
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=None,
        help="If specified, fix output feature height. If omitted, keep dynamic ONNX input.",
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=None,
        help="If specified, fix output feature width. If omitted, keep dynamic ONNX input.",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version for export (must be >= 16). Default: 17.",
    )
    parser.add_argument(
        "--yolo-batch-size",
        type=int,
        default=None,
        help="If specified, fix YOLO output axis-0 size. If omitted, keep dynamic axis-0.",
    )
    parser.add_argument(
        "--yolo-output-channels",
        type=int,
        default=None,
        help="If specified, fix YOLO output axis-1 size. If omitted, keep dynamic axis-1.",
    )
    parser.add_argument(
        "--yolo-num-candidates",
        type=int,
        default=None,
        help="If specified, fix YOLO output axis-2 size. If omitted, keep dynamic axis-2.",
    )
    parser.add_argument(
        "--yolo-box-format",
        type=str,
        choices=["xywh", "xyxy"],
        default="xywh",
        help="Interpretation of first 4 YOLO output channels.",
    )
    parser.add_argument(
        "--onnx-output-path",
        type=str,
        default="dynamic_roi_align_yolo.onnx",
        help="Export destination ONNX path.",
    )
    parser.add_argument(
        "--use-score-threshold",
        type=float,
        default=None,
        help="Enable score filtering with threshold in [0.001, 1.000].",
    )

    aligned_group = parser.add_mutually_exclusive_group()
    aligned_group.add_argument(
        "--aligned",
        dest="aligned",
        action="store_true",
        help="Enable aligned ROIAlign sampling (align_corners=True).",
    )
    aligned_group.add_argument(
        "--no-aligned",
        dest="aligned",
        action="store_false",
        help="Disable aligned ROIAlign sampling (align_corners=False).",
    )
    parser.set_defaults(aligned=False)

    args = parser.parse_args()

    if args.channels is not None and args.channels <= 0:
        raise ValueError("--channels must be a positive integer")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    if args.spatial_scale is not None and len(args.spatial_scale) not in (1, 2):
        raise ValueError("--spatial-scale must have 1 or 2 values")
    if args.output_height is not None and args.output_height <= 0:
        raise ValueError("--output-height must be a positive integer")
    if args.output_width is not None and args.output_width <= 0:
        raise ValueError("--output-width must be a positive integer")
    if args.opset_version < 16:
        raise ValueError("--opset-version must be >= 16")
    if args.yolo_batch_size is not None and args.yolo_batch_size <= 0:
        raise ValueError("--yolo-batch-size must be a positive integer")
    if args.yolo_output_channels is not None and args.yolo_output_channels < 4:
        raise ValueError("--yolo-output-channels must be >= 4")
    if args.yolo_num_candidates is not None and args.yolo_num_candidates <= 0:
        raise ValueError("--yolo-num-candidates must be a positive integer")
    if args.batch_size is not None and args.yolo_batch_size is not None and args.batch_size != args.yolo_batch_size:
        raise ValueError("--batch-size and --yolo-batch-size must match when both are specified")
    if args.use_score_threshold is not None and not (0.001 <= args.use_score_threshold <= 1.0):
        raise ValueError("--use-score-threshold must be in the range [0.001, 1.000]")
    if args.use_score_threshold is not None and args.yolo_output_channels is not None:
        if args.yolo_output_channels < 5:
            raise ValueError("--yolo-output-channels must be >= 5 when score filtering is enabled")

    if args.spatial_scale is None:
        spatial_scale_arg = None
    elif len(args.spatial_scale) == 1:
        spatial_scale_arg = args.spatial_scale[0]
    else:
        spatial_scale_arg = (args.spatial_scale[0], args.spatial_scale[1])

    test_input_batch_size = args.batch_size if args.batch_size is not None else (
        args.yolo_batch_size if args.yolo_batch_size is not None else 1
    )
    yolo_batch_size = args.yolo_batch_size if args.yolo_batch_size is not None else test_input_batch_size

    test_input_channels = args.channels if args.channels is not None else 256
    yolo_output_channels = args.yolo_output_channels if args.yolo_output_channels is not None else 29
    yolo_num_candidates = args.yolo_num_candidates if args.yolo_num_candidates is not None else 6300

    input_images_or_features = torch.randn(test_input_batch_size, test_input_channels, 56, 56)
    yolo_output = torch.randn(yolo_batch_size, yolo_output_channels, yolo_num_candidates)

    # Build valid-ish box values for first 4 channels.
    feat_h = input_images_or_features.shape[2]
    feat_w = input_images_or_features.shape[3]
    rand_x1 = torch.rand(yolo_batch_size, yolo_num_candidates) * (feat_w * 0.7)
    rand_y1 = torch.rand(yolo_batch_size, yolo_num_candidates) * (feat_h * 0.7)
    rand_x2 = rand_x1 + torch.rand(yolo_batch_size, yolo_num_candidates) * torch.clamp(feat_w - rand_x1, min=1.0)
    rand_y2 = rand_y1 + torch.rand(yolo_batch_size, yolo_num_candidates) * torch.clamp(feat_h - rand_y1, min=1.0)

    if args.yolo_box_format == "xywh":
        yolo_output[:, 0, :] = (rand_x1 + rand_x2) * 0.5
        yolo_output[:, 1, :] = (rand_y1 + rand_y2) * 0.5
        yolo_output[:, 2, :] = torch.clamp(rand_x2 - rand_x1, min=1.0)
        yolo_output[:, 3, :] = torch.clamp(rand_y2 - rand_y1, min=1.0)
    else:
        yolo_output[:, 0, :] = rand_x1
        yolo_output[:, 1, :] = rand_y1
        yolo_output[:, 2, :] = rand_x2
        yolo_output[:, 3, :] = rand_y2

    dynamic_output_height = args.output_height is None
    dynamic_output_width = args.output_width is None
    output_height = args.output_height if args.output_height is not None else 7
    output_width = args.output_width if args.output_width is not None else 7

    roi_align_module = DynamicRoIAlign(spatial_scale=spatial_scale_arg, aligned=args.aligned)
    integrated_module = DynamicRoIAlignFromYolo(
        roi_align_module,
        yolo_box_format=args.yolo_box_format,
        score_threshold=args.use_score_threshold,
    )

    output = integrated_module(input_images_or_features, yolo_output, output_height, output_width)
    print("Output shape:", output.shape)

    onnx_output_name = "aligned_features"
    dynamic_axes = {
        "input_images_or_features": {2: "H", 3: "W"},
        "yolo_output": {},
        onnx_output_name: {},
    }

    if args.batch_size is None:
        dynamic_axes["input_images_or_features"][0] = "batch_size"

    if args.yolo_batch_size is None:
        dynamic_axes["yolo_output"][0] = "yolo_batch_size"
    if args.yolo_output_channels is None:
        dynamic_axes["yolo_output"][1] = "yolo_output_channels"
    if args.yolo_num_candidates is None:
        dynamic_axes["yolo_output"][2] = "num_candidates"

    output_is_flattened = args.use_score_threshold is not None

    if output_is_flattened:
        # Output: [num_rois, C, H, W]
        dynamic_axes[onnx_output_name][0] = "num_rois"
        if args.channels is None:
            dynamic_axes["input_images_or_features"][1] = "channels"
            dynamic_axes[onnx_output_name][1] = "channels"
        output_axis_h = 2
        output_axis_w = 3
    else:
        # Output: [B, N, C, H, W]
        if args.yolo_batch_size is None:
            dynamic_axes[onnx_output_name][0] = "yolo_batch_size"
        if args.yolo_num_candidates is None:
            dynamic_axes[onnx_output_name][1] = "num_candidates"
        if args.channels is None:
            dynamic_axes["input_images_or_features"][1] = "channels"
            dynamic_axes[onnx_output_name][2] = "channels"
        output_axis_h = 3
        output_axis_w = 4

    input_names = ["input_images_or_features", "yolo_output"]

    if dynamic_output_height and dynamic_output_width:
        export_module = integrated_module
        export_args = (
            input_images_or_features,
            yolo_output,
            torch.tensor(output_height),
            torch.tensor(output_width),
        )
        input_names.extend(["output_height", "output_width"])
        dynamic_axes["output_height"] = {}
        dynamic_axes["output_width"] = {}
        dynamic_axes[onnx_output_name][output_axis_h] = "output_H"
        dynamic_axes[onnx_output_name][output_axis_w] = "output_W"
    elif dynamic_output_height:
        class ExportFixedWidth(torch.nn.Module):
            def __init__(self, module: DynamicRoIAlignFromYolo, fixed_width: int):
                super().__init__()
                self.module = module
                self.fixed_width = fixed_width

            def forward(
                self,
                input_images_or_features: torch.Tensor,
                yolo_output: torch.Tensor,
                output_height: torch.Tensor,
            ) -> torch.Tensor:
                return self.module(input_images_or_features, yolo_output, output_height, self.fixed_width)

        export_module = ExportFixedWidth(integrated_module, output_width)
        export_args = (input_images_or_features, yolo_output, torch.tensor(output_height))
        input_names.append("output_height")
        dynamic_axes["output_height"] = {}
        dynamic_axes[onnx_output_name][output_axis_h] = "output_H"
    elif dynamic_output_width:
        class ExportFixedHeight(torch.nn.Module):
            def __init__(self, module: DynamicRoIAlignFromYolo, fixed_height: int):
                super().__init__()
                self.module = module
                self.fixed_height = fixed_height

            def forward(
                self,
                input_images_or_features: torch.Tensor,
                yolo_output: torch.Tensor,
                output_width: torch.Tensor,
            ) -> torch.Tensor:
                return self.module(input_images_or_features, yolo_output, self.fixed_height, output_width)

        export_module = ExportFixedHeight(integrated_module, output_height)
        export_args = (input_images_or_features, yolo_output, torch.tensor(output_width))
        input_names.append("output_width")
        dynamic_axes["output_width"] = {}
        dynamic_axes[onnx_output_name][output_axis_w] = "output_W"
    else:
        class ExportFixedOutputSize(torch.nn.Module):
            def __init__(self, module: DynamicRoIAlignFromYolo, fixed_height: int, fixed_width: int):
                super().__init__()
                self.module = module
                self.fixed_height = fixed_height
                self.fixed_width = fixed_width

            def forward(self, input_images_or_features: torch.Tensor, yolo_output: torch.Tensor) -> torch.Tensor:
                return self.module(input_images_or_features, yolo_output, self.fixed_height, self.fixed_width)

        export_module = ExportFixedOutputSize(integrated_module, output_height, output_width)
        export_args = (input_images_or_features, yolo_output)

    onnx_model_path = args.onnx_output_path
    print("\nTesting ONNX export...")
    torch.onnx.export(
        export_module,
        export_args,
        onnx_model_path,
        input_names=input_names,
        output_names=[onnx_output_name],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset_version,
        dynamo=False,
    )
    print(f"ONNX model exported successfully to: {onnx_model_path}")

    print("Running onnxsim...")
    simplified_model, check = simplify(onnx_model_path)
    if not check:
        raise RuntimeError("onnxsim validation failed")

    # Preserve symbolic dimensions where requested.
    output_channel_axis = 1 if output_is_flattened else 2
    if args.batch_size is None:
        _set_input_dim_param(simplified_model, "input_images_or_features", 0, "batch_size")
    if args.channels is None:
        _set_input_dim_param(simplified_model, "input_images_or_features", 1, "channels")
        _set_output_dim_param(simplified_model, onnx_output_name, output_channel_axis, "channels")
    if args.yolo_batch_size is None:
        _set_input_dim_param(simplified_model, "yolo_output", 0, "yolo_batch_size")
    if args.yolo_output_channels is None:
        _set_input_dim_param(simplified_model, "yolo_output", 1, "yolo_output_channels")
    if args.yolo_num_candidates is None:
        _set_input_dim_param(simplified_model, "yolo_output", 2, "num_candidates")
    if output_is_flattened:
        _set_output_dim_param(simplified_model, onnx_output_name, 0, "num_rois")
    else:
        if args.yolo_batch_size is None:
            _set_output_dim_param(simplified_model, onnx_output_name, 0, "yolo_batch_size")
        if args.yolo_num_candidates is None:
            _set_output_dim_param(simplified_model, onnx_output_name, 1, "num_candidates")
    if dynamic_output_height:
        _set_output_dim_param(simplified_model, onnx_output_name, output_axis_h, "output_H")
    if dynamic_output_width:
        _set_output_dim_param(simplified_model, onnx_output_name, output_axis_w, "output_W")

    existing_metadata = {item.key: item.value for item in simplified_model.metadata_props}
    reserved_metadata_keys = {
        "input_images_or_features",
        "yolo_output",
        "aligned",
        "yolo_box_format",
        "use_score_threshold",
        "score_threshold",
        "output_height",
        "output_width",
        onnx_output_name,
    }
    model_metadata = {
        key: value
        for key, value in existing_metadata.items()
        if key not in reserved_metadata_keys
    }
    model_metadata["input_images_or_features"] = (
        "Input tensor shape: [N, C, H, W]\\n"
        "N: batch size\\n"
        "C: channels\\n"
        "H/W: feature map size"
    )
    model_metadata["yolo_output"] = (
        "YOLO output tensor shape: [B, S, N]\\n"
        "B: batch size, S: channels, N: number of candidates\\n"
        f"First 4 channels are interpreted as box format: {args.yolo_box_format}"
    )
    model_metadata["yolo_box_format"] = args.yolo_box_format
    model_metadata["use_score_threshold"] = str(args.use_score_threshold is not None).lower()
    if args.use_score_threshold is not None:
        model_metadata["score_threshold"] = f"{args.use_score_threshold:.3f}"
    model_metadata["aligned"] = (
        "ROIAlign aligned mode (bool)\\n"
        f"Current export value: {str(args.aligned).lower()}"
    )
    if not dynamic_output_height:
        model_metadata["output_height"] = (
            "Fixed output feature height (int)\\n"
            f"Current export value: {output_height}"
        )
    if not dynamic_output_width:
        model_metadata["output_width"] = (
            "Fixed output feature width (int)\\n"
            f"Current export value: {output_width}"
        )
    if output_is_flattened:
        model_metadata[onnx_output_name] = (
            "Aligned ROI features shape: [num_rois, C, output_H, output_W]\\n"
            "num_rois: number of candidates kept after enabled filtering"
        )
    else:
        model_metadata[onnx_output_name] = (
            "Aligned ROI features shape: [B, N, C, output_H, output_W]\\n"
            "B: yolo batch size, N: number of candidates"
        )
    onnx.helper.set_model_props(simplified_model, model_metadata)

    onnx.save(simplified_model, onnx_model_path)
    print(f"Simplified ONNX model exported successfully to: {onnx_model_path}")
