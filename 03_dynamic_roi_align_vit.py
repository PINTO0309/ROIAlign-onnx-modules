"""Dynamic ROI Align with integrated ViT output preprocessing.

This module exports an ONNX model that:
1. receives ViT output tensor [B, Q, F] (default F=6),
2. converts it into ROIAlign rois [B*Q, 5],
3. runs DynamicRoIAlign.
"""

import argparse
import typing
import warnings
from typing import Optional

import onnx
import torch
from onnxsim import simplify

from dynamic_roi_align_core import DynamicRoIAlign


class VitOutputToRois(torch.nn.Module):
    """Convert ViT output [B, Q, F] to ROIAlign rois [B*Q, 5]."""

    def __init__(self, box_format: str = "xyxy", score_threshold: Optional[float] = None):
        super().__init__()
        if box_format not in ("xyxy", "xywh"):
            raise ValueError("box_format must be 'xyxy' or 'xywh'")
        self.box_format = box_format
        self.score_threshold = score_threshold

    def forward(
        self,
        vit_output: torch.Tensor,
        input_images_or_features: torch.Tensor,
        score_threshold: Optional[torch.Tensor | float] = None,
    ) -> torch.Tensor:
        # vit_output: [B, Q, F], expected field order:
        # [label, x1, y1, x2, y2, score] for xyxy
        # [label, cx, cy, w, h, score] for xywh
        boxes = vit_output[:, :, 1:5]
        if self.box_format == "xywh":
            cx = boxes[:, :, 0]
            cy = boxes[:, :, 1]
            w = boxes[:, :, 2]
            h = boxes[:, :, 3]
            half_w = w * 0.5
            half_h = h * 0.5
            x1 = cx - half_w
            y1 = cy - half_h
            x2 = cx + half_w
            y2 = cy + half_h
        else:
            x1 = boxes[:, :, 0]
            y1 = boxes[:, :, 1]
            x2 = boxes[:, :, 2]
            y2 = boxes[:, :, 3]

        # Normalize coordinates to [0, 1] for DynamicRoIAlign.
        feat_h, feat_w = input_images_or_features.shape[2], input_images_or_features.shape[3]
        x1 = x1 / feat_w
        y1 = y1 / feat_h
        x2 = x2 / feat_w
        y2 = y2 / feat_h

        x_min = torch.clamp(torch.minimum(x1, x2), 0.0, 1.0)
        y_min = torch.clamp(torch.minimum(y1, y2), 0.0, 1.0)
        x_max = torch.clamp(torch.maximum(x1, x2), 0.0, 1.0)
        y_max = torch.clamp(torch.maximum(y1, y2), 0.0, 1.0)

        batch_size = vit_output.shape[0]
        num_queries = vit_output.shape[1]
        batch_indices = (
            torch.arange(batch_size, device=vit_output.device, dtype=vit_output.dtype)
            .unsqueeze(1)
            .expand(batch_size, num_queries)
        )

        rois = torch.stack([batch_indices, x_min, y_min, x_max, y_max], dim=2).reshape(-1, 5)

        effective_score_threshold = score_threshold
        if effective_score_threshold is None:
            effective_score_threshold = self.score_threshold

        if effective_score_threshold is not None:
            if not torch.is_tensor(effective_score_threshold):
                effective_score_threshold = torch.tensor(
                    effective_score_threshold,
                    dtype=vit_output.dtype,
                    device=vit_output.device,
                )
            else:
                effective_score_threshold = effective_score_threshold.to(
                    dtype=vit_output.dtype,
                    device=vit_output.device,
                ).reshape(())

            score = vit_output[:, :, 5]
            score_mask = score >= effective_score_threshold
            rois = rois[score_mask.reshape(-1)]

        return rois


class DynamicRoIAlignFromVit(torch.nn.Module):
    """Integrated pipeline: ViT output preprocessing + DynamicRoIAlign."""

    def __init__(
        self,
        roi_align_module: DynamicRoIAlign,
        vit_box_format: str = "xyxy",
        score_threshold: Optional[float] = None,
    ):
        super().__init__()
        self.roi_align = roi_align_module
        self.vit_to_rois = VitOutputToRois(
            box_format=vit_box_format,
            score_threshold=score_threshold,
        )

    def forward(
        self,
        input_images_or_features: torch.Tensor,
        vit_output: torch.Tensor,
        output_height: Optional[torch.Tensor | int],
        output_width: Optional[torch.Tensor | int],
        score_threshold: Optional[torch.Tensor | float] = None,
    ) -> torch.Tensor:
        runtime_score_threshold = score_threshold
        if runtime_score_threshold is None:
            runtime_score_threshold = self.vit_to_rois.score_threshold

        rois = typing.cast(
            torch.Tensor,
            self.vit_to_rois(
                vit_output,
                input_images_or_features,
                runtime_score_threshold,
            ),
        )
        aligned_features = typing.cast(
            torch.Tensor,
            self.roi_align(input_images_or_features, rois, output_height, output_width),
        )

        if runtime_score_threshold is not None:
            # Filtering makes per-batch ROI counts data-dependent.
            # Keep flattened [num_rois, C, H, W] representation.
            return aligned_features

        # Preserve batch axis for non-filtering case: [B, Q, C, H, W].
        batch_size = vit_output.shape[0]
        num_queries = vit_output.shape[1]
        channels = aligned_features.shape[1]
        out_h = aligned_features.shape[2]
        out_w = aligned_features.shape[3]
        return aligned_features.reshape(batch_size, num_queries, channels, out_h, out_w)


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
        "--input-channels",
        type=int,
        default=None,
        help="If specified, fix the feature-map channel dimension in ONNX export.",
    )
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=None,
        help="If specified, fix the feature-map batch dimension in ONNX export.",
    )
    parser.add_argument(
        "--input-hw-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="If specified, fix input_images_or_features height/width in ONNX export.",
    )
    parser.add_argument(
        "--spatial-scale",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Spatial scale for ROI coordinates. Specify 1 value (shared) or 2 values "
            "(height width). If omitted, uses --input-hw-size when provided, "
            "otherwise uses runtime input_images_or_features H/W."
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
        "--vit-batch-size",
        type=int,
        default=None,
        help="If specified, fix ViT output axis-0 size. If omitted, keep dynamic axis-0.",
    )
    parser.add_argument(
        "--vit-num-queries",
        type=int,
        default=None,
        help="If specified, fix ViT output axis-1 size. If omitted, keep dynamic axis-1.",
    )
    parser.add_argument(
        "--vit-output-fields",
        type=int,
        default=None,
        help="If specified, fix ViT output axis-2 size. If omitted, keep dynamic axis-2.",
    )
    parser.add_argument(
        "--vit-box-format",
        type=str,
        choices=["xyxy", "xywh"],
        default="xyxy",
        help="Interpretation of fields 1:5 in ViT output.",
    )
    parser.add_argument(
        "--onnx-output-path",
        type=str,
        default="dynamic_roi_align_vit.onnx",
        help="Export destination ONNX path.",
    )
    parser.add_argument(
        "--use-score-threshold",
        type=float,
        default=None,
        help="Enable score filtering with threshold in [0.001, 1.000].",
    )
    parser.add_argument(
        "--score-threshold-as-input",
        action="store_true",
        help="Enable score filtering and expose score_threshold as a runtime scalar ONNX input.",
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

    if args.input_channels is not None and args.input_channels <= 0:
        raise ValueError("--input-channels must be a positive integer")
    if args.input_batch_size is not None and args.input_batch_size <= 0:
        raise ValueError("--input-batch-size must be a positive integer")
    if args.input_hw_size is not None:
        if args.input_hw_size[0] <= 0 or args.input_hw_size[1] <= 0:
            raise ValueError("--input-hw-size values must be positive integers")
    if args.spatial_scale is not None and len(args.spatial_scale) not in (1, 2):
        raise ValueError("--spatial-scale must have 1 or 2 values")
    if args.output_height is not None and args.output_height <= 0:
        raise ValueError("--output-height must be a positive integer")
    if args.output_width is not None and args.output_width <= 0:
        raise ValueError("--output-width must be a positive integer")
    if args.opset_version < 16:
        raise ValueError("--opset-version must be >= 16")
    if args.vit_batch_size is not None and args.vit_batch_size <= 0:
        raise ValueError("--vit-batch-size must be a positive integer")
    if args.vit_num_queries is not None and args.vit_num_queries <= 0:
        raise ValueError("--vit-num-queries must be a positive integer")
    if args.vit_output_fields is not None and args.vit_output_fields < 5:
        raise ValueError("--vit-output-fields must be >= 5")
    if args.input_batch_size is not None and args.vit_batch_size is not None and args.input_batch_size != args.vit_batch_size:
        raise ValueError("--input-batch-size and --vit-batch-size must match when both are specified")
    if args.use_score_threshold is not None and not (0.001 <= args.use_score_threshold <= 1.0):
        raise ValueError("--use-score-threshold must be in the range [0.001, 1.000]")
    if args.use_score_threshold is not None and args.score_threshold_as_input:
        raise ValueError("--use-score-threshold and --score-threshold-as-input are mutually exclusive")
    if (args.use_score_threshold is not None or args.score_threshold_as_input) and args.vit_output_fields is not None and args.vit_output_fields < 6:
        raise ValueError("--vit-output-fields must be >= 6 when score filtering is enabled")

    if args.spatial_scale is None:
        if args.input_hw_size is not None:
            spatial_scale_arg = (args.input_hw_size[0], args.input_hw_size[1])
        else:
            spatial_scale_arg = None
    elif len(args.spatial_scale) == 1:
        spatial_scale_arg = args.spatial_scale[0]
    else:
        spatial_scale_arg = (args.spatial_scale[0], args.spatial_scale[1])

    test_input_batch_size = args.input_batch_size if args.input_batch_size is not None else (
        args.vit_batch_size if args.vit_batch_size is not None else 1
    )
    vit_batch_size = args.vit_batch_size if args.vit_batch_size is not None else test_input_batch_size

    test_input_channels = args.input_channels if args.input_channels is not None else 256
    vit_num_queries = args.vit_num_queries if args.vit_num_queries is not None else 680
    vit_output_fields = args.vit_output_fields if args.vit_output_fields is not None else 6
    input_height = args.input_hw_size[0] if args.input_hw_size is not None else 56
    input_width = args.input_hw_size[1] if args.input_hw_size is not None else 56

    input_images_or_features = torch.randn(test_input_batch_size, test_input_channels, input_height, input_width)
    vit_output = torch.randn(vit_batch_size, vit_num_queries, vit_output_fields)

    # Build valid-ish box values for fields 1:5.
    feat_h = input_images_or_features.shape[2]
    feat_w = input_images_or_features.shape[3]
    rand_x1 = torch.rand(vit_batch_size, vit_num_queries) * (feat_w * 0.7)
    rand_y1 = torch.rand(vit_batch_size, vit_num_queries) * (feat_h * 0.7)
    rand_x2 = rand_x1 + torch.rand(vit_batch_size, vit_num_queries) * torch.clamp(feat_w - rand_x1, min=1.0)
    rand_y2 = rand_y1 + torch.rand(vit_batch_size, vit_num_queries) * torch.clamp(feat_h - rand_y1, min=1.0)

    vit_output[:, :, 0] = torch.randint(low=0, high=34, size=(vit_batch_size, vit_num_queries)).to(vit_output.dtype)
    if args.vit_box_format == "xywh":
        vit_output[:, :, 1] = (rand_x1 + rand_x2) * 0.5
        vit_output[:, :, 2] = (rand_y1 + rand_y2) * 0.5
        vit_output[:, :, 3] = torch.clamp(rand_x2 - rand_x1, min=1.0)
        vit_output[:, :, 4] = torch.clamp(rand_y2 - rand_y1, min=1.0)
    else:
        vit_output[:, :, 1] = rand_x1
        vit_output[:, :, 2] = rand_y1
        vit_output[:, :, 3] = rand_x2
        vit_output[:, :, 4] = rand_y2
    if vit_output_fields >= 6:
        vit_output[:, :, 5] = torch.rand(vit_batch_size, vit_num_queries)

    dynamic_output_height = args.output_height is None
    dynamic_output_width = args.output_width is None
    dynamic_score_threshold = args.score_threshold_as_input
    output_height = args.output_height if args.output_height is not None else 7
    output_width = args.output_width if args.output_width is not None else 7

    roi_align_module = DynamicRoIAlign(spatial_scale=spatial_scale_arg, aligned=args.aligned)
    integrated_module = DynamicRoIAlignFromVit(
        roi_align_module,
        vit_box_format=args.vit_box_format,
        score_threshold=args.use_score_threshold,
    )

    if dynamic_score_threshold:
        output = integrated_module(
            input_images_or_features,
            vit_output,
            output_height,
            output_width,
            torch.tensor(0.25, dtype=vit_output.dtype),
        )
    else:
        output = integrated_module(input_images_or_features, vit_output, output_height, output_width)
    print("Output shape:", output.shape)

    onnx_output_name = "aligned_features"
    dynamic_axes = {
        "input_images_or_features": {},
        "vit_output": {},
        onnx_output_name: {},
    }

    if args.input_batch_size is None:
        dynamic_axes["input_images_or_features"][0] = "batch_size"
    if args.input_hw_size is None:
        dynamic_axes["input_images_or_features"][2] = "H"
        dynamic_axes["input_images_or_features"][3] = "W"

    if args.vit_batch_size is None:
        dynamic_axes["vit_output"][0] = "vit_batch_size"
    if args.vit_num_queries is None:
        dynamic_axes["vit_output"][1] = "num_queries"
    if args.vit_output_fields is None:
        dynamic_axes["vit_output"][2] = "vit_output_fields"

    output_is_flattened = args.use_score_threshold is not None or dynamic_score_threshold

    if output_is_flattened:
        # Output: [num_rois, C, H, W]
        dynamic_axes[onnx_output_name][0] = "num_rois"
        if args.input_channels is None:
            dynamic_axes["input_images_or_features"][1] = "channels"
            dynamic_axes[onnx_output_name][1] = "channels"
        output_axis_h = 2
        output_axis_w = 3
    else:
        # Output: [B, Q, C, H, W]
        if args.vit_batch_size is None:
            dynamic_axes[onnx_output_name][0] = "vit_batch_size"
        if args.vit_num_queries is None:
            dynamic_axes[onnx_output_name][1] = "num_queries"
        if args.input_channels is None:
            dynamic_axes["input_images_or_features"][1] = "channels"
            dynamic_axes[onnx_output_name][2] = "channels"
        output_axis_h = 3
        output_axis_w = 4

    input_names = ["input_images_or_features", "vit_output"]

    if dynamic_output_height and dynamic_output_width:
        export_module = integrated_module
        export_args = (
            input_images_or_features,
            vit_output,
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
            def __init__(self, module: DynamicRoIAlignFromVit, fixed_width: int):
                super().__init__()
                self.module = module
                self.fixed_width = fixed_width

            def forward(
                self,
                input_images_or_features: torch.Tensor,
                vit_output: torch.Tensor,
                output_height: torch.Tensor,
                score_threshold: Optional[torch.Tensor | float] = None,
            ) -> torch.Tensor:
                return typing.cast(
                    torch.Tensor,
                    self.module(
                        input_images_or_features,
                        vit_output,
                        output_height,
                        self.fixed_width,
                        score_threshold,
                    ),
                )

        export_module = ExportFixedWidth(integrated_module, output_width)
        export_args = (input_images_or_features, vit_output, torch.tensor(output_height))
        input_names.append("output_height")
        dynamic_axes["output_height"] = {}
        dynamic_axes[onnx_output_name][output_axis_h] = "output_H"
    elif dynamic_output_width:
        class ExportFixedHeight(torch.nn.Module):
            def __init__(self, module: DynamicRoIAlignFromVit, fixed_height: int):
                super().__init__()
                self.module = module
                self.fixed_height = fixed_height

            def forward(
                self,
                input_images_or_features: torch.Tensor,
                vit_output: torch.Tensor,
                output_width: torch.Tensor,
                score_threshold: Optional[torch.Tensor | float] = None,
            ) -> torch.Tensor:
                return typing.cast(
                    torch.Tensor,
                    self.module(
                        input_images_or_features,
                        vit_output,
                        self.fixed_height,
                        output_width,
                        score_threshold,
                    ),
                )

        export_module = ExportFixedHeight(integrated_module, output_height)
        export_args = (input_images_or_features, vit_output, torch.tensor(output_width))
        input_names.append("output_width")
        dynamic_axes["output_width"] = {}
        dynamic_axes[onnx_output_name][output_axis_w] = "output_W"
    else:
        class ExportFixedOutputSize(torch.nn.Module):
            def __init__(self, module: DynamicRoIAlignFromVit, fixed_height: int, fixed_width: int):
                super().__init__()
                self.module = module
                self.fixed_height = fixed_height
                self.fixed_width = fixed_width

            def forward(
                self,
                input_images_or_features: torch.Tensor,
                vit_output: torch.Tensor,
                score_threshold: Optional[torch.Tensor | float] = None,
            ) -> torch.Tensor:
                return typing.cast(
                    torch.Tensor,
                    self.module(
                        input_images_or_features,
                        vit_output,
                        self.fixed_height,
                        self.fixed_width,
                        score_threshold,
                    ),
                )

        export_module = ExportFixedOutputSize(integrated_module, output_height, output_width)
        export_args = (input_images_or_features, vit_output)

    if dynamic_score_threshold:
        export_args = (*export_args, torch.tensor(0.25, dtype=vit_output.dtype))
        input_names.append("score_threshold")
        dynamic_axes["score_threshold"] = {}

    onnx_model_path = args.onnx_output_path
    print("\nTesting ONNX export...")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are using the legacy TorchScript-based ONNX export.*",
            category=DeprecationWarning,
        )
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
    if args.input_batch_size is None:
        _set_input_dim_param(simplified_model, "input_images_or_features", 0, "batch_size")
    if args.input_channels is None:
        _set_input_dim_param(simplified_model, "input_images_or_features", 1, "channels")
        _set_output_dim_param(simplified_model, onnx_output_name, output_channel_axis, "channels")
    if args.input_hw_size is None:
        _set_input_dim_param(simplified_model, "input_images_or_features", 2, "H")
        _set_input_dim_param(simplified_model, "input_images_or_features", 3, "W")
    if args.vit_batch_size is None:
        _set_input_dim_param(simplified_model, "vit_output", 0, "vit_batch_size")
    if args.vit_num_queries is None:
        _set_input_dim_param(simplified_model, "vit_output", 1, "num_queries")
    if args.vit_output_fields is None:
        _set_input_dim_param(simplified_model, "vit_output", 2, "vit_output_fields")
    if output_is_flattened:
        _set_output_dim_param(simplified_model, onnx_output_name, 0, "num_rois")
    else:
        if args.vit_batch_size is None:
            _set_output_dim_param(simplified_model, onnx_output_name, 0, "vit_batch_size")
        if args.vit_num_queries is None:
            _set_output_dim_param(simplified_model, onnx_output_name, 1, "num_queries")
    if dynamic_output_height:
        _set_output_dim_param(simplified_model, onnx_output_name, output_axis_h, "output_H")
    if dynamic_output_width:
        _set_output_dim_param(simplified_model, onnx_output_name, output_axis_w, "output_W")

    existing_metadata = {item.key: item.value for item in simplified_model.metadata_props}
    reserved_metadata_keys = {
        "input_images_or_features",
        "vit_output",
        "aligned",
        "vit_box_format",
        "use_score_threshold",
        "score_threshold_as_input",
        "score_threshold",
        "input_hw_size",
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
        "Input tensor shape: [N, C, H, W]\n"
        "N: batch size\n"
        "C: channels\n"
        "H/W: feature map size"
    )
    model_metadata["vit_output"] = (
        "ViT output tensor shape: [B, Q, F]\n"
        "B: batch size, Q: number of queries, F: fields\n"
        f"Fields 1:5 are interpreted as box format: {args.vit_box_format}\n"
        "Expected common layout: [label, x1, y1, x2, y2, score]"
    )
    model_metadata["vit_box_format"] = args.vit_box_format
    model_metadata["use_score_threshold"] = str(args.use_score_threshold is not None).lower()
    model_metadata["score_threshold_as_input"] = str(dynamic_score_threshold).lower()
    if args.use_score_threshold is not None:
        model_metadata["score_threshold"] = f"{args.use_score_threshold:.3f}"
    elif dynamic_score_threshold:
        model_metadata["score_threshold"] = "Runtime scalar input: score_threshold"
    model_metadata["aligned"] = (
        "ROIAlign aligned mode (bool)\n"
        f"Current export value: {str(args.aligned).lower()}"
    )
    if args.input_hw_size is not None:
        model_metadata["input_hw_size"] = (
            "Fixed input feature-map size [H, W] (int, int)\n"
            f"Current export value: [{input_height}, {input_width}]"
        )
    if not dynamic_output_height:
        model_metadata["output_height"] = (
            "Fixed output feature height (int)\n"
            f"Current export value: {output_height}"
        )
    if not dynamic_output_width:
        model_metadata["output_width"] = (
            "Fixed output feature width (int)\n"
            f"Current export value: {output_width}"
        )
    if output_is_flattened:
        model_metadata[onnx_output_name] = (
            "Aligned ROI features shape: [num_rois, C, output_H, output_W]\n"
            "num_rois: number of queries kept after score filtering"
        )
    else:
        model_metadata[onnx_output_name] = (
            "Aligned ROI features shape: [B, Q, C, output_H, output_W]\n"
            "B: vit batch size, Q: number of queries"
        )
    onnx.helper.set_model_props(simplified_model, model_metadata)

    onnx.save(simplified_model, onnx_model_path)
    print(f"Simplified ONNX model exported successfully to: {onnx_model_path}")
