"""Dynamic ROI Align implementation with variable output sizes.

This module performs ROI Align operation with dynamic output heights and widths,
allowing different ROI sizes to be processed in a single forward pass.
"""

import argparse
import typing
import onnx
import torch
from onnxsim import simplify
from dynamic_roi_align_core import DynamicRoIAlign


if __name__ == '__main__':
    """Test DynamicRoIAlign with example data and ONNX export."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channels",
        type=int,
        default=None,
        help="If specified, fix the channel dimension to this value in ONNX export.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="If specified, fix the batch dimension to this value in ONNX export.",
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
        "--onnx-output-path",
        type=str,
        default="dynamic_roi_align.onnx",
        help="Export destination ONNX path.",
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

    if args.spatial_scale is None:
        if args.input_hw_size is not None:
            spatial_scale_arg = (args.input_hw_size[0], args.input_hw_size[1])
        else:
            spatial_scale_arg = None
    elif len(args.spatial_scale) == 1:
        spatial_scale_arg = args.spatial_scale[0]
    else:
        spatial_scale_arg = (args.spatial_scale[0], args.spatial_scale[1])

    feature_channels = args.channels
    test_input_channels = feature_channels if feature_channels is not None else 256
    feature_batch_size = args.batch_size
    test_input_batch_size = feature_batch_size if feature_batch_size is not None else 2
    input_height = args.input_hw_size[0] if args.input_hw_size is not None else 56
    input_width = args.input_hw_size[1] if args.input_hw_size is not None else 56

    # Create dummy input data for testing.
    # Feature map: batch_size=test_input_batch_size, channels=test_input_channels, height=input_height, width=input_width
    # When --batch-size/--channels are omitted (None), use 2/256 for this local test input only.
    input_images_or_features = torch.randn(test_input_batch_size, test_input_channels, input_height, input_width)

    # Define ROIs: [batch_idx, x1, y1, x2, y2]
    # Note: Coordinates are in feature map space (0-56 range)
    roi_second_batch_idx = min(1, test_input_batch_size - 1)
    rois = torch.tensor([
        [0, 10, 10, 50, 50],  # ROI from first image: 40x40 region
        [0, 20, 20, 60, 60],  # Another ROI from first image (overlaps edge)
        [roi_second_batch_idx, 5, 5, 45, 45]  # ROI from another valid image in the batch
    ], dtype=torch.float32)

    # Define desired output size for local test run.
    # If CLI values are omitted, use representative defaults for this script run only.
    dynamic_output_height = args.output_height is None
    dynamic_output_width = args.output_width is None
    output_height = args.output_height if args.output_height is not None else 7
    output_width = args.output_width if args.output_width is not None else 7

    # Create and test the module
    # When spatial_scale_arg is None, DynamicRoIAlign uses input_images_or_features H/W dynamically.
    roi_align_module = DynamicRoIAlign(spatial_scale=spatial_scale_arg, aligned=args.aligned)
    output = typing.cast(
        torch.Tensor,
        roi_align_module(input_images_or_features, rois, output_height, output_width),
    )

    print("Output shape:", output.shape)
    print(
        f"Expected: (num_rois={rois.shape[0]}, channels={test_input_channels}, "
        f"height={output_height}, width={output_width})"
    )

    # ONNX Export Test
    print("\nTesting ONNX export...")

    # Define dynamic axes for flexible input/output shapes
    # This allows the ONNX model to accept variable batch sizes, ROI counts, and output sizes
    onnx_output_name = "aligned_features"

    dynamic_axes = {
        "input_images_or_features": {},
        "rois": {
            0: "num_rois"       # Variable number of ROIs
        },
        onnx_output_name: {
            0: "num_rois",      # Matches number of input ROIs
        }
    }
    if args.batch_size is None:
        # Default behavior: keep batch size dynamic.
        dynamic_axes["input_images_or_features"][0] = "batch_size"
    if args.input_hw_size is None:
        # Default behavior: keep input feature map height/width dynamic.
        dynamic_axes["input_images_or_features"][2] = "H"
        dynamic_axes["input_images_or_features"][3] = "W"
    if args.channels is None:
        # Default behavior: keep channels dynamic.
        dynamic_axes["input_images_or_features"][1] = "channels"
        dynamic_axes[onnx_output_name][1] = "channels"
    input_names = ["input_images_or_features", "rois"]

    if dynamic_output_height and dynamic_output_width:
        export_module = roi_align_module
        export_args = (
            input_images_or_features,
            rois,
            torch.tensor(output_height),
            torch.tensor(output_width),
        )
        input_names.extend(["output_height", "output_width"])
        dynamic_axes["output_height"] = {}  # Scalar, but can vary between calls.
        dynamic_axes["output_width"] = {}  # Scalar, but can vary between calls.
        dynamic_axes[onnx_output_name][2] = "output_H"
        dynamic_axes[onnx_output_name][3] = "output_W"
    elif dynamic_output_height:
        class ExportFixedWidth(torch.nn.Module):
            def __init__(self, module: DynamicRoIAlign, fixed_width: int):
                super().__init__()
                self.module = module
                self.fixed_width = fixed_width

            def forward(self, input_images_or_features: torch.Tensor, rois: torch.Tensor, output_height: torch.Tensor) -> torch.Tensor:
                return typing.cast(
                    torch.Tensor,
                    self.module(input_images_or_features, rois, output_height, self.fixed_width),
                )

        export_module = ExportFixedWidth(roi_align_module, output_width)
        export_args = (input_images_or_features, rois, torch.tensor(output_height))
        input_names.append("output_height")
        dynamic_axes["output_height"] = {}  # Scalar, but can vary between calls.
        dynamic_axes[onnx_output_name][2] = "output_H"
    elif dynamic_output_width:
        class ExportFixedHeight(torch.nn.Module):
            def __init__(self, module: DynamicRoIAlign, fixed_height: int):
                super().__init__()
                self.module = module
                self.fixed_height = fixed_height

            def forward(self, input_images_or_features: torch.Tensor, rois: torch.Tensor, output_width: torch.Tensor) -> torch.Tensor:
                return typing.cast(
                    torch.Tensor,
                    self.module(input_images_or_features, rois, self.fixed_height, output_width),
                )

        export_module = ExportFixedHeight(roi_align_module, output_height)
        export_args = (input_images_or_features, rois, torch.tensor(output_width))
        input_names.append("output_width")
        dynamic_axes["output_width"] = {}  # Scalar, but can vary between calls.
        dynamic_axes[onnx_output_name][3] = "output_W"
    else:
        class ExportFixedOutputSize(torch.nn.Module):
            def __init__(self, module: DynamicRoIAlign, fixed_height: int, fixed_width: int):
                super().__init__()
                self.module = module
                self.fixed_height = fixed_height
                self.fixed_width = fixed_width

            def forward(self, input_images_or_features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
                return typing.cast(
                    torch.Tensor,
                    self.module(input_images_or_features, rois, self.fixed_height, self.fixed_width),
                )

        export_module = ExportFixedOutputSize(roi_align_module, output_height, output_width)
        export_args = (input_images_or_features, rois)

    onnx_model_path = args.onnx_output_path

    # Export the model to ONNX format
    torch.onnx.export(
        export_module,
        export_args,
        onnx_model_path,
        input_names=input_names,
        output_names=[onnx_output_name],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset_version,  # Opset 16+ required for grid_sample with dynamic shapes
        dynamo=False,  # Supports mixed input/constant export arguments with dynamic_axes.
    )
    print(f"ONNX model exported successfully to: {onnx_model_path}")

    # Simplify the exported ONNX model with onnxsim Python API.
    print("Running onnxsim...")
    simplified_model, check = simplify(onnx_model_path)
    if not check:
        raise RuntimeError("onnxsim validation failed")

    if args.batch_size is None:
        # Keep symbolic batch axis on ONNX input even after simplification.
        input_batch_dim = simplified_model.graph.input[0].type.tensor_type.shape.dim[0]
        input_batch_dim.ClearField("dim_value")
        input_batch_dim.dim_param = "batch_size"

    if args.channels is None:
        # Keep symbolic channel axis on ONNX input/output even after simplification.
        input_channel_dim = simplified_model.graph.input[0].type.tensor_type.shape.dim[1]
        input_channel_dim.ClearField("dim_value")
        input_channel_dim.dim_param = "channels"
        output_channel_dim = simplified_model.graph.output[0].type.tensor_type.shape.dim[1]
        output_channel_dim.ClearField("dim_value")
        output_channel_dim.dim_param = "channels"
    if args.input_hw_size is None:
        input_h_dim = simplified_model.graph.input[0].type.tensor_type.shape.dim[2]
        input_h_dim.ClearField("dim_value")
        input_h_dim.dim_param = "H"
        input_w_dim = simplified_model.graph.input[0].type.tensor_type.shape.dim[3]
        input_w_dim.ClearField("dim_value")
        input_w_dim.dim_param = "W"

    # Append input descriptions to ONNX model metadata.
    existing_metadata = {item.key: item.value for item in simplified_model.metadata_props}
    reserved_metadata_keys = {
        "input_images_or_features",
        "rois",
        "aligned",
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
        "H: feature height\n"
        "W: feature width"
    )
    model_metadata["rois"] = (
        "ROI tensor shape: [num_rois, 5]\n"
        "Each ROI: [batch_idx, x1, y1, x2, y2]\n"
        "Coordinates x1/y1/x2/y2 are normalized to [0, 1]"
    )
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
    model_metadata[onnx_output_name] = (
        "Aligned ROI features shape: [num_rois, C, output_H, output_W]\n"
        "num_rois: number of ROI entries\n"
        "C: same as input channels\n"
        "output_H/output_W: from output_height/output_width inputs when dynamic,\n"
        "or fixed by CLI constants when specified"
    )
    onnx.helper.set_model_props(simplified_model, model_metadata)

    onnx.save(simplified_model, onnx_model_path)
    print(f"Simplified ONNX model exported successfully to: {onnx_model_path}")
