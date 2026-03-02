"""Dynamic ROI Align implementation with variable output sizes.

This module performs ROI Align operation with dynamic output heights and widths,
allowing different ROI sizes to be processed in a single forward pass.
"""

import argparse
import torch
import onnx
from onnxsim import simplify
from typing import Optional, Tuple, cast

class DynamicRoIAlign(torch.nn.Module):
    """Dynamic ROI Align layer that supports variable output sizes.

    This implementation allows ROI Align to produce outputs of different sizes
    for each ROI, which is useful for hierarchical segmentation models where
    different ROIs may require different levels of detail.

    Attributes:
        spatial_scale (float, tuple, or None): Scale factor to convert ROI coordinates to feature map space.
            Can be a single float for square images, or a tuple (scale_h, scale_w)
            for non-square images. If None, use input feature map (H, W) dynamically.
            For example:
            - Square: If ROIs are normalized [0,1] and features are 640x640, spatial_scale=640
            - Non-square: If ROIs are normalized [0,1] and features are 480x640, spatial_scale=(480, 640)
        sampling_ratio (int): Number of sampling points in each bin. -1 means adaptive.
            (Note: Currently not used in this implementation)
        aligned (bool): If True, use pixel-aligned grid_sample. Default is False.
    """

    def __init__(self, spatial_scale: Optional[Tuple[float, float] | float]=(640, 640), sampling_ratio=-1, aligned=False):
        """Initialize DynamicRoIAlign module.

        Args:
            spatial_scale (float, tuple, or None): Scale factor to convert ROI coordinates to feature map space.
                Can be a single float for square images, or a tuple (scale_h, scale_w)
                for non-square images. If None, use input feature map (H, W) dynamically.
                For example:
                - Square: If ROIs are normalized [0,1] and features are 640x640, spatial_scale=640
                - Non-square: If ROIs are normalized [0,1] and features are 480x640, spatial_scale=(480, 640)
            sampling_ratio (int): Number of sampling points per bin. -1 for adaptive.
                (Currently not implemented in this version)
            aligned (bool): Whether to use pixel-aligned sampling. This affects how
                coordinates are normalized for grid_sample.
        """
        super().__init__()
        # Handle None, single value, and tuple for spatial_scale
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

    def forward(self, input_images_or_features: torch.Tensor, rois: torch.Tensor, output_height: Optional[torch.Tensor | int], output_width: Optional[torch.Tensor | int]) -> torch.Tensor:
        """Perform dynamic ROI Align on input features.

        Args:
            input_images_or_features (torch.Tensor): Input feature tensor of shape (N, C, H, W)
                where N is batch size, C is channels,
                H and W are spatial dimensions.
            rois (torch.Tensor): ROI tensor of shape (K, 5) where K is number of ROIs.
                Each ROI is [batch_idx, x1, y1, x2, y2].
                Coordinates should be in the same space as specified
                by spatial_scale.
                All coordinate values must be normalized to the range 0.0 to 1.0.
            output_height (int or torch.Tensor): Desired output height for aligned features.
                Can be different for each forward pass.
            output_width (int or torch.Tensor): Desired output width for aligned features.
                Can be different for each forward pass.

        Returns:
            torch.Tensor: Aligned features of shape (K, C, output_height, output_width)
                where K is the number of ROIs and C matches input channels.
        """
        # Extract batch indices and scale ROI coordinates to feature map space
        batch_indices = rois[:, 0].long()

        if self.spatial_scale is None:
            spatial_scale_h = input_images_or_features.shape[2]
            spatial_scale_w = input_images_or_features.shape[3]
        else:
            assert self.spatial_scale_h is not None
            assert self.spatial_scale_w is not None
            spatial_scale_h = self.spatial_scale_h
            spatial_scale_w = self.spatial_scale_w

        # Scale x and y coordinates separately for non-square images
        # rois format: [batch_idx, x1, y1, x2, y2] where x,y are normalized [0,1]
        x1 = rois[:, 1] * spatial_scale_w
        y1 = rois[:, 2] * spatial_scale_h
        x2 = rois[:, 3] * spatial_scale_w
        y2 = rois[:, 4] * spatial_scale_h

        # Stack back into boxes tensor
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        # ROI dimensions are already computed
        roi_width = x2 - x1
        roi_height = y2 - y1

        # Generate a grid for each ROI
        # This part is vectorized to avoid Python loops over ROIs

        # Create normalized coordinates [0, 1] for output grid points.
        # Keep output_height/output_width as tensors to preserve dynamic ONNX behavior.
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

        # `arange` accepts tensor scalar end at runtime/export time, but type stubs
        # declare numeric scalars. Cast here is for static type checkers only.
        x_indices = torch.arange(cast(int, output_width), device=rois.device, dtype=rois.dtype)
        y_indices = torch.arange(cast(int, output_height), device=rois.device, dtype=rois.dtype)

        # Match linspace(0, 1, n) while keeping n dynamic.
        x_denominator = torch.clamp(output_width_f - 1, min=1)
        y_denominator = torch.clamp(output_height_f - 1, min=1)
        x_coords_normalized = x_indices / x_denominator
        y_coords_normalized = y_indices / y_denominator

        # Create 2D grid of normalized coordinates
        # grid_y, grid_x shape: (output_height, output_width)
        grid_y, grid_x = torch.meshgrid(y_coords_normalized, x_coords_normalized, indexing='ij')

        # Calculate coordinates in feature map space for each bin center
        # The +0.5 offset centers the sampling point within each bin
        # fx = x1 + (grid_x + 0.5) * bin_width
        # fy = y1 + (grid_y + 0.5) * bin_height

        # Note: bin dimensions are implicitly handled by the grid normalization
        # Each output pixel samples from the corresponding portion of the ROI

        # Calculate actual feature map coordinates for each sampling point
        # The grid already represents positions from 0 to 1 within the ROI
        # Shape: (num_rois, output_height, output_width)
        # Broadcasting handles [num_rois, 1, 1] with [output_height, output_width].
        fx = x1.unsqueeze(1).unsqueeze(2) + grid_x * roi_width.unsqueeze(1).unsqueeze(2)
        fy = y1.unsqueeze(1).unsqueeze(2) + grid_y * roi_height.unsqueeze(1).unsqueeze(2)

        # Normalize coordinates to [-1, 1] range required by grid_sample
        # grid_sample expects -1 for top-left corner and 1 for bottom-right
        H_feat, W_feat = input_images_or_features.shape[2], input_images_or_features.shape[3]
        if self.aligned:
            # When align_corners=True, corners map to corners
            normalized_fx = (fx / (W_feat - 1)) * 2 - 1
            normalized_fy = (fy / (H_feat - 1)) * 2 - 1
        else:
            # When align_corners=False, pixel centers are used
            normalized_fx = (fx / W_feat) * 2 - 1
            normalized_fy = (fy / H_feat) * 2 - 1

        # Stack x and y coordinates to form sampling grid
        # Shape: (num_rois, output_height, output_width, 2)
        # Last dimension contains [x, y] coordinates for each sampling point
        grids_tensor = torch.stack([normalized_fx, normalized_fy], dim=-1)

        # Select the appropriate feature map for each ROI based on its batch index
        # This handles ROIs from different images in the batch
        # Shape: (num_rois, C, H, W)
        selected_feature_maps = torch.index_select(input_images_or_features, 0, batch_indices)

        # Apply bilinear interpolation to sample features at grid points
        # grid_sample performs differentiable bilinear interpolation
        # Input: (num_rois, C, H_feat, W_feat)
        # Grid: (num_rois, output_height, output_width, 2)
        # Output: (num_rois, C, output_height, output_width)
        pooled_features = torch.nn.functional.grid_sample(
            selected_feature_maps,
            grids_tensor,
            mode='bilinear',  # Use bilinear interpolation for smooth gradients
            padding_mode='zeros',  # Pad with zeros outside feature map boundaries
            align_corners=self.aligned  # Controls pixel alignment behavior
        )

        return pooled_features


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

    if args.spatial_scale is None:
        spatial_scale_arg = None
    elif len(args.spatial_scale) == 1:
        spatial_scale_arg = args.spatial_scale[0]
    else:
        spatial_scale_arg = (args.spatial_scale[0], args.spatial_scale[1])

    feature_channels = args.channels
    test_input_channels = feature_channels if feature_channels is not None else 256
    feature_batch_size = args.batch_size
    test_input_batch_size = feature_batch_size if feature_batch_size is not None else 2

    # Create dummy input data for testing
    # Feature map: batch_size=test_input_batch_size, channels=test_input_channels, height=56, width=56
    # When --batch-size/--channels are omitted (None), use 2/256 for this local test input only.
    input_images_or_features = torch.randn(test_input_batch_size, test_input_channels, 56, 56)

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
    roi_align_module = DynamicRoIAlign(spatial_scale=spatial_scale_arg)
    output = roi_align_module.forward(input_images_or_features, rois, output_height, output_width)

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
        "input_images_or_features": {
            2: "H",             # Variable feature map height
            3: "W"              # Variable feature map width
        },
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
                return self.module(input_images_or_features, rois, output_height, self.fixed_width)

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
                return self.module(input_images_or_features, rois, self.fixed_height, output_width)

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
                return self.module(input_images_or_features, rois, self.fixed_height, self.fixed_width)

        export_module = ExportFixedOutputSize(roi_align_module, output_height, output_width)
        export_args = (input_images_or_features, rois)

    onnx_model_path = "dynamic_roi_align.onnx"

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

    # Append input descriptions to ONNX model metadata.
    existing_metadata = {item.key: item.value for item in simplified_model.metadata_props}
    reserved_metadata_keys = {
        "input_images_or_features",
        "rois",
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
    if dynamic_output_height:
        model_metadata["output_height"] = (
            "Output feature height scalar input (int64)\n"
            "Used only when output height is dynamic at export time"
        )
    if dynamic_output_width:
        model_metadata["output_width"] = (
            "Output feature width scalar input (int64)\n"
            "Used only when output width is dynamic at export time"
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
