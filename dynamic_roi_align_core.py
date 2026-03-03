"""Shared DynamicRoIAlign module used by ONNX export scripts."""

from typing import Optional, Tuple, cast

import torch


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
            # Use torch.full instead of torch.tensor to avoid trace-time constant warnings.
            output_width = torch.full((), int(output_width), dtype=torch.int64, device=rois.device)
        else:
            output_width = output_width.to(dtype=torch.int64, device=rois.device).reshape(())
        if not torch.is_tensor(output_height):
            # Use torch.full instead of torch.tensor to avoid trace-time constant warnings.
            output_height = torch.full((), int(output_height), dtype=torch.int64, device=rois.device)
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
