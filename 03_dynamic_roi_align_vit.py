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

TopKGroupSpec = tuple[str, int, list[int]]
TopKGroupOutputSize = tuple[int, int]
VitForwardOutput = torch.Tensor | tuple[torch.Tensor, ...]
GroupedRoisOutput = tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]
VitToRoisOutput = tuple[torch.Tensor, torch.Tensor] | GroupedRoisOutput


def _parse_topk_group_specs(raw_specs: list[str]) -> list[TopKGroupSpec]:
    parsed_specs: list[TopKGroupSpec] = []
    seen_names: set[str] = set()
    for raw_spec in raw_specs:
        spec_parts = raw_spec.split(":", 2)
        if len(spec_parts) != 3:
            raise ValueError(
                "--use-topk-group must follow NAME:K:LABEL_IDS (e.g. body:8:0,1,2)"
            )

        group_name = spec_parts[0].strip()
        topk_text = spec_parts[1].strip()
        label_ids_text = spec_parts[2].strip()
        if not group_name:
            raise ValueError("--use-topk-group NAME cannot be empty")
        if group_name in seen_names:
            raise ValueError(f"Duplicate topk group name: {group_name}")
        seen_names.add(group_name)

        try:
            group_topk = int(topk_text)
        except ValueError as exc:
            raise ValueError(f"Invalid K in --use-topk-group '{raw_spec}'") from exc
        if group_topk <= 0:
            raise ValueError(f"K must be > 0 in --use-topk-group '{raw_spec}'")

        label_id_tokens = [token.strip() for token in label_ids_text.split(",") if token.strip()]
        if not label_id_tokens:
            raise ValueError(f"LABEL_IDS cannot be empty in --use-topk-group '{raw_spec}'")

        label_ids: list[int] = []
        for token in label_id_tokens:
            try:
                label_id = int(token)
            except ValueError as exc:
                raise ValueError(f"Invalid label id '{token}' in --use-topk-group '{raw_spec}'") from exc
            if label_id < 0:
                raise ValueError(f"Label ids must be >= 0 in --use-topk-group '{raw_spec}'")
            label_ids.append(label_id)

        if len(set(label_ids)) != len(label_ids):
            raise ValueError(f"Duplicate label ids in --use-topk-group '{raw_spec}'")

        parsed_specs.append((group_name, group_topk, label_ids))
    return parsed_specs


def _parse_topk_group_output_sizes(
    raw_sizes: list[str],
    expected_num_groups: int,
) -> list[TopKGroupOutputSize]:
    parsed_sizes: list[TopKGroupOutputSize] = []
    for raw_size in raw_sizes:
        size_parts = [part.strip() for part in raw_size.split(",")]
        if len(size_parts) != 2:
            raise ValueError(
                "--topk-group-output-sizes must use H,W format (e.g. 128,64)"
            )
        try:
            out_h = int(size_parts[0])
            out_w = int(size_parts[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid --topk-group-output-sizes item '{raw_size}', expected H,W with integers"
            ) from exc
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"--topk-group-output-sizes item '{raw_size}' must have positive integers"
            )
        parsed_sizes.append((out_h, out_w))

    if expected_num_groups <= 0:
        raise ValueError("expected_num_groups must be >= 1")
    if expected_num_groups == 1:
        if len(parsed_sizes) != 1:
            raise ValueError(
                "--topk-group-output-sizes expects exactly 1 item when --use-topk-group is not used"
            )
        return parsed_sizes

    if len(parsed_sizes) == 1:
        return parsed_sizes * expected_num_groups
    if len(parsed_sizes) != expected_num_groups:
        raise ValueError(
            f"--topk-group-output-sizes expects 1 item or {expected_num_groups} items, got {len(parsed_sizes)}"
        )
    return parsed_sizes


def _sanitize_output_suffix(name: str) -> str:
    sanitized = "".join(char if (char.isalnum() or char == "_") else "_" for char in name)
    return sanitized if sanitized else "group"


class VitOutputToRois(torch.nn.Module):
    """Convert ViT output [B, Q, F] to ROIAlign rois [B*Q, 5]."""

    def __init__(
        self,
        box_format: str = "xyxy",
        score_threshold: Optional[float] = None,
        use_topk: Optional[int] = None,
        topk_groups: Optional[list[TopKGroupSpec]] = None,
    ):
        super().__init__()
        if box_format not in ("xyxy", "xywh"):
            raise ValueError("box_format must be 'xyxy' or 'xywh'")
        self.box_format = box_format
        self.score_threshold = score_threshold
        self.use_topk = use_topk
        self.topk_groups = topk_groups
        self.topk_groups_total = sum(spec[1] for spec in topk_groups) if topk_groups is not None else None

    def forward(
        self,
        vit_output: torch.Tensor,
        input_images_or_features: torch.Tensor,
        score_threshold: Optional[torch.Tensor | float] = None,
        use_topk: Optional[int] = None,
        topk_groups: Optional[list[TopKGroupSpec]] = None,
        return_classids: bool = False,
        return_grouped: bool = False,
    ) -> VitToRoisOutput:
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

        rois_per_query = torch.stack([batch_indices, x_min, y_min, x_max, y_max], dim=2)
        labels = vit_output[:, :, 0].to(torch.int64)

        effective_score_threshold = score_threshold
        if effective_score_threshold is None:
            effective_score_threshold = self.score_threshold

        effective_use_topk = use_topk
        if effective_use_topk is None:
            effective_use_topk = self.use_topk

        effective_topk_groups = topk_groups
        if effective_topk_groups is None:
            effective_topk_groups = self.topk_groups

        if effective_use_topk is not None and effective_topk_groups is not None:
            raise ValueError("use_topk and topk_groups cannot be enabled at the same time")
        if effective_score_threshold is not None and (
            effective_use_topk is not None or effective_topk_groups is not None
        ):
            raise ValueError("score_threshold cannot be enabled together with topk options")
        if return_grouped and effective_topk_groups is None:
            raise ValueError("return_grouped requires topk_groups")
        if (
            effective_score_threshold is not None
            or effective_use_topk is not None
            or effective_topk_groups is not None
        ) and vit_output.shape[2] < 6:
            raise ValueError("ViT output must have at least 6 fields when score/topk filtering is enabled")

        class_ids = torch.empty((0,), dtype=torch.int64, device=vit_output.device)
        if effective_topk_groups is not None:
            score = vit_output[:, :, 5]
            grouped_rois: list[torch.Tensor] = []
            grouped_class_ids: list[torch.Tensor] = []
            for _, group_topk, group_label_ids in effective_topk_groups:
                group_mask = labels == group_label_ids[0]
                for label_id in group_label_ids[1:]:
                    group_mask = torch.logical_or(group_mask, labels == label_id)
                min_score = torch.finfo(score.dtype).min
                masked_scores = torch.where(group_mask, score, torch.full_like(score, min_score))
                topk_indices = torch.topk(
                    masked_scores,
                    k=group_topk,
                    dim=1,
                    largest=True,
                    sorted=False,
                ).indices
                gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, 5)
                grouped_rois.append(torch.gather(rois_per_query, dim=1, index=gather_indices))
                if return_classids:
                    grouped_class_ids.append(torch.gather(labels, dim=1, index=topk_indices))
            if return_grouped:
                return tuple(grouped_rois), tuple(grouped_class_ids)
            rois = torch.cat(grouped_rois, dim=1).reshape(-1, 5)
            if return_classids:
                class_ids = torch.cat(grouped_class_ids, dim=1).reshape(-1)
        elif effective_use_topk is not None:
            if effective_use_topk <= 0:
                raise ValueError("use_topk must be a positive integer")
            score = vit_output[:, :, 5]
            topk_indices = torch.topk(
                score,
                k=effective_use_topk,
                dim=1,
                largest=True,
                sorted=False,
            ).indices
            gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, 5)
            rois = torch.gather(rois_per_query, dim=1, index=gather_indices).reshape(-1, 5)
            if return_classids:
                class_ids = torch.gather(labels, dim=1, index=topk_indices).reshape(-1)
        elif effective_score_threshold is not None:
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
            rois = rois_per_query.reshape(-1, 5)[score_mask.reshape(-1)]
            if return_classids:
                class_ids = labels.reshape(-1)[score_mask.reshape(-1)]
        else:
            rois = rois_per_query.reshape(-1, 5)
            if return_classids:
                class_ids = labels.reshape(-1)

        return rois, class_ids


class DynamicRoIAlignFromVit(torch.nn.Module):
    """Integrated pipeline: ViT output preprocessing + DynamicRoIAlign."""

    def __init__(
        self,
        roi_align_module: DynamicRoIAlign,
        vit_box_format: str = "xyxy",
        score_threshold: Optional[float] = None,
        use_topk: Optional[int] = None,
        topk_groups: Optional[list[TopKGroupSpec]] = None,
        topk_group_output_sizes: Optional[list[TopKGroupOutputSize]] = None,
        enable_output_classids: bool = False,
    ):
        super().__init__()
        self.roi_align = roi_align_module
        self.enable_output_classids = enable_output_classids
        self.topk_group_output_sizes = topk_group_output_sizes
        self.vit_to_rois = VitOutputToRois(
            box_format=vit_box_format,
            score_threshold=score_threshold,
            use_topk=use_topk,
            topk_groups=topk_groups,
        )

    def forward(
        self,
        input_images_or_features: torch.Tensor,
        vit_output: torch.Tensor,
        output_height: Optional[torch.Tensor | int],
        output_width: Optional[torch.Tensor | int],
        score_threshold: Optional[torch.Tensor | float] = None,
    ) -> VitForwardOutput:
        runtime_score_threshold = score_threshold
        if runtime_score_threshold is None:
            runtime_score_threshold = self.vit_to_rois.score_threshold
        runtime_use_topk = self.vit_to_rois.use_topk
        runtime_topk_groups = self.vit_to_rois.topk_groups
        runtime_topk_group_output_sizes = self.topk_group_output_sizes

        if runtime_use_topk is not None and runtime_topk_groups is not None:
            raise ValueError("use_topk and topk_groups cannot be enabled at the same time")
        if runtime_score_threshold is not None and (runtime_use_topk is not None or runtime_topk_groups is not None):
            raise ValueError("score_threshold cannot be enabled together with topk options")

        if runtime_topk_groups is not None and runtime_topk_group_output_sizes is not None:
            grouped_rois, grouped_class_ids = typing.cast(
                GroupedRoisOutput,
                self.vit_to_rois(
                    vit_output,
                    input_images_or_features,
                    runtime_score_threshold,
                    runtime_use_topk,
                    runtime_topk_groups,
                    self.enable_output_classids,
                    True,
                ),
            )
            batch_size = vit_output.shape[0]
            grouped_features: list[torch.Tensor] = []
            for rois_group, (group_h, group_w) in zip(grouped_rois, runtime_topk_group_output_sizes):
                group_aligned_features = typing.cast(
                    torch.Tensor,
                    self.roi_align(
                        input_images_or_features,
                        rois_group.reshape(-1, 5),
                        group_h,
                        group_w,
                    ),
                )
                channels = group_aligned_features.shape[1]
                grouped_features.append(
                    group_aligned_features.reshape(
                        batch_size,
                        rois_group.shape[1],
                        channels,
                        group_aligned_features.shape[2],
                        group_aligned_features.shape[3],
                    )
                )
            if self.enable_output_classids:
                return (*grouped_features, *grouped_class_ids)
            return tuple(grouped_features)

        rois, class_ids = typing.cast(
            tuple[torch.Tensor, torch.Tensor],
            self.vit_to_rois(
                vit_output,
                input_images_or_features,
                runtime_score_threshold,
                runtime_use_topk,
                runtime_topk_groups,
                self.enable_output_classids,
            ),
        )
        aligned_features = typing.cast(
            torch.Tensor,
            self.roi_align(input_images_or_features, rois, output_height, output_width),
        )

        if runtime_score_threshold is not None:
            # Filtering makes per-batch ROI counts data-dependent.
            # Keep flattened [num_rois, C, H, W] representation.
            if self.enable_output_classids:
                return aligned_features, class_ids
            return aligned_features

        # Preserve batch axis for non-filtering case: [B, Q, C, H, W].
        batch_size = vit_output.shape[0]
        if runtime_use_topk is not None:
            num_queries = runtime_use_topk
        elif runtime_topk_groups is not None:
            num_queries = typing.cast(int, self.vit_to_rois.topk_groups_total)
        else:
            num_queries = vit_output.shape[1]
        channels = aligned_features.shape[1]
        out_h = aligned_features.shape[2]
        out_w = aligned_features.shape[3]
        reshaped_features = aligned_features.reshape(batch_size, num_queries, channels, out_h, out_w)
        if self.enable_output_classids:
            reshaped_class_ids = class_ids.reshape(batch_size, num_queries)
            return reshaped_features, reshaped_class_ids
        return reshaped_features


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
    parser.add_argument(
        "--use-topk",
        type=int,
        default=None,
        help="Enable fixed-count filtering by selecting top-K ViT queries by score.",
    )
    parser.add_argument(
        "--use-topk-group",
        action="append",
        nargs="+",
        default=None,
        metavar="NAME:K:LABEL_IDS",
        help=(
            "Enable grouped top-K filtering. Accepts one or more specs per flag use. "
            "Example: --use-topk-group body:8:0,1,2 head:12:7,8,9"
        ),
    )
    parser.add_argument(
        "--enable-output-classids",
        action="store_true",
        help="Add class_ids as an additional ONNX output aligned with selected ROIs.",
    )
    parser.add_argument(
        "--topk-group-output-sizes",
        nargs="+",
        default=None,
        metavar="H,W",
        help=(
            "ROI output size(s) as H,W. Without --use-topk-group, specify exactly one pair. "
            "With --use-topk-group, specify one shared pair or one pair per group."
        ),
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
    raw_topk_group_specs: Optional[list[str]] = None
    if args.use_topk_group is not None:
        raw_topk_group_specs = [
            group_spec
            for group_spec_list in args.use_topk_group
            for group_spec in group_spec_list
        ]
    topk_groups = _parse_topk_group_specs(raw_topk_group_specs) if raw_topk_group_specs is not None else None
    topk_group_output_sizes: Optional[list[TopKGroupOutputSize]] = None
    if args.topk_group_output_sizes is not None:
        expected_num_groups = len(topk_groups) if topk_groups is not None else 1
        topk_group_output_sizes = _parse_topk_group_output_sizes(
            args.topk_group_output_sizes,
            expected_num_groups=expected_num_groups,
        )

    if args.input_channels is not None and args.input_channels <= 0:
        raise ValueError("--input-channels must be a positive integer")
    if args.input_batch_size is not None and args.input_batch_size <= 0:
        raise ValueError("--input-batch-size must be a positive integer")
    if args.input_hw_size is not None:
        if args.input_hw_size[0] <= 0 or args.input_hw_size[1] <= 0:
            raise ValueError("--input-hw-size values must be positive integers")
    if args.spatial_scale is not None and len(args.spatial_scale) not in (1, 2):
        raise ValueError("--spatial-scale must have 1 or 2 values")
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
    if args.use_topk is not None and args.use_topk <= 0:
        raise ValueError("--use-topk must be a positive integer")
    if args.use_topk is not None and (args.use_score_threshold is not None or args.score_threshold_as_input):
        raise ValueError("--use-topk is mutually exclusive with score threshold options")
    if topk_groups is not None and (args.use_score_threshold is not None or args.score_threshold_as_input):
        raise ValueError("--use-topk-group is mutually exclusive with score threshold options")
    if args.use_topk is not None and topk_groups is not None:
        raise ValueError("--use-topk and --use-topk-group cannot be enabled at the same time")
    if (
        args.use_score_threshold is not None
        or args.score_threshold_as_input
        or args.use_topk is not None
        or topk_groups is not None
    ) and args.vit_output_fields is not None and args.vit_output_fields < 6:
        raise ValueError("--vit-output-fields must be >= 6 when score/topk filtering is enabled")
    if args.use_topk is not None and args.vit_num_queries is not None and args.use_topk > args.vit_num_queries:
        raise ValueError("--use-topk must be <= --vit-num-queries when both are specified")
    if topk_groups is not None and args.vit_num_queries is not None:
        for group_name, group_topk, _ in topk_groups:
            if group_topk > args.vit_num_queries:
                raise ValueError(
                    f"--use-topk-group '{group_name}' has K={group_topk} > --vit-num-queries ({args.vit_num_queries})"
                )

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

    grouped_output_sizes_enabled = (topk_groups is not None) and (topk_group_output_sizes is not None)
    fixed_global_output_size = (
        topk_group_output_sizes[0]
        if (topk_group_output_sizes is not None and topk_groups is None)
        else None
    )
    dynamic_output_height = (not grouped_output_sizes_enabled) and (fixed_global_output_size is None)
    dynamic_output_width = (not grouped_output_sizes_enabled) and (fixed_global_output_size is None)
    dynamic_score_threshold = args.score_threshold_as_input
    if grouped_output_sizes_enabled:
        output_height = typing.cast(list[TopKGroupOutputSize], topk_group_output_sizes)[0][0]
        output_width = typing.cast(list[TopKGroupOutputSize], topk_group_output_sizes)[0][1]
    elif fixed_global_output_size is not None:
        output_height, output_width = fixed_global_output_size
    else:
        output_height = 7
        output_width = 7

    roi_align_module = DynamicRoIAlign(spatial_scale=spatial_scale_arg, aligned=args.aligned)
    integrated_module = DynamicRoIAlignFromVit(
        roi_align_module,
        vit_box_format=args.vit_box_format,
        score_threshold=args.use_score_threshold,
        use_topk=args.use_topk,
        topk_groups=topk_groups,
        topk_group_output_sizes=topk_group_output_sizes,
        enable_output_classids=args.enable_output_classids,
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
    if isinstance(output, tuple):
        print("Output shapes:", [tensor.shape for tensor in output])
    else:
        print("Output shape:", output.shape)

    onnx_output_name = "aligned_features"
    class_ids_output_name = "class_ids"
    output_names: list[str] = []
    feature_output_names: list[str] = []
    class_ids_output_names: list[str] = []
    if grouped_output_sizes_enabled:
        typed_topk_groups = typing.cast(list[TopKGroupSpec], topk_groups)
        for group_index, (group_name, _, _) in enumerate(typed_topk_groups):
            safe_group_name = _sanitize_output_suffix(group_name)
            feature_output_names.append(f"aligned_features_g{group_index}_{safe_group_name}")
        output_names.extend(feature_output_names)
        if args.enable_output_classids:
            for group_index, (group_name, _, _) in enumerate(typed_topk_groups):
                safe_group_name = _sanitize_output_suffix(group_name)
                class_ids_output_names.append(f"class_ids_g{group_index}_{safe_group_name}")
            output_names.extend(class_ids_output_names)
    else:
        feature_output_names = [onnx_output_name]
        output_names = [onnx_output_name]
        if args.enable_output_classids:
            class_ids_output_names = [class_ids_output_name]
            output_names.append(class_ids_output_name)

    dynamic_axes = {
        "input_images_or_features": {},
        "vit_output": {},
    }
    for output_name in output_names:
        dynamic_axes[output_name] = {}

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
    output_axis_h = 2 if output_is_flattened else 3
    output_axis_w = 3 if output_is_flattened else 4

    if grouped_output_sizes_enabled:
        if args.vit_batch_size is None:
            for feature_output_name in feature_output_names:
                dynamic_axes[feature_output_name][0] = "vit_batch_size"
            for class_ids_name in class_ids_output_names:
                dynamic_axes[class_ids_name][0] = "vit_batch_size"
        if args.input_channels is None:
            dynamic_axes["input_images_or_features"][1] = "channels"
            for feature_output_name in feature_output_names:
                dynamic_axes[feature_output_name][2] = "channels"
    elif output_is_flattened:
        # Output: [num_rois, C, H, W]
        dynamic_axes[onnx_output_name][0] = "num_rois"
        if args.enable_output_classids:
            dynamic_axes[class_ids_output_name][0] = "num_rois"
        if args.input_channels is None:
            dynamic_axes["input_images_or_features"][1] = "channels"
            dynamic_axes[onnx_output_name][1] = "channels"
    else:
        # Output: [B, Q, C, H, W]
        if args.vit_batch_size is None:
            dynamic_axes[onnx_output_name][0] = "vit_batch_size"
            if args.enable_output_classids:
                dynamic_axes[class_ids_output_name][0] = "vit_batch_size"
        if args.vit_num_queries is None and args.use_topk is None and topk_groups is None:
            dynamic_axes[onnx_output_name][1] = "num_queries"
            if args.enable_output_classids:
                dynamic_axes[class_ids_output_name][1] = "num_queries"
        if args.input_channels is None:
            dynamic_axes["input_images_or_features"][1] = "channels"
            dynamic_axes[onnx_output_name][2] = "channels"

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
            ) -> VitForwardOutput:
                return self.module(
                    input_images_or_features,
                    vit_output,
                    output_height,
                    self.fixed_width,
                    score_threshold,
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
            ) -> VitForwardOutput:
                return self.module(
                    input_images_or_features,
                    vit_output,
                    self.fixed_height,
                    output_width,
                    score_threshold,
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
            ) -> VitForwardOutput:
                return self.module(
                    input_images_or_features,
                    vit_output,
                    self.fixed_height,
                    self.fixed_width,
                    score_threshold,
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
            output_names=output_names,
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
    output_channel_axis = 2 if grouped_output_sizes_enabled else (1 if output_is_flattened else 2)
    if args.input_batch_size is None:
        _set_input_dim_param(simplified_model, "input_images_or_features", 0, "batch_size")
    if args.input_channels is None:
        _set_input_dim_param(simplified_model, "input_images_or_features", 1, "channels")
        if grouped_output_sizes_enabled:
            for feature_output_name in feature_output_names:
                _set_output_dim_param(simplified_model, feature_output_name, output_channel_axis, "channels")
        else:
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
    if grouped_output_sizes_enabled:
        if args.vit_batch_size is None:
            for output_name in output_names:
                _set_output_dim_param(simplified_model, output_name, 0, "vit_batch_size")
    elif output_is_flattened:
        _set_output_dim_param(simplified_model, onnx_output_name, 0, "num_rois")
        if args.enable_output_classids:
            _set_output_dim_param(simplified_model, class_ids_output_name, 0, "num_rois")
    else:
        if args.vit_batch_size is None:
            _set_output_dim_param(simplified_model, onnx_output_name, 0, "vit_batch_size")
            if args.enable_output_classids:
                _set_output_dim_param(simplified_model, class_ids_output_name, 0, "vit_batch_size")
        if args.vit_num_queries is None and args.use_topk is None and topk_groups is None:
            _set_output_dim_param(simplified_model, onnx_output_name, 1, "num_queries")
            if args.enable_output_classids:
                _set_output_dim_param(simplified_model, class_ids_output_name, 1, "num_queries")
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
        "use_topk",
        "use_topk_groups",
        "topk_groups",
        "topk_group_output_sizes",
        "enable_output_classids",
        "input_hw_size",
        "output_height",
        "output_width",
    }
    reserved_metadata_keys.update(output_names)
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
    model_metadata["use_topk"] = "disabled" if args.use_topk is None else str(args.use_topk)
    model_metadata["use_topk_groups"] = str(topk_groups is not None).lower()
    model_metadata["enable_output_classids"] = str(args.enable_output_classids).lower()
    if topk_groups is not None:
        model_metadata["topk_groups"] = "; ".join(
            f"{name}:{group_topk}:{','.join(str(label_id) for label_id in label_ids)}"
            for name, group_topk, label_ids in topk_groups
        )
    if topk_group_output_sizes is not None:
        model_metadata["topk_group_output_sizes"] = "; ".join(
            f"{out_h},{out_w}" for out_h, out_w in topk_group_output_sizes
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
    if (not dynamic_output_height) and (not grouped_output_sizes_enabled):
        model_metadata["output_height"] = (
            "Fixed output feature height (int)\n"
            f"Current export value: {output_height}"
        )
    if (not dynamic_output_width) and (not grouped_output_sizes_enabled):
        model_metadata["output_width"] = (
            "Fixed output feature width (int)\n"
            f"Current export value: {output_width}"
        )
    if grouped_output_sizes_enabled:
        typed_topk_groups = typing.cast(list[TopKGroupSpec], topk_groups)
        typed_group_sizes = typing.cast(list[TopKGroupOutputSize], topk_group_output_sizes)
        for group_index, (group_name, group_topk, _) in enumerate(typed_topk_groups):
            out_h, out_w = typed_group_sizes[group_index]
            model_metadata[feature_output_names[group_index]] = (
                f"Aligned ROI features for group '{group_name}' shape: [B, K, C, {out_h}, {out_w}]\n"
                f"K: top-K for group '{group_name}' (current export value: {group_topk})"
            )
            if args.enable_output_classids:
                model_metadata[class_ids_output_names[group_index]] = (
                    f"Class id shape for group '{group_name}': [B, K]\n"
                    "Each value is the label id for corresponding ROI"
                )
    elif output_is_flattened:
        model_metadata[onnx_output_name] = (
            "Aligned ROI features shape: [num_rois, C, output_H, output_W]\n"
            "num_rois: number of queries kept after enabled filtering"
        )
        if args.enable_output_classids:
            model_metadata[class_ids_output_name] = (
                "Class id shape: [num_rois]\n"
                "Each value is the label id for corresponding ROI"
            )
    elif topk_groups is not None:
        total_group_topk = sum(group_topk for _, group_topk, _ in topk_groups)
        model_metadata[onnx_output_name] = (
            "Aligned ROI features shape: [B, K_total, C, output_H, output_W]\n"
            f"K_total: sum of top-K values across groups (current export value: {total_group_topk})"
        )
        if args.enable_output_classids:
            model_metadata[class_ids_output_name] = (
                "Class id shape: [B, K_total]\n"
                "K_total follows grouped top-K ROI ordering"
            )
    elif args.use_topk is not None:
        model_metadata[onnx_output_name] = (
            "Aligned ROI features shape: [B, K, C, output_H, output_W]\n"
            "K: top-K queries kept per batch from ViT score field"
        )
        if args.enable_output_classids:
            model_metadata[class_ids_output_name] = (
                "Class id shape: [B, K]\n"
                "K follows top-K ROI ordering"
            )
    else:
        model_metadata[onnx_output_name] = (
            "Aligned ROI features shape: [B, Q, C, output_H, output_W]\n"
            "B: vit batch size, Q: number of queries"
        )
        if args.enable_output_classids:
            model_metadata[class_ids_output_name] = (
                "Class id shape: [B, Q]\n"
                "Q: number of queries"
            )
    onnx.helper.set_model_props(simplified_model, model_metadata)

    onnx.save(simplified_model, onnx_model_path)
    print(f"Simplified ONNX model exported successfully to: {onnx_model_path}")
