"""Merge ViT detector ONNX and DynamicRoIAlign-for-ViT ONNX.

This script connects:
- ViT detector graph output (e.g. `output0`)
to:
- DynamicRoIAlign ViT input (e.g. `vit_output`)

and keeps the detector output as a graph output in the merged model.
It also appends filtered detector outputs using `vit_query_indices*`, then
removes `vit_query_indices*` from final graph outputs.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import onnx
from onnx import ModelProto, TensorProto, ValueInfoProto, compose, helper
from onnxsim import simplify


def _value_info_by_name(values: Iterable[ValueInfoProto], name: str) -> ValueInfoProto:
    for value in values:
        if value.name == name:
            return value
    raise ValueError(f"Tensor name '{name}' was not found.")


def _basename(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _prefix(name: str) -> str:
    base = _basename(name)
    return name[: len(name) - len(base)]


def _sanitize_name_token(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name)


def _collect_graph_tensor_names(model: ModelProto) -> set[str]:
    graph = model.graph
    tensor_names: set[str] = set()
    for graph_input in graph.input:
        tensor_names.add(graph_input.name)
    for graph_output in graph.output:
        tensor_names.add(graph_output.name)
    for value in graph.value_info:
        tensor_names.add(value.name)
    for initializer in graph.initializer:
        tensor_names.add(initializer.name)
    for node in graph.node:
        for input_name in node.input:
            if input_name:
                tensor_names.add(input_name)
        for output_name in node.output:
            if output_name:
                tensor_names.add(output_name)
    return tensor_names


def _unique_name(base: str, used_names: set[str]) -> str:
    if base not in used_names:
        used_names.add(base)
        return base
    suffix = 1
    while f"{base}_{suffix}" in used_names:
        suffix += 1
    unique = f"{base}_{suffix}"
    used_names.add(unique)
    return unique


def _unique_node_name(base: str, used_node_names: set[str]) -> str:
    if base not in used_node_names:
        used_node_names.add(base)
        return base
    suffix = 1
    while f"{base}_{suffix}" in used_node_names:
        suffix += 1
    unique = f"{base}_{suffix}"
    used_node_names.add(unique)
    return unique


def _make_int64_initializer(graph: onnx.GraphProto, name: str, values: list[int]) -> None:
    tensor = helper.make_tensor(
        name=name,
        data_type=TensorProto.INT64,
        dims=[len(values)],
        vals=values,
    )
    graph.initializer.extend([tensor])


def _resolve_dimension_value(dim: onnx.TensorShapeProto.Dimension) -> int | str | None:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    if dim.HasField("dim_param"):
        return dim.dim_param
    return None


def _resolve_filtered_output_shape(
    detector_output_info: ValueInfoProto,
    query_indices_info: ValueInfoProto,
) -> list[int | str | None]:
    detector_dims = detector_output_info.type.tensor_type.shape.dim
    query_dims = query_indices_info.type.tensor_type.shape.dim
    batch_dim = _resolve_dimension_value(detector_dims[0])
    field_dim = _resolve_dimension_value(detector_dims[2])
    if len(query_dims) == 2:
        selected_dim = _resolve_dimension_value(query_dims[1])
    elif len(query_dims) == 1:
        selected_dim = _resolve_dimension_value(query_dims[0])
    else:
        selected_dim = None
    return [batch_dim, selected_dim, field_dim]


def _find_query_index_output_names(model: ModelProto) -> list[str]:
    names: list[str] = []
    for output in model.graph.output:
        base = _basename(output.name)
        if base == "vit_query_indices" or base.startswith("vit_query_indices_g"):
            names.append(output.name)
    return names


def _drop_graph_outputs(model: ModelProto, output_names: Iterable[str]) -> list[str]:
    names_to_drop = set(output_names)
    if not names_to_drop:
        return []
    removed: list[str] = []
    kept: list[ValueInfoProto] = []
    for graph_output in model.graph.output:
        if graph_output.name in names_to_drop:
            removed.append(graph_output.name)
            continue
        kept.append(graph_output)
    del model.graph.output[:]
    model.graph.output.extend(kept)
    return removed


def _build_filtered_output_name(query_indices_output_name: str, used_tensor_names: set[str]) -> str:
    base = _basename(query_indices_output_name)
    base_prefix = _prefix(query_indices_output_name)
    if base == "vit_query_indices":
        filtered_base = "vit_output_filtered"
    else:
        filtered_base = f"vit_output_filtered_{base.removeprefix('vit_query_indices_')}"
    return _unique_name(f"{base_prefix}{filtered_base}", used_tensor_names)


def _append_filtered_detector_outputs(
    merged: ModelProto,
    detector_output_name: str,
    detector_output_elem_type: int,
) -> list[str]:
    detector_output_info = _value_info_by_name(merged.graph.output, detector_output_name)
    detector_rank = len(detector_output_info.type.tensor_type.shape.dim)
    if detector_rank != 3:
        raise ValueError(
            "Detector output must be rank-3 [B, Q, F] to build filtered outputs, "
            f"but '{detector_output_name}' has rank {detector_rank}."
        )

    query_index_outputs = _find_query_index_output_names(merged)
    if not query_index_outputs:
        raise ValueError(
            "No vit_query_indices* outputs were found in merged graph outputs. "
            "Export DynamicRoIAlign-ViT model with --enable-output-indices."
        )

    graph = merged.graph
    used_tensor_names = _collect_graph_tensor_names(merged)
    used_node_names = {node.name for node in graph.node if node.name}

    idx0_name = _unique_name("merge_filter_const_idx0", used_tensor_names)
    idx1_name = _unique_name("merge_filter_const_idx1", used_tensor_names)
    idx2_name = _unique_name("merge_filter_const_idx2", used_tensor_names)
    unsqueeze_axis2_name = _unique_name("merge_filter_const_unsqueeze_axis2", used_tensor_names)
    _make_int64_initializer(graph, idx0_name, [0])
    _make_int64_initializer(graph, idx1_name, [1])
    _make_int64_initializer(graph, idx2_name, [2])
    _make_int64_initializer(graph, unsqueeze_axis2_name, [2])

    added_output_names: list[str] = []

    for query_name in query_index_outputs:
        query_info = _value_info_by_name(graph.output, query_name)
        query_elem_type = query_info.type.tensor_type.elem_type
        if query_elem_type not in (TensorProto.INT64, TensorProto.INT32):
            raise ValueError(
                f"Query indices output '{query_name}' must be int32/int64, "
                f"but got elem_type={query_elem_type}."
            )

        query_rank = len(query_info.type.tensor_type.shape.dim)
        safe = _sanitize_name_token(query_name)
        filtered_output_name = _build_filtered_output_name(query_name, used_tensor_names)
        added_output_names.append(filtered_output_name)

        if query_rank == 2:
            detector_shape = _unique_name(f"merge_filter/{safe}/detector_shape", used_tensor_names)
            detector_batch = _unique_name(f"merge_filter/{safe}/detector_batch", used_tensor_names)
            detector_fields = _unique_name(f"merge_filter/{safe}/detector_fields", used_tensor_names)
            index_shape = _unique_name(f"merge_filter/{safe}/index_shape", used_tensor_names)
            topk_count = _unique_name(f"merge_filter/{safe}/topk_count", used_tensor_names)
            expand_shape = _unique_name(f"merge_filter/{safe}/expand_shape", used_tensor_names)
            index_unsqueezed = _unique_name(f"merge_filter/{safe}/index_unsqueezed", used_tensor_names)
            index_expanded = _unique_name(f"merge_filter/{safe}/index_expanded", used_tensor_names)

            graph.node.extend(
                [
                    helper.make_node(
                        "Shape",
                        inputs=[detector_output_name],
                        outputs=[detector_shape],
                        name=_unique_node_name(f"merge_filter/shape_detector/{safe}", used_node_names),
                    ),
                    helper.make_node(
                        "Gather",
                        inputs=[detector_shape, idx0_name],
                        outputs=[detector_batch],
                        axis=0,
                        name=_unique_node_name(f"merge_filter/gather_batch/{safe}", used_node_names),
                    ),
                    helper.make_node(
                        "Gather",
                        inputs=[detector_shape, idx2_name],
                        outputs=[detector_fields],
                        axis=0,
                        name=_unique_node_name(f"merge_filter/gather_fields/{safe}", used_node_names),
                    ),
                    helper.make_node(
                        "Shape",
                        inputs=[query_name],
                        outputs=[index_shape],
                        name=_unique_node_name(f"merge_filter/shape_indices/{safe}", used_node_names),
                    ),
                    helper.make_node(
                        "Gather",
                        inputs=[index_shape, idx1_name],
                        outputs=[topk_count],
                        axis=0,
                        name=_unique_node_name(f"merge_filter/gather_topk/{safe}", used_node_names),
                    ),
                    helper.make_node(
                        "Concat",
                        inputs=[detector_batch, topk_count, detector_fields],
                        outputs=[expand_shape],
                        axis=0,
                        name=_unique_node_name(f"merge_filter/concat_expand_shape/{safe}", used_node_names),
                    ),
                    helper.make_node(
                        "Unsqueeze",
                        inputs=[query_name, unsqueeze_axis2_name],
                        outputs=[index_unsqueezed],
                        name=_unique_node_name(f"merge_filter/unsqueeze_indices/{safe}", used_node_names),
                    ),
                    helper.make_node(
                        "Expand",
                        inputs=[index_unsqueezed, expand_shape],
                        outputs=[index_expanded],
                        name=_unique_node_name(f"merge_filter/expand_indices/{safe}", used_node_names),
                    ),
                    helper.make_node(
                        "GatherElements",
                        inputs=[detector_output_name, index_expanded],
                        outputs=[filtered_output_name],
                        axis=1,
                        name=_unique_node_name(f"merge_filter/gather_elements/{safe}", used_node_names),
                    ),
                ]
            )
        elif query_rank == 1:
            graph.node.extend(
                [
                    helper.make_node(
                        "Gather",
                        inputs=[detector_output_name, query_name],
                        outputs=[filtered_output_name],
                        axis=1,
                        name=_unique_node_name(f"merge_filter/gather/{safe}", used_node_names),
                    ),
                ]
            )
        else:
            raise ValueError(
                f"Query indices output '{query_name}' rank must be 1 or 2, "
                f"but got rank {query_rank}."
            )

        output_shape = _resolve_filtered_output_shape(detector_output_info, query_info)
        graph.output.extend(
            [
                helper.make_tensor_value_info(
                    name=filtered_output_name,
                    elem_type=detector_output_elem_type,
                    shape=output_shape,
                )
            ]
        )

    return added_output_names


def _infer_detector_output_name(model: ModelProto) -> str:
    if not model.graph.output:
        raise ValueError("Detector model has no graph outputs.")
    return model.graph.output[-1].name


def _infer_roi_vit_input_name(model: ModelProto) -> str:
    input_names = [graph_input.name for graph_input in model.graph.input]
    if "vit_output" in input_names:
        return "vit_output"
    if len(input_names) == 1:
        return input_names[0]
    raise ValueError(
        "Could not infer DynamicRoIAlign ViT input name. "
        "Please pass --roi-vit-input-name explicitly."
    )


def _get_domain_version_map(model: ModelProto) -> dict[str, int]:
    domain_version: dict[str, int] = {}
    for opset in model.opset_import:
        domain = opset.domain
        version = int(opset.version)
        if domain in domain_version and domain_version[domain] != version:
            raise ValueError(
                f"Model has conflicting opset versions in domain '{domain}': "
                f"{domain_version[domain]} and {version}"
            )
        domain_version[domain] = version
    return domain_version


def _validate_opset_compatibility(detector: ModelProto, roi_align: ModelProto) -> None:
    detector_map = _get_domain_version_map(detector)
    roi_map = _get_domain_version_map(roi_align)
    shared_domains = set(detector_map.keys()) & set(roi_map.keys())
    for domain in sorted(shared_domains):
        if detector_map[domain] != roi_map[domain]:
            name = domain if domain else "ai.onnx (default domain)"
            raise ValueError(
                f"Opset mismatch in {name}: detector={detector_map[domain]}, "
                f"roi_align={roi_map[domain]}"
            )


def _normalize_ir_version(detector: ModelProto, roi_align: ModelProto) -> None:
    merged_ir_version = max(int(detector.ir_version), int(roi_align.ir_version))
    detector.ir_version = merged_ir_version
    roi_align.ir_version = merged_ir_version


def _deduplicate_opset_imports(model: ModelProto) -> None:
    unique_pairs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for opset in model.opset_import:
        pair = (opset.domain, int(opset.version))
        if pair in seen:
            continue
        seen.add(pair)
        unique_pairs.append(pair)

    del model.opset_import[:]
    for domain, version in unique_pairs:
        opset = model.opset_import.add()
        opset.domain = domain
        opset.version = version


def _compatible_or_dynamic_dims(d1: onnx.TensorShapeProto.Dimension, d2: onnx.TensorShapeProto.Dimension) -> bool:
    d1_has_value = d1.HasField("dim_value")
    d2_has_value = d2.HasField("dim_value")
    if d1_has_value and d2_has_value:
        return int(d1.dim_value) == int(d2.dim_value)
    return True


def _validate_image_input_compatibility(
    detector: ModelProto,
    roi_align: ModelProto,
    detector_image_input_name: str,
    roi_image_input_name: str,
) -> None:
    detector_input = _value_info_by_name(detector.graph.input, detector_image_input_name)
    roi_input = _value_info_by_name(roi_align.graph.input, roi_image_input_name)

    detector_tensor = detector_input.type.tensor_type
    roi_tensor = roi_input.type.tensor_type
    if detector_tensor.elem_type != roi_tensor.elem_type:
        raise ValueError(
            "Image input dtype mismatch: "
            f"detector({detector_image_input_name})={detector_tensor.elem_type}, "
            f"roi_align({roi_image_input_name})={roi_tensor.elem_type}"
        )

    detector_dims = detector_tensor.shape.dim
    roi_dims = roi_tensor.shape.dim
    if len(detector_dims) != len(roi_dims):
        raise ValueError(
            "Image input rank mismatch: "
            f"detector({detector_image_input_name}) rank={len(detector_dims)}, "
            f"roi_align({roi_image_input_name}) rank={len(roi_dims)}"
        )

    for axis, (dim1, dim2) in enumerate(zip(detector_dims, roi_dims)):
        if not _compatible_or_dynamic_dims(dim1, dim2):
            raise ValueError(
                f"Image input dim mismatch at axis {axis}: "
                f"detector({detector_image_input_name})={dim1.dim_value}, "
                f"roi_align({roi_image_input_name})={dim2.dim_value}"
            )


def _share_image_input_name(
    merged: ModelProto,
    detector_image_input_name: str,
    roi_image_input_name: str,
) -> None:
    if detector_image_input_name == roi_image_input_name:
        return

    for node in merged.graph.node:
        for i, input_name in enumerate(node.input):
            if input_name == roi_image_input_name:
                node.input[i] = detector_image_input_name

    kept_inputs = [graph_input for graph_input in merged.graph.input if graph_input.name != roi_image_input_name]
    del merged.graph.input[:]
    merged.graph.input.extend(kept_inputs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge ViT detector ONNX and DynamicRoIAlign-ViT ONNX. "
            "The detector output is connected to DynamicRoIAlign vit_output input, "
            "detector output is kept as graph output, and filtered detector outputs "
            "are appended using vit_query_indices* outputs. "
            "The merged graph is simplified with onnxsim before save "
            "(query-index outputs are dropped from final graph outputs)."
        )
    )
    parser.add_argument(
        "--detector-onnx-path",
        type=Path,
        required=True,
        help="Path to ViT detector ONNX.",
    )
    parser.add_argument(
        "--roi-align-onnx-path",
        type=Path,
        required=True,
        help="Path to DynamicRoIAlign-ViT ONNX.",
    )
    parser.add_argument(
        "--output-onnx-path",
        type=Path,
        default=Path("merged_vit_dynamic_roi_align.onnx"),
        help="Output path for merged ONNX.",
    )
    parser.add_argument(
        "--detector-output-name",
        type=str,
        default=None,
        help="Detector output name to connect. Default: last graph output.",
    )
    parser.add_argument(
        "--roi-vit-input-name",
        type=str,
        default=None,
        help="DynamicRoIAlign input name for ViT output. Default: auto (`vit_output` if present).",
    )
    parser.add_argument(
        "--detector-image-input-name",
        type=str,
        default=None,
        help="Detector image input name used by --share-image-input. Default: first detector input.",
    )
    parser.add_argument(
        "--roi-image-input-name",
        type=str,
        default="input_images_or_features",
        help="DynamicRoIAlign image input name used by --share-image-input.",
    )
    parser.add_argument(
        "--share-image-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If enabled, DynamicRoIAlign image input is replaced with detector image input "
            "so merged graph has one image input. Use --no-share-image-input to keep both."
        ),
    )
    parser.add_argument(
        "--prefix-detector",
        type=str,
        default=None,
        help="Optional prefix added to all detector graph names before merge (for collision avoidance).",
    )
    parser.add_argument(
        "--prefix-roi-align",
        type=str,
        default=None,
        help="Optional prefix added to all DynamicRoIAlign graph names before merge (for collision avoidance).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="merged_vit_dynamic_roi_align",
        help="Merged model graph name.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    detector = onnx.load(str(args.detector_onnx_path))
    roi_align = onnx.load(str(args.roi_align_onnx_path))

    _validate_opset_compatibility(detector, roi_align)
    _normalize_ir_version(detector, roi_align)

    detector_output_name = args.detector_output_name or _infer_detector_output_name(detector)
    roi_vit_input_name = args.roi_vit_input_name or _infer_roi_vit_input_name(roi_align)

    detector_output = _value_info_by_name(detector.graph.output, detector_output_name)
    roi_vit_input = _value_info_by_name(roi_align.graph.input, roi_vit_input_name)

    if detector_output.type.tensor_type.elem_type != roi_vit_input.type.tensor_type.elem_type:
        raise ValueError(
            "Detector output and ROI vit input dtype mismatch: "
            f"{detector_output_name}={detector_output.type.tensor_type.elem_type}, "
            f"{roi_vit_input_name}={roi_vit_input.type.tensor_type.elem_type}"
        )

    if args.share_image_input:
        detector_image_input_name = args.detector_image_input_name or detector.graph.input[0].name
        _validate_image_input_compatibility(
            detector=detector,
            roi_align=roi_align,
            detector_image_input_name=detector_image_input_name,
            roi_image_input_name=args.roi_image_input_name,
        )
    else:
        detector_image_input_name = args.detector_image_input_name or detector.graph.input[0].name

    merged_roi_output_names = [graph_output.name for graph_output in roi_align.graph.output]
    merged_output_names: list[str] = [detector_output_name, *merged_roi_output_names]
    if args.prefix_detector:
        merged_output_names[0] = f"{args.prefix_detector}{detector_output_name}"
    if args.prefix_roi_align:
        merged_output_names = [merged_output_names[0], *[f"{args.prefix_roi_align}{name}" for name in merged_roi_output_names]]

    merged = compose.merge_models(
        detector,
        roi_align,
        io_map=[(detector_output_name, roi_vit_input_name)],
        outputs=merged_output_names,
        prefix1=args.prefix_detector,
        prefix2=args.prefix_roi_align,
        name=args.model_name,
    )

    if args.share_image_input:
        merged_detector_image_input_name = detector_image_input_name
        merged_roi_image_input_name = args.roi_image_input_name
        if args.prefix_detector:
            merged_detector_image_input_name = f"{args.prefix_detector}{detector_image_input_name}"
        if args.prefix_roi_align:
            merged_roi_image_input_name = f"{args.prefix_roi_align}{args.roi_image_input_name}"
        _share_image_input_name(
            merged=merged,
            detector_image_input_name=merged_detector_image_input_name,
            roi_image_input_name=merged_roi_image_input_name,
        )

    added_filtered_outputs = _append_filtered_detector_outputs(
        merged=merged,
        detector_output_name=merged_output_names[0],
        detector_output_elem_type=detector_output.type.tensor_type.elem_type,
    )
    query_index_outputs = _find_query_index_output_names(merged)
    removed_query_outputs = _drop_graph_outputs(merged, query_index_outputs)

    _deduplicate_opset_imports(merged)
    onnx.checker.check_model(merged)
    print("Running onnxsim...")
    simplified_model, check = simplify(merged)
    if not check:
        raise RuntimeError("onnxsim validation failed")
    onnx.checker.check_model(simplified_model)
    onnx.save(simplified_model, str(args.output_onnx_path))

    print(f"Merged and simplified model saved to: {args.output_onnx_path}")
    print("Graph inputs:")
    for graph_input in simplified_model.graph.input:
        print(f"  - {graph_input.name}")
    print("Graph outputs:")
    for graph_output in simplified_model.graph.output:
        print(f"  - {graph_output.name}")
    print("Added filtered detector outputs:")
    for output_name in added_filtered_outputs:
        print(f"  - {output_name}")
    print("Removed query-index outputs from graph outputs:")
    for output_name in removed_query_outputs:
        print(f"  - {output_name}")


if __name__ == "__main__":
    main()
