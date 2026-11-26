from collections import Counter
from pathlib import Path
import re

import bpy
import numpy as np
from bpy_extras.io_utils import ExportHelper

from .frame_dump import DumpError, FrameDump, frame_log
from .mod_export import (
    ATTRIBUTE_FORMATS,
    MeshBuffers,
    get_component,
    output_target,
    replacement_textures,
    write_mod,
)
from .options import ExportOptions, draw_export_options
from .skinning_variants import BLEND, add_variants, blend_dtype, field_draws


BONE_GROUP = re.compile(r"Bone (\d+)")
NORMAL_ROUND_TRIP_TOLERANCE = 0.002


def values_from(data, prop, count, width, dtype=np.float32):
    values = np.empty((count, width), dtype=dtype)
    data.foreach_get(prop, values.ravel())
    return values


def normalize(vectors):
    magnitude = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, magnitude, out=np.zeros_like(vectors), where=magnitude > 1e-10)


def game_axes(vectors):
    return vectors[:, [0, 2, 1]] * [1, 1, -1]


def blend_data(obj, mesh, ids, palette_size, dtype=BLEND):
    limit = dtype["weights"].shape[0]
    groups = {}
    for group in obj.vertex_groups:
        match = BONE_GROUP.fullmatch(group.name)
        if match:
            groups[group.index] = int(match[1])

    blend = np.zeros(len(mesh.vertices), dtype=dtype)
    for vertex_id in np.unique(ids):
        weights = {}
        for element in mesh.vertices[int(vertex_id)].groups:
            bone = groups.get(element.group)
            weight = element.weight
            if bone is None or not weight:
                continue
            if bone >= palette_size or not np.isfinite(weight) or weight < 0:
                raise DumpError(f"Vertex {vertex_id} has an invalid bone index or weight")
            weights[bone] = weights.get(bone, 0.0) + weight
        if not weights or len(weights) > limit:
            raise DumpError(f"Vertex {vertex_id} must have weights for one to {limit} Bone groups")
        total = sum(weights.values())
        for column, (bone, weight) in enumerate(sorted(weights.items(), key=lambda item: (-item[1], item[0]))):
            blend["weights"][vertex_id, column] = weight / total
            blend["indices"][vertex_id, column] = bone
    return blend[ids]


def texture_coordinates(mesh, number):
    layer = mesh.uv_layers.get(f"TEXCOORD{number}")
    if layer is None:
        return None
    return values_from(layer.uv, "vector", len(mesh.loops), 2)


def attribute_streams(mesh, loops, ids, streams, uv0):
    uv = {0: uv0}
    color = None
    result = {}
    for slot, (stride, fields) in sorted(streams.items()):
        stream = np.zeros((len(loops), stride), dtype=np.uint8)
        ranges = {}
        for field in sorted(fields, key=lambda field: (field.offset, field.semantic, field.uv)):
            dtype, width = ATTRIBUTE_FORMATS[field.format]
            end = field.offset + dtype.itemsize * width
            if field.offset < 0 or end > stride:
                raise DumpError(f"{field.semantic}{field.uv} is outside VB{slot}")
            used = (field.offset, end)
            overlapping = next((span for span in ranges if used[0] < span[1] and span[0] < used[1]), None)
            if overlapping and used != overlapping:
                raise DumpError(f"Overlapping fields in VB{slot}")

            if field.semantic == "TEXCOORD":
                if field.uv not in uv:
                    uv[field.uv] = texture_coordinates(mesh, field.uv)
                if uv[field.uv] is None:
                    if used in ranges:
                        continue
                    raise DumpError(f"Missing UV map TEXCOORD{field.uv}")
                values = uv[field.uv][loops]
                values[:, 1] = 1.0 - values[:, 1]
            else:
                if color is None:
                    attr = mesh.color_attributes.get("COLOR0")
                    if attr is None or attr.domain not in {"POINT", "CORNER"}:
                        raise DumpError("Missing COLOR0 vertex color attribute")
                    data = values_from(attr.data, "color", len(attr.data), 4)
                    if not np.isfinite(data).all():
                        raise DumpError("COLOR0 contains non-finite values")
                    color = np.rint(np.clip(data[ids if attr.domain == "POINT" else loops], 0, 1) * 255.0)
                values = color
            with np.errstate(over="ignore", invalid="ignore"):
                encoded = np.ascontiguousarray(values, dtype=dtype)
            if not np.isfinite(encoded).all():
                raise DumpError(f"{field.semantic}{field.uv} does not fit {field.format}")
            encoded = encoded.view(np.uint8).reshape(len(loops), -1)
            if used in ranges and not np.array_equal(stream[:, field.offset:end], encoded):
                raise DumpError(
                    f"TEXCOORD{field.uv} and TEXCOORD{ranges[used]} share storage in VB{slot}. "
                    "Use matching values or remove the redundant UV map."
                )
            stream[:, field.offset:end] = encoded
            ranges[used] = field.uv
        result[f"VB{slot}"] = stream
    return result


def position_data(mesh, loops, ids, matrix):
    count = len(mesh.loops)
    xyz = values_from(mesh.vertices, "co", len(mesh.vertices), 3)[ids]
    normal = values_from(mesh.corner_normals, "vector", count, 3)[loops]
    tangent = values_from(mesh.loops, "tangent", count, 3)[loops]
    sign = values_from(mesh.loops, "bitangent_sign", count, 1)[loops]
    linear = matrix[:3, :3]
    normal = normalize(normal @ np.linalg.inv(linear))
    tangent = tangent @ linear.T
    tangent = normalize(tangent - np.sum(tangent * normal, axis=1, keepdims=True) * normal)
    undefined = np.linalg.norm(tangent, axis=1) <= 1e-10
    if np.any(undefined):
        normals = normal[undefined]
        basis = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
        tangent[undefined] = normalize(np.cross(basis, normals))
    sign = np.where(sign < 0, 1, -1) * np.sign(np.linalg.det(linear))
    output = np.column_stack((
        game_axes(xyz @ linear.T + matrix[:3, 3]), game_axes(normal), game_axes(tangent), sign,
    )).astype("<f4", order="C")
    if not np.isfinite(output).all():
        raise DumpError("Position, normal or tangent is not finite")
    return output


def encode_original_indices(mesh, loops, ids, triangles, matrix, source, blend):
    vertex_ids = mesh.attributes.get("APMI_VERTEX_ID")
    if (
        vertex_ids is None or vertex_ids.domain != "POINT" or vertex_ids.data_type != "INT"
        or not np.array_equal(matrix, np.eye(4))
    ):
        return None

    original_ids = values_from(vertex_ids.data, "value", len(mesh.vertices), 1, np.int32).ravel()
    source_rows = np.searchsorted(source.vertex_ids, original_ids)
    if np.any(source_rows == len(source.vertex_ids)) or not np.array_equal(source.vertex_ids[source_rows], original_ids):
        return None

    indices = original_ids[ids[np.searchsorted(loops, triangles)]]
    source_indices = source.vertex_ids[source.triangles]
    unchanged = np.array_equal(indices, source_indices)
    if not unchanged and Counter(map(tuple, indices)) - Counter(map(tuple, source_indices)):
        return None

    positions = values_from(mesh.vertices, "co", len(mesh.vertices), 3)
    if not np.array_equal(game_axes(positions), source.positions[source_rows]):
        return None

    corner_rows = source_rows[ids]
    for number in range(3):
        uv = source.uvs.get(number)
        layer = texture_coordinates(mesh, number)
        if (layer is None) != (uv is None) or (layer is not None and not np.array_equal(layer[loops], uv[corner_rows])):
            return None
    color = mesh.color_attributes.get("COLOR0")
    if (color is None) != (source.color is None):
        return None
    if color is not None:
        if color.domain not in {"POINT", "CORNER"}:
            return None
        values = values_from(color.data, "color", len(color.data), 4)
        if not np.array_equal(values[ids if color.domain == "POINT" else loops], source.color[corner_rows]):
            return None

    weights = np.minimum(source.weights[corner_rows], 1).astype(np.float64)
    bones = source.bones[corner_rows]
    active = blend["weights"] > 0
    matches = (blend["indices"][:, :, None] == bones[:, None, :]) & active[:, :, None]
    if np.any((weights > 0) & ~matches.any(axis=1)):
        return None
    normalized = (matches * weights[:, None, :]).sum(axis=2) / weights.sum(axis=1, keepdims=True)
    if not np.allclose(blend["weights"], normalized, rtol=1e-6, atol=0):
        return None

    # Edit Mode deletion can change packed corner normals on surviving faces.
    if indices.size == source_indices.size:
        if not mesh.has_custom_normals:
            return None
        reference = mesh.copy()
        try:
            normals = source.normals[source_rows][:, [0, 2, 1]] * np.array([1, -1, 1], dtype=np.float32)
            reference.normals_split_custom_set_from_vertices(normals.tolist())
            actual = values_from(mesh.corner_normals, "vector", len(mesh.loops), 3)
            expected = values_from(reference.corner_normals, "vector", len(mesh.loops), 3)
            differences = np.linalg.norm(actual[loops] - expected[loops], axis=1)
            if not np.all(differences <= NORMAL_ROUND_TRIP_TOLERANCE):
                return None
        finally:
            bpy.data.meshes.remove(reference)

    files = {} if unchanged else {"IB": (indices.astype(np.int64) - source.draw.base).astype("<u4").tobytes()}
    return MeshBuffers(0, indices.size, files)


def encode_mesh(obj, depsgraph, component, source=None):
    if any(modifier.type == "ARMATURE" and modifier.show_viewport for modifier in obj.modifiers):
        raise DumpError("Disable the Armature modifier to export pre-skinning geometry")
    matrix = np.array(obj.matrix_world, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise DumpError("The object transform is not finite")
    determinant = np.linalg.det(matrix[:3, :3])
    if determinant == 0:
        raise DumpError("The object transform is singular")
    palette = component.skinning.palette
    if palette is None or palette.path.stat().st_size % 64:
        raise DumpError("The source bone palette is missing or has an invalid size")
    bone_count = palette.path.stat().st_size // 64

    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        mesh.calc_loop_triangles()
        if not mesh.loop_triangles:
            raise DumpError("The mesh has no triangles")
        triangles = values_from(mesh.loop_triangles, "loops", len(mesh.loop_triangles), 3, np.int32)
        if determinant < 0:
            triangles = triangles[:, [0, 2, 1]]
        loops = np.unique(triangles)
        ids = values_from(mesh.loops, "vertex_index", len(mesh.loops), 1, np.int32).ravel()[loops]
        uv0 = texture_coordinates(mesh, 0)
        if uv0 is None:
            raise DumpError("Missing UV map TEXCOORD0")
        blend = blend_data(obj, mesh, ids, bone_count, blend_dtype(component.skinning.influence_count))
        if source is not None:
            original = encode_original_indices(mesh, loops, ids, triangles, matrix, source, blend)
            if original is not None:
                return original
        mesh.calc_tangents(uvmap="TEXCOORD0")
        attributes = attribute_streams(mesh, loops, ids, component.streams, uv0)
        position = position_data(mesh, loops, ids, matrix)
    finally:
        evaluated.to_mesh_clear()

    size = len(loops)
    data = {"Position": position.view(np.uint8), "Blend": blend.view(np.uint8).reshape(size, -1), **attributes}
    keys = np.column_stack((ids.astype("<u4").view(np.uint8).reshape(size, 4), *data.values()))
    packed = np.ascontiguousarray(keys).view(f"V{keys.shape[1]}").ravel()
    _, first, remap = np.unique(packed, return_index=True, return_inverse=True)
    order = np.argsort(first)
    indices = np.argsort(order)[remap[np.searchsorted(loops, triangles)]].astype("<u4")
    files = {name: values[first[order]].tobytes() for name, values in data.items()}
    files["IB"] = indices.tobytes()
    return MeshBuffers(len(first), indices.size, files)


def merge_buffers(meshes):
    files = {name: bytearray() for name in meshes[0].files}
    vertex_count = index_count = 0
    for mesh in meshes:
        if vertex_count + mesh.vertices > np.iinfo(np.uint32).max:
            raise DumpError("The component exceeds the 32-bit index limit")
        for name, data in mesh.files.items():
            if name == "IB":
                values = np.frombuffer(data, dtype="<u4")
                data = (values + np.uint32(vertex_count)).astype("<u4").tobytes()
            files[name].extend(data)
        vertex_count += mesh.vertices
        index_count += mesh.indices
    return MeshBuffers(vertex_count, index_count, files)


def collect_parts(context, full_mesh=False, field_log=None):
    objects = sorted((obj for obj in context.selected_objects if obj.type == "MESH"), key=lambda obj: obj.name)
    if not objects:
        raise DumpError("Select one or more imported APMI meshes")
    depsgraph = context.evaluated_depsgraph_get()
    dumps, by_log, groups = {}, {}, {}
    for obj in objects:
        try:
            if obj.get("APMI_Pose") != "PRE_SKINNING":
                raise DumpError("Import the source again using Pre-Skinning")
            if not obj.get("APMI_SourceLog"):
                raise DumpError("The source log path is missing")
            log = Path(bpy.path.abspath(obj["APMI_SourceLog"])).resolve(strict=True)
            if log not in dumps:
                dumps[log] = FrameDump(log)
            dump = dumps[log]
            call = int(obj["APMI_Draw"])
            key = log, call
            if key not in by_log:
                by_log[key] = get_component(dump, call)
            component = by_log[key]
            draw = component.draw
            expected = (component.ib, component.vb0, draw.first, draw.count, draw.base, component.skinning.call)
            actual = tuple(obj.get("APMI_" + name) for name in (
                "IB", "VB0", "FirstIndex", "IndexCount", "BaseVertex", "Dispatch",
            ))
            if actual != expected:
                raise DumpError("The source log no longer matches the imported component")
            runtime_key = component.ib, draw.first, draw.count
            part_key = log, dump.component_key(draw, component.skinning)
            if runtime_key not in groups:
                groups[runtime_key] = part_key, component, []
            assigned_key, _, group = groups[runtime_key]
            if assigned_key != part_key:
                raise DumpError("The selection includes two source components with the same draw match")
            group.append(obj)
        except (KeyError, ValueError, OSError, TypeError, RuntimeError, np.linalg.LinAlgError) as error:
            raise DumpError(f"{obj.name}: {error}") from error

    parts, variants = [], []
    for (log, _), component, group in sorted(groups.values(), key=lambda entry: (entry[1].draw.call, entry[1].ib)):
        source = None
        if len(group) == 1 and not full_mesh:
            try:
                source = dumps[log].decode_part(component.draw, component.skinning, True, True)
            except (KeyError, ValueError, OSError):
                pass
        buffers = []
        for obj in group:
            try:
                buffers.append(encode_mesh(obj, depsgraph, component, source))
            except (KeyError, ValueError, OSError, TypeError, RuntimeError, np.linalg.LinAlgError) as error:
                raise DumpError(f"{obj.name}: {error}") from error
        combined = buffers[0] if len(buffers) == 1 else merge_buffers(buffers)
        parts.append((component, combined))
        if "Blend" in combined.files:
            variants.append((log, component, combined))
    if field_log is not None and variants:
        field_dump = FrameDump(field_log)
        add_variants(dumps, field_dump, field_draws(field_dump), variants)
    return parts


class APMI_OT_export_mod(bpy.types.Operator, ExportHelper, ExportOptions):
    bl_idname = "export_scene.apmi_mod"
    bl_label = "Export APMI Mod"
    bl_description = "Export selected pre-skinning meshes as an APMI mod folder"

    filename_ext = ""
    check_extension = False

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and any(obj.type == "MESH" for obj in context.selected_objects)

    def draw(self, context):
        draw_export_options(self.layout, self)

    def invoke(self, context, event):
        if not self.filepath:
            active_file = Path(bpy.data.filepath)
            self.filepath = str(active_file.parent / (active_file.stem or "APMI_Mod"))
        return super().invoke(context, event)

    def execute(self, context):
        try:
            target = output_target(bpy.path.abspath(self.filepath))
            field_log = frame_log(bpy.path.abspath(self.field_capture)) if self.field_capture.strip() else None
            parts = collect_parts(context, self.full_mesh, field_log)
            folder = Path(bpy.path.abspath(self.dds_folder)) if self.dds_folder else None
            replacements, ignored = replacement_textures(folder, [part for part, _ in parts])
            parts = [
                (part, buffers) for part, buffers in parts
                if buffers.files or part.textures.keys() & replacements.keys()
            ]
            write_mod(target, parts, replacements, self.overwrite)
        except (ValueError, OSError, KeyError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        if ignored:
            print("APMI-Tools: no matching source texture for " + ", ".join(ignored))
            self.report({"WARNING"}, f"Ignored {len(ignored)} DDS files with no source hash match")
        warnings = [f"{part.ib}: {buffers.variant_warning}" for part, buffers in parts if buffers.variant_warning]
        if warnings:
            for warning in warnings:
                print(f"APMI-Tools: {warning}")
            self.report({"WARNING"}, "Some draws use the original mesh for " + ", ".join(
                part.ib for part, buffers in parts if buffers.variant_warning
            ))
        self.report({"INFO"}, f"Exported {len(parts)} APMI components and {len(replacements)} DDS files")
        return {"FINISHED"}


def menu_export(self, context):
    self.layout.operator(APMI_OT_export_mod.bl_idname, text="APMI Mod")
