import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import bmesh
import bpy
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "apmi_tools", root / "__init__.py", submodule_search_locations=[str(root)],
    )
    addon = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = addon
    spec.loader.exec_module(addon)

    from apmi_tools.blender_export import blend_data, encode_original_indices, values_from

    source = SimpleNamespace(
        positions=np.array([[0, 0, 0], [1, 0, 0], [1, 1, .3], [0, 1, -.2], [-1, 0, .4]], dtype="<f4"),
        normals=np.array([[.13, .4, .8], [.27, -.1, .9], [.3, -.07, .9], [-.13, -.01, .7], [.1, .03, .9]], dtype="<f4"),
        triangles=np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]], dtype="<i4"),
        tangents=np.zeros((5, 4), dtype="<f4"),
        vertex_ids=np.arange(5, dtype="<i4"),
        weights=np.ones((5, 1), dtype="<f4"),
        bones=np.zeros((5, 1), dtype="<u4"),
        draw=SimpleNamespace(base=0),
        uvs={},
        color=None,
    )
    mesh = bpy.data.meshes.new("Delete Vertex")
    addon.build_mesh(mesh, source)
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    addon.add_vertex_groups(obj, source)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    def original_buffers():
        mesh.calc_loop_triangles()
        triangles = values_from(mesh.loop_triangles, "loops", len(mesh.loop_triangles), 3, np.int32)
        loops = np.unique(triangles)
        ids = values_from(mesh.loops, "vertex_index", len(mesh.loops), 1, np.int32).ravel()[loops]
        blend = blend_data(obj, mesh, ids, palette_size=1)
        return encode_original_indices(mesh, loops, ids, triangles, np.array(obj.matrix_world), source, blend)

    mesh.normals_split_custom_set_from_vertices([(1.0, 0.0, 0.0)] * len(mesh.vertices))
    assert original_buffers() is None
    mesh.normals_split_custom_set_from_vertices(addon.y_up_to_z_up(source.normals).tolist())

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action="DESELECT")
    edit_mesh = bmesh.from_edit_mesh(mesh)
    edit_mesh.verts.ensure_lookup_table()
    edit_mesh.verts[4].select = True
    bmesh.update_edit_mesh(mesh, destructive=False)
    bpy.ops.mesh.delete(type="VERT")
    bpy.ops.object.mode_set(mode="OBJECT")

    mesh.normals_split_custom_set_from_vertices([(1.0, 0.0, 0.0)] * len(mesh.vertices))
    buffers = original_buffers()
    assert buffers is not None and buffers.files == {"IB": source.triangles[:2].astype("<u4").tobytes()}


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
