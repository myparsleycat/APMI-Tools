bl_info = {
    "name": "APMI-Tools",
    "description": "Import APMI Frame Analysis meshes and export APMI mods",
    "version": (1, 0, 0),
    "blender": (5, 2, 0),
    "location": "3D Viewport > Sidebar > APMI Tools",
    "category": "Import-Export",
}

if "bpy" in locals():
    from importlib import reload
    from sys import modules

    for name in ("frame_dump", "options", "mod_export", "blender_export", "sidebar"):
        if (module := modules.get(f"{__name__}.{name}")) is not None:
            reload(module)

from pathlib import Path

import bpy
import numpy as np
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper

from .blender_export import APMI_OT_export_mod, menu_export
from .frame_dump import DumpError, FrameDump
from .options import ImportOptions, draw_import_options
from . import sidebar


def y_up_to_z_up(values):
    return values[:, [0, 2, 1]] * np.array([1, -1, 1], dtype=np.float32)


def build_mesh(mesh, part):
    mesh.from_pydata(y_up_to_z_up(part.positions).tolist(), [], part.triangles.tolist(), shade_flat=False)
    mesh.validate()
    mesh.update()

    mesh.normals_split_custom_set_from_vertices(y_up_to_z_up(part.normals).tolist())
    loops = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loops)
    for number, values in sorted(part.uvs.items()):
        layer = mesh.uv_layers.new(name=f"TEXCOORD{number}", do_init=False)
        layer.uv.foreach_set("vector", values[loops].ravel())

    if part.color is not None:
        color = mesh.color_attributes.new(name="COLOR0", type="FLOAT_COLOR", domain="POINT")
        color.data.foreach_set("color", part.color.ravel())
    vertex_id = mesh.attributes.new(name="APMI_VERTEX_ID", type="INT", domain="POINT")
    vertex_id.data.foreach_set("value", part.vertex_ids)
    tangent = mesh.attributes.new(name="APMI_TANGENT", type="FLOAT_VECTOR", domain="POINT")
    tangent.data.foreach_set("vector", y_up_to_z_up(part.tangents[:, :3]).ravel())
    sign = mesh.attributes.new(name="APMI_TANGENT_SIGN", type="FLOAT", domain="POINT")
    sign.data.foreach_set("value", -part.tangents[:, 3] if 0 in part.uvs else part.tangents[:, 3].copy())


def add_vertex_groups(obj, part):
    if part.weights is None:
        return
    groups = {
        int(bone): obj.vertex_groups.new(name=f"Bone {bone:03d}")
        for bone in np.unique(part.bones[part.weights > 0])
    }
    for vertex, (indices, weights) in enumerate(zip(part.bones, part.weights)):
        for index, weight in zip(indices, weights):
            if weight > 0:
                groups[int(index)].add([vertex], float(weight), "ADD")


def add_material(mesh, part, created, images, pack_textures, warnings):
    if not part.textures:
        return
    material = bpy.data.materials.new(f"APMI {part.index_hash}")
    created.append(material)
    mesh.materials.append(material)
    material.use_nodes = True
    tree = material.node_tree
    shader = next(node for node in tree.nodes if node.type == "BSDF_PRINCIPLED")
    for row, (slot, texture) in enumerate(sorted(part.textures.items())):
        if texture.path not in images:
            try:
                image = bpy.data.images.load(str(texture.path))
                created.append(image)
                if pack_textures:
                    try:
                        image.pack()
                    except RuntimeError as error:
                        warnings.append(f"{texture.path.name}: {error}")
            except RuntimeError as error:
                warnings.append(f"{texture.path.name}: {error}")
                image = None
            images[texture.path] = image
        image = images[texture.path]
        if image is None:
            continue
        node = tree.nodes.new("ShaderNodeTexImage")
        node.label = f"ps-t{slot} {texture.hash}"
        node.image = image
        node.location = (-360, 200 - 300 * row)
        if slot == 0 and 0 in part.uvs and "SRGB" in texture.path.stem.upper():
            tree.links.new(node.outputs["Color"], shader.inputs["Base Color"])


class APMI_OT_import_frame_analysis(bpy.types.Operator, ImportHelper, ImportOptions):
    bl_idname = "import_scene.apmi_frame_analysis"
    bl_label = "Import APMI Frame Analysis"
    bl_description = "Import character meshes from an APMI Frame Analysis log"
    bl_options = {"UNDO", "PRESET"}

    filename_ext = ".txt"
    filter_glob: StringProperty(default="log.txt", options={"HIDDEN"})
    def draw(self, context):
        self.layout.prop(self, "pose")
        draw_import_options(self.layout, self)

    def execute(self, context):
        log = Path(bpy.path.abspath(self.filepath))
        if log.is_dir():
            log /= "log.txt"
        if not log.is_file() or log.name.lower() != "log.txt":
            self.report({"ERROR"}, "Select the Frame Analysis log.txt")
            return {"CANCELLED"}
        try:
            dump = FrameDump(log)
            parts, warnings = dump.parts(
                self.pose == "PRE_SKINNING", self.import_weights, self.include_without_uv,
                self.draw_filter, self.import_textures,
            )
        except (OSError, ValueError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        if not parts:
            message = warnings[0] if warnings else "No matching APMI skinned meshes with the selected options"
            self.report({"ERROR"}, message)
            return {"CANCELLED"}

        collection = bpy.data.collections.new(f"APMI {log.parent.name}")
        created, objects, images = [collection], [], {}
        try:
            for part in parts:
                name = f"{part.draw.call:06d} {part.index_hash}"
                mesh = bpy.data.meshes.new(name)
                created.append(mesh)
                build_mesh(mesh, part)
                obj = bpy.data.objects.new(name, mesh)
                created.append(obj)
                collection.objects.link(obj)
                objects.append(obj)
                obj["APMI_SourceLog"] = str(log.resolve())
                obj["APMI_Pose"] = self.pose
                obj["APMI_Draw"] = part.draw.call
                obj["APMI_Dispatch"] = part.skinning.call
                obj["APMI_IB"] = part.index_hash
                obj["APMI_VB0"] = part.vertex_hash
                obj["APMI_FirstIndex"] = part.draw.first
                obj["APMI_IndexCount"] = part.draw.count
                obj["APMI_BaseVertex"] = part.draw.base
                add_vertex_groups(obj, part)
                add_material(mesh, part, created, images, self.pack_textures, warnings)
            context.scene.collection.children.link(collection)
        except (DumpError, OSError, RuntimeError, TypeError, ValueError) as error:
            bpy.data.batch_remove(created)
            self.report({"ERROR"}, f"APMI import failed: {error}")
            return {"CANCELLED"}

        for obj in context.selected_objects:
            obj.select_set(False)
        for obj in objects:
            obj.select_set(True)
        context.view_layer.objects.active = max(objects, key=lambda obj: len(obj.data.polygons))

        if warnings:
            for warning in warnings:
                print(f"APMI-Tools: {warning}")
            self.report({"WARNING"}, f"Imported {len(objects)} meshes with {len(warnings)} warnings. See the system console")
        else:
            self.report({"INFO"}, f"Imported {len(objects)} APMI meshes")
        return {"FINISHED"}


def menu_import(self, context):
    self.layout.operator(APMI_OT_import_frame_analysis.bl_idname, text="APMI Frame Analysis (log.txt)")


def register():
    bpy.utils.register_class(APMI_OT_import_frame_analysis)
    bpy.utils.register_class(APMI_OT_export_mod)
    sidebar.register()
    bpy.types.TOPBAR_MT_file_import.append(menu_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    sidebar.unregister()
    bpy.utils.unregister_class(APMI_OT_export_mod)
    bpy.utils.unregister_class(APMI_OT_import_frame_analysis)
