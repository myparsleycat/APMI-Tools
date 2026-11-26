from pathlib import Path

import bpy
from bpy.props import PointerProperty, StringProperty

from .mod_export import valid_mod_name
from .options import ExportOptions, ImportOptions, draw_export_options, draw_import_options, mesh_pose


class APMI_PG_settings(bpy.types.PropertyGroup, ImportOptions, ExportOptions):
    frame_log: StringProperty(
        name="Frame Analysis Log",
        description="Path to the Frame Analysis log.txt",
        subtype="FILE_PATH",
    )
    pose: mesh_pose("PRE_SKINNING")
    output_folder: StringProperty(
        name="Output Folder",
        description="Parent folder for the exported mod",
        subtype="DIR_PATH",
    )
    mod_name: StringProperty(
        name="Mod Name",
        description="Name of the mod folder to create inside the output folder",
        default="APMI_Mod",
    )


class APMI_PT_import(bpy.types.Panel):
    bl_label = "Import Frame Analysis"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "APMI Tools"
    bl_order = 0

    def draw(self, context):
        settings = context.scene.apmi_tools
        layout = self.layout
        layout.prop(settings, "frame_log", text="Log")
        layout.prop(settings, "pose", text="Pose")

        header, body = layout.panel("apmi_import_options", default_closed=True)
        header.label(text="Import Options")
        if body is not None:
            draw_import_options(body, settings)

        row = layout.row()
        row.enabled = bool(settings.frame_log.strip()) and context.mode == "OBJECT"
        row.operator_context = "EXEC_DEFAULT"
        op = row.operator("import_scene.apmi_frame_analysis", text="Import Frame Analysis", icon="IMPORT")
        op.filepath = settings.frame_log.strip()
        for name in ImportOptions.__annotations__:
            setattr(op, name, getattr(settings, name))


class APMI_PT_export(bpy.types.Panel):
    bl_label = "Export Mod"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "APMI Tools"
    bl_order = 1

    def draw(self, context):
        settings = context.scene.apmi_tools
        layout = self.layout
        layout.prop(settings, "output_folder")
        layout.prop(settings, "mod_name")
        draw_export_options(layout, settings)

        meshes = [obj for obj in context.selected_objects if obj.type == "MESH"]
        name = settings.mod_name.strip()
        valid_name = valid_mod_name(name)
        ready = False
        if context.mode != "OBJECT":
            layout.label(text="Switch to Object Mode", icon="INFO")
        elif not meshes:
            layout.label(text="Select meshes to export", icon="INFO")
        elif any(not obj.get("APMI_SourceLog") or obj.get("APMI_Pose") != "PRE_SKINNING" for obj in meshes):
            layout.label(text="Select Pre-Skinning APMI meshes", icon="INFO")
        else:
            layout.label(text=f"Selected meshes: {len(meshes)}", icon="MESH_DATA")
            ready = True

        if name and not valid_name:
            layout.label(text="Enter a valid mod folder name", icon="ERROR")

        row = layout.row()
        row.enabled = ready and bool(settings.output_folder.strip()) and valid_name
        row.operator_context = "EXEC_DEFAULT"
        op = row.operator("export_scene.apmi_mod", text="Export Mod", icon="EXPORT")
        if valid_name and settings.output_folder.strip():
            folder = bpy.path.abspath(settings.output_folder.strip())
            op.filepath = str(Path(folder) / name)
        for prop in ExportOptions.__annotations__:
            setattr(op, prop, getattr(settings, prop))


CLASSES = (APMI_PG_settings, APMI_PT_import, APMI_PT_export)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.apmi_tools = PointerProperty(type=APMI_PG_settings)


def unregister():
    del bpy.types.Scene.apmi_tools
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
