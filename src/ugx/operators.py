"""Blender operators for native model import, export and mesh adoption."""

from bpy.props import BoolProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper, orientation_helper
import bpy
import os
from ..visuals.scene import load_ugx_visual, load_visual_animations, visual_owner
from ..visuals.assembly import load_visual_attachments
from ..textures.export import packed_ugx_texture_recoveries, ugx_loaded_texture_images
from .bones import ugx_should_export_as_new
from .data import ugx_game_name
from .naming import current_ugx_mesh_name
from .scene import ugx_authoring_mesh_objects, ugx_export_object_for, ugx_mesh_has_imported_topology, ugx_lod_sibling_objects_for
from .scene_export import export_new_v4_ugx, export_new_v6_ugx, export_ugx_copy
from .scene_import import import_ugx
from .scene_validation import synchronize_ugx_export_meshes
from .transfer import adopt_ugx_mesh_into_target

__all__ = ('HWEK_OT_import_ugx', 'HWEK_OT_import_hwde_ugx', 'HWEK_OT_import_hw2_ugx', 'HWEK_OT_export_ugx', 'HWEK_OT_export_hwde_ugx', 'HWEK_OT_export_hw2_ugx', 'HWEK_OT_adopt_ugx_mesh')

class HWEK_OT_import_ugx(bpy.types.Operator, ImportHelper):
    bl_idname = "hwek.import_ugx"
    bl_label = "Import UGX"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".ugx"
    filter_glob: StringProperty(default="*.ugx", options={"HIDDEN"})
    expected_cache_version = None
    load_visual: BoolProperty(name='Load Associated VIS', default=False)
    load_animations: BoolProperty(name='Load Linked and Nearby UAX', default=False)

    def execute(self, context):
        try:
            obj = import_ugx(self.filepath, self.expected_cache_version)
            if self.load_visual or self.load_animations:
                report = load_ugx_visual(obj, self.load_animations) if self.load_visual else load_visual_animations(obj)
                if self.load_visual:
                    attached = load_visual_attachments(obj, self.load_animations)
                    report['imported'] += attached['imported']
                    report['issues'].extend(attached['issues'])
                if report is not None:
                    self.report({'WARNING'} if report['issues'] else {'INFO'}, f"Loaded {report['imported']} animation actions. " + '; '.join(report['issues'][:3]))
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        owner = visual_owner(obj)
        if owner is not None and owner.type == 'ARMATURE':
            for selected in context.selected_objects:
                selected.select_set(False)
            owner.select_set(True)
            context.view_layer.objects.active = owner
        return {"FINISHED"}


class HWEK_OT_import_hwde_ugx(HWEK_OT_import_ugx):
    bl_idname = "hwek.import_hwde_ugx"
    bl_label = "Import Halo Wars: Definitive Edition UGX (v4)"
    expected_cache_version = 4


class HWEK_OT_import_hw2_ugx(HWEK_OT_import_ugx):
    bl_idname = "hwek.import_hw2_ugx"
    bl_label = "Import Halo Wars 2 UGX (v6)"
    expected_cache_version = 6


class HWEK_OT_export_ugx(bpy.types.Operator, ExportHelper):
    bl_idname = "hwek.export_ugx"
    bl_label = "Export UGX"
    filename_ext = ".ugx"
    filter_glob: StringProperty(default="*.ugx", options={"HIDDEN"})
    export_loaded_textures: BoolProperty(
        name="Export Loaded Material Textures",
        description="Also write loaded DDX dependencies under the art root required by the UGX material paths",
        default=True,
    )
    expected_cache_version = None
    export_visual: BoolProperty(name='Export VIS, Damage and UAX', description='Write visual definitions, damage data, bone points and linked animation actions alongside the model', default=False)

    def report_texture_exports(self, outputs):
        if outputs:
            self.report({"INFO"}, f"Exported UGX and {len(outputs)} loaded material texture(s).")
        missing = {str(image.get("hwek_ugx_texture_path", image.name)) for image in ugx_loaded_texture_images(bpy.context.object)
                   if image.get("hwek_ugx_missing_texture")}
        if missing:
            self.report({"WARNING"}, f"Export kept {len(missing)} unavailable source texture reference(s); their image files were not written.")
        recovered = packed_ugx_texture_recoveries(bpy.context.object) if self.export_loaded_textures else set()
        if recovered:
            self.report({'WARNING'}, f"Rebuilt {len(recovered)} texture(s) from packed images because their source DDS files were unavailable.")

    def execute(self, context):
        obj = context.object
        restore_edit_mode = obj is not None and obj.mode == "EDIT"
        try:
            if restore_edit_mode:
                bpy.ops.object.mode_set(mode="OBJECT")
            return self.execute_export(context)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            if restore_edit_mode and obj.name in bpy.data.objects:
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                bpy.ops.object.mode_set(mode="EDIT")

    def execute_export(self, context):
        selected_obj = context.object
        synchronize_ugx_export_meshes(selected_obj)
        obj = ugx_export_object_for(selected_obj)
        if self.expected_cache_version in (4, 6):
            authoring_objects = ugx_authoring_mesh_objects(selected_obj)
            needs_armature_export = bool(authoring_objects) and (
                obj is None
                or obj.type != "MESH"
                or ugx_should_export_as_new(obj, self.expected_cache_version)
                or (self.expected_cache_version == 4 and len(authoring_objects) > 1
                    and set(authoring_objects) != set(ugx_lod_sibling_objects_for(obj)))
                or (
                    self.expected_cache_version == 4
                    and (
                        getattr(self, "axis_forward", "Z") != "Z"
                        or getattr(self, "axis_up", "Y") != "Y"
                    )
                )
                or any(not ugx_mesh_has_imported_topology(item) for item in authoring_objects)
            )
            if needs_armature_export:
                try:
                    if self.expected_cache_version == 4:
                        outputs = export_new_v4_ugx(
                            selected_obj,
                            self.filepath,
                            getattr(self, "axis_forward", "Z"),
                            getattr(self, "axis_up", "Y"),
                            export_loaded_textures=self.export_loaded_textures,
                            export_visual=self.export_visual,
                        )
                    else:
                        outputs = export_new_v6_ugx(selected_obj, self.filepath, export_loaded_textures=self.export_loaded_textures, export_visual=self.export_visual)
                    self.report_texture_exports(outputs)
                    self.report(
                        {"INFO"},
                        f"Exported {len(authoring_objects)} armature-bound {ugx_game_name(self.expected_cache_version)} mesh object(s).",
                    )
                except Exception as exc:
                    self.report({"ERROR"}, str(exc))
                    return {"CANCELLED"}
                return {"FINISHED"}
        export_as_new = obj is not None and ugx_should_export_as_new(obj, self.expected_cache_version)
        if obj is None or export_as_new:
            new_obj = obj if export_as_new else context.object
            if self.expected_cache_version in (4, 6) and new_obj is not None and new_obj.type == "MESH":
                try:
                    if self.expected_cache_version == 4:
                        outputs = export_new_v4_ugx(
                            new_obj,
                            self.filepath,
                            getattr(self, "axis_forward", "Z"),
                            getattr(self, "axis_up", "Y"),
                            export_loaded_textures=self.export_loaded_textures,
                            export_visual=self.export_visual,
                        )
                    else:
                        outputs = export_new_v6_ugx(new_obj, self.filepath, export_loaded_textures=self.export_loaded_textures, export_visual=self.export_visual)
                    self.report_texture_exports(outputs)
                except Exception as exc:
                    self.report({"ERROR"}, str(exc))
                    return {"CANCELLED"}
                return {"FINISHED"}
            self.report({"ERROR"}, "Select a mesh, an armature with bound meshes, or an imported UGX object first.")
            return {"CANCELLED"}
        if self.expected_cache_version is not None and int(obj.get("hwek_ugx_cache_version", 0)) != self.expected_cache_version:
            self.report({"ERROR"}, f"Select an imported {ugx_game_name(self.expected_cache_version)} UGX first.")
            return {"CANCELLED"}
        try:
            outputs = export_ugx_copy(obj, self.filepath, export_loaded_textures=self.export_loaded_textures, export_visual=self.export_visual)
            self.report_texture_exports(outputs)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


@orientation_helper(axis_forward="Z", axis_up="Y")
class HWEK_OT_export_hwde_ugx(HWEK_OT_export_ugx):
    bl_idname = "hwek.export_hwde_ugx"
    bl_label = "Export Halo Wars: Definitive Edition UGX (v4)"
    expected_cache_version = 4
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "export_loaded_textures")
        layout.prop(self, "export_visual")
        layout.separator()
        column = layout.column()
        column.label(text="Animation compatibility axes")
        row = column.row(align=True)
        row.prop(self, "axis_forward", text="Forward")
        row.prop(self, "axis_up", text="Up")
        column.label(text="Adjusts the UGX skeleton bind basis. Existing UAX files stay untouched.", icon="INFO")


class HWEK_OT_export_hw2_ugx(HWEK_OT_export_ugx):
    bl_idname = "hwek.export_hw2_ugx"
    bl_label = "Export Halo Wars 2 UGX (v6)"
    expected_cache_version = 6


class HWEK_OT_adopt_ugx_mesh(bpy.types.Operator):
    bl_idname = "hwek.adopt_ugx_mesh"
    bl_label = "Adopt Other Selected UGX Mesh"
    bl_description = "Copy one selected mesh from another HW2 UGX into the active imported UGX, including material and skeleton remapping"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        target = ugx_export_object_for(context.object)
        if target is None:
            self.report({"ERROR"}, "Make an imported target UGX mesh active.")
            return {"CANCELLED"}
        donors = []
        for candidate in context.selected_objects:
            source = ugx_export_object_for(candidate)
            if source is None or source == target or source in donors:
                continue
            if str(source.get("hwek_ugx_import_id", "")) != str(target.get("hwek_ugx_import_id", "")):
                donors.append(source)
        if len(donors) != 1:
            self.report({"ERROR"}, "Select exactly one mesh from another UGX, then make the target UGX mesh active.")
            return {"CANCELLED"}
        try:
            adopted = adopt_ugx_mesh_into_target(target, donors[0])
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Adopted {current_ugx_mesh_name(adopted)} into {os.path.basename(str(target.get('hwek_ugx_source_path', '')))}.")
        return {"FINISHED"}
