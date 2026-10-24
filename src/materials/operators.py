"""Blender operators for material assignment, repair and baking."""

from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ImportHelper
from typing import Optional, Tuple
import bpy
import json
import os
from ..errors import UgxError
from ..textures.context import ugx_texture_context_path
from ..ugx.data import ugx_game_name
from ..ugx.scene import ugx_material_context_object
from .assignment import assign_ugx_material_data
from .baking import bake_ugx_texture_input, ugx_bake_input_items
from .bindings import ugx_material_node_bindings
from .context import active_ugx_material, context_blender_material, ugx_material_cache_version
from .hw2 import material_current_ufx_value
from .properties import ufx_shader_file_property_name, ufx_shader_pass_label
from .ufx import set_material_ufx_shader_pass_from_file, ufx_shader_path_for_name
from .textures import repair_ugx_material_texture_links
from .uv import add_ugx_uv_animation

__all__ = ('HWEK_OT_add_uv_animation', 'HWEK_OT_validate_ugx_material', 'HWEK_OT_repair_ugx_texture_links', 'HWEK_OT_bake_ugx_texture_input', 'HWEK_OT_assign_ugx_material_data', 'HWEK_OT_select_ufx_shader_pass')

class HWEK_OT_repair_ugx_texture_links(bpy.types.Operator):
    bl_idname = "hwek.repair_ugx_texture_links"
    bl_label = "Repair Texture Links"
    bl_description = "Reconnect tagged imported textures to their material inputs; ambiguous and procedural inputs stay unchanged"
    bl_options = {"REGISTER", "UNDO"}

    material_name: StringProperty(default="", options={"HIDDEN"})

    def execute(self, context):
        mat = bpy.data.materials.get(self.material_name) if self.material_name else context_blender_material(context)
        if mat is None:
            self.report({"ERROR"}, "Select a material first.")
            return {"CANCELLED"}
        try:
            count, skipped = repair_ugx_material_texture_links(mat)
        except UgxError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        message = f"Repaired {count} texture links."
        if skipped:
            message += " Ambiguous inputs left unchanged: " + ", ".join(skipped) + "."
        self.report({"WARNING"} if skipped else {"INFO"}, message)
        return {"FINISHED"}

class HWEK_OT_add_uv_animation(bpy.types.Operator):
    bl_idname = "hwek.add_uv_animation"
    bl_label = "Add UV Animation"
    bl_options = {"REGISTER", "UNDO"}

    material_name: StringProperty(default="", options={"HIDDEN"})

    def execute(self, context):
        mat = bpy.data.materials.get(self.material_name) if self.material_name else context_blender_material(context)
        if mat is None:
            self.report({"ERROR"}, "Select a material first.")
            return {"CANCELLED"}
        try:
            add_ugx_uv_animation(mat)
        except UgxError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class HWEK_OT_validate_ugx_material(bpy.types.Operator):
    bl_idname = "hwek.validate_ugx_material"
    bl_label = "Validate Material"

    material_name: StringProperty(default="", options={"HIDDEN"})

    def execute(self, context):
        mat = bpy.data.materials.get(self.material_name) if self.material_name else context_blender_material(context)
        if mat is None:
            self.report({"ERROR"}, "Select a material first.")
            return {"CANCELLED"}
        try:
            bindings = ugx_material_node_bindings(mat, ugx_material_cache_version(mat, context.object), ugx_texture_context_path(context.object))
        except UgxError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        missing = sum(bool(binding["image"].get("hwek_ugx_missing_texture")) for binding in bindings)
        if missing:
            self.report({"WARNING"}, f"Material {mat.name}: keeps {missing} unavailable source texture reference(s); their image files will not be exported.")
            return {"FINISHED"}
        self.report({"INFO"}, f"Material {mat.name}: {len(bindings)} texture connections ready for export.")
        return {"FINISHED"}


class HWEK_OT_bake_ugx_texture_input(bpy.types.Operator):
    bl_idname = "hwek.bake_ugx_texture_input"
    bl_label = "Bake Texture Input"
    bl_description = "Bake one material input using only meshes and faces assigned to that material, then connect the image"
    bl_options = {"REGISTER", "UNDO"}

    material_name: StringProperty(default="", options={"HIDDEN"})
    usage: EnumProperty(name="Input", items=ugx_bake_input_items)
    resolution: IntProperty(name="Resolution", default=1024, min=8, max=8192)
    uv_layer: StringProperty(name="UV Map", default="UV0")
    all_users: BoolProperty(name="All Meshes Using Material", default=True,
                           description="Include UV coverage from every mesh using this material in the current scene")

    def invoke(self, context, event):
        obj = context.object
        if obj is not None and obj.type == "MESH" and obj.data.uv_layers.active is not None:
            self.uv_layer = obj.data.uv_layers.active.name
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        self.layout.prop(self, "usage")
        self.layout.prop(self, "resolution")
        self.layout.prop(self, "uv_layer")
        self.layout.prop(self, "all_users")
        self.layout.label(text="Chosen material and input only.")
        self.layout.label(text="UV animation stays connected after baking.")

    def execute(self, context):
        mat = bpy.data.materials.get(self.material_name) if self.material_name else context_blender_material(context)
        if mat is None:
            self.report({"ERROR"}, "Select a material first.")
            return {"CANCELLED"}
        try:
            image = bake_ugx_texture_input(context, context.object, mat, self.usage, self.resolution, self.uv_layer, self.all_users)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Baked {image.name} and connected it to {mat.name}.")
        return {"FINISHED"}


class HWEK_OT_assign_ugx_material_data(bpy.types.Operator):
    bl_idname = "hwek.assign_ugx_material_data"
    bl_label = "Assign UGX Material Data"
    bl_options = {"REGISTER", "UNDO"}

    cache_version: IntProperty(default=4, options={"HIDDEN"})
    skinned: BoolProperty(
        name="Skinned",
        description="Use skinned HW2 shader defaults for this material",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        return context_blender_material(context) is not None

    def execute(self, context):
        mat = context_blender_material(context)
        if mat is None:
            self.report({"ERROR"}, "Select a Blender material first.")
            return {"CANCELLED"}
        try:
            assign_ugx_material_data(mat, self.cache_version, self.skinned)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Assigned {ugx_game_name(self.cache_version)} data to material {mat.name}.")
        return {"FINISHED"}


class HWEK_OT_select_ufx_shader_pass(bpy.types.Operator, ImportHelper):
    bl_idname = "hwek.select_ufx_shader_pass"
    bl_label = "Select Halo Wars 2 UFX Shader"
    bl_description = "Set this exported v6 shader pass from a UFX file name and its embedded header hash"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".ufx"
    filter_glob: StringProperty(default="*.ufx;*.UFX", options={"HIDDEN"})
    pass_index: IntProperty(default=0, min=0, max=3, options={"HIDDEN"})
    material_index: IntProperty(default=-1, options={"HIDDEN"})
    material_name: StringProperty(default="", options={"HIDDEN"})

    def target(self, context) -> Tuple[Optional[bpy.types.Object], Optional[bpy.types.Material]]:
        obj = ugx_material_context_object(getattr(context, "object", None))
        if self.material_name:
            return obj, bpy.data.materials.get(self.material_name)
        mat = getattr(context, "material", None)
        if mat is not None and (obj is None or all(candidate != mat for candidate in obj.data.materials)):
            mat = None
        if mat is not None and self.material_index >= 0:
            if int(mat.get("hwek_ugx_material_index", -1)) != self.material_index:
                mat = None
        if mat is None and obj is not None and self.material_index >= 0:
            for candidate in obj.data.materials:
                if candidate is not None and int(candidate.get("hwek_ugx_material_index", -1)) == self.material_index:
                    mat = candidate
                    break
        if mat is None:
            mat = active_ugx_material(obj, True)
        return obj, mat

    def invoke(self, context, event):
        obj, mat = self.target(context)
        if obj is not None and mat is not None:
            selected_path = str(mat.get(ufx_shader_file_property_name(self.pass_index), "")).strip()
            attrs = json.loads(mat.get("hwek_ugx_ufx_attrs", "{}"))
            permutation_name = str(material_current_ufx_value(
                mat,
                attrs,
                f"shaderPermutationName{self.pass_index}",
                "",
            ) or "")
            resolved_path = ufx_shader_path_for_name(str(obj.get("hwek_ugx_source_path", "")), permutation_name)
            if selected_path and os.path.isfile(selected_path):
                self.filepath = selected_path
            elif resolved_path:
                self.filepath = resolved_path
        return super().invoke(context, event)

    def execute(self, context):
        obj, mat = self.target(context)
        if obj is None or mat is None or ugx_material_cache_version(mat, obj) != 6:
            self.report({"ERROR"}, "Select an imported Halo Wars 2 v6 UGX object and material first.")
            return {"CANCELLED"}
        try:
            permutation_name, header_hash = set_material_ufx_shader_pass_from_file(mat, self.pass_index, self.filepath)
            obj.data.update()
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Set {ufx_shader_pass_label(self.pass_index)} to {permutation_name} (hash {header_hash}).",
        )
        return {"FINISHED"}
