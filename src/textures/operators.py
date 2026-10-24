"""Blender operators for texture discovery, replacement and export."""

from bpy.props import BoolProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper
from typing import Any, Dict, Optional, Tuple
import bpy
import json
import os
from ..materials.context import ugx_material_cache_version
from ..materials.context import (
    active_ugx_material,
    active_ugx_texture_node,
    context_blender_material,
    material_texture_slot_current_value,
)
from ..materials.data import v4_material_usage
from ..materials.discovery import discover_material_textures
from ..materials.slots import (
    apply_v4_source_material_context,
    assign_image_to_ugx_texture_node,
    assign_image_to_ugx_texture_slot,
    assign_texture_path_to_ugx_texture_slot,
    clear_v4_texture_auto_context,
    sync_ugx_material_texture_slots_from_props,
)
from ..ugx.scene import ugx_material_context_object
from .context import (
    IMAGE_TEXTURE_FILTER,
    texture_usage_is_non_color,
    ugx_texture_context_path,
    ugx_texture_path_from_image,
)
from .discovery import resolve_texture_path, texture_set_usage_candidates
from .export import active_ugx_texture_image, export_loaded_ugx_textures
from .images import image_from_ddx, set_ugx_image_source, write_dds_image
from .paths import texture_path_hint_for_slot

__all__ = ('HWEK_OT_import_ugx_texture', 'HWEK_OT_replace_ugx_texture_slot', 'HWEK_OT_reset_ugx_texture_slot', 'HWEK_OT_discover_ugx_textures', 'HWEK_OT_apply_ugx_texture_folder', 'HWEK_OT_export_ugx_texture', 'HWEK_OT_export_ugx_textures')

class HWEK_OT_import_ugx_texture(bpy.types.Operator, ImportHelper):
    bl_idname = "hwek.import_ugx_texture"
    bl_label = "Import Halo Wars Texture"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".ddx"
    filter_glob: StringProperty(default=IMAGE_TEXTURE_FILTER, options={"HIDDEN"})
    non_color: BoolProperty(
        name="Non-Color Data",
        description="Load the texture as non-color data. The addon enables this automatically for normal, specular, packed-map, and mask slots.",
        default=False,
    )
    assign_active_node: BoolProperty(
        name="Assign Active UGX Texture Node",
        description="Replace the selected UGX material texture node with this texture and update the exported UGX texture path.",
        default=True,
    )
    texture_path: StringProperty(
        name="UGX Texture Path",
        description="Optional exported UGX texture path. Leave blank to derive it from the selected file or the slot's original folder.",
        default="",
    )

    def invoke(self, context, event):
        _, mat, node = active_ugx_texture_node(context)
        usage = str(node.get("hwek_ugx_usage", "")).lower() if node else ""
        if texture_usage_is_non_color(usage, ugx_material_cache_version(mat) if mat else 0):
            self.non_color = True
        return super().invoke(context, event)

    def execute(self, context):
        obj, mat, node = active_ugx_texture_node(context)
        usage = str(node.get("hwek_ugx_usage", "")).lower() if node else ""
        try:
            image = image_from_ddx(self.filepath, self.non_color or texture_usage_is_non_color(usage, ugx_material_cache_version(mat) if mat else 0))
            set_ugx_image_source(image, self.filepath)
            if self.assign_active_node and obj is not None and mat is not None and node is not None:
                target_path = assign_image_to_ugx_texture_node(obj, mat, node, image, self.texture_path)
            else:
                target_path = ""
            space = getattr(context, "space_data", None)
            if space is not None and getattr(space, "type", "") == "IMAGE_EDITOR":
                space.image = image
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if target_path:
            self.report({"INFO"}, f"Imported texture and set UGX path to {target_path}.")
        else:
            self.report({"INFO"}, f"Imported texture {image.name}.")
        return {"FINISHED"}


class HWEK_OT_replace_ugx_texture_slot(bpy.types.Operator, ImportHelper):
    bl_idname = "hwek.replace_ugx_texture_slot"
    bl_label = "Replace UGX Texture Slot"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".ddx"
    filter_glob: StringProperty(default=IMAGE_TEXTURE_FILTER, options={"HIDDEN"})
    slot_index: IntProperty(default=-1, options={"HIDDEN"})
    material_index: IntProperty(default=-1, options={"HIDDEN"})
    texture_path: StringProperty(
        name="UGX Texture Path",
        description="Optional exported UGX texture path. Leave blank to derive it from the chosen file and this slot's original folder.",
        default="",
    )
    non_color: BoolProperty(
        name="Non-Color Data",
        description="Load this texture as non-color data.",
        default=False,
    )

    def slot_info(self, context) -> Tuple[Optional[bpy.types.Object], Optional[bpy.types.Material], Dict[str, Any]]:
        obj = ugx_material_context_object(getattr(context, "object", None))
        mat = None
        if obj is not None and self.material_index >= 0:
            for candidate in obj.data.materials:
                if candidate is not None and int(candidate.get("hwek_ugx_material_index", -1)) == self.material_index:
                    mat = candidate
                    break
        if mat is None:
            mat = active_ugx_material(obj, True)
        if mat is None:
            return obj, mat, {}
        slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
        if self.slot_index < 0 or self.slot_index >= len(slots):
            return obj, mat, {}
        return obj, mat, slots[self.slot_index]

    def invoke(self, context, event):
        obj, mat, slot = self.slot_info(context)
        usage = str(slot.get("usage", "")).lower()
        self.non_color = texture_usage_is_non_color(usage, ugx_material_cache_version(mat) if mat else 0)
        if obj is not None and slot:
            current_path = material_texture_slot_current_value(mat, self.slot_index, slot)
            usage_file = str(mat.get(f"hwek_ugx_texture_{usage}_file", "")) if mat and usage else ""
            resolved = usage_file or resolve_texture_path(ugx_texture_context_path(obj), current_path)
            if resolved:
                self.filepath = resolved
        return super().invoke(context, event)

    def execute(self, context):
        obj, mat, slot = self.slot_info(context)
        if obj is None or mat is None or not slot:
            self.report({"ERROR"}, "Select an imported UGX object and a valid texture slot.")
            return {"CANCELLED"}
        usage = str(slot.get("usage", "")).lower()
        try:
            image = image_from_ddx(self.filepath, self.non_color or texture_usage_is_non_color(usage, ugx_material_cache_version(mat) if mat else 0))
            set_ugx_image_source(image, self.filepath)
            target_path = assign_image_to_ugx_texture_slot(obj, mat, self.slot_index, image, self.texture_path)
            sync_ugx_material_texture_slots_from_props(obj, mat)
            obj.data.update()
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Replaced {usage or ('slot ' + str(self.slot_index))} with {target_path or image.name}.")
        return {"FINISHED"}


class HWEK_OT_reset_ugx_texture_slot(bpy.types.Operator):
    bl_idname = "hwek.reset_ugx_texture_slot"
    bl_label = "Reset UGX Texture Slot"
    bl_options = {"REGISTER", "UNDO"}
    slot_index: IntProperty(default=-1, options={"HIDDEN"})
    material_index: IntProperty(default=-1, options={"HIDDEN"})

    def execute(self, context):
        obj = ugx_material_context_object(getattr(context, "object", None))
        mat = None
        if obj is not None and self.material_index >= 0:
            for candidate in obj.data.materials:
                if candidate is not None and int(candidate.get("hwek_ugx_material_index", -1)) == self.material_index:
                    mat = candidate
                    break
        if mat is None:
            mat = active_ugx_material(obj, True)
        if obj is None or mat is None:
            self.report({"ERROR"}, "Select an imported UGX object and material first.")
            return {"CANCELLED"}
        try:
            slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
        except Exception:
            slots = []
        if self.slot_index < 0 or self.slot_index >= len(slots):
            self.report({"ERROR"}, "Selected texture slot is no longer available.")
            return {"CANCELLED"}
        slot = slots[self.slot_index]
        usage = v4_material_usage(slot.get("usage", ""))
        source_path = str(slot.get("path", ""))
        try:
            assign_texture_path_to_ugx_texture_slot(obj, mat, self.slot_index, source_path)
            clear_v4_texture_auto_context(mat, self.slot_index)
            if usage == "diffuse":
                apply_v4_source_material_context(obj, mat, "diffuse", source_path)
            sync_ugx_material_texture_slots_from_props(obj, mat)
            obj.data.update()
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Reset {usage or ('slot ' + str(self.slot_index))}.")
        return {"FINISHED"}


class HWEK_OT_discover_ugx_textures(bpy.types.Operator):
    bl_idname = "hwek.discover_ugx_textures"
    bl_label = "Discover Textures"
    bl_description = "Fill empty or missing texture connections from the model or Blender scene folder"
    bl_options = {"REGISTER", "UNDO"}
    material_name: StringProperty(default="", options={"HIDDEN"})
    directory: StringProperty(subtype="DIR_PATH")

    def invoke(self, context, event):
        if self.directory:
            return self.execute(context)
        mat = bpy.data.materials.get(self.material_name) or context_blender_material(context)
        source = str(mat.get("hwek_ugx_source_path", "")) if mat else ""
        source = source or bpy.data.filepath
        if source:
            self.directory = os.path.dirname(source)
            return self.execute(context)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        mat = bpy.data.materials.get(self.material_name) or context_blender_material(context)
        if mat is None:
            self.report({"ERROR"}, "Select an Anders material first.")
            return {"CANCELLED"}
        try:
            source = str(mat.get("hwek_ugx_source_path", "")) or bpy.data.filepath
            changed = discover_material_textures(mat, bpy.path.abspath(self.directory), source)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Connected {changed} texture(s). Ambiguous names and existing connections are left in place.")
        return {"FINISHED"}


class HWEK_OT_apply_ugx_texture_folder(bpy.types.Operator):
    bl_idname = "hwek.apply_ugx_texture_folder"
    bl_label = "Apply UGX Texture Folder"
    bl_options = {"REGISTER", "UNDO"}
    directory: StringProperty(subtype="DIR_PATH")
    material_index: IntProperty(default=-1, options={"HIDDEN"})

    def invoke(self, context, event):
        obj = ugx_material_context_object(getattr(context, "object", None))
        mat = active_ugx_material(obj, True)
        if obj is not None and mat is not None and not self.directory:
            source_path = ""
            try:
                slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
            except Exception:
                slots = []
            for slot_index, slot in enumerate(slots):
                if v4_material_usage(slot.get("usage", "")) != "diffuse":
                    continue
                source_path = material_texture_slot_current_value(mat, slot_index, slot)
                break
            resolved = resolve_texture_path(ugx_texture_context_path(obj), source_path)
            if resolved:
                self.directory = os.path.dirname(resolved)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        obj = ugx_material_context_object(getattr(context, "object", None))
        mat = None
        if obj is not None and self.material_index >= 0:
            for candidate in obj.data.materials:
                if candidate is not None and int(candidate.get("hwek_ugx_material_index", -1)) == self.material_index:
                    mat = candidate
                    break
        if mat is None:
            mat = active_ugx_material(obj, True)
        if obj is None or mat is None:
            self.report({"ERROR"}, "Select an imported UGX object and material first.")
            return {"CANCELLED"}
        candidates = texture_set_usage_candidates(self.directory)
        if not candidates:
            self.report({"ERROR"}, "The selected folder contains no recognized UGX texture files.")
            return {"CANCELLED"}
        try:
            slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
        except Exception:
            slots = []
        changed = 0
        errors = []
        for slot_index, slot in enumerate(slots):
            usage = v4_material_usage(slot.get("usage", ""))
            if not usage or usage not in candidates:
                continue
            if int(slot.get("offset", 0) or 0) <= 0:
                continue
            path = candidates[usage]
            try:
                image = image_from_ddx(path, texture_usage_is_non_color(usage, ugx_material_cache_version(mat)))
                set_ugx_image_source(image, path)
                target = ugx_texture_path_from_image(ugx_texture_context_path(obj), texture_path_hint_for_slot(slot), image)
                assign_image_to_ugx_texture_slot(obj, mat, slot_index, image, target)
                if usage != "diffuse":
                    auto_key = f"hwek_ugx_texture_slot_{slot_index}_auto_context"
                    mat[auto_key] = True
                    mat[auto_key + "_value"] = target
                changed += 1
            except Exception as exc:
                errors.append(f"{usage}: {exc}")
        if changed:
            sync_ugx_material_texture_slots_from_props(obj, mat)
            obj.data.update()
        if errors and not changed:
            self.report({"ERROR"}, "; ".join(errors[:3]))
            return {"CANCELLED"}
        if not changed:
            self.report({"ERROR"}, "No texture slots matched the selected folder.")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Applied {changed} texture slot(s) from folder.")
        return {"FINISHED"}


class HWEK_OT_export_ugx_texture(bpy.types.Operator, ExportHelper):
    bl_idname = "hwek.export_ugx_texture"
    bl_label = "Export Active UGX Texture"
    filename_ext = ".dds"
    filter_glob: StringProperty(default="*.dds;*.ddx", options={"HIDDEN"})

    def invoke(self, context, event):
        image = active_ugx_texture_image(context.object)
        if image and not self.filepath:
            source = str(image.get("hwek_ugx_ddx_path", ""))
            self.filepath = os.path.join(os.path.dirname(source), os.path.basename(source))
        return super().invoke(context, event)

    def execute(self, context):
        image = active_ugx_texture_image(context.object)
        if image is None:
            self.report({"ERROR"}, "Active UGX material has no loaded texture image.")
            return {"CANCELLED"}
        try:
            write_dds_image(image, self.filepath)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class HWEK_OT_export_ugx_textures(bpy.types.Operator):
    bl_idname = "hwek.export_ugx_textures"
    bl_label = "Export Loaded UGX Textures"
    directory: StringProperty(subtype="DIR_PATH")

    def invoke(self, context, event):
        obj = ugx_material_context_object(context.object)
        if obj and not self.directory:
            self.directory = os.path.dirname(str(obj.get("hwek_ugx_source_path", "")))
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        obj = ugx_material_context_object(context.object)
        if obj is None:
            self.report({"ERROR"}, "Select an imported UGX object first.")
            return {"CANCELLED"}
        try:
            outputs = export_loaded_ugx_textures(obj, self.directory)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if not outputs:
            self.report({"ERROR"}, "Imported UGX has no loaded texture images.")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported {len(outputs)} UGX textures.")
        return {"FINISHED"}
