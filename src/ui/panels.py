"""Draw model metadata and material authoring panels."""

from typing import Optional
import bpy
import json
import os
from ..animation.actions import uax_action_group_metadata
from ..animation.placement import PLACEMENT_PROPERTY
from ..materials.context import ugx_materials_for_object
from ..materials.hw2 import material_current_ufx_value
from ..materials.operators import HWEK_OT_select_ufx_shader_pass
from ..materials.properties import (
    split_ufx_permutation_name,
    ufx_attr_property_name,
    ufx_shader_file_property_name,
    ufx_shader_pass_label,
)
from ..materials.ufx import material_ufx_permutation_status
from ..porting.operators import HWEK_OT_port_hw1_to_hw2, HWEK_OT_remove_ugx_data, HWEK_OT_view_hw2_port_report
from ..textures.operators import HWEK_OT_export_ugx_textures
from ..ugx.data import is_no_rigid_bone_index
from ..ugx.metadata import (
    bone_bound_property_name,
    bone_bounds_are_empty,
    section_entry_source_index,
    section_property_key,
)
from ..ugx.operators import HWEK_OT_adopt_ugx_mesh
from ..ugx.scene import ugx_armature_for, ugx_authoring_mesh_objects, ugx_export_object_for

__all__ = ('draw_ugx_material_controls', 'HWEK_PT_ugx_object', 'HWEK_PT_uax_action', 'HWEK_PT_uax_placement')


class HWEK_PT_uax_action(bpy.types.Panel):
    bl_label = 'Halo Wars'
    bl_idname = 'HWEK_PT_uax_action'
    bl_space_type = 'DOPESHEET_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Action'
    bl_parent_id = 'DOPESHEET_PT_action'

    @classmethod
    def poll(cls, context):
        action = context.active_action
        owner = context.object
        return action is not None and (bool(action.get('hwek_uax_source_path'))
                                       or (owner is not None and owner.type == 'ARMATURE'))

    def draw(self, context):
        action = context.active_action
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.label(text='Animation Defaults')
        if action.get('hwek_uax_source_snapshot') and any(
                name not in action for name in ('hwek_uax_default_loop_count', 'hwek_uax_animation_flags')):
            layout.label(text=action.get('hwek_uax_options_error', 'Reading native animation settings'), icon='INFO')
            return
        layout.prop(action, 'hwek_uax_motion_extraction')
        if action.hwek_uax_motion_extraction == 'CONSTANT':
            groups, index = uax_action_group_metadata(action)
            row = layout.row()
            row.enabled = not (index < len(groups) and groups[index].get('periodic_loop_offset', 0))
            row.prop(action, 'hwek_uax_loop_translation')
        layout.prop(action, 'hwek_uax_override_loop_count')
        row = layout.row()
        row.enabled = action.hwek_uax_override_loop_count
        row.prop(action, 'hwek_uax_loop_count')
        layout.prop(action, 'hwek_uax_clamp_loop_boundaries')


class HWEK_PT_uax_placement(bpy.types.Panel):
    bl_label = 'Initial Placement'
    bl_idname = 'HWEK_PT_uax_placement'
    bl_space_type = 'DOPESHEET_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Action'
    bl_parent_id = 'HWEK_PT_uax_action'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        action = context.active_action
        layout = self.layout
        layout.use_property_decorate = False
        if action.get('hwek_uax_source_snapshot') and PLACEMENT_PROPERTY not in action:
            layout.label(text=action.get('hwek_uax_options_error', 'Reading native placement'), icon='INFO')
            return
        layout.prop(action, 'hwek_uax_placement_position')
        layout.prop(action, 'hwek_uax_placement_rotation')
        layout.label(text='Scale / Shear')
        for index in range(3):
            row = layout.row(align=True)
            for column in range(3):
                row.prop(action, 'hwek_uax_placement_scale_shear', index=index*3+column, text='')


def draw_ugx_material_controls(layout, obj: Optional[bpy.types.Object], mat: bpy.types.Material, section: str = "settings", shader_node=None):
    material_index = int(mat.get("hwek_ugx_material_index", 0))
    if section == "settings" and "hwek_ugx_export_name" in mat:
        layout.prop(mat, '["hwek_ugx_export_name"]', text="UGX name")
    ufx_attrs = json.loads(mat.get("hwek_ugx_ufx_attrs", "{}"))
    if ufx_attrs:
        ufx_box = layout
        if section == "settings":
            for name in (
                "blendMode",
                "skinned",
            ):
                if name not in ufx_attrs:
                    continue
                prop = ufx_attr_property_name(name)
                if prop in mat:
                    ufx_box.prop(mat, f'["{prop}"]', text={"blendMode": "Blend Mode", "skinned": "Skinned"}.get(name, name))

        if section == "passes":
            for name in ("ufxVersion", "shadowRequiresConsts", "terrainBlending"):
                prop = ufx_attr_property_name(name)
                if prop in mat:
                    ufx_box.prop(mat, f'["{prop}"]', text=name)
            source_path = str(mat.get("hwek_ugx_source_path", ""))
            permutation_box = ufx_box.box()
            permutation_box.label(text="Shader passes")
            permutation_box.label(text="UGX exports the permutation name and UFX header hash.", icon="INFO")
            for index in range(4):
                name_prop = ufx_attr_property_name(f"shaderPermutationName{index}")
                hash_prop = ufx_attr_property_name(f"shaderPermutationHash{index}")
                permutation_name = str(material_current_ufx_value(mat, ufx_attrs, f"shaderPermutationName{index}", "") or "")
                permutation_hash = material_current_ufx_value(mat, ufx_attrs, f"shaderPermutationHash{index}", "")
                status, icon = material_ufx_permutation_status(
                    source_path,
                    mat,
                    index,
                    permutation_name,
                    permutation_hash,
                )
                family, key = split_ufx_permutation_name(permutation_name)
                pass_box = permutation_box.box()
                row = pass_box.row(align=True)
                row.label(text=ufx_shader_pass_label(index))
                row.label(text=status, icon=icon)
                name_row = pass_box.row(align=True)
                if name_prop in mat:
                    name_row.prop(mat, f'["{name_prop}"]', text="Permutation")
                else:
                    name_row.label(text=f"Permutation: {permutation_name or '<empty>'}")
                operator = name_row.operator(HWEK_OT_select_ufx_shader_pass.bl_idname, text="", icon="FILE_FOLDER")
                operator.pass_index = index
                operator.material_index = material_index
                operator.material_name = mat.name
                if hash_prop in mat:
                    hash_row = pass_box.row(align=True)
                    hash_row.label(text="UFX header hash")
                    hash_row.prop(mat, f'["{hash_prop}"]', text="")
                elif permutation_hash not in ("", None):
                    pass_box.label(text=f"UFX header hash: {permutation_hash}")
                pass_box.label(text=f"Shader family: {family}" if family else "Shader family: <empty>")
                if key:
                    pass_box.label(text=f"Permutation key: {key}")
                selected_path = str(mat.get(ufx_shader_file_property_name(index), "")).strip()
                if selected_path:
                    pass_box.label(text=f"Selected file: {os.path.basename(selected_path)}")


class HWEK_PT_ugx_object(bpy.types.Panel):
    bl_label = "New Arcadia UGX"
    bl_idname = "HWEK_PT_ugx_object"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        return (
            ugx_export_object_for(context.object) is not None
            or bool(ugx_authoring_mesh_objects(context.object))
        )

    def draw(self, context):
        layout = self.layout
        obj = ugx_export_object_for(context.object)
        authoring_objects = ugx_authoring_mesh_objects(context.object)
        armature = ugx_armature_for(context.object)
        if armature is not None and 'hwek_ugx_granny_model_name' in armature:
            layout.prop(armature, '["hwek_ugx_granny_model_name"]', text="Animation Model")
        if armature is not None:
            if 'hwek_ugx_granny_lod_type' not in armature or int(armature['hwek_ugx_granny_lod_type']) in (0, 1, 2):
                layout.prop(armature, 'hwek_ugx_skeleton_lod_mode')
            else:
                layout.prop(armature, '["hwek_ugx_granny_lod_type"]', text="Skeleton LOD")
        remove_box = layout.box()
        remove_box.label(text="Cross-game port", icon="UNLINKED")
        remove_box.operator(HWEK_OT_port_hw1_to_hw2.bl_idname, icon="DUPLICATE")
        if context.object is not None and context.object.get("hwek_ugx_port_target") == 6:
            remove_box.label(text="HW2 copy ready. Export as Halo Wars 2 UGX.", icon="CHECKMARK")
            remove_box.operator(HWEK_OT_view_hw2_port_report.bl_idname, icon="TEXT")
        remove_box.operator(HWEK_OT_remove_ugx_data.bl_idname)
        remove_box.label(text="Keeps geometry, rig, materials, textures, and actions.", icon="INFO")
        if obj is None or obj.type != "MESH":
            if not authoring_objects:
                return
            armature = ugx_armature_for(context.object)
            material_count = len({
                int(material.as_pointer())
                for item in authoring_objects
                for material in item.data.materials
                if material is not None
            })
            bone_count = len(armature.data.bones) if armature is not None else 1
            layout.label(text="Custom HW2 armature export")
            row = layout.row(align=True)
            row.label(text=f"{len(authoring_objects)} meshes")
            row.label(text=f"{bone_count} bones")
            row.label(text=f"{max(1, material_count)} materials")
            box = layout.box()
            box.label(text="UGX source data is optional for new HW2 models.", icon="INFO")
            box.label(text="Every mesh bound to this armature is exported as a Granny mesh/accessory.")
            return
        layout.label(text=os.path.basename(str(obj.get("hwek_ugx_source_path", ""))) or obj.name)
        row = layout.row(align=True)
        row.label(text=f"v{int(obj.get('hwek_ugx_cache_version', 0))}")
        row.label(text=f"{int(obj.get('hwek_ugx_section_count', 0))} sections")
        row.label(text=f"{int(obj.get('hwek_ugx_bone_count', 0))} bones")
        row.label(text=f"{int(obj.get('hwek_ugx_accessory_count', 0))} accessories")
        row.label(text=f"{int(obj.get('hwek_ugx_material_count', 0))} materials")
        if "hwek_ugx_mesh_name" in obj:
            mesh_row = layout.row(align=True)
            mesh_row.label(text=f"Mesh {int(obj.get('hwek_ugx_mesh_index', -1))}")
            mesh_row.label(text=str(obj.get("hwek_ugx_mesh_name", obj.name)))
            lod_count = int(obj.get("hwek_ugx_lod_object_count", 1))
            if int(obj.get("hwek_ugx_cache_version", 0)) == 6:
                mesh_row.label(text=f"{lod_count} LOD range{'s' if lod_count != 1 else ''}")
            if lod_count > 1:
                mesh_row.label(text=f"LOD {int(obj.get('hwek_ugx_lod_index', 0))}")
        if obj.get("hwek_ugx_damage_path"):
            layout.label(text=f"Damage: {os.path.basename(str(obj.get('hwek_ugx_damage_path')))}")
        if obj.get("hwek_ugx_reimport_required"):
            box = layout.box()
            box.label(text="Legacy rebuilt scene; the next export will refresh it automatically.", icon="INFO")
            if obj.get("hwek_ugx_exported_damage_path"):
                box.label(text=f"Paired damage: {os.path.basename(str(obj.get('hwek_ugx_exported_damage_path')))}")
        if int(obj.get("hwek_ugx_cache_version", 0)) == 6 and not obj.get("hwek_ugx_reimport_required"):
            adopt_box = layout.box()
            adopt_box.label(text="Cross-file mesh transfer")
            adopt_box.label(text="Select a donor mesh too; keep this target active.")
            adopt_box.operator(HWEK_OT_adopt_ugx_mesh.bl_idname, icon="DUPLICATE")
        layout.operator(HWEK_OT_export_ugx_textures.bl_idname, text="Export Loaded Textures")
        if ugx_materials_for_object(obj):
            layout.label(text="Edit UGX data in Material Properties.", icon="MATERIAL")

        armature = context.object if context.object and context.object.type == "ARMATURE" else ugx_armature_for(obj)
        active_bone = getattr(context, "active_bone", None)
        if armature and (active_bone is None or active_bone.name not in armature.data.bones):
            active_bone = armature.data.bones.active
        if active_bone and "hwek_ugx_bone_index" in active_bone:
            box = layout.box()
            box.label(text=f"Bone {int(active_bone['hwek_ugx_bone_index'])}: {active_bone.name}")
            if "hwek_ugx_export_name" in active_bone:
                box.prop(active_bone, '["hwek_ugx_export_name"]', text="UGX name")
            if 'hwek_ugx_granny_lod_error' in active_bone:
                box.prop(active_bone, '["hwek_ugx_granny_lod_error"]', text="LOD Error")
            if bone_bounds_are_empty(active_bone):
                box.label(text="No geometry uses this bone (empty UGX bounds).", icon="INFO")
            else:
                for key, label in (("low", "Bounds low"), ("high", "Bounds high")):
                    for axis, axis_name in enumerate("XYZ"):
                        prop_key = bone_bound_property_name(key, axis)
                        if prop_key in active_bone:
                            box.prop(active_bone, f'["{prop_key}"]', text=f"{label} {axis_name}")

        sections = json.loads(obj.get("hwek_ugx_sections", "[]"))
        if sections:
            section_index = max(0, min(len(sections) - 1, int(obj.get("hwek_ugx_section_edit_index", 0))))
            section = sections[section_index]
            source_section_index = section_entry_source_index(section_index, section)
            box = layout.box()
            box.prop(obj, '["hwek_ugx_section_edit_index"]', text="Section index")
            box.label(text=f"Section {source_section_index}: {int(section.get('face_count', 0))} faces")
            for key, label in (
                ("accessory_index", "Accessory index"),
                ("max_bones", "Max bones"),
                ("rigid_bone_index", "Rigid bone index"),
                ("rigid_only", "Rigid only"),
                ("lod_min_distance", "LOD min"),
                ("lod_max_distance", "LOD max"),
                ("sorting_bias", "Sorting bias"),
            ):
                prop_key = section_property_key(source_section_index, key)
                if prop_key in obj:
                    if key == "rigid_bone_index" and is_no_rigid_bone_index(obj[prop_key]):
                        box.label(text="Rigid bone index: None (skinned section)")
                        box.label(text="Raw UGX sentinel: 0x7FFFFFFF", icon="INFO")
                        continue
                    box.prop(obj, f'["{prop_key}"]', text=label)
