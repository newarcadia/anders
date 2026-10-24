"""Register Blender classes, menus, handlers and session caches."""

from bpy.props import BoolProperty, EnumProperty, FloatVectorProperty, IntProperty, PointerProperty
import bpy
import os
from .animation.operators import HWEK_OT_export_uax, HWEK_OT_import_uax
from .animation.actions import set_uax_animation_flag, uax_loop_translation, set_uax_loop_translation
from .animation.placement import uax_placement_component, set_uax_placement_component
from .ugx.skeleton_scene import set_skeleton_lod_mode
from .animation.playback import ensure_uax_playback_handler, remove_uax_playback_handler
from .animation.loading import clear_pending_uax_actions, resume_pending_uax_actions, prepare_active_uax_actions
from .materials.migration import (
    ensure_builtin_hw_shader_groups_after_startup,
    ensure_builtin_hw_shader_groups_on_load,
)
from .materials.node_types import (
    ShaderNodeAndersHW1PlayerColorMask,
    ShaderNodeAndersHW1UV,
    ShaderNodeAndersHW2,
    ShaderNodeAndersHWDE,
    ShaderNodeAndersParameters,
    ShaderNodeAndersShaderPasses,
    ShaderNodeAndersUVAnimation,
    ensure_ugx_material_node_types,
    menu_ugx_material_nodes,
)
from .materials.operators import (
    HWEK_OT_add_uv_animation,
    HWEK_OT_assign_ugx_material_data,
    HWEK_OT_bake_ugx_texture_input,
    HWEK_OT_select_ufx_shader_pass,
    HWEK_OT_validate_ugx_material,
    HWEK_OT_repair_ugx_texture_links,
)
from .materials.uv import sync_hw1_texture_sampling
from .materials.ownership import sync_material_shader_states
from .materials.player_color import (
    UGX_PLAYER_COLOR_PROPERTY,
    sync_ugx_player_colors_from_depsgraph,
    update_ugx_player_color,
)
from .porting.operators import HWEK_OT_port_hw1_to_hw2, HWEK_OT_remove_ugx_data, HWEK_OT_view_hw2_port_report
from .textures.images import pack_ugx_texture_previews_before_save
from .textures.operators import (
    HWEK_OT_apply_ugx_texture_folder,
    HWEK_OT_discover_ugx_textures,
    HWEK_OT_export_ugx_texture,
    HWEK_OT_export_ugx_textures,
    HWEK_OT_import_ugx_texture,
    HWEK_OT_replace_ugx_texture_slot,
    HWEK_OT_reset_ugx_texture_slot,
)
from .ugx.operators import (
    HWEK_OT_adopt_ugx_mesh,
    HWEK_OT_export_hw2_ugx,
    HWEK_OT_export_hwde_ugx,
    HWEK_OT_export_ugx,
    HWEK_OT_import_hw2_ugx,
    HWEK_OT_import_hwde_ugx,
    HWEK_OT_import_ugx,
)
from .ui.panels import HWEK_PT_ugx_object, HWEK_PT_uax_action, HWEK_PT_uax_placement
from .ui.preferences import HWEK_AddonPreferences
from .porting.fbx import HWEK_OT_import_fbx, HWEK_OT_export_fbx_ugx, HWEK_PT_fbx
from .toolkit import HWEK_OT_enable_terrain_tools, restore_terrain_tools, disable_terrain_tools
from .visuals.scene import synchronize_visual_documents
from .visuals.playback import synchronize_visual_playback, update_visual_attachments
from .visuals.properties import HWEK_VisualUV, HWEK_VisualUVSet, HWEK_VisualSettings
from .visuals.damage_properties import HWEK_DamageField, HWEK_DamageAction, HWEK_DamageEvent, HWEK_DamageRegion, HWEK_DamageSettings
from .visuals.damage_authoring import HWEK_OT_make_detachable, HWEK_OT_reset_damage_preview, menu_damage_authoring
from .visuals.damage_ui import (
    HWEK_DamageAttribute, HWEK_OT_load_damage, HWEK_OT_edit_damage_xml, HWEK_OT_edit_damage_action,
    HWEK_OT_select_damage_event, HWEK_MT_damage_events, HWEK_PT_damage,
    HWEK_UL_damage_events, HWEK_UL_damage_actions, HWEK_OT_damage_entry,
)
from .visuals.operators import (
    HWEK_OT_import_vis, HWEK_OT_export_vis, HWEK_OT_edit_vis,
    HWEK_OT_add_vis_point, HWEK_OT_create_vis, HWEK_PT_visual, HWEK_PT_visual_uv,
    HWEK_OT_activate_vis_action, HWEK_MT_visual_actions,
)

__all__ = ('menu_import', 'menu_export', 'classes', 'unregister_legacy_panel', 'remove_stale_anders_menus', 'register', 'unregister')

def menu_import(self, context):
    self.layout.operator(HWEK_OT_import_fbx.bl_idname, text='FBX for Halo Wars')
    self.layout.operator(HWEK_OT_load_damage.bl_idname, text='Halo Wars Damage Template (DMG)')
    self.layout.operator(HWEK_OT_import_vis.bl_idname, text='Halo Wars Visual (VIS)')
    self.layout.operator(HWEK_OT_import_hwde_ugx.bl_idname, text="Halo Wars: Definitive Edition UGX (v4)")
    self.layout.operator(HWEK_OT_import_hw2_ugx.bl_idname, text="Halo Wars 2 UGX (v6)")
    self.layout.operator(HWEK_OT_import_uax.bl_idname, text="Halo Wars UAX Animation")
    self.layout.operator(HWEK_OT_import_ugx_texture.bl_idname, text="Halo Wars Texture (DDX, DDS, Images)")


def menu_export(self, context):
    self.layout.operator(HWEK_OT_export_vis.bl_idname, text='Halo Wars Visual (VIS)')
    self.layout.operator(HWEK_OT_export_hwde_ugx.bl_idname, text="Halo Wars: Definitive Edition UGX (v4)")
    self.layout.operator(HWEK_OT_export_hw2_ugx.bl_idname, text="Halo Wars 2 UGX (v6)")
    self.layout.operator(HWEK_OT_export_uax.bl_idname, text="Halo Wars Animation UAX")


classes = (
    HWEK_AddonPreferences,
    HWEK_OT_import_fbx, HWEK_OT_export_fbx_ugx, HWEK_PT_fbx, HWEK_OT_enable_terrain_tools,
    HWEK_VisualUV, HWEK_VisualUVSet, HWEK_VisualSettings,
    HWEK_DamageField, HWEK_DamageAction, HWEK_DamageEvent, HWEK_DamageRegion, HWEK_DamageSettings,
    HWEK_OT_make_detachable, HWEK_OT_reset_damage_preview,
    HWEK_DamageAttribute, HWEK_OT_load_damage, HWEK_OT_edit_damage_xml, HWEK_OT_edit_damage_action,
    HWEK_OT_select_damage_event, HWEK_MT_damage_events, HWEK_PT_damage,
    HWEK_UL_damage_events, HWEK_UL_damage_actions, HWEK_OT_damage_entry,
    HWEK_OT_import_vis, HWEK_OT_export_vis, HWEK_OT_edit_vis,
    HWEK_OT_add_vis_point, HWEK_OT_create_vis, HWEK_PT_visual, HWEK_PT_visual_uv,
    HWEK_OT_activate_vis_action, HWEK_MT_visual_actions,
    HWEK_OT_remove_ugx_data,
    HWEK_OT_port_hw1_to_hw2,
    HWEK_OT_view_hw2_port_report,
    HWEK_OT_import_hwde_ugx,
    HWEK_OT_import_hw2_ugx,
    HWEK_OT_import_uax,
    HWEK_OT_export_uax,
    HWEK_OT_export_ugx,
    HWEK_OT_export_hwde_ugx,
    HWEK_OT_export_hw2_ugx,
    HWEK_OT_adopt_ugx_mesh,
    HWEK_OT_assign_ugx_material_data,
    HWEK_OT_import_ugx_texture,
    HWEK_OT_discover_ugx_textures,
    HWEK_OT_select_ufx_shader_pass,
    HWEK_OT_replace_ugx_texture_slot,
    HWEK_OT_reset_ugx_texture_slot,
    HWEK_OT_apply_ugx_texture_folder,
    HWEK_OT_export_ugx_texture,
    HWEK_OT_export_ugx_textures,
    HWEK_OT_add_uv_animation,
    HWEK_OT_validate_ugx_material,
    HWEK_OT_repair_ugx_texture_links,
    HWEK_OT_bake_ugx_texture_input,
    HWEK_PT_ugx_object,
    HWEK_PT_uax_action,
    HWEK_PT_uax_placement,
)


def unregister_legacy_panel():
    for name in ("HWEK_PT_ugx_constant_buffers", "HWEK_PT_ugx_shader_passes", "HWEK_PT_ugx_material_advanced", "HWEK_PT_ugx_material"):
        panel = getattr(bpy.types, name, None)
        if panel is not None:
            bpy.utils.unregister_class(panel)
    try:
        bpy.utils.unregister_class(HWEK_OT_import_ugx)
    except RuntimeError:
        pass
    panel = getattr(bpy.types, "HWEK_PT_ugx_object", None)
    if panel is not None:
        try:
            bpy.utils.unregister_class(panel)
        except RuntimeError:
            pass


def remove_stale_anders_menus():
    for menu in (bpy.types.TOPBAR_MT_file_import, bpy.types.TOPBAR_MT_file_export, bpy.types.NODE_MT_add, bpy.types.VIEW3D_MT_object_context_menu):
        for callback in tuple(getattr(menu.draw, "_draw_funcs", ())):
            code = getattr(callback, "__code__", None)
            filename = os.path.basename(code.co_filename) if code else ""
            package = __package__.removesuffix(".src")
            owner = getattr(callback, "__module__", "") or ""
            owned = owner == package or owner.startswith(package + ".")
            legacy = filename in ("anders.py", "terrain.py", "terrain_fbx.py")
            if (owned or legacy) and callback.__name__ in ("menu_import", "menu_export", "menu_ugx_material_nodes", "menu_damage_authoring"):
                menu.remove(callback)
    for name in ("ANDERS_PT_terrain", "ANDERS_OT_import_terrain", "ANDERS_OT_export_terrain", "ANDERS_OT_bake_terrain",
                 "ANDERS_OT_import_terrain_fbx", "ANDERS_OT_export_terrain_fbx", "ANDERS_OT_terrain_fbx_role", "ANDERS_OT_terrain_world_bounds"):
        cls = getattr(bpy.types, name, None)
        if cls is not None:
            bpy.utils.unregister_class(cls)


def clear_session_caches(_unused=None):
    from .textures.discovery import TEXTURE_LOOKUP_CACHE
    from .materials.source import V4_SOURCE_MATERIAL_LOOKUP_CACHE
    TEXTURE_LOOKUP_CACHE.clear()
    V4_SOURCE_MATERIAL_LOOKUP_CACHE.clear()


@bpy.app.handlers.persistent
def reset_session_caches_on_load(_unused):
    clear_session_caches()


def register():
    remove_stale_anders_menus()
    unregister_legacy_panel()
    clear_session_caches()
    if hasattr(bpy.types.Object, UGX_PLAYER_COLOR_PROPERTY):
        delattr(bpy.types.Object, UGX_PLAYER_COLOR_PROPERTY)
    setattr(
        bpy.types.Object,
        UGX_PLAYER_COLOR_PROPERTY,
        FloatVectorProperty(
            name="Player Color",
            description="Shared player-color preview for every Halo Wars material on this armature",
            subtype="COLOR",
            size=4,
            min=0.0,
            max=1.0,
            default=(0.28, 0.37, 0.09, 1.0),
            update=update_ugx_player_color,
        ),
    )
    ensure_ugx_material_node_types()
    bpy.types.Object.hwek_ugx_skeleton_lod_mode = EnumProperty(
        name='Skeleton LOD',
        description='Native policy for reducing the animated skeleton',
        items=(
            ('AUTO', 'Automatic', 'Use source data when available, otherwise use export defaults', -1),
            ('NONE', 'Disabled', 'Animate the complete skeleton at every LOD', 0),
            ('ESTIMATED', 'Estimated', 'Use estimated per-bone LOD errors', 1),
            ('MEASURED', 'Measured', 'Use measured per-bone LOD errors', 2),
        ),
        options=set(),
        get=lambda obj: int(obj.get('hwek_ugx_granny_lod_type', -1)),
        set=set_skeleton_lod_mode,
    )
    bpy.types.Action.hwek_uax_override_loop_count = BoolProperty(
        name='Override Loop Count',
        description='Store a loop-count default; VIS playback can override it',
        options=set(),
        get=lambda action: bool(int(action.get('hwek_uax_animation_flags', 0)) & 1),
        set=lambda action, value: set_uax_animation_flag(action, 1, value),
    )
    bpy.types.Action.hwek_uax_loop_count = IntProperty(
        name='Loop Count',
        description='Native animation default; zero repeats indefinitely',
        min=0, max=2147483647, options=set(),
        get=lambda action: int(action.get('hwek_uax_default_loop_count', 0)),
        set=lambda action, value: action.__setitem__('hwek_uax_default_loop_count', value),
    )
    bpy.types.Action.hwek_uax_motion_extraction = EnumProperty(
        name='Motion Extraction',
        description='How the game uses the root track for movement',
        items=(
            ('NONE', 'None', 'Keep root transforms in the skeletal pose', 0),
            ('CONSTANT', 'Constant', 'Use the native loop translation or periodic loop for movement', 1),
            ('VARIABLE', 'Variable', 'Extract movement from changes in the root transform curves', 4),
        ),
        options=set(),
        get=lambda action: 4 if int(action.get('hwek_uax_accumulation_flags', 0)) & 4
            else int(action.get('hwek_uax_accumulation_flags', 0)) & 1,
        set=lambda action, value: action.__setitem__('hwek_uax_accumulation_flags',
            (int(action.get('hwek_uax_accumulation_flags', 0)) & ~5) | value),
    )
    bpy.types.Action.hwek_uax_loop_translation = FloatVectorProperty(
        name='Loop Translation',
        description='Model-space movement over one animation loop, in game axes; unused when the group has a periodic loop',
        size=3, options=set(), get=uax_loop_translation, set=set_uax_loop_translation,
    )
    bpy.types.Action.hwek_uax_placement_position = FloatVectorProperty(
        name='Location', description='Animation-group initial location in game axes, used for paired and cinematic alignment',
        size=3, options=set(),
        get=lambda action: uax_placement_component(action, 'position'),
        set=lambda action, value: set_uax_placement_component(action, 'position', value),
    )
    bpy.types.Action.hwek_uax_placement_rotation = FloatVectorProperty(
        name='Rotation', description='Animation-group initial rotation in game axes, ordered W, X, Y, Z',
        size=4, subtype='QUATERNION', options=set(),
        get=lambda action: uax_placement_component(action, 'rotation'),
        set=lambda action, value: set_uax_placement_component(action, 'rotation', value),
    )
    bpy.types.Action.hwek_uax_placement_scale_shear = FloatVectorProperty(
        name='Scale / Shear', description='Animation-group initial scale and shear matrix in native row order',
        size=9, options=set(),
        get=lambda action: uax_placement_component(action, 'scale_shear'),
        set=lambda action, value: set_uax_placement_component(action, 'scale_shear', value),
    )
    bpy.types.Action.hwek_uax_clamp_loop_boundaries = BoolProperty(
        name='Clamp Loop Boundaries',
        description='Stored boundary-clamp default; VIS playback can override it',
        options=set(),
        get=lambda action: bool(int(action.get('hwek_uax_animation_flags', 0)) & 2),
        set=lambda action, value: set_uax_animation_flag(action, 2, value),
    )
    for cls in classes:
        if not cls.is_registered:
            bpy.utils.register_class(cls)
    bpy.types.Object.hwek_visual = PointerProperty(type=HWEK_VisualSettings)
    bpy.types.Object.hwek_damage = PointerProperty(type=HWEK_DamageSettings)
    bpy.types.NODE_MT_add.append(menu_ugx_material_nodes)
    bpy.types.TOPBAR_MT_file_import.append(menu_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_export)
    bpy.types.VIEW3D_MT_object_context_menu.append(menu_damage_authoring)
    ensure_uax_playback_handler()
    if update_visual_attachments not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(update_visual_attachments)
    if resume_pending_uax_actions not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(resume_pending_uax_actions)
    if resume_pending_uax_actions() is not None and not bpy.app.timers.is_registered(resume_pending_uax_actions):
        bpy.app.timers.register(resume_pending_uax_actions, first_interval=0.1)
    if prepare_active_uax_actions not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(prepare_active_uax_actions)
    if synchronize_visual_playback not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(synchronize_visual_playback)
    if reset_session_caches_on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.insert(0, reset_session_caches_on_load)
    if pack_ugx_texture_previews_before_save not in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.append(pack_ugx_texture_previews_before_save)
    if ensure_builtin_hw_shader_groups_on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(ensure_builtin_hw_shader_groups_on_load)
    if sync_hw1_texture_sampling not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(sync_hw1_texture_sampling)
    if sync_material_shader_states not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(sync_material_shader_states)
    if sync_ugx_player_colors_from_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(sync_ugx_player_colors_from_depsgraph)
    if ensure_builtin_hw_shader_groups_after_startup() is not None:
        bpy.app.timers.register(ensure_builtin_hw_shader_groups_after_startup, first_interval=0.1)
    if not bpy.app.timers.is_registered(synchronize_visual_documents):
        bpy.app.timers.register(synchronize_visual_documents, first_interval=0.5, persistent=True)
    if not bpy.app.timers.is_registered(restore_terrain_tools):
        bpy.app.timers.register(restore_terrain_tools, first_interval=0.1, persistent=True)


def unregister(keep_node_types=False):
    disable_terrain_tools()
    if synchronize_visual_playback in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(synchronize_visual_playback)
    if update_visual_attachments in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(update_visual_attachments)
    clear_session_caches()
    clear_pending_uax_actions()
    if bpy.app.timers.is_registered(resume_pending_uax_actions):
        bpy.app.timers.unregister(resume_pending_uax_actions)
    if prepare_active_uax_actions in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(prepare_active_uax_actions)
    if resume_pending_uax_actions in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(resume_pending_uax_actions)
    if bpy.app.timers.is_registered(synchronize_visual_documents):
        bpy.app.timers.unregister(synchronize_visual_documents)
    if reset_session_caches_on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(reset_session_caches_on_load)
    remove_stale_anders_menus()
    if bpy.app.timers.is_registered(ensure_builtin_hw_shader_groups_after_startup):
        bpy.app.timers.unregister(ensure_builtin_hw_shader_groups_after_startup)
    if ensure_builtin_hw_shader_groups_on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(ensure_builtin_hw_shader_groups_on_load)
    if sync_hw1_texture_sampling in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(sync_hw1_texture_sampling)
    if sync_material_shader_states in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(sync_material_shader_states)
    if sync_ugx_player_colors_from_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(sync_ugx_player_colors_from_depsgraph)
    if pack_ugx_texture_previews_before_save in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(pack_ugx_texture_previews_before_save)
    remove_uax_playback_handler()
    bpy.types.NODE_MT_add.remove(menu_ugx_material_nodes)
    bpy.types.VIEW3D_MT_object_context_menu.remove(menu_damage_authoring)
    try:
        bpy.types.TOPBAR_MT_file_export.remove(menu_export)
    except Exception:
        pass
    try:
        bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    except Exception:
        pass
    for name in ('hwek_visual', 'hwek_damage', 'hwek_ugx_skeleton_lod_mode'):
        if hasattr(bpy.types.Object, name):
            delattr(bpy.types.Object, name)
    for name in ('hwek_uax_override_loop_count', 'hwek_uax_loop_count', 'hwek_uax_motion_extraction',
                 'hwek_uax_loop_translation', 'hwek_uax_clamp_loop_boundaries',
                 'hwek_uax_placement_position',
                 'hwek_uax_placement_rotation', 'hwek_uax_placement_scale_shear'):
        if hasattr(bpy.types.Action, name):
            delattr(bpy.types.Action, name)
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
    if hasattr(bpy.types.Object, UGX_PLAYER_COLOR_PROPERTY):
        delattr(bpy.types.Object, UGX_PLAYER_COLOR_PROPERTY)
    if not keep_node_types:
        for node_type in (ShaderNodeAndersHW1UV, ShaderNodeAndersShaderPasses, ShaderNodeAndersParameters, ShaderNodeAndersUVAnimation, ShaderNodeAndersHW1PlayerColorMask, ShaderNodeAndersHWDE, ShaderNodeAndersHW2):
            if node_type.is_registered:
                bpy.utils.unregister_class(node_type)
    unregister_legacy_panel()
