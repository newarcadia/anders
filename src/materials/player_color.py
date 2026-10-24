"""Synchronize player-color previews across bound model materials."""

from bpy.app.handlers import persistent
from typing import Any, Iterable, List, Optional
import bpy
from ..ugx.scene import ugx_armature_for, ugx_authoring_mesh_objects
from .parameters import retarget_ugx_parameter_drivers

__all__ = ('UGX_PLAYER_COLOR_PROPERTY', 'UGX_PLAYER_COLOR_SYNC_ACTIVE', 'ugx_player_color_owner', 'ugx_player_color_nodes', 'ugx_player_color_materials', 'sync_ugx_player_color', 'ugx_player_color_owners_for_material', 'sync_ugx_player_color_from_material', 'sync_ugx_player_colors_from_depsgraph', 'initialize_ugx_player_color', 'initialize_all_ugx_player_colors', 'update_ugx_player_color')

UGX_PLAYER_COLOR_PROPERTY = "hwek_ugx_player_color"


UGX_PLAYER_COLOR_SYNC_ACTIVE = False


def ugx_player_color_owner(obj: Optional[bpy.types.Object]) -> Optional[bpy.types.Object]:
    if obj is None:
        return None
    return ugx_armature_for(obj) or (obj if obj.type == "MESH" else None)


def ugx_player_color_nodes(mat: Optional[bpy.types.Material]) -> List[Any]:
    if mat is None or not getattr(mat, "use_nodes", False) or not getattr(mat, "node_tree", None):
        return []
    return [
        node
        for node in mat.node_tree.nodes
        if node.type in ("GROUP", "CUSTOM GROUP")
        and (node.inputs.get("Player_color") is not None or node.inputs.get("Player Color") is not None)
    ]


def ugx_player_color_materials(owner: Optional[bpy.types.Object]) -> List[bpy.types.Material]:
    if owner is None:
        return []
    objects = ugx_authoring_mesh_objects(owner)
    if not objects and owner.type == "MESH":
        objects = [owner]
    materials: List[bpy.types.Material] = []
    seen = set()
    for obj in objects:
        for mat in getattr(getattr(obj, "data", None), "materials", []):
            if mat is None or not ugx_player_color_nodes(mat):
                continue
            identity = int(mat.as_pointer())
            if identity in seen:
                continue
            seen.add(identity)
            materials.append(mat)
    return materials


def sync_ugx_player_color(owner: Optional[bpy.types.Object], color: Optional[Iterable[float]] = None) -> int:
    global UGX_PLAYER_COLOR_SYNC_ACTIVE
    owner = ugx_player_color_owner(owner)
    if UGX_PLAYER_COLOR_SYNC_ACTIVE or owner is None or not hasattr(owner, UGX_PLAYER_COLOR_PROPERTY):
        return 0
    values = tuple(float(value) for value in (color if color is not None else getattr(owner, UGX_PLAYER_COLOR_PROPERTY)))
    if len(values) < 3:
        return 0
    rgba = values[:4] if len(values) >= 4 else values[:3] + (1.0,)
    changed_nodes = 0
    UGX_PLAYER_COLOR_SYNC_ACTIVE = True
    try:
        for mat in ugx_player_color_materials(owner):
            material_changed = False
            for node in ugx_player_color_nodes(mat):
                socket = node.inputs.get("Player_color")
                if socket is None:
                    socket = node.inputs.get("Player Color")
                if socket is None:
                    continue
                if any(abs(float(actual) - expected) > 0.000001 for actual, expected in zip(socket.default_value, rgba)):
                    socket.default_value = rgba
                    material_changed = True
                changed_nodes += 1
            if material_changed:
                mat.node_tree.update_tag()
                mat.update_tag()
    finally:
        UGX_PLAYER_COLOR_SYNC_ACTIVE = False
    return changed_nodes


def ugx_player_color_owners_for_material(mat: Optional[bpy.types.Material]) -> List[bpy.types.Object]:
    if mat is None:
        return []
    owners = {}
    for obj in getattr(bpy.data, "objects", []):
        materials = getattr(getattr(obj, "data", None), "materials", [])
        if obj.type != "MESH" or not any(candidate == mat for candidate in materials):
            continue
        owner = ugx_player_color_owner(obj)
        if owner is not None:
            owners[int(owner.as_pointer())] = owner
    return list(owners.values())


def sync_ugx_player_color_from_material(mat: Optional[bpy.types.Material]) -> int:
    if UGX_PLAYER_COLOR_SYNC_ACTIVE:
        return 0
    nodes = ugx_player_color_nodes(mat)
    if not nodes:
        return 0
    socket = nodes[0].inputs.get("Player_color")
    if socket is None:
        socket = nodes[0].inputs.get("Player Color")
    if socket is None:
        return 0
    color = tuple(float(value) for value in socket.default_value)
    changed_owners = 0
    for owner in ugx_player_color_owners_for_material(mat):
        if not hasattr(owner, UGX_PLAYER_COLOR_PROPERTY):
            continue
        current = tuple(float(value) for value in getattr(owner, UGX_PLAYER_COLOR_PROPERTY))
        if all(abs(actual - expected) <= 0.000001 for actual, expected in zip(current, color)):
            continue
        setattr(owner, UGX_PLAYER_COLOR_PROPERTY, color)
        changed_owners += 1
    return changed_owners


@persistent
def sync_ugx_player_colors_from_depsgraph(_scene, depsgraph):
    if UGX_PLAYER_COLOR_SYNC_ACTIVE:
        return
    materials = {}
    for update in getattr(depsgraph, "updates", []):
        data = getattr(update, "id", None)
        data = getattr(data, "original", data)
        if isinstance(data, bpy.types.Material) and ugx_player_color_nodes(data):
            materials[int(data.as_pointer())] = data
    for mat in materials.values():
        retarget_ugx_parameter_drivers(mat)
        sync_ugx_player_color_from_material(mat)


def initialize_ugx_player_color(owner: Optional[bpy.types.Object]) -> bool:
    owner = ugx_player_color_owner(owner)
    if owner is None or not hasattr(owner, UGX_PLAYER_COLOR_PROPERTY):
        return False
    materials = ugx_player_color_materials(owner)
    if not materials:
        return False
    if not owner.is_property_set(UGX_PLAYER_COLOR_PROPERTY):
        for mat in materials:
            nodes = ugx_player_color_nodes(mat)
            if not nodes:
                continue
            socket = nodes[0].inputs.get("Player_color")
            if socket is None:
                socket = nodes[0].inputs.get("Player Color")
            if socket is not None:
                setattr(owner, UGX_PLAYER_COLOR_PROPERTY, tuple(socket.default_value))
                break
    sync_ugx_player_color(owner)
    return True


def initialize_all_ugx_player_colors():
    owners = {}
    for obj in getattr(bpy.data, "objects", []):
        owner = ugx_player_color_owner(obj)
        if owner is not None:
            owners[int(owner.as_pointer())] = owner
    for owner in owners.values():
        initialize_ugx_player_color(owner)


def update_ugx_player_color(owner: bpy.types.Object, _context):
    sync_ugx_player_color(owner)
