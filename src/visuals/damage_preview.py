"""Derived debris geometry for inspecting detached parts without editing the model."""
import bpy
from mathutils import Vector

__all__ = ('update_damage_debris',)


def update_damage_debris(owner, meshes, detached, enabled, distance):
    collection = next((item for item in bpy.data.collections if item.get('hwek_damage_preview_owner') == owner), None)
    if collection is not None:
        for obj in list(collection.objects):
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
    parts = [(name, obj) for name, obj in meshes if name in detached and not obj.get('hwek_damage_original_hidden', False)] if enabled else []
    if not parts:
        if collection is not None:
            bpy.data.collections.remove(collection)
        return
    if collection is None:
        collection = bpy.data.collections.new(owner.name + ' Detached Preview')
        collection['hwek_damage_preview_owner'] = owner
        owner.users_collection[0].children.link(collection)
        collection.hide_render = True
    bounds = [obj.matrix_world @ Vector(corner) for _, obj in meshes for corner in obj.bound_box]
    center = Vector(tuple((min(point[axis] for point in bounds) + max(point[axis] for point in bounds)) * 0.5 for axis in range(3)))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for _, source in parts:
        evaluated = source.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
        obj = bpy.data.objects.new(source.name + ' Detached Preview', mesh)
        obj['hwek_ugx_export_enabled'] = False
        obj['hwek_damage_preview_source'] = source
        obj.hide_select = True
        obj.hide_render = True
        collection.objects.link(obj)
        obj.matrix_world = source.matrix_world
        part_center = source.matrix_world @ (sum((Vector(corner) for corner in source.bound_box), Vector()) / 8)
        direction = part_center - center
        if direction.length < 1e-5:
            direction = Vector((0, 0, 1))
        obj.matrix_world.translation += direction.normalized() * distance
