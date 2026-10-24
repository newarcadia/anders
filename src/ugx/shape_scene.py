"""Bake native shape evaluation into temporary export meshes."""
from array import array
from contextlib import ExitStack, contextmanager

import bpy

from .scene import ugx_authoring_mesh_objects, ugx_export_object_for, ugx_lod_sibling_objects_for

__all__ = ('prepared_shape_key_edits',)


def restore_shape_edit_mode(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')


@contextmanager
def prepared_shape_key_edits(obj):
    objects = ugx_authoring_mesh_objects(obj)
    source = ugx_export_object_for(obj)
    if source is not None:
        objects.extend(ugx_lod_sibling_objects_for(source))
    objects = [item for item in dict.fromkeys(objects) if item.data.shape_keys is not None
               and (not item.data.shape_keys.use_relative or item.show_only_shape_key)]
    with ExitStack() as resources:
        if any(item.mode == 'EDIT' for item in objects):
            active = bpy.context.view_layer.objects.active
            bpy.ops.object.mode_set(mode='OBJECT')
            resources.callback(restore_shape_edit_mode, active)
        for item in objects:
            mesh = item.data.copy()
            resources.callback(bpy.data.meshes.remove, mesh)
            temporary = item.copy()
            try:
                temporary.data = mesh
                mixed = temporary.shape_key_add(name='UGX Export Shape', from_mix=True)
                positions = array('f', [0.0]) * (len(mesh.vertices) * 3)
                mixed.data.foreach_get('co', positions)
            finally:
                bpy.data.objects.remove(temporary)
            mesh.vertices.foreach_set('co', positions)
            mesh.shape_keys.key_blocks[0].data.foreach_set('co', positions)
            # Keep shape ownership visible to source-retention guards. The
            # temporary mesh already contains the mix and must not apply it twice.
            mesh.shape_keys.animation_data_clear()
            mesh.shape_keys.use_relative = True
            for key in mesh.shape_keys.key_blocks[1:]:
                key.mute = True
            mesh.update()
            resources.callback(setattr, item, 'data', item.data)
            resources.callback(setattr, item, 'show_only_shape_key', item.show_only_shape_key)
            item.show_only_shape_key = False
            item.data = mesh
        yield
