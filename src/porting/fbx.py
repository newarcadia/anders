"""Universal FBX input and authored UGX export."""

import os
from pathlib import Path
import re
import tempfile
import uuid
import math
import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ImportHelper, ExportHelper
from mathutils import Vector
from ..errors import UgxError
from ..formats.files import write_ugx_export_files
from ..materials.conversion import convert_mesh_materials
from ..materials.parameters import retarget_ugx_parameter_drivers
from ..textures.export import ugx_texture_export_root
from ..ugx.scene import ugx_armature_for, ugx_bound_mesh_objects, ugx_ordered_armature_bones
from ..ugx.scene_export import export_new_v4_ugx, export_new_v6_ugx
from ..visuals.export import prepare_visual_export

__all__ = ('import_fbx', 'HWEK_OT_import_fbx', 'HWEK_OT_export_fbx_ugx', 'HWEK_PT_fbx')


def import_fbx(path, *, animation=True, image_search=True, scale=1.0):
    # The native importer also accepts ASCII FBX. Keep binary support on 4.2.
    if bpy.app.version >= (4, 5, 0):
        result = bpy.ops.wm.fbx_import(filepath=str(path), use_custom_props=True,
                                       use_anim=animation, global_scale=scale)
    else:
        with open(path, 'rb') as source:
            if not source.read(23).startswith(b'Kaydara FBX Binary'):
                raise UgxError('ASCII FBX input requires Blender 4.5 or newer. Binary FBX also works on Blender 4.2.')
        result = bpy.ops.import_scene.fbx(filepath=str(path), use_custom_props=True,
                                         use_anim=animation, use_image_search=image_search,
                                         global_scale=scale, automatic_bone_orientation=False)
    if 'FINISHED' not in result:
        raise UgxError('Blender could not import the FBX file.')


def selected_models(context, batch):
    groups, static = {}, []
    for obj in context.selected_objects:
        rig = ugx_armature_for(obj)
        if rig is not None:
            groups[rig] = ugx_bound_mesh_objects(rig)
        elif obj.type == 'MESH' and obj.data.polygons:
            static.append(obj)
    result = [(rig.name, rig, meshes) for rig, meshes in groups.items() if meshes]
    if batch:
        result.extend((obj.name, None, [obj]) for obj in static)
    elif static:
        result.append((static[0].name, None, static))
    if not result:
        raise UgxError('Select meshes or an armature to export.')
    return result


def export_models(context, path, version, resolution=2048, batch=False, visual=True, limit_weights=True):
    if context.object is not None and context.object.mode != 'OBJECT':
        raise UgxError('Switch to Object Mode before converting FBX models.')
    models = selected_models(context, batch)
    if len(models) > 1 and not batch:
        raise UgxError('Selection contains separate rigs or rigged and static meshes. Enable Batch Export.')
    destination = Path(path).resolve()
    previous_selection, previous_active = list(context.selected_objects), context.view_layer.objects.active
    stores = (bpy.data.objects, bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.images)
    before = [set(store) for store in stores]
    collection = bpy.data.collections.new('Anders Model Export')
    context.scene.collection.children.link(collection)
    payloads, staged_outputs, names = [], [], set()
    try:
        with tempfile.TemporaryDirectory(prefix='anders-model-') as staging:
            for number, (name, rig, meshes) in enumerate(models):
                model_before = [set(store) for store in stores]
                bpy.ops.object.select_all(action='DESELECT')
                cloned_rig = None
                if rig is not None:
                    cloned_rig = rig.copy()
                    cloned_rig.data = rig.data.copy()
                    cloned_rig.animation_data_clear()
                    cloned_rig.parent = None
                    cloned_rig.constraints.clear()
                    cloned_rig.matrix_world = rig.matrix_world.copy()
                    collection.objects.link(cloned_rig)
                    cloned_rig.hide_set(False)
                    cloned_rig.hide_viewport = False
                    cloned_rig.hide_select = False
                    bones = ugx_ordered_armature_bones(cloned_rig)
                    if version == 4 and bones and (bones[0].parent is not None or
                            (cloned_rig.matrix_world @ bones[0].head_local).length > 0.0001):
                        # HWDE multipart models reserve an unweighted origin root.
                        # Keep the source pelvis and its bind transforms intact.
                        source_bone_names = [bone.name for bone in bones]
                        cloned_rig.select_set(True)
                        context.view_layer.objects.active = cloned_rig
                        bpy.ops.object.mode_set(mode='EDIT')
                        try:
                            root = cloned_rig.data.edit_bones.new('GrannyRootBone')
                            root.head = cloned_rig.matrix_world.inverted().translation
                            root.tail = root.head + Vector((0, 0, 1))
                            root_name = root.name
                            for bone_name in source_bone_names:
                                bone = cloned_rig.data.edit_bones[bone_name]
                                if bone.parent is None:
                                    matrix = bone.matrix.copy()
                                    bone.parent = root
                                    bone.use_connect = False
                                    bone.matrix = matrix
                        finally:
                            bpy.ops.object.mode_set(mode='OBJECT')
                        for index, bone_name in enumerate([root_name, *source_bone_names]):
                            cloned_rig.data.bones[bone_name]['hwek_ugx_bone_index'] = index
                        cloned_rig.select_set(False)
                clones = []
                native_materials = {}
                for source in meshes:
                    clone = source.copy()
                    clone.data = source.data.copy()
                    for index, material in enumerate(list(clone.data.materials)):
                        if material is None or not material.get('hwek_ugx_native_material'):
                            continue
                        if material not in native_materials:
                            copied = material.copy()
                            retarget_ugx_parameter_drivers(copied)
                            for node in copied.node_tree.nodes:
                                if node.type == 'TEX_IMAGE' and node.image:
                                    node.image = node.image.copy()
                            native_materials[material] = copied
                        clone.data.materials[index] = native_materials[material]
                    clone.animation_data_clear()
                    world = source.matrix_world.copy()
                    clone.constraints.clear()
                    clone.parent = cloned_rig
                    clone.matrix_world = world
                    collection.objects.link(clone)
                    clone.hide_set(False)
                    clone.hide_viewport = False
                    clone.hide_select = False
                    clone.select_set(True)
                    context.view_layer.objects.active = clone
                    armature_modifiers = [(modifier, modifier.show_viewport, modifier.show_render)
                                          for modifier in clone.modifiers if modifier.type == 'ARMATURE']
                    for modifier, _, _ in armature_modifiers:
                        modifier.show_viewport = False
                        modifier.show_render = False
                    for modifier in list(clone.modifiers):
                        if modifier.type == 'ARMATURE':
                            modifier.object = cloned_rig
                        elif modifier.show_viewport:
                            result = bpy.ops.object.modifier_apply(modifier=modifier.name)
                            if 'FINISHED' not in result:
                                raise UgxError(f'Mesh {source.name}: could not apply modifier {modifier.name}.')
                    for modifier, viewport, render in armature_modifiers:
                        modifier.show_viewport, modifier.show_render = viewport, render
                    if limit_weights and cloned_rig is not None:
                        bone_names = set(cloned_rig.data.bones.keys())
                        for vertex in clone.data.vertices:
                            if any(not math.isfinite(item.weight) or item.weight < 0 for item in vertex.groups):
                                raise UgxError(f'Mesh {source.name}: vertex {vertex.index} has invalid bone weights.')
                            weights = sorted(((clone.vertex_groups[item.group], item.weight) for item in vertex.groups
                                              if clone.vertex_groups[item.group].name in bone_names and item.weight > 0),
                                             key=lambda item: item[1], reverse=True)
                            for group, _ in weights[4:]:
                                group.remove([vertex.index])
                            total = sum(weight for _, weight in weights[:4])
                            if total:
                                for group, weight in weights[:4]:
                                    group.add([vertex.index], weight / total, 'REPLACE')
                    clones.append(clone)
                context.view_layer.update()
                namespace = 'anders/' + str(uuid.uuid4()).replace('-', '')
                convert_mesh_materials(context, clones, version, resolution, namespace)
                if cloned_rig is None and len(clones) > 1:
                    bpy.ops.object.select_all(action='DESELECT')
                    for clone in clones:
                        clone.select_set(True)
                    context.view_layer.objects.active = clones[0]
                    if 'FINISHED' not in bpy.ops.object.join():
                        raise UgxError('Could not combine the selected static meshes.')
                    clones = [context.object]
                target = cloned_rig or clones[0]
                stem = destination.stem if len(models) == 1 else re.sub(r'[^A-Za-z0-9_-]', '_', name).strip('_') or f'model_{number + 1}'
                candidate = stem
                suffix = 2
                while candidate.casefold() in names:
                    candidate = f'{stem}_{suffix}'
                    suffix += 1
                names.add(candidate.casefold())
                folder = Path(staging) / candidate if batch else Path(staging)
                folder.mkdir(parents=True, exist_ok=True)
                model_path = folder / (candidate + '.ugx')
                final_folder = destination.parent / candidate if batch else destination.parent
                final_model = final_folder / model_path.name
                if version == 4:
                    textures = export_new_v4_ugx(target, str(model_path), export_loaded_textures=True)
                else:
                    textures = export_new_v6_ugx(target, str(model_path), export_loaded_textures=True)
                staged_outputs.append((model_path, final_model))
                staged_texture_root = Path(ugx_texture_export_root(str(model_path), version))
                final_texture_root = Path(ugx_texture_export_root(str(final_model), version))
                for texture in textures:
                    staged_texture = Path(texture)
                    staged_outputs.append((staged_texture, final_texture_root / staged_texture.relative_to(staged_texture_root)))
                if visual:
                    payloads.extend(prepare_visual_export(target, str(final_model.with_suffix('.vis')),
                                                         model_path=str(final_model), binary=version == 6))
                for store, previous in zip(stores, model_before):
                    for item in set(store) - previous:
                        store.remove(item, do_unlink=True)
                # Finish every conversion before replacing any destination.
            for staged_file, final_file in staged_outputs:
                payloads.append((str(final_file), staged_file.read_bytes()))
            write_ugx_export_files(payloads)
        return [file for file, _ in payloads if file.lower().endswith('.ugx')]
    finally:
        for store, previous in zip(stores, before):
            for item in set(store) - previous:
                store.remove(item, do_unlink=True)
        bpy.data.collections.remove(collection)
        bpy.ops.object.select_all(action='DESELECT')
        for obj in previous_selection:
            obj.select_set(True)
        context.view_layer.objects.active = previous_active


class HWEK_OT_import_fbx(bpy.types.Operator, ImportHelper):
    bl_idname = 'hwek.import_fbx'
    bl_label = 'Import FBX for Halo Wars'
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = '.fbx'
    filter_glob: StringProperty(default='*.fbx', options={'HIDDEN'})
    texture_directory: StringProperty(name='Texture Folder', subtype='DIR_PATH')
    import_animation: BoolProperty(name='Import Animation', default=True)
    scale: FloatProperty(name='Scale', default=1.0, min=0.0001, max=10000)

    def execute(self, context):
        stores = (bpy.data.objects, bpy.data.collections, bpy.data.meshes, bpy.data.armatures,
                  bpy.data.materials, bpy.data.images, bpy.data.actions)
        previous = [set(store) for store in stores]
        before = previous[0]
        selected, active = list(context.selected_objects), context.view_layer.objects.active
        collection = None
        try:
            import_fbx(self.filepath, animation=self.import_animation, scale=self.scale)
            objects = set(bpy.data.objects) - before
            if not any(obj.type == 'MESH' and obj.data.polygons for obj in objects):
                raise UgxError('The FBX contains no mesh geometry.')
            collection = bpy.data.collections.new(Path(self.filepath).stem)
            context.scene.collection.children.link(collection)
            for obj in objects:
                for owner in list(obj.users_collection):
                    owner.objects.unlink(obj)
                collection.objects.link(obj)
                obj['anders_fbx_source'] = os.path.abspath(self.filepath)
            if self.texture_directory:
                files = {}
                for file in Path(bpy.path.abspath(self.texture_directory)).rglob('*'):
                    if file.is_file():
                        files.setdefault(file.name.casefold(), []).append(file)
                materials = {slot.material for obj in objects if obj.type == 'MESH' for slot in obj.material_slots if slot.material}
                for mat in materials:
                    for node in mat.node_tree.nodes if mat.use_nodes else []:
                        if node.type != 'TEX_IMAGE' or node.image is None:
                            continue
                        image = node.image
                        if min(image.size) > 0 and image.has_data:
                            continue
                        filename = Path(image.filepath.replace('\\', '/')).name.casefold()
                        matches = files.get(filename, [])
                        if len(matches) > 1:
                            raise UgxError(f'Texture {filename} is ambiguous in the selected folder. Keep matching FBX texture paths.')
                        if len(matches) == 1:
                            if image in previous[5]:
                                image = image.copy()
                                node.image = image
                            image.filepath = str(matches[0])
                            image.reload()
            self.report({'INFO'}, f'Imported {sum(obj.type == "MESH" for obj in objects)} mesh objects. Select models, then Convert to UGX.')
            return {'FINISHED'}
        except Exception as exc:
            for store, original in zip(stores, previous):
                for item in set(store) - original:
                    store.remove(item, do_unlink=True)
            for obj in selected:
                obj.select_set(True)
            context.view_layer.objects.active = active
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class HWEK_OT_export_fbx_ugx(bpy.types.Operator, ExportHelper):
    bl_idname = 'hwek.export_fbx_ugx'
    bl_label = 'Convert to UGX'
    filename_ext = '.ugx'
    filter_glob: StringProperty(default='*.ugx', options={'HIDDEN'})
    version: EnumProperty(name='Target', items=(('4', 'Halo Wars: Definitive Edition (v4)', ''), ('6', 'Halo Wars 2 (v6)', '')), default='4')
    resolution: IntProperty(name='Material Bake Size', default=2048, min=64, max=8192)
    batch: BoolProperty(name='Batch Export', description='Export each armature or static mesh into its own folder', default=False)
    visual: BoolProperty(name='Export VIS', default=True)
    limit_weights: BoolProperty(name='Limit Weights to Four', description='Keep the four strongest bone weights and normalize them on export copies', default=True)
    send_to_project: BoolProperty(name='Add to Forge Project', description='Create placeable HW1 scenery in the project selected in preferences', default=False)

    def execute(self, context):
        try:
            if self.send_to_project and self.version != '4':
                raise UgxError('Forge project object creation currently supports HW1 v4 models.')
            if self.send_to_project:
                from ..toolkit import tool_command, toolkit_preferences
                preferences = toolkit_preferences()
                if not preferences.project_file or not Path(bpy.path.abspath(preferences.project_file)).is_file():
                    raise UgxError('Select a Halo Wars project in Anders preferences.')
                tool_command('forge.exe', [])
            outputs = export_models(context, self.filepath, int(self.version), self.resolution, self.batch, self.visual, self.limit_weights)
            if self.send_to_project:
                from ..toolkit import add_models_to_project
                add_models_to_project(outputs)
            self.report({'INFO'}, f'Exported {len(outputs)} UGX models with textures.')
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class HWEK_PT_fbx(bpy.types.Panel):
    bl_label = 'FBX Conversion'
    bl_idname = 'HWEK_PT_fbx'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anders'

    def draw(self, context):
        self.layout.operator('hwek.import_fbx', icon='IMPORT')
        row = self.layout.row(align=True)
        row.operator('hwek.export_fbx_ugx', text='UGX v4').version = '4'
        row.operator('hwek.export_fbx_ugx', text='UGX v6').version = '6'
        self.layout.label(text='Select meshes or rigs before export.')
        from ..toolkit import terrain_enabled
        if not terrain_enabled():
            self.layout.operator('hwek.enable_terrain_tools', text='Enable Terrain Authoring', icon='MESH_GRID')
