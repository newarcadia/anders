"""Convert ordinary Blender materials to authored Halo Wars materials."""

import json
import bpy
import numpy as np
from ..errors import UgxError
from .assignment import assign_ugx_material_data
from .baking import bake_ugx_texture_input
from .data import hw_shader_socket_name
from .parameters import hw2_parameter_property
from .properties import ufx_attr_property_name, v4_attr_property_name
from .sockets import active_ugx_material_shader

__all__ = ('convert_mesh_materials',)


def input_signal(tree, socket, default):
    if socket is not None and socket.is_linked:
        return socket.links[0].from_socket
    value = socket.default_value if socket is not None else default
    node = tree.nodes.new('ShaderNodeRGB' if hasattr(value, '__len__') else 'ShaderNodeValue')
    node.outputs[0].default_value = value
    return node.outputs[0]


def convert_mesh_materials(context, objects, version, resolution=2048, namespace='imported'):
    """Replace source materials only on the supplied export meshes."""
    converted = {}
    for obj in objects:
        if not obj.data.uv_layers:
            if any(mat and mat.use_nodes and any(node.type == 'BSDF_PRINCIPLED' and any(socket.is_linked for socket in node.inputs)
                                                for node in mat.node_tree.nodes) for mat in obj.data.materials):
                raise UgxError(f'Mesh {obj.name}: unwrap its textured geometry before conversion.')
            obj.data.uv_layers.new(name='UV0')
        # Keep source UV links valid while giving the writer canonical channels.
        if len(obj.data.uv_layers) > 3:
            raise UgxError(f'Mesh {obj.name}: select at most three UV maps before conversion.')
        renames = [(layer.name, f'UV{index}') for index, layer in enumerate(obj.data.uv_layers)]
        for index, layer in enumerate(obj.data.uv_layers):
            layer.name = f'__anders_uv_{index}'
        for index, layer in enumerate(obj.data.uv_layers):
            layer.name = f'UV{index}'
        obj.data.uv_layers.active_index = 0
        if not obj.data.materials:
            obj.data.materials.append(None)
        for index, original in enumerate(list(obj.data.materials)):
            if original is not None and original.get('hwek_ugx_native_material'):
                if int(original.get('hwek_ugx_cache_version', 0)) != version:
                    raise UgxError(f'Material {original.name}: choose its native UGX version or convert from the original FBX.')
                continue
            key = (original, obj)
            if key in converted:
                obj.data.materials[index] = converted[key]
                continue
            material = original.copy() if original else bpy.data.materials.new('FBX Material')
            material.use_nodes = True
            if original is None or not original.use_nodes:
                shader = next(node for node in material.node_tree.nodes if node.type == 'BSDF_PRINCIPLED')
                shader.inputs['Base Color'].default_value = material.diffuse_color
            for node in material.node_tree.nodes:
                if node.type in ('UVMAP', 'NORMAL_MAP'):
                    node.uv_map = dict(renames).get(node.uv_map, node.uv_map)
            obj.data.materials[index] = material
            convert_material(context, obj, material, version, resolution,
                             f'{namespace}/material_{len(converted):03d}')
            converted[key] = material
    return len(converted)


def convert_material(context, obj, material, version, resolution, resource):
    tree = material.node_tree
    output = next((node for node in tree.nodes if node.type == 'OUTPUT_MATERIAL' and node.is_active_output), None)
    source = output.inputs['Surface'].links[0].from_node if output and output.inputs['Surface'].is_linked else None
    if source is not None and source.type != 'BSDF_PRINCIPLED':
        raise UgxError(f'Material {material.name}: use a Principled BSDF surface before FBX conversion.')
    for node in tree.nodes:
        if node.type == 'TEX_IMAGE' and any(socket.is_linked for socket in node.outputs):
            if node.image is None or min(node.image.size) <= 0 or not node.image.has_data:
                raise UgxError(f'Material {material.name}: texture {node.image.name if node.image else node.name} is missing. Choose its texture folder.')
    inputs = source.inputs if source else {}
    for name in ('Transmission Weight', 'Coat Weight', 'Subsurface Weight'):
        socket = inputs.get(name)
        if socket is not None and (socket.is_linked or socket.default_value > 0):
            raise UgxError(f'Material {material.name}: {name} has no native Halo Wars equivalent. Bake or simplify this surface first.')
    color = input_signal(tree, inputs.get('Base Color'), tuple(material.diffuse_color))
    alpha = inputs.get('Alpha')
    transparent = alpha is not None and (alpha.is_linked or alpha.default_value < 1.0)
    alpha_signal = input_signal(tree, alpha, 1.0) if transparent else None
    rough = inputs.get('Roughness')
    roughness_linked = rough is not None and rough.is_linked
    roughness_value = float(rough.default_value) if rough is not None else 0.5
    roughness = input_signal(tree, rough, 0.5)
    specular = input_signal(tree, inputs.get('Specular IOR Level'), 0.5)
    metallic_value = inputs.get('Metallic')
    if metallic_value is not None and (metallic_value.is_linked or metallic_value.default_value > 0):
        metallic = tree.nodes.new('ShaderNodeMixRGB')
        metallic.blend_type = 'MIX'
        tree.links.new(input_signal(tree, metallic_value, 0.0), metallic.inputs[0])
        tree.links.new(specular, metallic.inputs[1])
        tree.links.new(color, metallic.inputs[2])
        specular = metallic.outputs[0]
    normal = inputs.get('Normal')
    normal_signal = None
    if normal is not None and normal.is_linked:
        normal_node = normal.links[0].from_node
        if normal_node.type != 'NORMAL_MAP' or normal_node.space != 'TANGENT':
            raise UgxError(f'Material {material.name}: bake its normal input to a tangent-space Normal Map first.')
        if normal_node.inputs['Strength'].is_linked or abs(normal_node.inputs['Strength'].default_value - 1.0) > 0.0001:
            raise UgxError(f'Material {material.name}: bake normal strength into its image before conversion.')
        normal_signal = input_signal(tree, normal_node.inputs['Color'], (0.5, 0.5, 1, 1))
    emission = inputs.get('Emission Color')
    emissive = emission is not None and (emission.is_linked or any(emission.default_value[:3]))
    emission_signal = None
    if emissive:
        multiply = tree.nodes.new('ShaderNodeVectorMath')
        multiply.operation = 'SCALE'
        tree.links.new(input_signal(tree, emission, (0, 0, 0, 1)), multiply.inputs[0])
        tree.links.new(input_signal(tree, inputs.get('Emission Strength'), 1.0), multiply.inputs['Scale'])
        emission_signal = multiply.outputs[0]
    assign_ugx_material_data(material, version, skinned=bool(obj.find_armature()))
    material['hwek_ugx_native_material'] = True
    shader = active_ugx_material_shader(material, version)
    signals = {'diffuse': color, 'gloss' if version == 4 else 'specular': specular}
    if normal_signal is not None:
        signals['normal'] = normal_signal
    if emission_signal is not None:
        signals['emissive'] = emission_signal
    if transparent and version == 4:
        signals['opacity'] = alpha_signal
        material[v4_attr_property_name('BlendType')] = 2
    if version == 4:
        if not roughness_linked:
            material[v4_attr_property_name('SpecPower')] = min(8192.0, max(1.0, 2.0 / max(roughness_value, 0.04) ** 4 - 2.0))
        else:
            smoothness = tree.nodes.new('ShaderNodeMath')
            smoothness.operation = 'SUBTRACT'
            smoothness.inputs[0].default_value = 1.0
            tree.links.new(roughness, smoothness.inputs[1])
            gloss = tree.nodes.new('ShaderNodeVectorMath')
            gloss.operation = 'SCALE'
            tree.links.new(specular, gloss.inputs[0])
            tree.links.new(smoothness.outputs[0], gloss.inputs['Scale'])
            signals['gloss'] = gloss.outputs[0]
    else:
        if roughness.node.type == 'VALUE':
            masks = tree.nodes.new('ShaderNodeRGB')
            masks.outputs[0].default_value = (roughness.default_value, 1, 0, 1)
        else:
            masks = tree.nodes.new('ShaderNodeCombineColor')
            masks.mode = 'RGB'
            tree.links.new(roughness, masks.inputs['Red'])
            masks.inputs['Green'].default_value = 1.0
            masks.inputs['Blue'].default_value = 0.0
        signals['packedmask'] = masks.outputs[0]
        if transparent:
            material[ufx_attr_property_name('blendMode')] = 2
    uv = tree.nodes.new('ShaderNodeUVMap')
    uv.uv_map = 'UV0'
    for usage, signal in signals.items():
        socket = shader.inputs.get(hw_shader_socket_name(usage, version))
        if socket is None:
            raise UgxError(f'Material {material.name}: the target shader has no {usage} input.')
        image = image_for_signal(context, obj, material, shader, socket, signal, usage, resolution, uv)
        image.filepath_raw = ''
        for key in list(image.keys()):
            if str(key).startswith('hwek_ugx_'):
                del image[key]
        image['anders_authored_texture'] = True
        image['hwek_ugx_texture_path'] = resource.replace('/', '\\') + '_' + usage
        socket.links[0].from_node['hwek_ugx_usage'] = usage
        if usage == 'normal':
            # Blender tangent maps use +Y; Halo Wars native normal maps use -Y.
            pixels = np.empty(len(image.pixels), dtype=np.float32)
            image.pixels.foreach_get(pixels)
            pixels[1::4] = 1.0 - pixels[1::4]
            image.pixels.foreach_set(pixels)
            image.update()
        image.pack()
        if version == 6:
            offset = {'diffuse': 0, 'normal': 1, 'packedmask': 2, 'specular': 3, 'emissive': 5}[usage]
            prop = hw2_parameter_property(material, 'PSCBData', offset)
            if prop is not None:
                material[prop] = 1.0
    if transparent and version == 6:
        # HW2 standard materials carry opacity in albedo alpha.
        socket = shader.inputs['Albedo']
        image = socket.links[0].from_node.image
        if min(image.size) < resolution:
            image.scale(max(image.size[0], resolution), max(image.size[1], resolution))
        original = socket.links[0].from_socket
        tree.links.new(alpha_signal, socket)
        mask = bake_ugx_texture_input(context, obj, material, 'diffuse', max(image.size), 'UV0', all_users=False)
        try:
            if tuple(mask.size) != tuple(image.size):
                mask.scale(*image.size)
            pixels = np.empty(len(image.pixels), dtype=np.float32)
            opacity = np.empty(len(mask.pixels), dtype=np.float32)
            image.pixels.foreach_get(pixels)
            mask.pixels.foreach_get(opacity)
            pixels[3::4] = opacity[0::4]
            image.pixels.foreach_set(pixels)
            image.update()
            image.pack()
            tree.links.new(original, socket)
        finally:
            bpy.data.images.remove(mask)
    if version == 4:
        slots = []
        for usage in signals:
            texture = shader.inputs[hw_shader_socket_name(usage, version)].links[0].from_node
            path = texture.image['hwek_ugx_texture_path']
            slots.append(dict(usage=usage, path=path, channel=0, flags=0))
        material['hwek_ugx_texture_slots'] = json.dumps(slots)
        for index, slot in enumerate(slots):
            material[f'hwek_ugx_texture_slot_{index}'] = slot['path']
            material[f'hwek_ugx_texture_slot_{index}_synced'] = slot['path']
            material['hwek_ugx_texture_' + slot['usage']] = slot['path']
    # The original graph remains available for editing and material rebaking.
    material['anders_fbx_material'] = True


def image_for_signal(context, obj, material, shader, socket, signal, usage, resolution, uv):
    tree = material.node_tree
    tree.links.new(signal, socket)
    node = signal.node
    vector = node.inputs.get('Vector') if node.type == 'TEX_IMAGE' else None
    plain_uv = vector is not None and (not vector.is_linked or vector.links[0].from_node.type == 'UVMAP')
    if node.type == 'TEX_IMAGE' and signal.name == 'Color' and plain_uv:
        image = node.image.copy()
        texture = tree.nodes.new('ShaderNodeTexImage')
        texture.image = image
        tree.links.new(vector.links[0].from_socket if vector.is_linked else uv.outputs['UV'], texture.inputs['Vector'])
        tree.links.new(texture.outputs['Color'], socket)
    elif node.type in ('RGB', 'VALUE'):
        value = signal.default_value
        color = tuple(value) if hasattr(value, '__len__') else (value, value, value, 1)
        image = bpy.data.images.new(f'{material.name}_{usage}', 4, 4, alpha=True)
        image.pixels.foreach_set(np.tile(color, 16).astype(np.float32))
        texture = tree.nodes.new('ShaderNodeTexImage')
        texture.image = image
        tree.links.new(uv.outputs['UV'], texture.inputs['Vector'])
        tree.links.new(texture.outputs['Color'], socket)
    else:
        image = bake_ugx_texture_input(context, obj, material, usage, resolution, 'UV0', all_users=False)
    if usage in ('normal', 'gloss', 'specular', 'packedmask', 'opacity'):
        image.colorspace_settings.name = 'Non-Color'
    return image
