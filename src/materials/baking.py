"""Bake material inputs into native texture slots in a temporary scene."""

import bmesh
import bpy
import json
import re
from ..errors import UgxError
from ..textures.context import texture_usage_is_non_color, ugx_texture_resource_key
from ..textures.images import set_non_color_image
from .context import context_blender_material, is_ugx_material, material_output_node, ugx_material_cache_version
from .data import V4_MATERIAL_MAP_USAGES, hw_shader_socket_name
from .parameters import hw2_parameter_property, retarget_ugx_parameter_drivers
from .sockets import active_ugx_material_shader, ugx_uv_map_name
from .uv import ugx_texture_vector

__all__ = ('ugx_bake_input_items', 'ugx_bake_uv_vector', 'bake_ugx_texture_input')

def ugx_bake_input_items(self, context):
    mat = bpy.data.materials.get(self.material_name) if self.material_name else context_blender_material(context)
    if mat is None or not is_ugx_material(mat):
        return []
    version = ugx_material_cache_version(mat)
    usages = ("diffuse", "normal", "packedmask", "specular", "emissive") if version == 6 else V4_MATERIAL_MAP_USAGES
    slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
    if version == 6:
        usages = [usage for usage in usages if any(slot.get("usage") == usage for slot in slots)]
    return [(usage, hw_shader_socket_name(usage, version), "Bake this material texture input") for usage in usages]


def ugx_bake_uv_vector(mat, socket, uv_layer):
    version = ugx_material_cache_version(mat)
    animation_type = "ShaderNodeAndersUVAnimation" if version == 6 else "ShaderNodeAndersHW1UV"
    pending = [link.from_node for link in socket.links]
    upstream = []
    visited = set()
    while pending:
        node = pending.pop()
        if node.as_pointer() in visited:
            continue
        visited.add(node.as_pointer())
        upstream.append(node)
        pending.extend(link.from_node for item in node.inputs for link in item.links)
    # Native HW2 scrolling applies to the material's whole texture array.
    # Keep that mapping live after baking a single color expression.
    candidates = upstream + [node for node in mat.node_tree.nodes if node not in upstream]
    for node in candidates:
        if node.bl_idname != animation_type:
            continue
        try:
            if ugx_uv_map_name(node, mat) == uv_layer:
                return node.outputs["UV"]
        except UgxError:
            continue
    return next((node.outputs["UV"] for node in mat.node_tree.nodes
                 if node.type == "UVMAP" and node.uv_map == uv_layer), None)


def bake_ugx_texture_input(context, obj, mat, usage, resolution, uv_layer, all_users=True):
    if obj is None or obj.type != "MESH" or obj.mode != "OBJECT":
        raise UgxError("Select a mesh in Object Mode before baking a material input.")
    indices = {index for index, material in enumerate(obj.data.materials) if material == mat}
    if not indices or uv_layer not in obj.data.uv_layers:
        raise UgxError(f"Material {mat.name}: the active mesh needs this material and UV map {uv_layer}.")
    version = ugx_material_cache_version(mat)
    shader = active_ugx_material_shader(mat, version)
    socket_name = hw_shader_socket_name(usage, version)
    socket = shader.inputs.get(socket_name)
    if socket is None or not socket.is_linked:
        raise UgxError(f"Material {mat.name}: connect {socket_name or usage} before baking.")
    uv_vector = ugx_bake_uv_vector(mat, socket, uv_layer)
    image = scene = material = None
    meshes, bake_objects = [], []
    try:
        material = mat.copy()
        retarget_ugx_parameter_drivers(material)
        tree = material.node_tree
        for node in tree.nodes:
            if node.bl_idname in ("ShaderNodeAndersUVAnimation", "ShaderNodeAndersHW1UV"):
                node.inputs["Time"].driver_remove("default_value")
                node.inputs["Time"].default_value = 0.0
        bake_socket = tree.nodes[shader.name].inputs[socket_name]
        emission = tree.nodes.new("ShaderNodeEmission")
        tree.links.new(bake_socket.links[0].from_socket, emission.inputs["Color"])
        output = material_output_node(material)
        output.target = "ALL"
        tree.links.new(emission.outputs[0], output.inputs["Surface"])
        image = bpy.data.images.new(f"{mat.name}_{usage}_baked", resolution, resolution, alpha=True)
        slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
        original = next((str(slot.get("path", "")) for slot in slots if slot.get("usage") == usage), "")
        original_dir = original.replace("/", "\\").rpartition("\\")[0]
        # A baked input must not overwrite a texture still used by another material.
        resource_name = re.sub(r"[^\w-]", "_", image.name)
        resource_prefix = original_dir + "\\" if original_dir else ""
        used_paths = {ugx_texture_resource_key(str(item.get("hwek_ugx_texture_path", "")))
                      for item in bpy.data.images if item != image}
        resource_path = resource_prefix + resource_name
        suffix = 2
        while ugx_texture_resource_key(resource_path) in used_paths:
            resource_path = resource_prefix + resource_name + "_" + str(suffix)
            suffix += 1
        image["hwek_ugx_texture_path"] = resource_path
        if texture_usage_is_non_color(usage, version):
            set_non_color_image(image)
        target = tree.nodes.new("ShaderNodeTexImage")
        target.image = image
        tree.nodes.active = target
        scene = bpy.data.scenes.new("UGX Texture Bake")
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 1
        scene.render.fps = context.scene.render.fps
        scene.render.fps_base = context.scene.render.fps_base
        scene.frame_set(context.scene.frame_current)
        layer = scene.view_layers[0]
        sources = list(context.scene.objects) if all_users else [obj]
        depsgraph = context.evaluated_depsgraph_get()
        for source in sources:
            if source.type != "MESH":
                continue
            indices = {index for index, item in enumerate(source.data.materials) if item == mat}
            if not indices:
                continue
            mesh = bpy.data.meshes.new_from_object(source.evaluated_get(depsgraph))
            meshes.append(mesh)
            faces = bmesh.new()
            try:
                faces.from_mesh(mesh)
                bmesh.ops.delete(faces, geom=[face for face in faces.faces if face.material_index not in indices], context="FACES")
                for face in faces.faces:
                    face.material_index = 0
                faces.to_mesh(mesh)
            finally:
                faces.free()
            if not mesh.polygons:
                continue
            if uv_layer not in mesh.uv_layers:
                raise UgxError(f"Material {mat.name}: mesh {source.name} needs UV map {uv_layer} for baking.")
            mesh.uv_layers.active = mesh.uv_layers[uv_layer]
            mesh.uv_layers[uv_layer].active_render = True
            mesh.materials.clear()
            mesh.materials.append(material)
            bake_obj = bpy.data.objects.new("UGX Texture Bake", mesh)
            bake_objects.append(bake_obj)
            bake_obj.matrix_world = source.matrix_world
            scene.collection.objects.link(bake_obj)
            bake_obj.select_set(True, view_layer=layer)
        if not bake_objects:
            raise UgxError(f"Material {mat.name}: no mesh faces use this material.")
        bake_obj = bake_objects[0]
        layer.objects.active = bake_obj
        layer.update()
        # A separate scene prevents other selected meshes and render settings
        # from becoming part of the bake.
        with context.temp_override(scene=scene, view_layer=layer, object=bake_obj, active_object=bake_obj,
                                   selected_objects=bake_objects, selected_editable_objects=bake_objects):
            result = bpy.ops.object.bake(type="EMIT", margin=8, use_clear=True, use_selected_to_active=False)
        if result != {"FINISHED"}:
            raise UgxError(f"Material {mat.name}: baking {socket_name} did not finish.")
        image.pack()
    except Exception:
        if image is not None:
            bpy.data.images.remove(image)
        raise
    finally:
        for bake_obj in bake_objects:
            bpy.data.objects.remove(bake_obj, do_unlink=True)
        if scene is not None:
            bpy.data.scenes.remove(scene)
        for mesh in meshes:
            bpy.data.meshes.remove(mesh)
        if material is not None:
            bpy.data.materials.remove(material)
    texture = mat.node_tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.label = socket_name + " Baked"
    texture.location = (shader.location.x - 320, shader.location.y)
    mat.node_tree.links.new(uv_vector or ugx_texture_vector(mat, uv_layer), texture.inputs["Vector"])
    mat.node_tree.links.new(texture.outputs["Color"], socket)
    # Texture scale is already in the image; scrolling remains in the UV node.
    standard = str(json.loads(mat.get("hwek_ugx_ufx_attrs", "{}")).get("shaderPermutationName0", {}).get("value", "")).upper().startswith("HOGAN_STANDARD_")
    if version == 6 and standard:
        offset = {"diffuse": 0, "normal": 1, "packedmask": 2, "specular": 3, "emissive": 5}[usage]
        prop = hw2_parameter_property(mat, "PSCBData", offset)
        if prop is not None:
            mat[prop] = 1.0
    mat.update_tag()
    return image
