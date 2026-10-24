"""Convert HW1 materials, textures and model bindings to HW2."""

import bpy
import hashlib
import json
import os
import re
import struct
import uuid
from ..errors import UgxError
from ..visuals.damage_format import model_damage_version, require_damage_version
from ..materials.baking import bake_ugx_texture_input
from ..materials.bindings import ugx_image_node_from_socket
from ..materials.context import ugx_material_cache_version
from ..materials.data import (
    HOGAN_TEXTURE_SLOT_ORDER,
    HW2_SKINNED_MATERIAL_DEFAULT,
    HW2_STATIC_MATERIAL_DEFAULT,
    build_v6_material_bdt_from_specs,
    hw_shader_socket_name,
)
from ..materials.nodes import create_hw_shader_node
from ..materials.parameters import ensure_ugx_parameter_nodes, retarget_ugx_parameter_drivers
from ..materials.player_color import initialize_ugx_player_color
from ..materials.properties import apply_ufx_material_metadata, v4_attr_property_name
from ..materials.sockets import active_ugx_material_shader, ugx_constant_socket_value
from ..materials.mapping import hw1_texture_mapping
from mathutils import Matrix
from ..materials.textures import link_material_texture
from ..materials.uv import arrange_ugx_material_nodes
from ..materials.uv_data import hw1_texture_coordinates
from ..textures.context import NON_COLOR_TEXTURE_USAGES
from ..textures.images import image_rgba_floats, set_non_color_image
from ..ugx.naming import current_ugx_mesh_name
from ..ugx.read import read_materials
from ..ugx.scene import ugx_armature_for
from .cleanup import clear_ugx_id_properties, ugx_portable_object_scope
from ..visuals.properties import populate_visual_settings
from ..visuals.damage_properties import populate_damage_settings

try:
    import numpy as np
except ImportError:
    np = None
























__all__ = ('HW2_PORT_SHADER_PASSES', 'hw1_port_input_image', 'hw2_port_image_pixels', 'hw2_port_generated_image', 'hw2_port_template_image', 'hw2_port_mask_image', 'hw2_port_material', 'retarget_hw2_port_object', 'port_hw1_asset_to_hw2')

HW2_PORT_SHADER_PASSES = {
    False: (
        ("HOGAN_STANDARD_00000001003009A0", 1408979380),
        ("HOGAN_STANDARD_00000001005009A0", 2251755014),
        ("HOGAN_STANDARD_00080001000009A0", 1778963691),
        ("HOGAN_STANDARD_00180001000009A0", 3578108153),
    ),
    True: (
        ("HOGAN_STANDARD_00000101003009A0", 3256523718),
        ("HOGAN_STANDARD_00000101005009A0", 3436429419),
        ("HOGAN_STANDARD_00080101000009A0", 997423763),
        ("HOGAN_STANDARD_00180101000009A0", 1639947673),
    ),
}


def hw1_port_input_image(context, obj, mat, usage, resolution, notes):
    shader = active_ugx_material_shader(mat, 4)
    socket = shader.inputs.get(hw_shader_socket_name(usage, 4))
    if socket is None or not socket.is_linked:
        return None
    pending, visited = [socket.links[0].from_node], set()
    while pending:
        upstream = pending.pop()
        if upstream.as_pointer() in visited:
            continue
        visited.add(upstream.as_pointer())
        if upstream.type == "TEX_IMAGE" and (upstream.image is None or upstream.image.get("hwek_ugx_missing_texture")):
            raise UgxError(f"Material {mat.name}: supply the missing {usage} image before porting.")
        if upstream.type == "REROUTE" and not upstream.inputs[0].is_linked:
            raise UgxError(f"Material {mat.name}: complete the {usage} texture reroute before porting.")
        pending.extend(link.from_node for item in upstream.inputs for link in item.links)
    try:
        node = ugx_image_node_from_socket(socket, mat, True)
    except UgxError:
        node = None
    if node is not None:
        image = node.image
        if image.get("hwek_ugx_missing_texture"):
            raise UgxError(f"Material {mat.name}: supply the missing {usage} image before porting.")
        try:
            uv, transform, _, _ = hw1_texture_mapping(node, mat)
        except UgxError:
            uv = ""
        coordinates = hw1_texture_coordinates(node)
        if coordinates is not None and any(abs(ugx_constant_socket_value(coordinates.inputs[name])) > 0.00001
                                           for name in ("Speed U", "Speed V", "Speed W")):
            notes.add(f"{mat.name}: texture scrolling was frozen at its initial position.")
        if uv == "UV0" and transform == Matrix.Identity(4):
            return image
    notes.add(f"{mat.name}: {usage} mapping was baked to UV0.")
    return bake_ugx_texture_input(context, obj, mat, usage, resolution, "UV0", True)


def hw2_port_image_pixels(image, width, height, fallback, linear=False):
    if image is None:
        return list(fallback) * (width * height)
    source_width, source_height = map(int, image.size)
    if source_width <= 0 or source_height <= 0:
        raise UgxError(f"Texture {image.name} has no pixels for the HW2 port.")
    if np is not None:
        pixels = np.empty(source_width * source_height * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        if linear and not image.is_float and image.colorspace_settings.name == 'sRGB':
            rgb = pixels.reshape(-1, 4)[:, :3]
            rgb[:] = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        rows = np.minimum(np.arange(height) * source_height // height, source_height - 1)
        columns = np.minimum(np.arange(width) * source_width // width, source_width - 1)
        return pixels.reshape(source_height, source_width, 4)[rows[:, None], columns].reshape(-1)
    pixels = image_rgba_floats(image)
    if linear and not image.is_float and image.colorspace_settings.name == 'sRGB':
        pixels = [value if index % 4 == 3 else (value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
                  for index, value in enumerate(pixels)]
    return [pixels[(y * source_height // height * source_width + x * source_width // width) * 4 + channel]
            for y in range(height) for x in range(width) for channel in range(4)]


def hw2_port_generated_image(name, width, height, pixels, non_color=True):
    image = bpy.data.images.new(name, width, height, alpha=True)
    if non_color:
        set_non_color_image(image)
    image.pixels.foreach_set(pixels)
    image.update()
    image.pack()
    return image


def hw2_port_template_image(usage, pixels):
    pixels = list(pixels)
    signature = hashlib.sha256(struct.pack('<64f', *pixels)).hexdigest()[:16]
    suffix = '[al]' if usage == 'diffuse' else '[pm]'
    path = f'anders/templates/{usage}_{signature}_{suffix}'
    for image in bpy.data.images:
        if image.get('hwek_ugx_port_template') != signature or tuple(image.size) != (4, 4):
            continue
        if image.get('hwek_ugx_port_template_usage') != usage:
            continue
        if image.colorspace_settings.name != ('Non-Color' if usage in NON_COLOR_TEXTURE_USAGES else 'sRGB'):
            continue
        if any(abs(float(actual) - expected) > 1 / 255 for actual, expected in zip(image.pixels, pixels)):
            continue
        return image
    # An edited template retains its texture path. Give a fresh default its own
    # path instead of creating conflicting exports in the same scene.
    if any(image.get('hwek_ugx_texture_path') == path for image in bpy.data.images):
        path = f'anders/templates/{usage}_{signature}_{uuid.uuid4().hex[:8]}_{suffix}'
    image = hw2_port_generated_image('HW2 Template ' + usage.title(), 4, 4, pixels,
                                   usage in NON_COLOR_TEXTURE_USAGES)
    image['hwek_ugx_port_template'] = signature
    image['hwek_ugx_port_template_usage'] = usage
    image['hwek_ugx_texture_path'] = path
    return image


def hw2_port_mask_image(mat, team, ao, resolution):
    images = [image for image in (team, ao) if image is not None]
    width = min(resolution, max((int(image.size[0]) for image in images), default=4))
    height = min(resolution, max((int(image.size[1]) for image in images), default=4))
    power = max(0.0, float(mat.get(v4_attr_property_name("SpecPower"), 32.0)))
    roughness = max(0.02, min(1.0, (2.0 / (power + 2.0)) ** 0.25))
    team_pixels = hw2_port_image_pixels(team, width, height, (1, 1, 1, 1), linear=True)
    ao_pixels = hw2_port_image_pixels(ao, width, height, (1, 1, 1, 1), linear=True)
    if np is not None:
        pixels = np.ones((width * height, 4), dtype=np.float32)
        pixels[:, 0] = roughness
        pixels[:, 1] = np.clip(np.asarray(ao_pixels).reshape(-1, 4)[:, 1], 0, 1)
        # HW1's green team mask is inverted; HW2 uses the packed blue channel.
        pixels[:, 2] = 1 - np.clip(np.asarray(team_pixels).reshape(-1, 4)[:, 1], 0, 1)
        pixels = pixels.reshape(-1)
    else:
        pixels = [value for index in range(width * height) for value in (
            roughness, max(0, min(1, ao_pixels[index * 4 + 1])),
            1 - max(0, min(1, team_pixels[index * 4 + 1])), 1)]
    if not images and (width, height) == (4, 4):
        return hw2_port_template_image('packedmask', pixels)
    return hw2_port_generated_image(mat.name + "_Masks", width, height, pixels)


def hw2_port_material(source, images, skinned, resource_root, index):
    defaults = HW2_SKINNED_MATERIAL_DEFAULT if skinned else HW2_STATIC_MATERIAL_DEFAULT
    spec = dict(defaults)
    if images.get('emissive') is not None:
        passes = HW2_PORT_SHADER_PASSES[skinned]
        spec["shader_names"] = tuple(name for name, _ in passes)
        spec["shader_hashes"] = tuple(value for _, value in passes)
    spec["pscb"] = struct.pack("<8f", 1, 1, 1, 1, 1, 1, 1 if images.get("emissive") else 0, 0)
    suffixes = ("[al]", "[nm]", "[pm]", "[sp]", "[em]")
    prefix = resource_root + f"/material_{index:03d}"
    images = dict(images)
    if images.get('diffuse') is None:
        images['diffuse'] = hw2_port_template_image('diffuse', [0.5, 0.5, 0.5, 1] * 16)
    usages = HOGAN_TEXTURE_SLOT_ORDER if images.get('emissive') is not None else HOGAN_TEXTURE_SLOT_ORDER[:4]
    paths = []
    for slot_index, usage in enumerate(usages):
        image = images.get(usage)
        if image is None:
            paths.append('0')
        elif image.get('hwek_ugx_port_template'):
            paths.append(str(image['hwek_ugx_texture_path']))
        else:
            paths.append(prefix + '_' + suffixes[slot_index])
    spec["textures"] = tuple(paths)
    parsed = read_materials(build_v6_material_bdt_from_specs([spec]), 1)[0]
    mat = bpy.data.materials.new(source.name + "_HW2")
    mat.use_nodes = True
    mat["hwek_ugx_cache_version"] = 6
    mat["hwek_ugx_native_material"] = True
    mat["hwek_ugx_texture_slots"] = json.dumps(parsed.slots)
    mat["hwek_ugx_export_name"] = source.get("hwek_ugx_export_name", source.name)
    apply_ufx_material_metadata(mat, parsed, None, 6)
    shader = create_hw_shader_node(mat, 6)
    apply_ufx_material_metadata(mat, parsed, shader, 6)
    old = active_ugx_material_shader(source, 4)
    shader.inputs["Player Color"].default_value = old.inputs["Player_color"].default_value
    for slot_index, usage in enumerate(usages):
        image = images.get(usage)
        if image is None:
            continue
        if not image.get('hwek_ugx_port_template'):
            source_image = image
            image = source_image.copy()
            image.name = mat.name + "_" + usage
            # Blender's image copy can reload the file instead of carrying unsaved pixels.
            if usage == 'specular' and not source_image.is_float and source_image.colorspace_settings.name == 'sRGB':
                pixels = hw2_port_image_pixels(source_image, *map(int, source_image.size), (1, 1, 1, 1), linear=True)
                set_non_color_image(image)
            elif np is not None:
                pixels = np.empty(int(source_image.size[0]) * int(source_image.size[1]) * 4, dtype=np.float32)
                source_image.pixels.foreach_get(pixels)
            else:
                pixels = image_rgba_floats(source_image)
            image.pixels.foreach_set(pixels)
            image.update()
            image.pack()
        path = spec["textures"][slot_index]
        image["hwek_ugx_texture_path"] = path
        link_material_texture(mat, usage, image, slot_index, path, shader_node=shader, cache_version=6)
        mat[f"hwek_ugx_texture_slot_{slot_index}"] = path
        mat[f"hwek_ugx_texture_slot_{slot_index}_synced"] = path
    arrange_ugx_material_nodes(mat, 6)
    ensure_ugx_parameter_nodes(mat)
    return mat


def retarget_hw2_port_object(obj, copies):
    obj.parent = copies.get(obj.parent, obj.parent)
    blocks = list(obj.modifiers) + list(obj.constraints)
    if obj.type == "ARMATURE":
        blocks += [constraint for bone in obj.pose.bones for constraint in bone.constraints]
    for block in blocks:
        for name in ("object", "target", "mirror_object", "offset_object"):
            target = getattr(block, name, None)
            if target in copies:
                setattr(block, name, copies[target])
    for block in (obj, obj.data):
        animation = getattr(block, "animation_data", None)
        if animation is not None:
            for curve in animation.drivers:
                for variable in curve.driver.variables:
                    for target in variable.targets:
                        if target.id in copies:
                            target.id = copies[target.id]


def port_hw1_asset_to_hw2(context, obj, resolution=1024):
    if obj is None or obj.mode != "OBJECT":
        raise UgxError("Select an imported HW1 mesh or rig in Object Mode.")
    scope = ugx_portable_object_scope([obj])
    meshes = [item for item in scope if item.type == "MESH" and item.data.polygons]
    if not meshes or any(int(item.get("hwek_ugx_cache_version", 0)) != 4 for item in meshes):
        raise UgxError("Select an imported HW1 asset before creating its HW2 copy.")
    materials = {mat for item in meshes for mat in item.data.materials if mat is not None}
    if any(ugx_material_cache_version(mat) != 4 for mat in materials):
        raise UgxError("All source materials must use HW1 material data before porting.")
    if any("UV0" not in item.data.uv_layers for item in meshes):
        raise UgxError("Every source mesh needs UV0 before porting to HW2.")
    owners = [owner for owner in scope if owner.get('hwek_vis_text') and not owner.get('hwek_vis_marker')]
    for owner in owners:
        populate_visual_settings(owner)
    scope = ugx_portable_object_scope([obj])
    snapshots = {name: set(getattr(bpy.data, name)) for name in ("objects", "meshes", "armatures", "materials", "images", "collections", "texts", "node_groups", "actions")}
    source_state = {source: (source.name, source.hide_get(), source.hide_render,
                             {key: source[key] for key in ('hwek_ugx_mesh_name', 'hwek_ugx_mesh_import_name') if key in source})
                    for source in scope}
    active, selected = context.view_layer.objects.active, list(context.selected_objects)
    notes, copies, material_copies = set(), {}, {}
    name = os.path.splitext(os.path.basename(str(meshes[0].get("hwek_ugx_source_path", meshes[0].name))))[0]
    resource_root = "ports/" + re.sub(r"[^\w.-]", "_", name) + "_" + uuid.uuid4().hex[:8]
    try:
        collection = bpy.data.collections.new(name + "_HW2")
        context.scene.collection.children.link(collection)
        for source in scope:
            target = source.copy()
            if source.data is not None:
                target.data = source.data.copy()
            target.name = source.name + "_HW2"
            collection.objects.link(target)
            copies[source] = target
        visual_texts = {}
        for source, target in copies.items():
            for key in ('hwek_vis_text', 'hwek_damage_text'):
                text = source.get(key)
                if text is not None:
                    if text not in visual_texts:
                        visual_texts[text] = text.copy()
                    target[key] = visual_texts[text]
                    if key == 'hwek_damage_text':
                        require_damage_version(source, model_damage_version(source))
                        target[key]['hwek_damage_game'] = text['hwek_damage_game']
                        notes.add('HW1 damage is retained for reference only. Load a native HW2 damage template before exporting VIS with damage.')
        action_copies = {}
        for action in list(bpy.data.actions):
            owner = action.get('hwek_vis_owner')
            if owner in copies:
                copy = action.copy()
                copy['hwek_vis_owner'] = copies[owner]
                action_copies[action] = copy
        for source, target in copies.items():
            retarget_hw2_port_object(target, copies)
            for key in ('hwek_vis_default_action', 'hwek_vis_requested_action', 'hwek_vis_active_action'):
                if target.get(key) in action_copies:
                    target[key] = action_copies[target[key]]
            if target.animation_data and target.animation_data.action in action_copies:
                target.animation_data.action = action_copies[target.animation_data.action]
            target.matrix_world = source.matrix_world
            clear_ugx_id_properties(target)
            clear_ugx_id_properties(target.data)
            # Mesh names and LOD ranges also describe fresh HW2 accessories.
            if source.type == "MESH":
                target["hwek_ugx_mesh_name"] = current_ugx_mesh_name(source)
                target["hwek_ugx_mesh_import_name"] = target.name
                for key in ("hwek_ugx_lod_min_distance", "hwek_ugx_lod_max_distance",
                            "hwek_ugx_sorting_bias", "hwek_ugx_lod_index", "hwek_ugx_mesh_index"):
                    if key in source:
                        target[key] = source[key]
            if target.type == "ARMATURE":
                if "hwek_ugx_granny_model_name" in source:
                    target["hwek_ugx_granny_model_name"] = source["hwek_ugx_granny_model_name"]
                for bone in target.data.bones:
                    clear_ugx_id_properties(bone)
                    original = source.data.bones[bone.name]
                    for key in ("hwek_ugx_bone_index", "hwek_ugx_source_name", "hwek_ugx_blender_name",
                                "hwek_ugx_export_name", "hwek_ugx_source_matrix", "hwek_ugx_export_matrix",
                                "hwek_ugx_skeleton_transform_mode", "hwek_ugx_source_display_matrix",
                                "hwek_ugx_import_display_matrix", "hwek_ugx_source_head_local"):
                        if key in original:
                            bone[key] = original[key]
                for bone in target.pose.bones:
                    clear_ugx_id_properties(bone)
            if target.type != "MESH":
                continue
            for attribute in list(target.data.attributes):
                if attribute.name.startswith("UGX_"):
                    target.data.attributes.remove(attribute)
            if not target.data.polygons:
                notes.add(f"{source.name}: the empty helper mesh is kept in Blender but has no exported geometry.")
                continue
            target.data.uv_layers.active = target.data.uv_layers["UV0"]
            target.data.uv_layers["UV0"].active_render = True
            used_slots = sorted({poly.material_index for poly in target.data.polygons})
            used_materials = [target.data.materials[index] if index < len(target.data.materials) else None for index in used_slots]
            if any(material is None for material in used_materials):
                raise UgxError(f"Mesh {source.name} has faces without a material. Assign their HW1 material before porting.")
            slot_map = {old: new for new, old in enumerate(used_slots)}
            polygon_slots = [poly.material_index for poly in target.data.polygons]
            target.data.materials.clear()
            for material in used_materials:
                target.data.materials.append(material)
            for poly, original_slot in zip(target.data.polygons, polygon_slots):
                poly.material_index = slot_map[original_slot]
            for slot in target.material_slots:
                if slot.material is None:
                    raise UgxError(f"Mesh {source.name} has an empty material slot. Assign its HW1 material before porting.")
                if slot.material not in material_copies:
                    material_copies[slot.material] = slot.material.copy()
                    retarget_ugx_parameter_drivers(material_copies[slot.material])
                slot.material = material_copies[slot.material]
        context.view_layer.update()
        converted = {}
        for source, working in material_copies.items():
            users = [target for target in copies.values() if target.type == "MESH" and working in target.data.materials[:]]
            mesh = next((target for target in users if any(target.data.materials[poly.material_index] == working for poly in target.data.polygons)), None)
            if mesh is None:
                mesh = users[0]
            images = {usage: hw1_port_input_image(context, mesh, working, usage, resolution, notes)
                      for usage in ("diffuse", "normal", "gloss", "xform", "ao", "emissive")}
            images["packedmask"] = hw2_port_mask_image(working, images.pop("xform"), images.pop("ao"), resolution)
            images["specular"] = images.pop("gloss")
            converted[working] = hw2_port_material(working, images, ugx_armature_for(mesh) is not None, resource_root, len(converted))
            if int(working.get(v4_attr_property_name("BlendType"), 0)) != 0 or float(working.get(v4_attr_property_name("Opacity"), 255)) < 255:
                notes.add(f"{source.name}: transparency needs a compatible HW2 shader; the port uses opaque shading.")
            for usage in ("opacity", "env", "envmask", "emxform", "distortion", "highlight", "modulate"):
                socket = active_ugx_material_shader(working, 4).inputs.get(hw_shader_socket_name(usage, 4))
                if socket is not None and socket.is_linked:
                    notes.add(f"{source.name}: {usage} needs a separate HW2 material setup.")
        for target in copies.values():
            if target.type == "MESH" and target.data.polygons:
                for slot in target.material_slots:
                    if slot.material not in converted:
                        raise UgxError(f"Material {slot.material.name} has no faces in this asset.")
                    slot.material = converted[slot.material]
            target["hwek_ugx_port_target"] = 6
        report = bpy.data.texts.new(name + "_HW2 Port Report")
        lines = ["HW1 to HW2 port", f"Created {len(meshes)} meshes and {len(converted)} materials.",
                 "Masks: red roughness, green occlusion, blue inverted HW1 team mask.",
                 "Connected emissive images use texture slot 4 and an emissive shader.",
                 "Export with File > Export > Halo Wars 2 UGX (v6).", "", "Items to check:"]
        lines.extend(sorted(notes) or ["No additional material effects were found."])
        if any(item.type == "ARMATURE" for item in copies.values()):
            lines.append("The HW1 rig and bone names are preserved. Stock HW2 animations need retargeting or a compatible HW2 rig.")
        lines.append("Unit definitions, effects, damage documents and animation files need separate porting.")
        report.write("\n".join(lines))
        for target in copies.values():
            target["hwek_ugx_port_report"] = report.name
        for item in context.selected_objects:
            item.select_set(False)
        for source, target in copies.items():
            source.hide_set(True)
            source.hide_render = True
            target.hide_set(target.get('hwek_vis_marker') == 'uvOffset')
            target.hide_render = False
        target = copies[obj] if copies[obj].type == "MESH" else copies[meshes[0]]
        target.select_set(True)
        context.view_layer.objects.active = target
        for source, ported in copies.items():
            if source.type != 'MESH':
                continue
            source_name = source.name
            source_mesh_name = current_ugx_mesh_name(source)
            source.name = source_name if source_name.endswith('_HW1') else source_name + '_HW1'
            source['hwek_ugx_mesh_name'] = source_mesh_name
            source['hwek_ugx_mesh_import_name'] = source.name
            ported.name = source_name
            ported['hwek_ugx_mesh_import_name'] = ported.name
        initialize_ugx_player_color(target)
        for owner in copies.values():
            if owner.get('hwek_vis_marker'):
                continue
            if owner.get('hwek_vis_text'):
                populate_visual_settings(owner)
            if owner.get('hwek_damage_text') and int(owner['hwek_damage_text'].get('hwek_damage_game', 4)) == 6:
                populate_damage_settings(owner)
        for working in material_copies.values():
            bpy.data.materials.remove(working)
        for image in list(bpy.data.images):
            if image not in snapshots["images"] and image.users == 0:
                bpy.data.images.remove(image)
        return {"object": target, "meshes": len(meshes), "materials": len(converted), "notes": sorted(notes), "report": report}
    except Exception:
        for name, original in snapshots.items():
            for item in list(getattr(bpy.data, name)):
                if item not in original:
                    getattr(bpy.data, name).remove(item, do_unlink=True)
        for source, (name, hidden, hide_render, properties) in source_state.items():
            source.name = name
            source.hide_set(hidden)
            source.hide_render = hide_render
            for key in ('hwek_ugx_mesh_name', 'hwek_ugx_mesh_import_name'):
                if key in properties:
                    source[key] = properties[key]
                elif key in source:
                    del source[key]
        context.view_layer.objects.active = active
        for item in selected:
            item.select_set(True)
        raise
