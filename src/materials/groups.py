"""Build and migrate Blender preview groups for native shaders."""

from typing import Any, Dict, List, Optional, Tuple
import math
import bpy
from ..errors import UgxError

__all__ = ('HW_SHADER_BUILTIN_VERSION', 'HW_SHADER_GROUP_SIGNATURES', 'hw_shader_group_signature', 'hw_shader_group_name', 'hw_shader_group_interface_names', 'hw_shader_group_matches_builtin', 'new_hw_shader_interface_socket', 'hw_shader_node', 'hw_shader_socket', 'link_hw_shader_nodes', 'add_hw_shader_normal_z_preview', 'add_hw_shader_normal_preview', 'upgrade_hw1_surface_preview', 'build_builtin_hwde_shader_group', 'build_builtin_hw2_shader_group', 'build_builtin_hw_shader_group', 'hw_shader_groups_have_same_graph', 'hw_shader_group_legacy_copy', 'migrate_builtin_hw_shader_group_users', 'ensure_hw_shader_group')

HW_SHADER_BUILTIN_VERSION = 13


HW_SHADER_GROUP_SIGNATURES = {
    4: {
        "name": "Halo Wars DE",
        "nodes": 26,
        "links": 42,
        "interface": [
            "BSDF", "Player_color", "Diffuse (df)", "Normal (nm)", "Specular (sp)", "Alpha (op)",
            "Emmissive (em)", "Emmissive_player_color (ef)", "Player_color (xf)", "Environment (en)",
            "EnvMask (rm)", "Occlusion (ao)", "Distortion (ds)", "Highlight (hl)", "Modulate (ml)",
            "SpecPower", "SpecColor", "Reflectivity", "Sharpness", "FresnelPower",
            "Fresnel", "AccIndex", "BlendType", "Opacity", "Flags",
        ],
    },
    6: {
        "name": "Halo Wars 2",
        "nodes": 19,
        "links": 30,
        "interface": [
            "BSDF", "Player Color", "Albedo", "Emissive", "Normal", "Specular", "Masks", "Emission", "Shader Passes",
        ],
    },
}


def hw_shader_group_signature(cache_version: int) -> Dict[str, Any]:
    return HW_SHADER_GROUP_SIGNATURES.get(cache_version, HW_SHADER_GROUP_SIGNATURES[4])


def hw_shader_group_name(cache_version: int) -> str:
    return str(hw_shader_group_signature(cache_version)["name"])


def hw_shader_group_interface_names(group: bpy.types.NodeTree) -> List[str]:
    interface = getattr(group, "interface", None)
    items = getattr(interface, "items_tree", None) if interface is not None else None
    if items is not None:
        return [item.name for item in items]
    names: List[str] = []
    input_node = next((node for node in group.nodes if node.type == "GROUP_INPUT"), None)
    output_node = next((node for node in group.nodes if node.type == "GROUP_OUTPUT"), None)
    if output_node is not None:
        names.extend(socket.name for socket in output_node.inputs)
    if input_node is not None:
        names.extend(socket.name for socket in input_node.outputs)
    return names


def hw_shader_group_matches_builtin(group: Optional[bpy.types.NodeTree], cache_version: int) -> bool:
    if group is None:
        return False
    signature = hw_shader_group_signature(cache_version)
    return (
        getattr(group, "bl_idname", "") == "ShaderNodeTree"
        and bool(group.get("hwek_ugx_shader_builtin", False))
        and int(group.get("hwek_ugx_shader_builtin_version", 0)) == HW_SHADER_BUILTIN_VERSION
        and int(group.get("hwek_ugx_shader_game_version", 0)) == cache_version
        and len(group.nodes) == int(signature["nodes"])
        and len(group.links) == int(signature["links"])
        and hw_shader_group_interface_names(group) == list(signature["interface"])
    )


def new_hw_shader_interface_socket(
    group: bpy.types.NodeTree,
    name: str,
    in_out: str,
    socket_type: str,
    default_value: Any = None,
    min_value: Any = None,
    max_value: Any = None,
    parent: Any = None,
    description: str = "",
):
    socket = group.interface.new_socket(
        name=name,
        description=description,
        in_out=in_out,
        socket_type=socket_type,
        parent=parent,
    )
    for attribute, value in (("default_value", default_value), ("min_value", min_value), ("max_value", max_value)):
        if value is None or not hasattr(socket, attribute):
            continue
        try:
            setattr(socket, attribute, value)
        except (AttributeError, TypeError, ValueError):
            pass
    return socket


def hw_shader_node(group: bpy.types.NodeTree, node_type: str, name: str, location: Tuple[float, float]):
    node = group.nodes.new(node_type)
    node.name = name
    node.label = name
    node.location = location
    return node


def hw_shader_socket(sockets: Any, *names: str):
    for name in names:
        socket = sockets.get(name)
        if socket is not None:
            return socket
    raise UgxError("Built-in shader node is missing socket: " + " / ".join(names))


def link_hw_shader_nodes(
    group: bpy.types.NodeTree,
    from_node: bpy.types.Node,
    from_names: Tuple[str, ...],
    to_node: bpy.types.Node,
    to_names: Tuple[str, ...],
):
    group.links.new(
        hw_shader_socket(from_node.outputs, *from_names),
        hw_shader_socket(to_node.inputs, *to_names),
    )


def add_hw_shader_normal_z_preview(group: bpy.types.NodeTree, group_input: bpy.types.Node, combine: bpy.types.Node, y: float):
    # The game reconstructs Z after sampling RG. Decoded DDS blue can be absent
    # or reconstructed before filtering, which produces different normal slopes.
    xy = hw_shader_node(group, "ShaderNodeVectorMath", "Normal XY", (-610, y - 260))
    xy.operation = "MULTIPLY_ADD"
    xy.inputs[1].default_value = (2.0, 2.0, 0.0)
    xy.inputs[2].default_value = (-1.0, -1.0, 0.0)
    length = hw_shader_node(group, "ShaderNodeVectorMath", "Normal XY Length Squared", (-390, y - 260))
    length.operation = "DOT_PRODUCT"
    remaining = hw_shader_node(group, "ShaderNodeMath", "Normal Z Squared", (-160, y - 260))
    remaining.operation = "SUBTRACT"
    remaining.use_clamp = True
    remaining.inputs[0].default_value = 1.0
    z = hw_shader_node(group, "ShaderNodeMath", "Normal Z", (60, y - 260))
    z.operation = "SQRT"
    encoded = hw_shader_node(group, "ShaderNodeMath", "Encode Normal Z", (280, y - 260))
    encoded.operation = "MULTIPLY_ADD"
    encoded.inputs[1].default_value = 0.5
    encoded.inputs[2].default_value = 0.5
    link_hw_shader_nodes(group, group_input, ("Normal", "Normal (nm)"), xy, ("Vector",))
    group.links.new(xy.outputs["Vector"], length.inputs[0])
    group.links.new(xy.outputs["Vector"], length.inputs[1])
    group.links.new(length.outputs["Value"], remaining.inputs[1])
    group.links.new(remaining.outputs[0], z.inputs[0])
    group.links.new(z.outputs[0], encoded.inputs[0])
    group.links.new(encoded.outputs[0], combine.inputs["Blue"])


def add_hw_shader_normal_preview(group: bpy.types.NodeTree, group_input: bpy.types.Node, y: float):
    separate = hw_shader_node(group, "ShaderNodeSeparateColor", "Decode Normal", (-610, y))
    invert_green = hw_shader_node(group, "ShaderNodeInvert", "Decode Normal Green", (-390, y - 40))
    combine = hw_shader_node(group, "ShaderNodeCombineColor", "Decoded Normal", (-160, y - 60))
    normal_map = hw_shader_node(group, "ShaderNodeNormalMap", "Normal Map", (60, y - 60))
    separate.mode = "RGB"
    combine.mode = "RGB"

    link_hw_shader_nodes(group, group_input, ("Normal", "Normal (nm)"), separate, ("Color",))
    link_hw_shader_nodes(group, separate, ("Red",), combine, ("Red",))
    hw_shader_socket(invert_green.inputs, "Factor", "Fac").default_value = 1.0
    link_hw_shader_nodes(group, separate, ("Green",), invert_green, ("Color",))
    link_hw_shader_nodes(group, invert_green, ("Color",), combine, ("Green",))
    add_hw_shader_normal_z_preview(group, group_input, combine, y)
    link_hw_shader_nodes(group, combine, ("Color",), normal_map, ("Color",))
    return normal_map


def upgrade_hw1_surface_preview(group):
    inputs = group.nodes['Group Input']
    surface = group.nodes['Halo Wars Surface']
    if group.nodes.get('Modulated Albedo') is not None:
        return
    modulate = hw_shader_node(group, 'ShaderNodeMixRGB', 'Modulated Albedo', (180, 420))
    modulate.blend_type = 'MULTIPLY'
    modulate.inputs[0].default_value = 1.0
    group.links.new(group.nodes['Player Color'].outputs['Color'], modulate.inputs[1])
    group.links.new(inputs.outputs['Modulate (ml)'], modulate.inputs[2])
    group.links.new(modulate.outputs[0], surface.inputs['Base Color'])
    emission = hw_shader_node(group, 'ShaderNodeMixRGB', 'Surface Emission', (180, -600))
    emission.blend_type = 'ADD'
    emission.inputs[0].default_value = 1.0
    group.links.new(group.nodes['Emissive Player Color'].outputs['Color'], emission.inputs[1])
    group.links.new(inputs.outputs['Highlight (hl)'], emission.inputs[2])
    group.links.new(emission.outputs[0], hw_shader_socket(surface.inputs, 'Emission Color', 'Emission'))
    surface.inputs['Emission Strength'].default_value = 1.0
    for name, default in (('Modulate (ml)', (1, 1, 1, 1)), ('Specular (sp)', (1, 1, 1, 1))):
        interface = next(item for item in group.interface.items_tree if item.name == name)
        interface.default_value = default


def _configure_hw1_specular_preview(group):
    # Match the half-width of HW1's r / (p - (p - 1) * r) lobe to GGX.
    # Blender squares roughness to obtain GGX alpha.
    coefficient = 2.0 * (math.sqrt(2.0) - 1.0)
    denominator = group.nodes['Specular Power Offset']
    denominator.operation = 'MULTIPLY_ADD'
    denominator.inputs[1].default_value = coefficient
    denominator.inputs[2].default_value = coefficient + 1.0
    ratio = group.nodes['Specular Roughness Ratio']
    ratio.inputs[0].default_value = 1.0
    roughness = group.nodes['Specular Roughness']
    roughness.operation = 'POWER'
    roughness.inputs[1].default_value = 0.25


def _configure_hw1_player_color_mask(group):
    channel = hw_shader_node(group, "ShaderNodeSeparateColor", "Player Mask Channel", (-830, 420))
    channel.mode = "RGB"
    group.links.new(group.nodes["Group Input"].outputs["Player_color (xf)"], channel.inputs["Color"])
    group.links.new(channel.outputs["Green"], group.nodes["Player Mask"].inputs["Color"])


def build_builtin_hwde_shader_group() -> bpy.types.NodeTree:
    group = bpy.data.node_groups.new(hw_shader_group_name(4), "ShaderNodeTree")
    try:
        new_hw_shader_interface_socket(group, "BSDF", "OUTPUT", "NodeSocketShader")
        new_hw_shader_interface_socket(group, "Player_color", "INPUT", "NodeSocketColor", (0.28, 0.37, 0.09, 1.0))
        new_hw_shader_interface_socket(group, "Diffuse (df)", "INPUT", "NodeSocketColor", (0.5, 0.5, 0.5, 1.0))
        new_hw_shader_interface_socket(group, "Normal (nm)", "INPUT", "NodeSocketColor", (0.5, 0.5, 1.0, 1.0))
        new_hw_shader_interface_socket(group, "Specular (sp)", "INPUT", "NodeSocketColor", (0.0, 0.0, 0.0, 1.0))
        new_hw_shader_interface_socket(group, "Alpha (op)", "INPUT", "NodeSocketColor", (1.0, 1.0, 1.0, 1.0))
        new_hw_shader_interface_socket(group, "Emmissive (em)", "INPUT", "NodeSocketColor", (0.0, 0.0, 0.0, 1.0))
        new_hw_shader_interface_socket(group, "Emmissive_player_color (ef)", "INPUT", "NodeSocketColor", (0.0, 0.0, 0.0, 1.0))
        new_hw_shader_interface_socket(
            group, "Player_color (xf)", "INPUT", "NodeSocketColor", (1.0, 1.0, 1.0, 1.0),
            description="Green channel: 0 applies player color; 1 keeps diffuse color.",
        )
        new_hw_shader_interface_socket(group, "Environment (en)", "INPUT", "NodeSocketColor", (0.0, 0.0, 0.0, 1.0))
        new_hw_shader_interface_socket(group, "EnvMask (rm)", "INPUT", "NodeSocketColor", (0.0, 0.0, 0.0, 1.0))
        new_hw_shader_interface_socket(
            group, "Occlusion (ao)", "INPUT", "NodeSocketColor", (0.0, 0.0, 0.0, 1.0),
            description="Legacy Halo Wars occlusion texture slot.",
        )
        new_hw_shader_interface_socket(group, "Distortion (ds)", "INPUT", "NodeSocketColor", (0.0, 0.0, 0.0, 1.0))
        new_hw_shader_interface_socket(group, "Highlight (hl)", "INPUT", "NodeSocketColor", (0.0, 0.0, 0.0, 1.0))
        new_hw_shader_interface_socket(group, "Modulate (ml)", "INPUT", "NodeSocketColor", (0.0, 0.0, 0.0, 1.0))

        settings = None
        new_hw_shader_interface_socket(group, "SpecPower", "INPUT", "NodeSocketFloat", 500.0, 0.0, 500.0, settings)
        new_hw_shader_interface_socket(group, "SpecColor", "INPUT", "NodeSocketColor", (1.0, 1.0, 1.0, 1.0), parent=settings)
        new_hw_shader_interface_socket(group, "Reflectivity", "INPUT", "NodeSocketFloat", 1.0, 0.0, 1.0, settings)
        new_hw_shader_interface_socket(group, "Sharpness", "INPUT", "NodeSocketFloat", 1.0, 0.0, 1.0, settings)
        new_hw_shader_interface_socket(group, "FresnelPower", "INPUT", "NodeSocketFloat", 4.0, parent=settings)
        new_hw_shader_interface_socket(group, "Fresnel", "INPUT", "NodeSocketFloat", 0.5, parent=settings)
        new_hw_shader_interface_socket(group, "AccIndex", "INPUT", "NodeSocketInt", 0, 0, None, settings)
        new_hw_shader_interface_socket(group, "BlendType", "INPUT", "NodeSocketInt", 0, 0, 5, settings)
        new_hw_shader_interface_socket(group, "Opacity", "INPUT", "NodeSocketInt", 255, 0, 255, settings)

        new_hw_shader_interface_socket(group, "Flags", "INPUT", "NodeSocketInt", 1)

        group_input = hw_shader_node(group, "NodeGroupInput", "Group Input", (-930, 260))
        group_output = hw_shader_node(group, "NodeGroupOutput", "Group Output", (900, 260))
        principled = hw_shader_node(group, "ShaderNodeBsdfPrincipled", "Halo Wars Surface", (430, 280))
        normal_map = add_hw_shader_normal_preview(group, group_input, -250)
        player_mask = hw_shader_node(group, "ShaderNodeInvert", "Player Mask", (-610, 300))
        player_mix = hw_shader_node(group, "ShaderNodeMixRGB", "Player Color", (-160, 360))
        emissive_mask = hw_shader_node(group, "ShaderNodeInvert", "Emissive Player Mask", (-610, -640))
        emissive_mix = hw_shader_node(group, "ShaderNodeMixRGB", "Emissive Player Color", (-160, -600))
        player_mix.blend_type = "MULTIPLY"
        player_mix.use_clamp = True
        emissive_mix.blend_type = "MULTIPLY"
        emissive_mix.use_clamp = True

        link_hw_shader_nodes(group, normal_map, ("Normal",), principled, ("Normal",))
        _configure_hw1_player_color_mask(group)
        link_hw_shader_nodes(group, group_input, ("Diffuse (df)",), player_mix, ("Color1",))
        link_hw_shader_nodes(group, group_input, ("Player_color",), player_mix, ("Color2",))
        link_hw_shader_nodes(group, player_mask, ("Color",), player_mix, ("Fac", "Factor"))
        link_hw_shader_nodes(group, player_mix, ("Color",), principled, ("Base Color",))
        link_hw_shader_nodes(group, group_input, ("Emmissive_player_color (ef)",), emissive_mask, ("Color",))
        link_hw_shader_nodes(group, group_input, ("Emmissive (em)",), emissive_mix, ("Color1",))
        link_hw_shader_nodes(group, group_input, ("Player_color",), emissive_mix, ("Color2",))
        link_hw_shader_nodes(group, emissive_mask, ("Color",), emissive_mix, ("Fac", "Factor"))
        link_hw_shader_nodes(group, emissive_mix, ("Color",), principled, ("Emission Color", "Emission"))
        link_hw_shader_nodes(group, principled, ("BSDF",), group_output, ("BSDF",))

        power_offset = hw_shader_node(group, "ShaderNodeMath", "Specular Power Offset", (-380, -900))
        roughness_ratio = hw_shader_node(group, "ShaderNodeMath", "Specular Roughness Ratio", (-160, -900))
        roughness_ratio.operation = "DIVIDE"
        roughness = hw_shader_node(group, "ShaderNodeMath", "Specular Roughness", (60, -900))
        _configure_hw1_specular_preview(group)
        group.links.new(group_input.outputs["SpecPower"], power_offset.inputs[0])
        group.links.new(power_offset.outputs[0], roughness_ratio.inputs[1])
        group.links.new(roughness_ratio.outputs[0], roughness.inputs[0])
        group.links.new(roughness.outputs[0], principled.inputs["Roughness"])
        specular = hw_shader_node(group, "ShaderNodeMixRGB", "Specular Color", (60, 100))
        specular.blend_type = "MULTIPLY"
        specular.inputs[0].default_value = 1.0
        group.links.new(group_input.outputs["Specular (sp)"], specular.inputs[1])
        group.links.new(group_input.outputs["SpecColor"], specular.inputs[2])
        group.links.new(specular.outputs[0], principled.inputs["Specular Tint"])
        specular_strength = hw_shader_node(group, "ShaderNodeRGBToBW", "Specular Strength", (230, 50))
        group.links.new(specular.outputs[0], specular_strength.inputs["Color"])
        group.links.new(specular_strength.outputs["Val"], principled.inputs["Specular IOR Level"])
        opacity = hw_shader_node(group, "ShaderNodeMath", "Material Opacity", (-160, -1100))
        opacity.operation = "DIVIDE"
        opacity.inputs[1].default_value = 255.0
        alpha = hw_shader_node(group, "ShaderNodeMath", "Texture Opacity", (60, -1100))
        alpha.operation = "MULTIPLY"
        alpha.use_clamp = True
        group.links.new(group_input.outputs["Opacity"], opacity.inputs[0])
        group.links.new(opacity.outputs[0], alpha.inputs[0])
        group.links.new(group_input.outputs["Alpha (op)"], alpha.inputs[1])
        group.links.new(alpha.outputs[0], principled.inputs["Alpha"])

        upgrade_hw1_surface_preview(group)
        group["hwek_ugx_shader_builtin"] = True
        group["hwek_ugx_shader_builtin_version"] = HW_SHADER_BUILTIN_VERSION
        group["hwek_ugx_shader_game_version"] = 4
        group["hwek_inline_parameters"] = True
        return group
    except Exception:
        bpy.data.node_groups.remove(group)
        raise


def build_builtin_hw2_shader_group() -> bpy.types.NodeTree:
    group = bpy.data.node_groups.new(hw_shader_group_name(6), "ShaderNodeTree")
    try:
        new_hw_shader_interface_socket(group, "BSDF", "OUTPUT", "NodeSocketShader")
        new_hw_shader_interface_socket(group, "Player Color", "INPUT", "NodeSocketColor", (1.0, 0.013, 0.013, 1.0))
        new_hw_shader_interface_socket(group, "Albedo", "INPUT", "NodeSocketColor", (0.5, 0.5, 0.5, 1.0))
        new_hw_shader_interface_socket(group, "Emissive", "INPUT", "NodeSocketColor", (0.0, 0.0, 0.0, 1.0))
        new_hw_shader_interface_socket(group, "Normal", "INPUT", "NodeSocketColor", (0.5, 0.5, 1.0, 1.0))
        new_hw_shader_interface_socket(group, "Specular", "INPUT", "NodeSocketColor", (0.0, 0.0, 0.0, 1.0))
        new_hw_shader_interface_socket(
            group, "Masks", "INPUT", "NodeSocketColor", (0.0, 1.0, 0.0, 1.0),
            description="Red: roughness. Green: albedo multiplier (1 keeps full brightness). Blue: player-color mask.",
        )

        new_hw_shader_interface_socket(group, "Emission", "INPUT", "NodeSocketFloat", 1.0, 0.0)
        new_hw_shader_interface_socket(group, "Shader Passes", "INPUT", "NodeSocketFloat")

        group_input = hw_shader_node(group, "NodeGroupInput", "Group Input", (-930, 260))
        group_output = hw_shader_node(group, "NodeGroupOutput", "Group Output", (900, 260))
        principled = hw_shader_node(group, "ShaderNodeBsdfPrincipled", "Halo Wars 2 Surface", (430, 280))
        normal_map = add_hw_shader_normal_preview(group, group_input, -250)
        packed = hw_shader_node(group, "ShaderNodeSeparateColor", "Packed Map", (-610, 80))
        player_mix = hw_shader_node(group, "ShaderNodeMixRGB", "Player Color", (-160, 360))
        packed_mix = hw_shader_node(group, "ShaderNodeMixRGB", "Packed Color", (80, 360))
        emissive_mix = hw_shader_node(group, "ShaderNodeMixRGB", "Emissive Player Color", (160, -190))
        emission = hw_shader_node(group, "ShaderNodeEmission", "Emissive", (430, -190))
        add_shader = hw_shader_node(group, "ShaderNodeAddShader", "Surface + Emissive", (900, 250))
        packed.mode = "RGB"
        player_mix.blend_type = "MULTIPLY"
        player_mix.use_clamp = True
        packed_mix.blend_type = "MULTIPLY"
        packed_mix.inputs[0].default_value = 1.0
        packed_mix.use_clamp = True
        emissive_mix.blend_type = "BURN"
        emissive_mix.inputs[0].default_value = 0.5
        emissive_mix.use_clamp = True
        principled.inputs["Metallic"].default_value = 0.0

        link_hw_shader_nodes(group, normal_map, ("Normal",), principled, ("Normal",))
        link_hw_shader_nodes(group, group_input, ("Masks",), packed, ("Color",))
        link_hw_shader_nodes(group, group_input, ("Albedo",), player_mix, ("Color1",))
        link_hw_shader_nodes(group, group_input, ("Player Color",), player_mix, ("Color2",))
        link_hw_shader_nodes(group, packed, ("Blue",), player_mix, ("Fac", "Factor"))
        link_hw_shader_nodes(group, player_mix, ("Color",), packed_mix, ("Color1",))
        link_hw_shader_nodes(group, packed, ("Green",), packed_mix, ("Color2",))
        link_hw_shader_nodes(group, packed_mix, ("Color",), principled, ("Base Color",))
        link_hw_shader_nodes(group, group_input, ("Specular",), principled, ("Specular Tint",))
        specular_strength = hw_shader_node(group, "ShaderNodeRGBToBW", "Specular Strength", (230, 50))
        link_hw_shader_nodes(group, group_input, ("Specular",), specular_strength, ("Color",))
        link_hw_shader_nodes(group, specular_strength, ("Val",), principled, ("Specular IOR Level",))
        link_hw_shader_nodes(group, group_input, ("Emissive",), emissive_mix, ("Color1",))
        link_hw_shader_nodes(group, group_input, ("Player Color",), emissive_mix, ("Color2",))
        link_hw_shader_nodes(group, emissive_mix, ("Color",), emission, ("Color",))
        link_hw_shader_nodes(group, group_input, ("Emission",), emission, ("Strength",))
        group.links.new(principled.outputs["BSDF"], add_shader.inputs[0])
        group.links.new(emission.outputs[0], add_shader.inputs[1])
        link_hw_shader_nodes(group, add_shader, ("Shader",), group_output, ("BSDF",))

        group["hwek_ugx_shader_builtin"] = True
        group["hwek_ugx_shader_builtin_version"] = HW_SHADER_BUILTIN_VERSION
        group["hwek_ugx_shader_game_version"] = 6
        group["hwek_inline_parameters"] = True
        return group
    except Exception:
        bpy.data.node_groups.remove(group)
        raise


def build_builtin_hw_shader_group(cache_version: int) -> bpy.types.NodeTree:
    return build_builtin_hw2_shader_group() if cache_version == 6 else build_builtin_hwde_shader_group()


def hw_shader_groups_have_same_graph(first, second):
    def links(tree):
        return {(link.from_node.name, link.from_socket.name, link.to_node.name, link.to_socket.name) for link in tree.links}

    if len(first.nodes) != len(second.nodes) or links(first) != links(second):
        return False
    for node in first.nodes:
        other = second.nodes.get(node.name)
        if other is None or node.bl_idname != other.bl_idname:
            return False
        for name in ("operation", "blend_type", "mode", "use_clamp", "space", "uv_map"):
            if getattr(node, name, None) != getattr(other, name, None):
                return False
        if node.type in ("GROUP_INPUT", "GROUP_OUTPUT"):
            continue
        for socket, other_socket in zip(node.inputs, other.inputs):
            if socket.name != other_socket.name:
                return False
            if not hasattr(socket, "default_value"):
                continue
            value, other_value = socket.default_value, other_socket.default_value
            try:
                value, other_value = tuple(value), tuple(other_value)
            except TypeError:
                pass
            if value != other_value:
                return False
    return True


def hw_shader_group_legacy_copy(group, version):
    candidate = group.copy()
    builtin_version = int(group.get('hwek_ugx_shader_builtin_version', 0))
    decode, combine = candidate.nodes.get('Decode Normal'), candidate.nodes.get('Decoded Normal')
    blue = combine.inputs.get('Blue') if combine is not None else None
    if builtin_version < 10 and decode is not None and blue is not None and len(blue.links) == 1:
        link = blue.links[0]
        if link.from_node == decode and link.from_socket.name == 'Blue':
            add_hw_shader_normal_z_preview(candidate, candidate.nodes['Group Input'], combine, decode.location.y)
    if version == 4 and builtin_version < 11:
        surface = candidate.nodes.get('Halo Wars Surface')
        if surface is not None and float(surface.inputs['Emission Strength'].default_value) in (0.0, 1.0):
            upgrade_hw1_surface_preview(candidate)
    if version == 4 and builtin_version < 12:
        offset = candidate.nodes.get('Specular Power Offset')
        ratio = candidate.nodes.get('Specular Roughness Ratio')
        roughness = candidate.nodes.get('Specular Roughness')
        if (offset is not None and ratio is not None and roughness is not None
                and offset.operation == 'ADD' and offset.inputs[1].default_value == 2.0
                and ratio.operation == 'DIVIDE' and ratio.inputs[0].default_value == 2.0
                and roughness.operation == 'SQRT'):
            _configure_hw1_specular_preview(candidate)
    if version == 4 and builtin_version < 13:
        mask = candidate.nodes.get('Player Mask')
        color = mask.inputs.get('Color') if mask is not None else None
        if color is not None and len(color.links) == 1:
            link = color.links[0]
            if link.from_node == candidate.nodes.get('Group Input') and link.from_socket.name == 'Player_color (xf)':
                _configure_hw1_player_color_mask(candidate)
    return candidate


def migrate_builtin_hw_shader_group_users(
    old_group: bpy.types.NodeTree,
    new_group: bpy.types.NodeTree,
):
    for mat in bpy.data.materials:
        tree = getattr(mat, "node_tree", None)
        if tree is None:
            continue
        for node in tree.nodes:
            if node.type not in ("GROUP", "CUSTOM GROUP") or getattr(node, "node_tree", None) != old_group:
                continue
            incoming = [
                (link.from_socket, link.to_socket.name)
                for socket in node.inputs
                for link in socket.links
            ]
            outgoing = [
                (link.from_socket.name, link.to_socket)
                for socket in node.outputs
                for link in socket.links
            ]
            defaults = {}
            for socket in node.inputs:
                if not hasattr(socket, "default_value"):
                    continue
                value = socket.default_value
                try:
                    defaults[socket.name] = tuple(value)
                except TypeError:
                    defaults[socket.name] = value

            node.node_tree = new_group
            for socket in list(node.inputs) + list(node.outputs):
                for link in list(socket.links):
                    tree.links.remove(link)
            old_names = {"Albedo (al)": "Albedo", "Emissive (em)": "Emissive", "Normal (nm)": "Normal", "Specular (sp)": "Specular", "Packed Map (pm)": "Masks"} if int(new_group.get("hwek_ugx_shader_game_version", 0)) == 6 else {}
            for socket_name, value in defaults.items():
                if socket_name == 'Modulate (ml)' and not any(name == socket_name for _, name in incoming):
                    value = (1, 1, 1, 1)
                socket = node.inputs.get(old_names.get(socket_name, socket_name))
                if socket is None or not hasattr(socket, "default_value"):
                    continue
                try:
                    socket.default_value = value
                except (TypeError, ValueError):
                    pass
            for from_socket, socket_name in incoming:
                socket = node.inputs.get(old_names.get(socket_name, socket_name))
                if socket is not None and not any(link.from_socket == from_socket for link in socket.links):
                    tree.links.new(from_socket, socket)
            for socket_name, to_socket in outgoing:
                socket = node.outputs.get(socket_name)
                if socket is not None and not any(link.to_socket == to_socket for link in socket.links):
                    tree.links.new(socket, to_socket)
            mat["hwek_ugx_shader_group"] = new_group.name


def ensure_hw_shader_group(cache_version: int) -> Optional[bpy.types.NodeTree]:
    group_name = hw_shader_group_name(cache_version)
    group = bpy.data.node_groups.get(group_name)
    if hw_shader_group_matches_builtin(group, cache_version):
        return group
    if group is not None:
        old_builtin = bool(group.get("hwek_ugx_shader_builtin", False))
        if getattr(group, "users", 0) == 0:
            bpy.data.node_groups.remove(group)
        else:
            group.name = group_name + " [Legacy]"
            new_group = build_builtin_hw_shader_group(cache_version)
            candidate = hw_shader_group_legacy_copy(group, cache_version) if old_builtin else None
            matches = candidate is not None and hw_shader_groups_have_same_graph(candidate, new_group)
            if candidate is not None:
                bpy.data.node_groups.remove(candidate)
            if matches:
                migrate_builtin_hw_shader_group_users(group, new_group)
                if getattr(group, "users", 0) == 0:
                    bpy.data.node_groups.remove(group)
            return new_group
    return build_builtin_hw_shader_group(cache_version)
