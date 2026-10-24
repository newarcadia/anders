"""Native material node types and reload-safe RNA registration."""

import bpy
from bpy.props import BoolProperty
from .context import material_for_ugx_shader_node
from .groups import ensure_hw_shader_group
from .ownership import copy_material_shader_state

__all__ = ('ShaderNodeAndersHW2', 'ShaderNodeAndersHWDE', 'ShaderNodeAndersHW1PlayerColorMask', 'ShaderNodeAndersUVAnimation', 'ShaderNodeAndersParameters', 'ShaderNodeAndersShaderPasses', 'ShaderNodeAndersHW1UV', 'ensure_ugx_material_node_types', 'menu_ugx_material_nodes')

def retain_registered_node_type(cls):
    registered = bpy.types.Node.bl_rna_get_subclass_py(cls.bl_idname)
    if registered is None or not registered.is_registered or registered.__module__ != cls.__module__:
        return cls
    # Keep live node instances bound to their RNA type during script reloads.
    for name in ('poll', 'init', 'copy', 'draw_buttons', 'draw_buttons_ext'):
        if name in cls.__dict__:
            setattr(registered, name, cls.__dict__[name])
    return registered


@retain_registered_node_type
class ShaderNodeAndersHW2(bpy.types.ShaderNodeCustomGroup):
    bl_idname = "ShaderNodeAndersHW2"
    bl_label = "Anders HW2 Material"
    bl_icon = "MATERIAL"

    def copy(self, source):
        copy_material_shader_state(self, source)

    @classmethod
    def poll(cls, tree):
        return tree.bl_idname == "ShaderNodeTree"

    def init(self, context):
        from .assignment import initialize_ugx_material_shader
        self.node_tree = ensure_hw_shader_group(6)
        self.width = 350
        self["hwek_material_ui_version"] = 1
        mat = material_for_ugx_shader_node(self, context)
        if mat is not None:
            initialize_ugx_material_shader(mat, self, 6)

    def draw_buttons(self, context, layout):
        from ..ui.material_nodes import draw_ugx_shader_node_controls
        draw_ugx_shader_node_controls(self, context, layout, 6)

    def draw_buttons_ext(self, context, layout):
        self.draw_buttons(context, layout)


@retain_registered_node_type
class ShaderNodeAndersHWDE(bpy.types.ShaderNodeCustomGroup):
    bl_idname = "ShaderNodeAndersHWDE"
    bl_label = "Anders HW1 Material"
    bl_icon = "MATERIAL"

    def copy(self, source):
        copy_material_shader_state(self, source)

    @classmethod
    def poll(cls, tree):
        return tree.bl_idname == "ShaderNodeTree"

    def init(self, context):
        from .assignment import initialize_ugx_material_shader
        self.node_tree = ensure_hw_shader_group(4)
        self.width = 350
        self["hwek_material_ui_version"] = 1
        mat = material_for_ugx_shader_node(self, context)
        if mat is not None:
            initialize_ugx_material_shader(mat, self, 4)

    def draw_buttons(self, context, layout):
        from ..ui.material_nodes import draw_ugx_shader_node_controls
        draw_ugx_shader_node_controls(self, context, layout, 4)

    def draw_buttons_ext(self, context, layout):
        self.draw_buttons(context, layout)


@retain_registered_node_type
class ShaderNodeAndersHW1PlayerColorMask(bpy.types.ShaderNodeCustomGroup):
    bl_idname = "ShaderNodeAndersHW1PlayerColorMask"
    bl_label = "Anders HW1 Player Color Mask"
    bl_icon = "MATERIAL"
    bl_description = "Extract an XF texture's green channel for the HW1 Player_color (xf) input"

    @classmethod
    def poll(cls, tree):
        return tree.bl_idname == "ShaderNodeTree"

    def init(self, context):
        tree = bpy.data.node_groups.new("Anders HW1 Player Color Mask", "ShaderNodeTree")
        self.node_tree = tree
        self.width = 220
        color = tree.interface.new_socket(name="Color", in_out="INPUT", socket_type="NodeSocketColor")
        color.default_value = (1.0, 1.0, 1.0, 1.0)
        color.description = "XF texture color. Green: 0 applies player color; 1 keeps diffuse color."
        tree.interface.new_socket(name="Mask", in_out="OUTPUT", socket_type="NodeSocketColor")
        inputs = tree.nodes.new("NodeGroupInput")
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        combine = tree.nodes.new("ShaderNodeCombineColor")
        output = tree.nodes.new("NodeGroupOutput")
        inputs.location, separate.location = (-420, 0), (-220, 0)
        combine.location, output.location = (0, 0), (200, 0)
        separate.mode = combine.mode = "RGB"
        tree.links.new(inputs.outputs["Color"], separate.inputs["Color"])
        for name in ("Red", "Green", "Blue"):
            tree.links.new(separate.outputs["Green"], combine.inputs[name])
        if "Alpha" in combine.inputs:
            combine.inputs["Alpha"].default_value = 1.0
        tree.links.new(combine.outputs["Color"], output.inputs["Mask"])


@retain_registered_node_type
class ShaderNodeAndersUVAnimation(bpy.types.ShaderNodeCustomGroup):
    bl_idname = "ShaderNodeAndersUVAnimation"
    bl_label = "Anders UV Animation"
    bl_icon = "UV"

    @classmethod
    def poll(cls, tree):
        return tree.bl_idname == "ShaderNodeTree"

    def init(self, context):
        tree = bpy.data.node_groups.new("Anders UV Animation", "ShaderNodeTree")
        self.node_tree = tree
        self.width = 220
        tree.interface.new_socket(name="Vector", in_out="INPUT", socket_type="NodeSocketVector")
        for name in ("Speed U", "Speed V", "Time"):
            tree.interface.new_socket(name=name, in_out="INPUT", socket_type="NodeSocketFloat")
        tree.interface.new_socket(name="UV", in_out="OUTPUT", socket_type="NodeSocketVector")
        nodes, links = tree.nodes, tree.links
        inputs = nodes.new("NodeGroupInput")
        output = nodes.new("NodeGroupOutput")
        offset = nodes.new("ShaderNodeCombineXYZ")
        for name, axis in (("Speed U", "X"), ("Speed V", "Y")):
            multiply = nodes.new("ShaderNodeMath")
            multiply.operation = "MULTIPLY"
            links.new(inputs.outputs[name], multiply.inputs[0])
            links.new(inputs.outputs["Time"], multiply.inputs[1])
            links.new(multiply.outputs[0], offset.inputs[axis])
        add = nodes.new("ShaderNodeVectorMath")
        add.operation = "ADD"
        links.new(inputs.outputs["Vector"], add.inputs[0])
        links.new(offset.outputs[0], add.inputs[1])
        links.new(add.outputs["Vector"], output.inputs["UV"])
        self.inputs["Time"].hide = True
        for name in ("Speed U", "Speed V", "Time"):
            self.inputs[name].hide_value = name == "Time"
        driver = self.inputs["Time"].driver_add("default_value").driver
        driver.expression = "(frame - 1) * fps_base / fps"
        for name, path in (("fps", "render.fps"), ("fps_base", "render.fps_base")):
            variable = driver.variables.new()
            variable.name = name
            variable.type = "SINGLE_PROP"
            variable.targets[0].id_type = "SCENE"
            variable.targets[0].id = (context or bpy.context).scene
            variable.targets[0].data_path = path

    def draw_buttons(self, context, layout):
        pass

    def draw_buttons_ext(self, context, layout):
        self.draw_buttons(context, layout)


@retain_registered_node_type
class ShaderNodeAndersParameters(bpy.types.ShaderNodeCustomGroup):
    bl_idname = "ShaderNodeAndersParameters"
    bl_label = "Anders Material Parameters"

    @classmethod
    def poll(cls, tree):
        return tree.bl_idname == "ShaderNodeTree"

    def draw_buttons(self, context, layout):
        if self.get("raw_buffer"):
            layout.prop(self, '["raw_buffer"]', text="Bytes")


@retain_registered_node_type
class ShaderNodeAndersShaderPasses(bpy.types.ShaderNodeCustomGroup):
    bl_idname = "ShaderNodeAndersShaderPasses"
    bl_label = "Anders Shader Passes"

    @classmethod
    def poll(cls, tree):
        return tree.bl_idname == "ShaderNodeTree"

    def draw_buttons(self, context, layout):
        from ..ui.panels import draw_ugx_material_controls
        mat = material_for_ugx_shader_node(self, context)
        if mat is not None:
            draw_ugx_material_controls(layout, getattr(context, "object", None), mat, "passes")


@retain_registered_node_type
class ShaderNodeAndersHW1UV(bpy.types.ShaderNodeCustomGroup):
    bl_idname = "ShaderNodeAndersHW1UV"
    bl_label = "Anders HW1 UV Animation"

    @classmethod
    def poll(cls, tree):
        return tree.bl_idname == "ShaderNodeTree"

    def get_clamp_u(self):
        return bool(int(self.inputs['Map Flags'].default_value) & 1)

    def set_clamp_u(self, value):
        self.inputs['Map Flags'].default_value = (int(self.inputs['Map Flags'].default_value) & ~1) | int(value)
        self.refresh_sampling()

    def get_clamp_v(self):
        return bool(int(self.inputs['Map Flags'].default_value) & 2)

    def set_clamp_v(self, value):
        self.inputs['Map Flags'].default_value = (int(self.inputs['Map Flags'].default_value) & ~2) | (int(value) << 1)
        self.refresh_sampling()

    clamp_u: BoolProperty(name='Clamp U', description='Native U clamp flag. The game repeats both axes if either axis wraps or this map scrolls', get=get_clamp_u, set=set_clamp_u)
    clamp_v: BoolProperty(name='Clamp V', description='Native V clamp flag. The game repeats both axes if either axis wraps or this map scrolls', get=get_clamp_v, set=set_clamp_v)

    def refresh_sampling(self):
        from .uv import refresh_hw1_texture_sampling
        mat = material_for_ugx_shader_node(self)
        if mat is not None:
            refresh_hw1_texture_sampling(mat)

    def draw_buttons(self, context, layout):
        row = layout.row(align=True)
        row.enabled = not self.inputs['Map Flags'].is_linked
        row.prop(self, 'clamp_u')
        row.prop(self, 'clamp_v')

    def init(self, context):
        ShaderNodeAndersUVAnimation.init(self, context)
        self.node_tree.interface.new_socket(name="Speed W", in_out="INPUT", socket_type="NodeSocketFloat")
        flags = self.node_tree.interface.new_socket(name="Map Flags", in_out="INPUT", socket_type="NodeSocketInt")
        flags.min_value, flags.max_value = 0, 65535
        inputs = next(node for node in self.node_tree.nodes if node.type == "GROUP_INPUT")
        offset = next(node for node in self.node_tree.nodes if node.type == "COMBXYZ")
        multiply = self.node_tree.nodes.new("ShaderNodeMath")
        multiply.operation = "MULTIPLY"
        self.node_tree.links.new(inputs.outputs["Speed W"], multiply.inputs[0])
        self.node_tree.links.new(inputs.outputs["Time"], multiply.inputs[1])
        self.node_tree.links.new(multiply.outputs[0], offset.inputs["Z"])
        source = offset.inputs['Y'].links[0].from_socket
        invert = self.node_tree.nodes.new('ShaderNodeMath')
        invert.operation = 'MULTIPLY'
        invert.inputs[1].default_value = -1.0
        self.node_tree.links.new(source, invert.inputs[0])
        self.node_tree.links.new(invert.outputs[0], offset.inputs['Y'])
        self.node_tree['hwek_hw1_uv_version'] = 2


def ensure_ugx_material_node_types():
    for node_type in (ShaderNodeAndersHW2, ShaderNodeAndersHWDE, ShaderNodeAndersHW1PlayerColorMask, ShaderNodeAndersUVAnimation, ShaderNodeAndersParameters, ShaderNodeAndersShaderPasses, ShaderNodeAndersHW1UV):
        if not node_type.is_registered:
            # Opening a blend can refresh Blender's Python wrapper for this RNA base.
            if not issubclass(node_type, bpy.types.ShaderNodeCustomGroup):
                node_type.__bases__ = (bpy.types.ShaderNodeCustomGroup,)
            bpy.utils.register_class(node_type)


def menu_ugx_material_nodes(self, context):
    space = getattr(context, "space_data", None)
    if space is None or space.type != "NODE_EDITOR" or space.tree_type != "ShaderNodeTree":
        return
    self.layout.separator()
    for node_type in (ShaderNodeAndersHW2, ShaderNodeAndersHWDE, ShaderNodeAndersHW1PlayerColorMask, ShaderNodeAndersUVAnimation, ShaderNodeAndersHW1UV):
        operator = self.layout.operator("node.add_node", text=node_type.bl_label, icon="MATERIAL")
        operator.type = node_type.bl_idname
        operator.use_transform = True
