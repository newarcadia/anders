"""Blender operators for removing source metadata and porting models."""

from bpy.props import IntProperty
import bpy
from .cleanup import object_scope_has_ugx_data, remove_ugx_data_for_port, ugx_portable_object_scope
from .hw1_to_hw2 import port_hw1_asset_to_hw2

__all__ = ('HWEK_OT_remove_ugx_data', 'HWEK_OT_port_hw1_to_hw2', 'HWEK_OT_view_hw2_port_report')

class HWEK_OT_remove_ugx_data(bpy.types.Operator):
    bl_idname = "hwek.remove_ugx_data"
    bl_label = "Remove UGX Data for Cross-Game Port"
    bl_description = "Detach imported model records in the active object's collection while preserving native part names, shader settings, textures, weights, and animation"
    bl_options = {"REGISTER", "UNDO"}

    @staticmethod
    def object_collection(context):
        obj = context.object
        if obj is None:
            return None
        collection = context.collection
        if collection is not None and obj.name in collection.objects:
            return collection
        return next(iter(obj.users_collection), None)

    @classmethod
    def poll(cls, context):
        collection = cls.object_collection(context)
        return collection is not None and object_scope_has_ugx_data(collection.objects)

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        collection = self.object_collection(context)
        if collection is None:
            self.report({"ERROR"}, "The active object has no collection.")
            return {"CANCELLED"}
        stats = remove_ugx_data_for_port(collection.objects)
        self.report({"INFO"}, (
            f"Detached UGX source data from {stats['objects']} object(s) in collection '{collection.name}'. "
            "Materials and weights were kept."
        ))
        return {"FINISHED"}


class HWEK_OT_port_hw1_to_hw2(bpy.types.Operator):
    bl_idname = "hwek.port_hw1_to_hw2"
    bl_label = "Port HW1 Asset to HW2"
    bl_description = "Create an HW2 copy with converted team masks, emissive textures, shader passes and a separate rig"
    bl_options = {"REGISTER", "UNDO"}

    resolution: IntProperty(name="Mask and Bake Resolution", default=1024, min=8, max=8192)

    @classmethod
    def poll(cls, context):
        obj = getattr(context, "object", None)
        return obj is not None and obj.mode == "OBJECT" and any(
            item.type == "MESH" and int(item.get("hwek_ugx_cache_version", 0)) == 4
            for item in ugx_portable_object_scope([obj]))

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        self.layout.prop(self, "resolution")
        self.layout.label(text="Creates a separate HW2 copy; hides the HW1 source.")
        self.layout.label(text="Converts team masks and includes emissive textures.")
        self.layout.label(text="Extra material effects are listed in the port report.")

    def execute(self, context):
        try:
            result = port_hw1_asset_to_hw2(context, context.object, self.resolution)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"WARNING"} if result["notes"] else {"INFO"},
                    f"Created {result['meshes']} HW2 meshes and {result['materials']} materials. Check the Port Report, then export HW2 UGX.")
        return {"FINISHED"}


class HWEK_OT_view_hw2_port_report(bpy.types.Operator):
    bl_idname = "hwek.view_hw2_port_report"
    bl_label = "Port Report"

    @classmethod
    def poll(cls, context):
        obj = getattr(context, "object", None)
        return obj is not None and bool(obj.get("hwek_ugx_port_report"))

    def execute(self, context):
        report = bpy.data.texts.get(str(context.object.get("hwek_ugx_port_report", "")))
        if report is None:
            self.report({"ERROR"}, "This object has no saved port report.")
            return {"CANCELLED"}
        lines = report.as_string().splitlines()
        def draw(menu, _context):
            for line in lines:
                menu.layout.label(text=line)
        context.window_manager.popup_menu(draw, title="HW2 Port Report", icon="INFO")
        return {"FINISHED"}
