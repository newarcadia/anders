"""Blender operators for UAX animation import and export."""

from bpy.props import BoolProperty, CollectionProperty, StringProperty
from bpy.types import OperatorFileListElement
from bpy_extras.io_utils import ExportHelper, ImportHelper, orientation_helper
from typing import List
import bpy
import os
from .export import export_uax
from .scene import activate_uax_action, import_uax, uax_context_armature, uax_target_armature
from .loading import queue_pending_uax_actions

__all__ = ('HWEK_OT_import_uax', 'HWEK_OT_export_uax')

class HWEK_OT_import_uax(bpy.types.Operator, ImportHelper):
    bl_idname = "hwek.import_uax"
    bl_label = "Import UAX Animation"
    filename_ext = ".uax"
    filter_glob: StringProperty(default="*.uax;*.UAX", options={"HIDDEN"})
    files: CollectionProperty(type=OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN", "SKIP_SAVE"})
    keep_matrix_source: BoolProperty(
        name="Keep Matrix Source Action",
        description="Also create the exact baked matrix_basis source action. This is slower and usually only needed for importer debugging.",
        default=False,
    )

    def execute(self, context):
        paths = [
            os.path.join(self.directory, item.name)
            for item in self.files
            if item.name.lower().endswith(".uax")
        ]
        if not paths:
            paths = [self.filepath]
        imported: List[bpy.types.Action] = []
        armature_target = uax_target_armature(getattr(context, "object", None))
        deferred = armature_target is not None and not self.keep_matrix_source
        try:
            for path in paths:
                imported.extend(import_uax(
                    path,
                    getattr(context, "object", None),
                    self.keep_matrix_source,
                    activate=False,
                    defer=deferred,
                ))
            if imported:
                activate_uax_action(armature_target, imported[0])
                queue_pending_uax_actions(imported[:1], active=True)
            self.report({"INFO"}, f"Imported {len(imported)} animation action(s) from {len(paths)} UAX file(s).")
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


@orientation_helper(axis_forward="Z", axis_up="Y")
class HWEK_OT_export_uax(bpy.types.Operator, ExportHelper):
    bl_idname = "hwek.export_uax"
    bl_label = "Export Animation UAX"
    bl_description = "Export the active action, retaining the source format and auxiliary tracks of imported UAX animations"
    filename_ext = ".uax"
    filter_glob: StringProperty(default="*.uax", options={"HIDDEN"})

    def draw(self, context):
        layout = self.layout
        layout.label(text="Animation axes in Blender")
        row = layout.row(align=True)
        row.prop(self, "axis_forward", text="Forward")
        row.prop(self, "axis_up", text="Up")
        layout.label(text="Exports to Halo Wars +Z forward, +Y up.", icon="INFO")

    @classmethod
    def poll(cls, context):
        armature = uax_context_armature(context)
        animation_data = getattr(armature, "animation_data", None) if armature is not None else None
        return armature is not None and getattr(animation_data, "action", None) is not None

    def execute(self, context):
        armature = uax_context_armature(context)
        animation_data = getattr(armature, "animation_data", None) if armature is not None else None
        action = getattr(animation_data, "action", None)
        if armature is None or action is None:
            self.report({"ERROR"}, "Select an armature with an active action before exporting UAX.")
            return {"CANCELLED"}
        try:
            result = export_uax(
                armature,
                action,
                self.filepath,
                self.axis_forward,
                self.axis_up,
            )
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, (
            f"Exported {result['tracks']} tracks and {result['samples']} samples "
            f"at {result['fps']:.3f} fps."
        ))
        return {"FINISHED"}
