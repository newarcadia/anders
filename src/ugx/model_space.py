"""Separate VIS attachment placement from editable model coordinates."""
import bpy
from mathutils import Matrix

__all__ = ('ugx_model_matrix', 'ugx_model_transform_objects')


def ugx_model_transform_objects(obj):
    anchor = obj.get('hwek_vis_anchor')
    current = obj
    while current is not None and current != anchor:
        yield current
        current = current.parent
    if isinstance(anchor, bpy.types.Object) and current is None:
        # A reparented model uses anchor-relative world space, so both
        # branches contribute to its exported transform.
        current = anchor
        while current is not None:
            yield current
            current = current.parent


def ugx_model_matrix(obj):
    anchor = obj.get('hwek_vis_anchor')
    if not isinstance(anchor, bpy.types.Object):
        return obj.matrix_world
    matrix = Matrix.Identity(4)
    current = obj
    while current != anchor:
        if current is None or current.constraints or current.parent_type != 'OBJECT':
            return anchor.matrix_world.inverted_safe() @ obj.matrix_world
        matrix = current.matrix_parent_inverse @ current.matrix_basis @ matrix
        current = current.parent
    return matrix
