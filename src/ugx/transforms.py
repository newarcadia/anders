"""Convert native transforms and preserve bone export axes."""

from bpy_extras.io_utils import axis_conversion
from mathutils import Matrix, Vector
from typing import List, Optional, Tuple
import bpy
from .model_space import ugx_model_matrix
import json
import math
from ..errors import UgxError
from ..formats.granny import GrannyTransform
from ..formats.matrix import mat3_inverse, mat3_mul, mat4_inverse, mat4_mul
from .coordinates import blender_to_game_vec3, game_to_blender_vec3
from .mesh import object_matrix_is_identity
from .vertices import changed, write_f32_endian

__all__ = ('game_rows_to_blender_matrix', 'granny_transform_from_local_rows', 'granny_transform_matches_local_rows', 'uax_export_axis_rows', 'ugx_export_axis_correction_rows', 'parse_matrix_rows', 'matrix_changed', 'matrix3_is_identity', 'blender_matrix3_to_game_rows', 'apply_game_delta_rows', 'game_orientation_delta_rows', 'pose_basis_delta_rows', 'blender_matrix_to_game_rows', 'bone_rest_display_rows', 'native_bone_export_matrix', 'write_matrix_rows', 'parse_vec3', 'bone_head_export_position')

def game_rows_to_blender_matrix(rows: List[List[float]]) -> Matrix:
    axis = Matrix(((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
    # ``axis`` changes handedness, while Blender edit-bone matrices must be
    # proper rotation frames.  Reflect the bone-local X axis as well so the
    # two reflections cancel.  This keeps local Y pointing along the visible
    # Blender bone and, critically, preserves the determinant of the game
    # transform.  Using a +90-degree rotation here left one uncompensated
    # reflection: imported bone rolls were unstable and every newly-authored
    # armature exported negative-determinant inverse-world matrices that the
    # HWDE Granny/render path never emits for ordinary skeletons.
    bone_local = Matrix(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    game_column = Matrix((
        (rows[0][0], rows[1][0], rows[2][0]),
        (rows[0][1], rows[1][1], rows[2][1]),
        (rows[0][2], rows[1][2], rows[2][2]),
    ))
    result = (axis @ game_column @ bone_local).to_4x4()
    result.translation = Vector(game_to_blender_vec3(tuple(rows[3][:3])))
    return result


def granny_transform_from_local_rows(rows: List[List[float]]) -> GrannyTransform:
    """Encode a row-vector local matrix as Granny.bt's granny_transform."""
    basis = Matrix(tuple(
        tuple(float(rows[row][column]) for column in range(3))
        for row in range(3)
    )).transposed()
    orientation = basis.to_quaternion().normalized()
    rotation = orientation.to_matrix()
    scale_shear_matrix = rotation.inverted() @ basis
    position = tuple(float(rows[3][axis]) for axis in range(3))
    scale_shear = tuple(
        tuple(float(scale_shear_matrix[row][column]) for column in range(3))
        for row in range(3)
    )
    flags = 0
    # Native flags select which stored components contribute to the bind pose.
    # Even a small nonidentity value must stay active.
    if any(component != 0.0 for component in position):
        flags |= 0x1
    if (
        float(orientation.x) != 0.0
        or float(orientation.y) != 0.0
        or float(orientation.z) != 0.0
        or abs(float(orientation.w)) != 1.0
    ):
        flags |= 0x2
    if any(
        scale_shear[row][column] != (1.0 if row == column else 0.0)
        for row in range(3)
        for column in range(3)
    ):
        flags |= 0x4
    return GrannyTransform(
        flags,
        position,
        (
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        ),
        scale_shear,
    )


def granny_transform_matches_local_rows(transform: GrannyTransform, rows: List[List[float]], epsilon: float = 0.001) -> bool:
    expected = granny_transform_from_local_rows(rows)
    if transform.flags != expected.flags:
        return False
    if changed(transform.position, expected.position, epsilon):
        return False
    actual_orientation = tuple(float(component) for component in transform.orientation)
    expected_orientation = tuple(float(component) for component in expected.orientation)
    actual_length = math.sqrt(sum(component * component for component in actual_orientation))
    expected_length = math.sqrt(sum(component * component for component in expected_orientation))
    if actual_length <= 0.000001 or expected_length <= 0.000001:
        return False
    orientation_dot = sum(
        actual_orientation[index] * expected_orientation[index]
        for index in range(4)
    ) / (actual_length * expected_length)
    if abs(abs(orientation_dot) - 1.0) > epsilon:
        return False
    return not any(
        changed(transform.scale_shear[row], expected.scale_shear[row], epsilon)
        for row in range(3)
    )


def uax_export_axis_rows(axis_forward: str = "Z", axis_up: str = "Y") -> List[List[float]]:
    """Convert the chosen Blender animation basis to Halo's +Z-forward, +Y-up basis."""
    try:
        correction = axis_conversion(
            from_forward=str(axis_forward),
            from_up=str(axis_up),
            to_forward="Z",
            to_up="Y",
        ).to_4x4()
    except (TypeError, ValueError) as exc:
        raise UgxError(f"Invalid UAX export axes: forward {axis_forward}, up {axis_up}.") from exc
    # axis_conversion returns a column-vector matrix. Granny/Halo transforms in
    # this module use row vectors, so transpose it before right-multiplication.
    return [
        [float(correction[column][row]) for column in range(4)]
        for row in range(4)
    ]


def ugx_export_axis_correction_rows(
    axis_forward: str = "Z",
    axis_up: str = "Y",
) -> List[List[float]]:
    """Move the animation-basis correction into UGX skeleton bind data."""
    return uax_export_axis_rows(axis_forward, axis_up)


def parse_matrix_rows(value, label: str) -> List[List[float]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise UgxError(f"{label} is not valid JSON.") from exc
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise UgxError(f"{label} must be a 4x4 matrix.")
    rows: List[List[float]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise UgxError(f"{label} must be a 4x4 matrix.")
        floats = [float(component) for component in row]
        if not all(math.isfinite(component) for component in floats):
            raise UgxError(f"{label} contains a non-finite float.")
        rows.append(floats)
    return rows


def matrix_changed(current: List[List[float]], source: List[List[float]], epsilon: float = 0.00001) -> bool:
    return any(changed(current[row], source[row], epsilon) for row in range(4))


def matrix3_is_identity(matrix: Matrix, epsilon: float = 0.000001) -> bool:
    for row in range(3):
        for column in range(3):
            expected = 1.0 if row == column else 0.0
            if abs(float(matrix[row][column]) - expected) > epsilon:
                return False
    return True


def blender_matrix3_to_game_rows(matrix: Matrix) -> List[List[float]]:
    axis = Matrix(((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
    # Must be the inverse pair of game_rows_to_blender_matrix.  Both basis
    # changes are reflections, so proper Blender bone rotations remain proper
    # game rotations instead of becoming mirrored matrices.
    bone_local = Matrix(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    game_column = axis.inverted() @ matrix @ bone_local.inverted()
    return [
        [float(game_column[0][column]), float(game_column[1][column]), float(game_column[2][column])]
        for column in range(3)
    ]


def apply_game_delta_rows(rows: List[List[float]], delta: List[List[float]]) -> List[List[float]]:
    result = [row[:] for row in rows]
    rotated = mat3_mul(delta, [row[:3] for row in rows[:3]])
    for row in range(3):
        result[row][0], result[row][1], result[row][2] = rotated[row]
    return result


def game_orientation_delta_rows(current: List[List[float]], source: List[List[float]]) -> List[List[float]]:
    return mat3_mul([row[:3] for row in current[:3]], mat3_inverse([row[:3] for row in source[:3]]))


def pose_basis_delta_rows(pose_bone) -> Optional[List[List[float]]]:
    if not pose_bone:
        return None
    animation = pose_bone.id_data.animation_data
    if animation is not None and (animation.action is not None or animation.nla_tracks):
        return None
    basis = pose_bone.matrix_basis.to_3x3()
    baseline = getattr(pose_bone, 'bone', {}).get('hwek_ugx_source_pose_basis')
    if baseline:
        basis = Matrix(json.loads(baseline)).inverted_safe() @ basis
    if matrix3_is_identity(basis):
        return None
    # matrix_basis is a bone-local delta, not a model-space bone frame.
    # Conjugate by the bone-local basis on both sides so identity remains
    # identity. The full frame conversion adds a quarter turn even for tiny
    # pose rotations left behind by editing an imported armature.
    bone_local = Matrix(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    game_column = bone_local @ basis @ bone_local
    return [[float(game_column[column][row]) for column in range(3)] for row in range(3)]


def blender_matrix_to_game_rows(matrix: Matrix) -> List[List[float]]:
    rows = [row + [0.0] for row in blender_matrix3_to_game_rows(matrix.to_3x3())]
    rows.append(list(blender_to_game_vec3(tuple(float(component) for component in matrix.translation))) + [1.0])
    return rows


def bone_rest_display_rows(armature: bpy.types.Object, bone: bpy.types.Bone) -> List[List[float]]:
    matrix = bone.matrix_local.copy()
    if armature and not object_matrix_is_identity(armature):
        matrix = ugx_model_matrix(armature) @ matrix
    rows = blender_matrix_to_game_rows(matrix)
    head = bone.head_local.copy()
    if armature and not object_matrix_is_identity(armature):
        head = ugx_model_matrix(armature) @ head
    rows[3][0], rows[3][1], rows[3][2] = blender_to_game_vec3(tuple(float(component) for component in head))
    return rows


def native_bone_export_matrix(armature, bone, source_rows=None):
    explicit = bone.get('hwek_ugx_export_matrix')
    if explicit:
        rows = parse_matrix_rows(explicit, f'Bone {bone.name} export matrix')
        if source_rows is None or rows != source_rows:
            return rows
    animation = armature.animation_data
    animated = animation is not None and (animation.action is not None or animation.nla_tracks)
    pose = armature.pose.bones.get(bone.name)
    current = (blender_matrix_to_game_rows(ugx_model_matrix(armature) @ pose.matrix)
               if pose is not None and not animated else bone_rest_display_rows(armature,bone))
    if source_rows is None:
        return mat4_inverse(current)
    baseline = (bone.get('hwek_ugx_import_pose_display_matrix') if not animated else None)
    baseline = baseline or bone.get('hwek_ugx_import_display_matrix')
    if not baseline:
        return mat4_inverse(current)
    reference = parse_matrix_rows(baseline, f'Bone {bone.name} imported frame')
    if not matrix_changed(current, reference):
        return source_rows
    return mat4_inverse(mat4_mul(mat4_mul(mat4_inverse(source_rows),mat4_inverse(reference)),current))


def write_matrix_rows(data: bytearray, offset: int, rows: List[List[float]], big: bool):
    if offset + 64 > len(data):
        raise UgxError("Bone matrix patch points outside the UGX file.")
    for row in range(4):
        for column in range(4):
            write_f32_endian(data, offset + (row * 4 + column) * 4, rows[row][column], big)


def parse_vec3(value, label: str) -> Tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise UgxError(f"{label} must be a 3D vector.")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise UgxError(f"{label} contains a non-finite float.")
    return result


def bone_head_export_position(armature: bpy.types.Object, bone: bpy.types.Bone) -> Tuple[float, float, float]:
    pose_bone = armature.pose.bones.get(bone.name) if armature and armature.pose else None
    animation = armature.animation_data if armature else None
    animated = animation is not None and (animation.action is not None or animation.nla_tracks)
    # Previewing an action must not bake its current frame into the bind skeleton.
    if pose_bone and not animated and not pose_bone.matrix_basis.is_identity:
        head = ugx_model_matrix(armature) @ pose_bone.head
        return (float(head.x), float(head.y), float(head.z))
    if object_matrix_is_identity(armature):
        return (float(bone.head_local.x), float(bone.head_local.y), float(bone.head_local.z))
    head = ugx_model_matrix(armature) @ bone.head_local
    return (float(head.x), float(head.y), float(head.z))
