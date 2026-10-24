"""Native skeleton LOD estimation and authored error constraints."""
from typing import Any, List, Optional
import math

from ..errors import UgxError
from ..formats.granny_transform import inverse_scale_shear
from ..formats.matrix import mat4_mul, uax_matrix4_identity

__all__ = ('estimate_granny_lod_errors',)


def estimate_granny_lod_errors(bones: List[Any], lod_errors: Optional[List[float]] = None) -> List[float]:
    """Match Granny's EstimatedLOD skeleton-builder policy without the SDK."""
    if lod_errors is not None and (len(lod_errors) != len(bones) or any(not math.isfinite(value) or value > 3.4028234663852886e38 for value in lod_errors)):
        raise UgxError("Native bone LOD overrides must be finite and match its bone count.")
    if not bones:
        return []

    parents: List[int] = []
    inverse_worlds: List[List[List[float]]] = []
    for bone in bones:
        if isinstance(bone, dict):
            parents.append(int(bone.get("parent", -1)))
            rows = bone.get("matrix", uax_matrix4_identity())
        else:
            parents.append(int(getattr(bone, "parent", -1)))
            rows = getattr(bone, "matrix", uax_matrix4_identity())
        inverse_worlds.append([[float(value) for value in row] for row in rows])
    state = [0] * len(bones)
    for index, parent in enumerate(parents):
        if parent < -1 or parent >= len(bones) or parent == index:
            raise UgxError(f"Bone {index}: its native parent index is invalid.")
    for index in range(len(bones)):
        walk = index
        while walk >= 0 and state[walk] == 0:
            state[walk] = 1
            walk = parents[walk]
        if walk >= 0 and state[walk] == 1:
            raise UgxError("Native skeleton parent relationships contain a cycle.")
        walk = index
        while walk >= 0 and state[walk] == 1:
            state[walk] = 2
            walk = parents[walk]
    local_rows: List[List[List[float]]] = []
    for index, rows in enumerate(inverse_worlds):
        if any(not math.isfinite(value) for row in rows for value in row) or any(rows[row][3] != float(row == 3) for row in range(4)):
            raise UgxError(f"Bone {index}: its inverse bind matrix must be finite and affine.")
        inverse = inverse_scale_shear([row[:3] for row in rows[:3]])
        world = [row + [0.] for row in inverse] + [[-sum(rows[3][i]*inverse[i][j] for i in range(3)) for j in range(3)] + [1.]]
        parent = parents[index]
        local_rows.append(mat4_mul(world, inverse_worlds[parent]) if parent >= 0 else world)

    # granny_skeleton_builder.cpp::EstimateSkeletonLOD starts with the child's
    # LocalTransform.Position. At each ancestor it records that vector length,
    # then applies only the ancestor's scale/shear and rotation. Ancestor
    # translations are deliberately not accumulated.
    errors = [-1.0 for _ in bones]
    for bone_index in range(len(bones)):
        offset = [float(value) for value in local_rows[bone_index][3][:3]]
        walk = parents[bone_index]
        while walk >= 0:
            estimated = math.sqrt(sum(float(value) * float(value) for value in offset))
            if not math.isfinite(estimated) or estimated > 3.4028234663852886e38:
                raise UgxError("Native bone LOD error exceeds its float range.")
            errors[walk] = max(errors[walk], estimated)
            ancestor = local_rows[walk]
            offset = [
                sum(float(offset[row]) * float(ancestor[row][column]) for row in range(3))
                for column in range(3)
            ]
            walk = parents[walk]

    if lod_errors is not None:
        errors = [value if value >= 0.0 else estimated for value, estimated in zip(lod_errors, errors)]

    # Granny assigns a leaf half its parent's error, then enforces the max-heap
    # property needed by GetBoneCountForLOD's prefix search.
    for bone_index in range(len(bones)):
        if errors[bone_index] < 0.0:
            parent = parents[bone_index]
            errors[bone_index] = errors[parent] * 0.5 if 0 <= parent < len(bones) else 0.0
        walk = bone_index
        while parents[walk] >= 0:
            parent = parents[walk]
            errors[parent] = max(errors[parent], errors[walk])
            walk = parent
    return errors
