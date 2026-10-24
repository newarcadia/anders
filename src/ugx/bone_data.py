"""Derive bone world transforms and Blender display matrices."""

from __future__ import annotations
from typing import List, Optional, Tuple
from ..formats.matrix import mat4_inverse, mat4_mul
from .data import Bone, UgxModel

__all__ = ('bone_world_matrices', 'matrices_close', 'cached_bones_match_granny_inverse_world', 'bone_display_matrices')

def bone_world_matrices(bones: List[Bone]) -> List[List[List[float]]]:
    worlds: List[Optional[List[List[float]]]] = [None] * len(bones)

    def build(index: int) -> List[List[float]]:
        cached = worlds[index]
        if cached is not None:
            return cached
        local = bones[index].matrix
        parent = bones[index].parent
        if 0 <= parent < len(bones) and parent != index:
            world = mat4_mul(local, build(parent))
        else:
            world = [row[:] for row in local]
        worlds[index] = world
        return world

    return [build(index) for index in range(len(bones))]


def matrices_close(left: List[List[float]], right: List[List[float]], epsilon: float = 0.001) -> bool:
    if len(left) != 4 or len(right) != 4:
        return False
    for row in range(4):
        if len(left[row]) != 4 or len(right[row]) != 4:
            return False
        for column in range(4):
            if abs(float(left[row][column]) - float(right[row][column])) > epsilon:
                return False
    return True


def cached_bones_match_granny_inverse_world(model: UgxModel) -> bool:
    if not model.granny_info.parsed or not model.bones or not model.granny_info.bones:
        return False
    compared = 0
    matched = 0
    for index in range(min(len(model.bones), len(model.granny_info.bones))):
        source_bone = model.bones[index]
        granny_bone = model.granny_info.bones[index]
        if source_bone.name != granny_bone.name:
            continue
        compared += 1
        if matrices_close(source_bone.matrix, granny_bone.inverse_world):
            matched += 1
    return compared > 0 and matched / compared >= 0.75


def bone_display_matrices(model: UgxModel) -> Tuple[str, List[List[List[float]]]]:
    first_name = model.bones[0].name.lower() if model.bones else ""
    native_pc = (model.cache_version in (4, 6) and model.bones and not model.bones[0].big_endian
                 and model.granny_info.parsed and model.granny_info.models)
    if native_pc or cached_bones_match_granny_inverse_world(model) or (model.cache_version == 4 and first_name == "grannyrootbone"):
        return "inverse_world", [mat4_inverse(bone.matrix) for bone in model.bones]
    return "local_hierarchy", bone_world_matrices(model.bones)
