"""Convert positions, triangle winding and UVs between game and Blender."""

from __future__ import annotations
from typing import Tuple

__all__ = ('game_to_blender_vec3', 'blender_to_game_vec3', 'reverse_triangle_winding', 'game_to_blender_uv', 'blender_to_game_uv', 'game_to_blender_uv_for_version', 'blender_to_game_uv_for_version')

def game_to_blender_vec3(value: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y, z = value
    # Halo Wars UGX data is left-handed. Keep the established Y-up to Z-up
    # rotation, then reflect X so asymmetric meshes are right-handed in Blender.
    return -x, -z, y


def blender_to_game_vec3(value: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y, z = value
    return -x, z, -y


def reverse_triangle_winding(values):
    """Convert triangle winding across the handedness-changing reflection."""
    return values[0], values[2], values[1]


def game_to_blender_uv(value: Tuple[float, float]) -> Tuple[float, float]:
    return value[0], 1.0 - value[1]


def blender_to_game_uv(value: Tuple[float, float]) -> Tuple[float, float]:
    return value[0], 1.0 - value[1]


def game_to_blender_uv_for_version(value: Tuple[float, float], cache_version: int) -> Tuple[float, float]:
    return game_to_blender_uv(value)


def blender_to_game_uv_for_version(value: Tuple[float, float], cache_version: int) -> Tuple[float, float]:
    return blender_to_game_uv(value)
