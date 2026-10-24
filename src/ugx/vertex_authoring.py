"""Encode authored vertex positions, tangent frames and skin weights."""

from __future__ import annotations
from mathutils import Vector
from typing import Optional, Tuple
from .vertices import (
    normalized_tuple,
    write_half,
    write_hw2_packed_normal,
    write_hw2_packed_tangent,
    write_vec2,
    write_vec3,
)

__all__ = ('orthonormal_tangent_tuple', 'normalized_game_vector', 'tangent_for_game_normal', 'v4_tangent_with_handedness', 'encoded_new_v4_vertex', 'encoded_new_v4_skinned_vertex', 'encoded_new_v6_vertex', 'encoded_new_v6_skinned_vertex')

def orthonormal_tangent_tuple(
    normal: Tuple[float, float, float],
    tangent: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    n = Vector(normalized_tuple(normal, (0.0, 1.0, 0.0)))
    t = Vector(tangent)
    t -= n * t.dot(n)
    if t.length_squared <= 0.0000001:
        axis = Vector((0.0, 1.0, 0.0)) if abs(n.y) < 0.9 else Vector((1.0, 0.0, 0.0))
        t = axis.cross(n)
    if t.length_squared <= 0.0000001:
        return (1.0, 0.0, 0.0)
    t.normalize()
    return tuple(float(component) for component in t)


def normalized_game_vector(value: Tuple[float, float, float], fallback: Tuple[float, float, float]) -> Tuple[float, float, float]:
    vector = Vector(value)
    if vector.length_squared <= 0.0000001:
        return fallback
    vector.normalize()
    return tuple(float(component) for component in vector)


def tangent_for_game_normal(normal: Tuple[float, float, float]) -> Tuple[float, float, float]:
    n = Vector(normal)
    if n.length_squared <= 0.0000001:
        return (1.0, 0.0, 0.0)
    n.normalize()
    axis = Vector((0.0, 1.0, 0.0)) if abs(n.y) < 0.9 else Vector((1.0, 0.0, 0.0))
    tangent = axis.cross(n)
    if tangent.length_squared <= 0.0000001:
        return (1.0, 0.0, 0.0)
    tangent.normalize()
    return tuple(float(component) for component in tangent)


def v4_tangent_with_handedness(
    normal: Tuple[float, float, float],
    tangent: Optional[Tuple[float, float, float]] = None,
    bitangent_sign: float = 1.0,
) -> Tuple[float, float, float]:
    tangent = orthonormal_tangent_tuple(normal, tangent or tangent_for_game_normal(normal))
    scale = 0.5 if float(bitangent_sign) < 0.0 else 1.0
    return tuple(component * scale for component in tangent)


def encoded_new_v4_vertex(
    position: Tuple[float, float, float],
    normal: Tuple[float, float, float],
    uv: Tuple[float, float],
    tangent: Optional[Tuple[float, float, float]] = None,
    bitangent_sign: float = 1.0,
) -> bytes:
    normal = normalized_game_vector(normal, (0.0, 1.0, 0.0))
    vertex = bytearray(36)
    write_vec3(vertex, 0, 17, position)
    write_half(vertex, 6, 1.0)
    write_vec3(vertex, 8, 3, normal)
    write_vec3(vertex, 20, 3, v4_tangent_with_handedness(normal, tangent, bitangent_sign))
    write_vec2(vertex, 32, 16, uv)
    return bytes(vertex)


def encoded_new_v4_skinned_vertex(
    position: Tuple[float, float, float],
    normal: Tuple[float, float, float],
    uv: Tuple[float, float],
    raw_indices: bytes,
    raw_weights: bytes,
    tangent: Optional[Tuple[float, float, float]] = None,
    bitangent_sign: float = 1.0,
) -> bytes:
    normal = normalized_game_vector(normal, (0.0, 1.0, 0.0))
    vertex = bytearray(44)
    write_vec3(vertex, 0, 17, position)
    write_half(vertex, 6, 1.0)
    write_vec3(vertex, 8, 3, normal)
    write_vec3(vertex, 20, 3, v4_tangent_with_handedness(normal, tangent, bitangent_sign))
    vertex[32:36] = raw_indices[:4].ljust(4, b"\0")
    vertex[36:40] = raw_weights[:4].ljust(4, b"\0")
    write_vec2(vertex, 40, 16, uv)
    return bytes(vertex)


def encoded_new_v6_vertex(
    position: Tuple[float, float, float],
    normal: Tuple[float, float, float],
    uv: Tuple[float, float],
    tangent: Optional[Tuple[float, float, float]] = None,
    bitangent_sign: float = 1.0,
) -> bytes:
    normal = normalized_game_vector(normal, (0.0, 1.0, 0.0))
    tangent = orthonormal_tangent_tuple(normal, tangent or tangent_for_game_normal(normal))
    vertex = bytearray(20)
    write_vec3(vertex, 0, 17, position)
    write_half(vertex, 6, 1.0)
    write_vec2(vertex, 8, 16, uv)
    write_hw2_packed_tangent(vertex, 12, tangent)
    write_hw2_packed_normal(vertex, 16, normal, bitangent_sign)
    return bytes(vertex)


def encoded_new_v6_skinned_vertex(
    position: Tuple[float, float, float],
    normal: Tuple[float, float, float],
    uv: Tuple[float, float],
    raw_indices: bytes,
    raw_weights: bytes,
    tangent: Optional[Tuple[float, float, float]] = None,
    bitangent_sign: float = 1.0,
) -> bytes:
    vertex = bytearray(28)
    vertex[:20] = encoded_new_v6_vertex(position, normal, uv, tangent, bitangent_sign)
    vertex[20:24] = raw_indices[:4].ljust(4, b"\0")
    vertex[24:28] = raw_weights[:4].ljust(4, b"\0")
    return bytes(vertex)
