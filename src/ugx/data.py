"""Native model records, vertex layouts and instance settings."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from ..errors import UgxError
from ..formats.granny import GrannyInfo
from ..materials.data import Material

__all__ = ('CACHED_SIGNATURE_HW1_BE', 'CACHED_SIGNATURE_HW1_LE', 'CACHED_SIGNATURE_HW1_PC_BYTES', 'CACHED_SIGNATURE_HW2', 'UGX_NO_RIGID_BONE', 'UV_SEMANTICS', 'V6_PACKED_BASIS_TYPE', 'UGX_MAX_SECTION_TRIANGLES', 'V4_MAX_SKELETON_BONES', 'V6_MAX_SKELETON_BONES', 'V4_MAX_SECTION_VERTICES', 'V6_MAX_SECTION_VERTICES', 'is_no_rigid_bone_index', 'VertexLayout', 'Vertex', 'Section', 'Bone', 'Accessory', 'CachedBounds', 'UgxModel', 'ugx_game_name', 'source_vertex_section_ranges', 'v4_instance_settings', 'v6_instance_settings')

CACHED_SIGNATURE_HW1_BE = 0xC2340004


CACHED_SIGNATURE_HW1_LE = 0x040034C2


CACHED_SIGNATURE_HW1_PC_BYTES = b"\x04\x00\x34\xC2"


CACHED_SIGNATURE_HW2 = 0x060034C2


UGX_NO_RIGID_BONE = 0x7FFFFFFF


UV_SEMANTICS = (("T0", "UV0", "uv0"), ("T1", "UV1", "uv1"), ("T2", "UV2", "uv2"))


# Private logical layout type. HW2 v6 stores a complete tangent-space vector in
# one float32; this is not an extra UnivertPacker enum value written to UGX.
V6_PACKED_BASIS_TYPE = 21


UGX_MAX_SECTION_TRIANGLES = 8192


V4_MAX_SKELETON_BONES = 127


# HW2 stores the runtime model's bone count in one byte.
V6_MAX_SKELETON_BONES = 255


V4_MAX_SECTION_VERTICES = 0x3FFF


V6_MAX_SECTION_VERTICES = 0x7FFF


def is_no_rigid_bone_index(value: Any) -> bool:
    try:
        index = int(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return index < 0 or index == UGX_NO_RIGID_BONE


@dataclass
class VertexLayout:
    semantic: str
    type_id: int
    offset: int
    size: int


@dataclass
class Vertex:
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    tangent: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    tangent_bits: int = 1
    bitangent_sign: float = 1.0
    uv0: Tuple[float, float] = (0.0, 0.0)
    uv1: Tuple[float, float] = (0.0, 0.0)
    uv2: Tuple[float, float] = (0.0, 0.0)
    bone_indices: Tuple[int, int, int, int] = (0, 0, 0, 0)
    bone_weights: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


@dataclass
class Section:
    cache_version: int = 0
    vertices: List[Vertex] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    index_offset: int = 0
    vertex_offset: int = 0
    vertex_buffer_size: int = 0
    vertex_file_offset: int = 0
    vertex_count: int = 0
    section_record_file_offset: int = 0
    index_offset_field_offset: int = 0
    tri_count_field_offset: int = 0
    vertex_offset_field_offset: int = 0
    vertex_size_field_offset: int = 0
    vertex_count_field_offset: int = 0
    material_index: int = -1
    material_index_offset: int = 0
    accessory_index: int = -1
    accessory_index_offset: int = 0
    accessory_first_bone: int = -1
    accessory_bone_count: int = 0
    big_endian: bool = False
    max_bones: int = 0
    max_bones_offset: int = 0
    rigid_bone_index: int = -1
    rigid_bone_index_offset: int = 0
    rigid_only: bool = False
    rigid_only_offset: int = 0
    lod_min_distance: float = 0.0
    lod_max_distance: float = 0.0
    sorting_bias: float = 0.0
    lod_min_distance_offset: int = 0
    lod_max_distance_offset: int = 0
    sorting_bias_offset: int = 0
    vertex_stride: int = 0
    pack_order: str = ""
    decl_order: str = ""
    layout: List[VertexLayout] = field(default_factory=list)
    local_to_global_bones: List[int] = field(default_factory=list)
    source_vertex_data: bytes = b""


@dataclass
class Bone:
    name: str
    matrix: List[List[float]]
    parent: int
    name_offset: int = 0
    name_capacity: int = 0
    name_offset_field_offset: int = 0
    name_offset_field_size: int = 0
    matrix_offset: int = 0
    big_endian: bool = False
    bounds_low: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_high: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_low_offset: int = 0
    bounds_high_offset: int = 0


@dataclass
class Accessory:
    first_bone: int = 0
    bone_count: int = 0
    section_indices: List[int] = field(default_factory=list)


@dataclass
class CachedBounds:
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    bounds_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    center_offset: int = 0
    radius_offset: int = 0
    bounds_min_offset: int = 0
    bounds_max_offset: int = 0
    big_endian: bool = False


@dataclass
class UgxModel:
    path: str
    original_bytes: bytes
    cache_version: int
    sections: List[Section]
    bones: List[Bone]
    materials: List[Material]
    cached_bounds: CachedBounds = field(default_factory=CachedBounds)
    accessories: List[Accessory] = field(default_factory=list)
    granny_info: GrannyInfo = field(default_factory=GrannyInfo)


def ugx_game_name(cache_version: int) -> str:
    if cache_version == 4:
        return "Halo Wars: Definitive Edition"
    if cache_version == 6:
        return "Halo Wars 2"
    return f"UGX v{cache_version}"


def source_vertex_section_ranges(model: UgxModel) -> List[Tuple[int, int]]:
    ranges = []
    cursor = 0
    for section in model.sections:
        start = cursor
        cursor += len(section.vertices)
        ranges.append((start, cursor))
    return ranges


def v4_instance_settings(section_infos: List[Dict]) -> Tuple[int, int]:
    max_section_vertices = max(
        (max(1, int(info["vertex_count"])) for info in section_infos),
        default=1,
    )
    instance_index_multiplier = 4
    while instance_index_multiplier < max_section_vertices:
        instance_index_multiplier <<= 1
    instance_index_multiplier = min(instance_index_multiplier, 0x4000)
    total_vertices = sum(max(0, int(info["vertex_count"])) for info in section_infos)
    max_instances = (
        4
        if instance_index_multiplier <= 0x1000 and total_vertices <= 0x1000
        else 1
    )
    if max_instances * instance_index_multiplier >= 0xFFFF:
        raise UgxError("New v4 UGX instance indices exceed the 16-bit index-buffer range.")
    return max_instances, instance_index_multiplier


def v6_instance_settings(section_infos: List[Dict]) -> Tuple[int, int]:
    """Match BUGXGeomInstancer's HW2 16-bit instanced-index policy."""
    max_local_index = max(
        (max(0, int(info.get("vertex_count", 0)) - 1) for info in section_infos),
        default=0,
    )
    instance_index_multiplier = 1
    while instance_index_multiplier <= max_local_index:
        instance_index_multiplier <<= 1
    if instance_index_multiplier > 0x8000:
        raise UgxError("New HW2 UGX section needs an instance-index multiplier larger than uint16 can encode.")

    max_instances = 1
    total_triangles = sum(max(0, int(info.get("tri_count", 0))) for info in section_infos)
    if total_triangles <= 2048:
        possible = max(1, 65534 // instance_index_multiplier)
        power_of_two = 1
        while power_of_two * 2 <= possible:
            power_of_two *= 2
        max_instances = max(1, min(4, power_of_two))
    if max_instances * instance_index_multiplier >= 0xFFFF:
        raise UgxError("New HW2 UGX instance indices exceed the 16-bit index-buffer range.")
    return max_instances, instance_index_multiplier
