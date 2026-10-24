"""Read and write Granny model records, transforms and animation curves."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import copy
import math
import struct
from ..errors import UgxError

__all__ = ('GrannyTransform', 'GrannyBoneInfo', 'GrannyBoneBindingInfo', 'GrannyMeshInfo', 'GrannyModelInfo', 'GrannyCurveInfo', 'GrannyVectorTrackInfo', 'GrannyTextTrackEntryInfo', 'GrannyTextTrackInfo', 'GrannyTransformTrackInfo', 'GrannyTrackGroupInfo', 'GrannyAnimationInfo', 'GrannyInfo', 'GRX_ROOT_SIZE', 'GRX_TRANSFORM_SIZE', 'GRX_BONE_SIZE', 'GRX_MESH_SIZE', 'GRX_BONE_BINDING_SIZE', 'GRX_MODEL_SIZE', 'GRX_TRACK_GROUP_SIZE', 'GRX_TYPE_RECORD_SIZE', 'GRX_FILE_INFO_SKELETONS', 'GRX_FILE_INFO_MESHES', 'GRX_FILE_INFO_MODELS', 'GRX_FILE_INFO_TRACK_GROUPS', 'GRX_FILE_INFO_ANIMATIONS', 'GRX_VECTOR_TRACK_SIZE', 'GRX_TRANSFORM_TRACK_SIZE', 'GRX_TEXT_TRACK_SIZE', 'GRX_TEXT_TRACK_ENTRY_SIZE', 'GRX_MEMBER_END', 'GRX_MEMBER_INLINE', 'GRX_MEMBER_REFERENCE', 'GRX_MEMBER_REFERENCE_TO_ARRAY', 'GRX_MEMBER_ARRAY_OF_REFERENCES', 'GRX_MEMBER_VARIANT_REFERENCE', 'GRX_MEMBER_REFERENCE_TO_VARIANT_ARRAY', 'GRX_MEMBER_STRING', 'GRX_MEMBER_TRANSFORM', 'GRX_MEMBER_REAL32', 'GRX_MEMBER_INT8', 'GRX_MEMBER_UINT8', 'GRX_MEMBER_BINORMAL_INT8', 'GRX_MEMBER_NORMAL_UINT8', 'GRX_MEMBER_INT16', 'GRX_MEMBER_UINT16', 'GRX_MEMBER_BINORMAL_INT16', 'GRX_MEMBER_NORMAL_UINT16', 'GRX_MEMBER_INT32', 'GRX_MEMBER_UINT32', 'GRX_MEMBER_REAL16', 'GRX_MEMBER_EMPTY_REFERENCE', 'GrxTypeMember', 'GRX_SCHEMA_DEFS', 'grx_cstring', 'grx_array_header', 'grx_ref_array', 'grx_ref_array_from_header', 'validate_granny_model_layout', 'repair_granny_model_binding_overlap', 'grx_curve_type_name', 'grx_u64', 'grx_i32', 'grx_member_width', 'grx_primitive_size', 'grx_read_primitive', 'grx_type_members', 'grx_type_static_size', 'grx_read_array', 'grx_read_typed_object', 'grx_curve_header_value', 'grx_curve_info', 'uax_numeric_list', 'uax_curve_field', 'uax_truncated_float', 'uax_knot_scale', 'uax_knot_times', 'uax_control_scale', 'uax_scaled_vector', 'uax_split_packed_knots_controls', 'uax_decode_d3_curve', 'uax_normalize_quaternion_xyzw', 'uax_blender_quaternion_xyzw', 'UAX_D4N_SCALE_TABLE', 'UAX_D4N_OFFSET_TABLE', 'uax_d4n_tables', 'uax_d4n_quaternion_xyzw', 'uax_decode_d4_curve', 'uax_identity_scale_shear', 'uax_scale_shear_matrix', 'uax_scale_shear_diag', 'uax_decode_scale_shear_curve', 'uax_decode_scale_curve', 'decode_granny_transform_track', 'read_grx_transform', 'parse_granny_vector_tracks', 'parse_granny_transform_lod_errors', 'parse_granny_text_tracks', 'parse_granny_track_groups', 'parse_granny_animations', 'parse_granny_info', 'granny_mesh_for_model_index', 'safe_grx_name', 'write_grx_array_header', 'write_grx_transform', 'identity_grx_transform', 'GrxWriter', 'write_grx_type_schema')

@dataclass
class GrannyTransform:
    flags: int = 0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale_shear: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


@dataclass
class GrannyBoneInfo:
    name: str
    parent: int
    transform: GrannyTransform = field(default_factory=GrannyTransform)
    inverse_world: List[List[float]] = field(default_factory=lambda: [[1.0 if row == column else 0.0 for column in range(4)] for row in range(4)])
    lod_error: float = 0.0


@dataclass
class GrannyBoneBindingInfo:
    bone_name: str
    bounds_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class GrannyMeshInfo:
    name: str
    bone_bindings: List[GrannyBoneBindingInfo] = field(default_factory=list)


@dataclass
class GrannyModelInfo:
    name: str
    mesh_names: List[str] = field(default_factory=list)
    initial_placement: GrannyTransform = field(default_factory=GrannyTransform)
    mesh_indices: List[int] = field(default_factory=list)


@dataclass
class GrannyCurveInfo:
    type_name: str = ""
    type_offset: int = 0
    object_offset: int = 0
    header_format: int = 0
    header_degree: int = 0
    fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GrannyVectorTrackInfo:
    name: str
    track_key: int = 0
    dimension: int = 0
    value_curve: Optional[GrannyCurveInfo] = None


@dataclass
class GrannyTextTrackEntryInfo:
    time_stamp: float
    text: str


@dataclass
class GrannyTextTrackInfo:
    name: str
    entries: List[GrannyTextTrackEntryInfo] = field(default_factory=list)


@dataclass
class GrannyTransformTrackInfo:
    name: str
    flags: int = 0
    orientation_curve: str = ""
    position_curve: str = ""
    scale_shear_curve: str = ""
    orientation_curve_info: Optional[GrannyCurveInfo] = None
    position_curve_info: Optional[GrannyCurveInfo] = None
    scale_shear_curve_info: Optional[GrannyCurveInfo] = None
    rotation_keys: List[Tuple[float, Tuple[float, float, float, float]]] = field(default_factory=list)
    position_keys: List[Tuple[float, Tuple[float, float, float]]] = field(default_factory=list)
    scale_keys: List[Tuple[float, Tuple[float, float, float]]] = field(default_factory=list)
    scale_shear_keys: List[Tuple[float, Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]] = field(default_factory=list)


@dataclass
class GrannyTrackGroupInfo:
    name: str
    offset: int = 0
    flags: int = 0
    loop_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    periodic_loop_offset: int = 0
    extended_data_type_offset: int = 0
    extended_data_object_offset: int = 0
    initial_placement: GrannyTransform = field(default_factory=GrannyTransform)
    vector_tracks: List[GrannyVectorTrackInfo] = field(default_factory=list)
    transform_tracks: List[GrannyTransformTrackInfo] = field(default_factory=list)
    transform_lod_errors: List[float] = field(default_factory=list)
    text_tracks: List[GrannyTextTrackInfo] = field(default_factory=list)


@dataclass
class GrannyAnimationInfo:
    name: str
    duration: float = 0.0
    time_step: float = 0.0
    oversampling: float = 0.0
    track_group_names: List[str] = field(default_factory=list)
    transform_track_names: List[str] = field(default_factory=list)
    transform_tracks: List[GrannyTransformTrackInfo] = field(default_factory=list)
    track_groups: List[GrannyTrackGroupInfo] = field(default_factory=list)
    accumulation_flags: int = 0
    default_loop_count: int = 0
    flags: int = 0


@dataclass
class GrannyInfo:
    from_file_name: str = "gr2ugx"
    skeleton_name: str = "GrannyRootBone"
    skeleton_lod_type: int = 0
    bones: List[GrannyBoneInfo] = field(default_factory=list)
    meshes: List[GrannyMeshInfo] = field(default_factory=list)
    models: List[GrannyModelInfo] = field(default_factory=list)
    track_groups: List[GrannyTrackGroupInfo] = field(default_factory=list)
    animations: List[GrannyAnimationInfo] = field(default_factory=list)
    parsed: bool = False


GRX_ROOT_SIZE = 148


GRX_TRANSFORM_SIZE = 68


GRX_BONE_SIZE = 164


GRX_MESH_SIZE = 76


GRX_BONE_BINDING_SIZE = 44


GRX_MODEL_SIZE = 112


GRX_TRACK_GROUP_SIZE = 164


GRX_TYPE_RECORD_SIZE = 44


GRX_FILE_INFO_SKELETONS = 48


GRX_FILE_INFO_MESHES = 84


GRX_FILE_INFO_MODELS = 96


GRX_FILE_INFO_TRACK_GROUPS = 108


GRX_FILE_INFO_ANIMATIONS = 120


GRX_VECTOR_TRACK_SIZE = 32


GRX_TRANSFORM_TRACK_SIZE = 60


GRX_TEXT_TRACK_SIZE = 20


GRX_TEXT_TRACK_ENTRY_SIZE = 12


GRX_MEMBER_END = 0


GRX_MEMBER_INLINE = 1


GRX_MEMBER_REFERENCE = 2


GRX_MEMBER_REFERENCE_TO_ARRAY = 3


GRX_MEMBER_ARRAY_OF_REFERENCES = 4


GRX_MEMBER_VARIANT_REFERENCE = 5


GRX_MEMBER_REFERENCE_TO_VARIANT_ARRAY = 7


GRX_MEMBER_STRING = 8


GRX_MEMBER_TRANSFORM = 9


GRX_MEMBER_REAL32 = 10


GRX_MEMBER_INT8 = 11


GRX_MEMBER_UINT8 = 12


GRX_MEMBER_BINORMAL_INT8 = 13


GRX_MEMBER_NORMAL_UINT8 = 14


GRX_MEMBER_INT16 = 15


GRX_MEMBER_UINT16 = 16


GRX_MEMBER_BINORMAL_INT16 = 17


GRX_MEMBER_NORMAL_UINT16 = 18


GRX_MEMBER_INT32 = 19


GRX_MEMBER_UINT32 = 20


GRX_MEMBER_REAL16 = 21


GRX_MEMBER_EMPTY_REFERENCE = 22


@dataclass(frozen=True)
class GrxTypeMember:
    member_type: int
    name: str
    reference_type: int
    array_width: int


GRX_SCHEMA_DEFS = {
    "FileInfo": [
        (2, "ArtToolInfo", "ArtToolInfo", 0),
        (2, "ExporterInfo", "ExporterInfo", 0),
        (8, "FromFileName", "", 0),
        (4, "Textures", "Texture", 0),
        (4, "Materials", "Material", 0),
        (4, "Skeletons", "Skeleton", 0),
        (4, "VertexDatas", "VertexData", 0),
        (4, "TriTopologies", "TriTopology", 0),
        (4, "Meshes", "Mesh", 0),
        (4, "Models", "Model", 0),
        (4, "TrackGroups", "TrackGroup", 0),
        (4, "Animations", "Animation", 0),
        (5, "ExtendedData", "", 0),
    ],
    "ArtToolInfo": [
        (8, "FromArtToolName", "", 0),
        (19, "ArtToolMajorRevision", "", 0),
        (19, "ArtToolMinorRevision", "", 0),
        (19, "ArtToolPointerSize", "", 0),
        (10, "UnitsPerMeter", "", 0),
        (10, "Origin", "", 3),
        (10, "RightVector", "", 3),
        (10, "UpVector", "", 3),
        (10, "BackVector", "", 3),
        (5, "ExtendedData", "", 0),
    ],
    "ExporterInfo": [
        (8, "ExporterName", "", 0),
        (19, "ExporterMajorRevision", "", 0),
        (19, "ExporterMinorRevision", "", 0),
        (19, "ExporterCustomization", "", 0),
        (19, "ExporterBuildNumber", "", 0),
        (5, "ExtendedData", "", 0),
    ],
    "Texture": [
        (8, "FromFileName", "", 0),
        (19, "TextureType", "", 0),
        (19, "Width", "", 0),
        (19, "Height", "", 0),
        (19, "Encoding", "", 0),
        (19, "SubFormat", "", 0),
        (1, "Layout", "TextureLayout", 0),
        (3, "Images", "TextureImage", 0),
        (5, "ExtendedData", "", 0),
    ],
    "TextureLayout": [
        (19, "BytesPerPixel", "", 0),
        (19, "ShiftForComponent", "", 4),
        (19, "BitsForComponent", "", 4),
    ],
    "TextureImage": [
        (3, "MIPLevels", "TextureMIPLevel", 0),
    ],
    "TextureMIPLevel": [
        (19, "Stride", "", 0),
        (3, "PixelBytes", "UInt8", 0),
    ],
    "UInt8": [
        (12, "UInt8", "", 0),
    ],
    "Material": [
        (8, "Name", "", 0),
        (3, "Maps", "MaterialMap", 0),
        (2, "Texture", "Texture", 0),
        (5, "ExtendedData", "", 0),
    ],
    "MaterialMap": [
        (8, "Usage", "", 0),
        (2, "Map", "Material", 0),
    ],
    "Skeleton": [
        (8, "Name", "", 0),
        (3, "Bones", "Bone", 0),
        (19, "LODType", "", 0),
        (5, "ExtendedData", "", 0),
    ],
    "Bone": [
        (8, "Name", "", 0),
        (19, "ParentIndex", "", 0),
        (9, "Transform", "", 0),
        (10, "InverseWorldTransform", "", 16),
        (10, "LODError", "", 0),
        (5, "ExtendedData", "", 0),
    ],
    "VertexData": [
        (7, "Vertices", "", 0),
        (3, "VertexComponentNames", "String", 0),
        (3, "VertexAnnotationSets", "VertexAnnotationSet", 0),
    ],
    "String": [
        (8, "String", "", 0),
    ],
    "VertexAnnotationSet": [
        (8, "Name", "", 0),
        (7, "VertexAnnotations", "", 0),
        (19, "IndicesMapFromVertexToAnnotation", "", 0),
        (3, "VertexAnnotationIndices", "Int32", 0),
    ],
    "Int32": [
        (19, "Int32", "", 0),
    ],
    "TriTopology": [
        (3, "Groups", "TriGroup", 0),
        (3, "Indices", "Int32", 0),
        (3, "Indices16", "Int16", 0),
        (3, "VertexToVertexMap", "Int32", 0),
        (3, "VertexToTriangleMap", "Int32", 0),
        (3, "SideToNeighborMap", "Int32", 0),
        (3, "PolygonIndexStarts", "Int32", 0),
        (3, "PolygonIndices", "Int32", 0),
        (3, "BonesForTriangle", "Int32", 0),
        (3, "TriangleToBoneIndices", "Int32", 0),
        (3, "TriAnnotationSets", "TriAnnotationSet", 0),
    ],
    "TriGroup": [
        (19, "MaterialIndex", "", 0),
        (19, "TriFirst", "", 0),
        (19, "TriCount", "", 0),
    ],
    "Int16": [
        (15, "Int16", "", 0),
    ],
    "TriAnnotationSet": [
        (8, "Name", "", 0),
        (7, "TriAnnotations", "", 0),
        (19, "IndicesMapFromTriToAnnotation", "", 0),
        (3, "TriAnnotationIndices", "Int32", 0),
    ],
    "Mesh": [
        (8, "Name", "", 0),
        (2, "PrimaryVertexData", "VertexData", 0),
        (3, "MorphTargets", "MorphTarget", 0),
        (2, "PrimaryTopology", "TriTopology", 0),
        (3, "MaterialBindings", "MaterialBinding", 0),
        (3, "BoneBindings", "BoneBinding", 0),
        (5, "ExtendedData", "", 0),
    ],
    "MorphTarget": [
        (8, "ScalarName", "", 0),
        (2, "VertexData", "VertexData", 0),
        (19, "DataIsDeltas", "", 0),
    ],
    "MaterialBinding": [
        (2, "Material", "Material", 0),
    ],
    "BoneBinding": [
        (8, "BoneName", "", 0),
        (10, "OBBMin", "", 3),
        (10, "OBBMax", "", 3),
        (3, "TriangleIndices", "Int32", 0),
    ],
    "Model": [
        (8, "Name", "", 0),
        (2, "Skeleton", "Skeleton", 0),
        (9, "InitialPlacement", "", 0),
        (3, "MeshBindings", "ModelMeshBinding", 0),
        (5, "ExtendedData", "", 0),
    ],
    "ModelMeshBinding": [
        (2, "Mesh", "Mesh", 0),
    ],
    "TrackGroup": [
        (8, "Name", "", 0),
        (3, "VectorTracks", "VectorTrack", 0),
        (3, "TransformTracks", "TransformTrack", 0),
        (3, "TransformLODErrors", "Real32", 0),
        (3, "TextTracks", "TextTrack", 0),
        (9, "InitialPlacement", "", 0),
        (19, "AccumulationFlags", "", 0),
        (10, "LoopTranslation", "", 3),
        (2, "PeriodicLoop", "PeriodicLoop", 0),
        (5, "ExtendedData", "", 0),
    ],
    "VectorTrack": [
        (8, "Name", "", 0),
        (20, "TrackKey", "", 0),
        (19, "Dimension", "", 0),
        (1, "ValueCurve", "Curve2", 0),
    ],
    "Curve2": [
        (5, "CurveData", "", 0),
    ],
    "TransformTrack": [
        (8, "Name", "", 0),
        (19, "Flags", "", 0),
        (1, "OrientationCurve", "Curve2", 0),
        (1, "PositionCurve", "Curve2", 0),
        (1, "ScaleShearCurve", "Curve2", 0),
    ],
    "Real32": [
        (10, "Real32", "", 0),
    ],
    "TextTrack": [
        (8, "Name", "", 0),
        (3, "Entries", "TextTrackEntry", 0),
    ],
    "TextTrackEntry": [
        (10, "TimeStamp", "", 0),
        (8, "Text", "", 0),
    ],
    "PeriodicLoop": [
        (10, "Radius", "", 0),
        (10, "dAngle", "", 0),
        (10, "dZ", "", 0),
        (10, "BasisX", "", 3),
        (10, "BasisY", "", 3),
        (10, "Axis", "", 3),
    ],
    "Animation": [
        (8, "Name", "", 0),
        (10, "Duration", "", 0),
        (10, "TimeStep", "", 0),
        (10, "Oversampling", "", 0),
        (4, "TrackGroups", "TrackGroup", 0),
        (19, "DefaultLoopCount", "", 0),
        (19, "Flags", "", 0),
        (5, "ExtendedData", "", 0),
    ],
}


def grx_cstring(data: bytes, offset: int) -> str:
    if offset <= 0 or offset >= len(data):
        return ""
    end = data.find(b"\0", offset)
    if end < 0:
        return ""
    return data[offset:end].decode("utf-8", "replace")


def grx_array_header(data: bytes, offset: int) -> Tuple[int, int]:
    if offset < 0 or offset + 12 > len(data):
        return 0, 0
    return struct.unpack_from("<IQ", data, offset)


def grx_ref_array(data: bytes, length: int, offset: int) -> List[int]:
    if length <= 0 or offset <= 0 or offset + length * 8 > len(data):
        return []
    return [struct.unpack_from("<Q", data, offset + index * 8)[0] for index in range(length)]


def grx_ref_array_from_header(data: bytes, offset: int) -> List[int]:
    return grx_ref_array(data, *grx_array_header(data, offset))


def validate_granny_model_layout(data: bytes):
    if len(data) < GRX_ROOT_SIZE:
        raise UgxError("UGX Granny file-info record is truncated.")
    count, offset = grx_array_header(data, GRX_FILE_INFO_MODELS)
    if count and (offset <= 0 or offset + count * 8 > len(data)):
        raise UgxError("UGX Granny model references point outside the data chunk.")
    for model in grx_ref_array(data, count, offset):
        if model <= 0 or model + GRX_MODEL_SIZE > len(data):
            raise UgxError("UGX Granny model record is truncated.")
        binding_count, bindings = grx_array_header(data, model + 84)
        if not binding_count:
            continue
        end = bindings + binding_count * 8
        if bindings <= 0 or end > len(data):
            raise UgxError("UGX Granny mesh bindings point outside the data chunk.")
        if bindings < model + GRX_MODEL_SIZE and end > model:
            raise UgxError("UGX Granny mesh bindings overwrite the model's ExtendedData pointers.")


def repair_granny_model_binding_overlap(data: bytes) -> bytes:
    repaired = None
    meshes = set(grx_ref_array_from_header(data, GRX_FILE_INFO_MESHES))
    models = grx_ref_array_from_header(data, GRX_FILE_INFO_MODELS)

    def relocate(header):
        count, offset = grx_array_header(data, header)
        references = grx_ref_array(data, count, offset)
        if len(references) != count:
            raise UgxError("Cannot repair overlapping UGX Granny model bindings: a shared array is outside the data chunk.")
        repaired.extend(b"\0" * (-len(repaired) % 16))
        new_offset = len(repaired)
        repaired.extend(data[offset:offset + count * 8])
        struct.pack_into("<Q", repaired, header + 4, new_offset)

    for model in models:
        if model <= 0 or model + GRX_MODEL_SIZE > len(data):
            continue
        count, bindings = grx_array_header(data, model + 84)
        if not count or bindings != model + 96:
            continue
        references = grx_ref_array(data, count, bindings)
        if len(references) != count or any(not mesh or mesh not in meshes for mesh in references):
            raise UgxError("Cannot repair overlapping UGX Granny model bindings: mesh references are invalid.")
        # Older exports allocated only the first 96 bytes of granny_model.
        # Move that array without rebuilding unknown types or extended data.
        if repaired is None:
            repaired = bytearray(data)
        relocate(model + 84)
        repaired[model + 96:model + 112] = b"\0" * 16
        # Each owning array is rebased independently by the game.
        if grx_array_header(data, GRX_FILE_INFO_MESHES)[1] == bindings:
            relocate(GRX_FILE_INFO_MESHES)
        for other in models:
            if other != model and 0 < other <= len(data) - GRX_MODEL_SIZE and grx_array_header(data, other + 84)[1] == bindings:
                relocate(other + 84)
    result = bytes(repaired) if repaired is not None else data
    validate_granny_model_layout(result)
    return result


def grx_curve_type_name(data: bytes, variant_offset: int) -> str:
    if variant_offset < 0 or variant_offset + 16 > len(data):
        return ""
    type_offset = struct.unpack_from("<Q", data, variant_offset)[0]
    if type_offset <= 0 or type_offset + 12 > len(data):
        return ""
    return grx_cstring(data, struct.unpack_from("<Q", data, type_offset + 4)[0])


def grx_u64(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 8 > len(data):
        return 0
    return struct.unpack_from("<Q", data, offset)[0]


def grx_i32(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        return 0
    return struct.unpack_from("<i", data, offset)[0]


def grx_member_width(array_width: int) -> int:
    return max(1, int(array_width))


def grx_primitive_size(member_type: int) -> int:
    if member_type in (GRX_MEMBER_INT8, GRX_MEMBER_UINT8, GRX_MEMBER_BINORMAL_INT8, GRX_MEMBER_NORMAL_UINT8):
        return 1
    if member_type in (GRX_MEMBER_INT16, GRX_MEMBER_UINT16, GRX_MEMBER_BINORMAL_INT16, GRX_MEMBER_NORMAL_UINT16, GRX_MEMBER_REAL16):
        return 2
    if member_type in (GRX_MEMBER_REAL32, GRX_MEMBER_INT32, GRX_MEMBER_UINT32):
        return 4
    return 0


_GRX_PRIMITIVE_FORMATS = {
    GRX_MEMBER_INT8: 'b', GRX_MEMBER_BINORMAL_INT8: 'b',
    GRX_MEMBER_UINT8: 'B', GRX_MEMBER_NORMAL_UINT8: 'B',
    GRX_MEMBER_INT16: 'h', GRX_MEMBER_BINORMAL_INT16: 'h',
    GRX_MEMBER_UINT16: 'H', GRX_MEMBER_NORMAL_UINT16: 'H',
    GRX_MEMBER_REAL16: 'e', GRX_MEMBER_INT32: 'i',
    GRX_MEMBER_UINT32: 'I', GRX_MEMBER_REAL32: 'f',
}


def grx_read_primitive(data: bytes, offset: int, member_type: int, array_width: int = 0) -> Tuple[Any, int]:
    count = grx_member_width(array_width)
    size = grx_primitive_size(member_type)
    total = size * count
    if size <= 0 or offset < 0 or offset + total > len(data):
        return None, max(0, total)
    values = struct.unpack_from(f'<{count}{_GRX_PRIMITIVE_FORMATS[member_type]}', data, offset)
    return (values[0] if count == 1 else values), total


def grx_type_members(data: bytes, type_offset: int) -> List[GrxTypeMember]:
    if type_offset <= 0 or type_offset + GRX_TYPE_RECORD_SIZE > len(data):
        return []
    members: List[GrxTypeMember] = []
    offset = type_offset
    for _ in range(256):
        if offset + GRX_TYPE_RECORD_SIZE > len(data):
            break
        member_type = struct.unpack_from("<I", data, offset)[0]
        if member_type == GRX_MEMBER_END:
            break
        members.append(GrxTypeMember(
            member_type,
            grx_cstring(data, grx_u64(data, offset + 4)),
            grx_u64(data, offset + 12),
            grx_i32(data, offset + 20),
        ))
        offset += GRX_TYPE_RECORD_SIZE
    return members


def grx_type_static_size(data: bytes, type_offset: int, depth: int = 0) -> int:
    if depth > 8:
        return 0
    total = 0
    for member in grx_type_members(data, type_offset):
        if member.member_type == GRX_MEMBER_INLINE:
            total += grx_type_static_size(data, member.reference_type, depth + 1) * grx_member_width(member.array_width)
        elif grx_primitive_size(member.member_type):
            total += grx_primitive_size(member.member_type) * grx_member_width(member.array_width)
        elif member.member_type in (GRX_MEMBER_REFERENCE, GRX_MEMBER_STRING, GRX_MEMBER_EMPTY_REFERENCE):
            total += 8
        elif member.member_type in (GRX_MEMBER_REFERENCE_TO_ARRAY, GRX_MEMBER_ARRAY_OF_REFERENCES):
            total += 12
        elif member.member_type == GRX_MEMBER_REFERENCE_TO_VARIANT_ARRAY:
            total += 20
        elif member.member_type == GRX_MEMBER_VARIANT_REFERENCE:
            total += 16
        elif member.member_type == GRX_MEMBER_TRANSFORM:
            total += GRX_TRANSFORM_SIZE * grx_member_width(member.array_width)
    return total


def grx_read_array(data: bytes, type_offset: int, length: int, array_offset: int, depth: int = 0) -> List[Any]:
    if length <= 0 or array_offset <= 0 or array_offset >= len(data) or depth > 8:
        return []
    members = grx_type_members(data, type_offset)
    if len(members) == 1 and grx_primitive_size(members[0].member_type):
        member = members[0]
        stride = grx_primitive_size(member.member_type) * grx_member_width(member.array_width)
        if stride <= 0 or array_offset + length * stride > len(data):
            return []
        width = grx_member_width(member.array_width)
        primitive_format = _GRX_PRIMITIVE_FORMATS[member.member_type]
        if width == 1:
            return list(struct.unpack_from(f'<{length}{primitive_format}', data, array_offset))
        return list(struct.iter_unpack(f'<{width}{primitive_format}', memoryview(data)[array_offset:array_offset+length*stride]))
    stride = grx_type_static_size(data, type_offset, depth + 1)
    if stride <= 0 or array_offset + length * stride > len(data):
        return []
    out = []
    for index in range(length):
        value, _ = grx_read_typed_object(data, type_offset, array_offset + index * stride, depth + 1)
        out.append(value)
    return out


def grx_read_typed_object(data: bytes, type_offset: int, object_offset: int, depth: int = 0) -> Tuple[Dict[str, Any], int]:
    if object_offset <= 0 or object_offset >= len(data) or depth > 8:
        return {}, 0
    fields: Dict[str, Any] = {}
    cursor = object_offset
    for index, member in enumerate(grx_type_members(data, type_offset)):
        name = member.name or f"member_{index}"
        if member.member_type == GRX_MEMBER_INLINE:
            width = grx_member_width(member.array_width)
            stride = grx_type_static_size(data, member.reference_type, depth + 1)
            values = [grx_read_typed_object(data, member.reference_type, cursor + index * stride, depth + 1)[0]
                      for index in range(width)]
            fields[name] = values[0] if width == 1 else values
            cursor += stride * width
        elif grx_primitive_size(member.member_type):
            value, size = grx_read_primitive(data, cursor, member.member_type, member.array_width)
            fields[name] = value
            cursor += size
        elif member.member_type == GRX_MEMBER_REFERENCE:
            target = grx_u64(data, cursor)
            fields[name] = target
            cursor += 8
        elif member.member_type == GRX_MEMBER_REFERENCE_TO_ARRAY:
            length, array_offset = grx_array_header(data, cursor)
            fields[name] = grx_read_array(data, member.reference_type, length, array_offset, depth + 1)
            fields[f"{name}Header"] = {"length": length, "offset": array_offset}
            cursor += 12
        elif member.member_type == GRX_MEMBER_ARRAY_OF_REFERENCES:
            length, array_offset = grx_array_header(data, cursor)
            fields[name] = grx_ref_array(data, length, array_offset)
            fields[f"{name}Header"] = {"length": length, "offset": array_offset}
            cursor += 12
        elif member.member_type == GRX_MEMBER_VARIANT_REFERENCE:
            fields[name] = grx_curve_info(data, cursor)
            cursor += 16
        elif member.member_type == GRX_MEMBER_REFERENCE_TO_VARIANT_ARRAY:
            element_type = grx_u64(data, cursor)
            length, array_offset = grx_array_header(data, cursor + 8)
            fields[name] = grx_read_array(data, element_type, length, array_offset, depth + 1)
            fields[f"{name}Header"] = {"type_offset": element_type, "length": length, "offset": array_offset}
            cursor += 20
        elif member.member_type == GRX_MEMBER_STRING:
            fields[name] = grx_cstring(data, grx_u64(data, cursor))
            cursor += 8
        elif member.member_type == GRX_MEMBER_TRANSFORM:
            width = grx_member_width(member.array_width)
            values = [read_grx_transform(data, cursor + index * GRX_TRANSFORM_SIZE) for index in range(width)]
            fields[name] = values[0] if width == 1 else values
            cursor += GRX_TRANSFORM_SIZE * width
        elif member.member_type == GRX_MEMBER_EMPTY_REFERENCE:
            fields[name] = grx_u64(data, cursor)
            cursor += 8
    return fields, max(0, cursor - object_offset)


def grx_curve_header_value(fields: Dict[str, Any], key: str) -> int:
    if key in fields and isinstance(fields[key], (int, float)):
        return int(fields[key])
    for value in fields.values():
        if isinstance(value, dict):
            found = grx_curve_header_value(value, key)
            if found:
                return found
    return 0


def grx_curve_info(data: bytes, variant_offset: int) -> GrannyCurveInfo:
    if variant_offset < 0 or variant_offset + 16 > len(data):
        return GrannyCurveInfo()
    type_offset = grx_u64(data, variant_offset)
    object_offset = grx_u64(data, variant_offset + 8)
    type_name = grx_curve_type_name(data, variant_offset)
    fields, _ = grx_read_typed_object(data, type_offset, object_offset)
    return GrannyCurveInfo(
        type_name,
        type_offset,
        object_offset,
        grx_curve_header_value(fields, "Format"),
        grx_curve_header_value(fields, "Degree"),
        fields,
    )


def uax_numeric_list(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, tuple):
        return [float(item) for item in value if isinstance(item, (int, float))]
    if isinstance(value, list):
        out: List[float] = []
        for item in value:
            if isinstance(item, (int, float)):
                out.append(float(item))
            elif isinstance(item, tuple):
                out.extend(float(part) for part in item if isinstance(part, (int, float)))
        return out
    return []


def uax_curve_field(curve: Optional[GrannyCurveInfo], name: str, default: Any = None) -> Any:
    if curve is None:
        return default
    return curve.fields.get(name, default)


def uax_truncated_float(value: Any) -> float:
    try:
        return struct.unpack("<f", struct.pack("<I", (int(value) & 0xFFFF) << 16))[0]
    except (TypeError, ValueError, struct.error):
        return 0.0


def uax_knot_scale(curve: Optional[GrannyCurveInfo]) -> float:
    direct = uax_curve_field(curve, "OneOverKnotScale")
    if isinstance(direct, (int, float)) and math.isfinite(float(direct)) and float(direct) > 0.0:
        return float(direct)
    return uax_truncated_float(uax_curve_field(curve, "OneOverKnotScaleTrunc"))


def uax_knot_times(knots: List[float], one_over_knot_scale: float, packed: bool) -> List[float]:
    if not packed:
        return [float(knot) for knot in knots]
    if not math.isfinite(one_over_knot_scale) or one_over_knot_scale <= 0.0:
        return [float(index) for index, _ in enumerate(knots)]
    return [float(knot) / one_over_knot_scale for knot in knots]


def uax_control_scale(value: float, max_value: float, packed: bool) -> float:
    return float(value)


def uax_scaled_vector(raw: List[float], scales: List[float], offsets: List[float], max_value: float, packed: bool) -> Tuple[float, ...]:
    out = []
    for index, value in enumerate(raw):
        scale = scales[index] if index < len(scales) else 1.0
        offset = offsets[index] if index < len(offsets) else 0.0
        out.append(offset + scale * uax_control_scale(value, max_value, packed))
    return tuple(out)


def uax_split_packed_knots_controls(values: List[float], dimension: int) -> Tuple[List[float], List[float]]:
    if dimension <= 0:
        return [], []
    key_count = len(values) // (dimension + 1)
    if key_count <= 0:
        return [], []
    return values[:key_count], values[key_count:key_count + key_count * dimension]


def uax_decode_d3_curve(curve: Optional[GrannyCurveInfo]) -> List[Tuple[float, Tuple[float, float, float]]]:
    if curve is None:
        return []
    name = curve.type_name
    if "DaIdentity" in name:
        return [(0.0, (0.0, 0.0, 0.0))]
    controls = uax_numeric_list(uax_curve_field(curve, "Controls"))
    if "Constant32f" in name and len(controls) >= 3:
        return [(0.0, (controls[0], controls[1], controls[2]))]
    if "DaK32fC32f" in name:
        knots = uax_numeric_list(uax_curve_field(curve, "Knots"))
        controls = uax_numeric_list(uax_curve_field(curve, "Controls"))
        dimension = len(controls) // len(knots) if knots else 0
        if dimension < 3:
            return []
        return [
            (float(knots[index]), tuple(controls[index * dimension:index * dimension + 3]))
            for index in range(min(len(knots), len(controls) // dimension))
        ]
    if "D3" not in name:
        return []
    raw_values = uax_numeric_list(uax_curve_field(curve, "KnotsControls"))
    packed = "K32fC32f" not in name
    max_value = 255.0 if "K8uC8u" in name else 65535.0
    if "D3I1" in name:
        knots, controls = uax_split_packed_knots_controls(raw_values, 1)
        times = uax_knot_times(knots, uax_knot_scale(curve), packed)
        scales = uax_numeric_list(uax_curve_field(curve, "ControlScales"))
        offsets = uax_numeric_list(uax_curve_field(curve, "ControlOffsets"))
        out: List[Tuple[float, Tuple[float, float, float]]] = []
        for index, time in enumerate(times):
            if index >= len(controls):
                break
            raw = [controls[index], controls[index], controls[index]]
            value = uax_scaled_vector(raw, scales, offsets, max_value, packed)
            out.append((time, (value[0], value[1], value[2])))
        return out
    knots, controls = uax_split_packed_knots_controls(raw_values, 3)
    times = uax_knot_times(knots, uax_knot_scale(curve), packed)
    scales = uax_numeric_list(uax_curve_field(curve, "ControlScales"))
    offsets = uax_numeric_list(uax_curve_field(curve, "ControlOffsets"))
    out: List[Tuple[float, Tuple[float, float, float]]] = []
    for index, time in enumerate(times):
        raw = controls[index * 3:index * 3 + 3]
        if len(raw) < 3:
            break
        value = uax_scaled_vector(raw, scales, offsets, max_value, packed)
        out.append((time, (value[0], value[1], value[2])))
    return out


def uax_normalize_quaternion_xyzw(value: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    length = math.sqrt(sum(component * component for component in value))
    if not math.isfinite(length) or length <= 0.000001:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(component / length for component in value)


def uax_blender_quaternion_xyzw(value: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    # Spline controls retain their magnitude. Normalize the sampled rotation,
    # since normalizing each control first changes the spline's shape.
    x, y, z, w = value
    return (w, x, y, z)


UAX_D4N_SCALE_TABLE = (
    1.4142135, 0.70710677, 0.35355338, 0.35355338,
    0.35355338, 0.17677669, 0.17677669, 0.17677669,
    -1.4142135, -0.70710677, -0.35355338, -0.35355338,
    -0.35355338, -0.17677669, -0.17677669, -0.17677669,
)


UAX_D4N_OFFSET_TABLE = (
    -0.70710677, -0.35355338, -0.53033006, -0.17677669,
    0.17677669, -0.17677669, -0.088388346, 0.0,
    0.70710677, 0.35355338, 0.53033006, 0.17677669,
    -0.17677669, 0.17677669, 0.088388346, -0.0,
)


def uax_d4n_tables(selector: int, bits: int) -> Tuple[List[float], List[float]]:
    multiplier = 0.0078740157 if bits == 8 else 0.000030518509
    scales = []
    offsets = []
    for shift in (0, 4, 8, 12):
        table_index = (int(selector) >> shift) & 0x0F
        scales.append(UAX_D4N_SCALE_TABLE[table_index] * multiplier)
        offsets.append(UAX_D4N_OFFSET_TABLE[table_index])
    return scales, offsets


def uax_d4n_quaternion_xyzw(a: float, b: float, c: float, bits: int, scales: List[float], offsets: List[float]) -> Tuple[float, float, float, float]:
    if bits == 8:
        raw_a = int(a) & 0xFF
        raw_b = int(b) & 0xFF
        raw_c = int(c) & 0xFF
        sign_mask = 0x80
        data_mask = 0x7F
        swizzle1 = ((raw_b & sign_mask) >> 6) | ((raw_c & sign_mask) >> 7)
    else:
        raw_a = int(a) & 0xFFFF
        raw_b = int(b) & 0xFFFF
        raw_c = int(c) & 0xFFFF
        sign_mask = 0x8000
        data_mask = 0x7FFF
        swizzle1 = ((raw_b & sign_mask) >> 14) | ((raw_c & sign_mask) >> 15)
    swizzle2 = (swizzle1 + 1) & 3
    swizzle3 = (swizzle2 + 1) & 3
    swizzle4 = (swizzle3 + 1) & 3
    data_a = (raw_a & data_mask) * scales[swizzle2] + offsets[swizzle2]
    data_b = (raw_b & data_mask) * scales[swizzle3] + offsets[swizzle3]
    data_c = (raw_c & data_mask) * scales[swizzle4] + offsets[swizzle4]
    missing = math.sqrt(max(0.0, 1.0 - (data_a * data_a + data_b * data_b + data_c * data_c)))
    if raw_a & sign_mask:
        missing = -missing
    components = [0.0, 0.0, 0.0, 0.0]
    components[swizzle2] = data_a
    components[swizzle3] = data_b
    components[swizzle4] = data_c
    components[swizzle1] = missing
    return components[0], components[1], components[2], components[3]


def uax_decode_d4_curve(curve: Optional[GrannyCurveInfo]) -> List[Tuple[float, Tuple[float, float, float, float]]]:
    if curve is None:
        return []
    name = curve.type_name
    if "DaIdentity" in name:
        return [(0.0, (1.0, 0.0, 0.0, 0.0))]
    controls = uax_numeric_list(uax_curve_field(curve, "Controls"))
    if "Constant32f" in name and len(controls) >= 4:
        return [(0.0, uax_blender_quaternion_xyzw((controls[0], controls[1], controls[2], controls[3])))]
    if "DaK32fC32f" in name:
        knots = uax_numeric_list(uax_curve_field(curve, "Knots"))
        controls = uax_numeric_list(uax_curve_field(curve, "Controls"))
        dimension = len(controls) // len(knots) if knots else 0
        if dimension < 4:
            return []
        return [
            (float(knots[index]), uax_blender_quaternion_xyzw(tuple(controls[index * dimension:index * dimension + 4])))
            for index in range(min(len(knots), len(controls) // dimension))
        ]
    if "D4nK" not in name:
        return []
    raw_values = uax_numeric_list(uax_curve_field(curve, "KnotsControls"))
    knots, controls = uax_split_packed_knots_controls(raw_values, 3)
    times = uax_knot_times(knots, uax_knot_scale(curve), True)
    bits = 8 if "K8uC7u" in name else 16
    scales, offsets = uax_d4n_tables(int(uax_curve_field(curve, "ScaleOffsetTableEntries", 0) or 0), bits)
    out: List[Tuple[float, Tuple[float, float, float, float]]] = []
    for index, time in enumerate(times):
        raw = controls[index * 3:index * 3 + 3]
        if len(raw) < 3:
            break
        out.append((time, uax_blender_quaternion_xyzw(uax_d4n_quaternion_xyzw(raw[0], raw[1], raw[2], bits, scales, offsets))))
    return out


def uax_identity_scale_shear() -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def uax_scale_shear_matrix(values: List[float]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    if len(values) >= 9:
        return (
            (values[0], values[1], values[2]),
            (values[3], values[4], values[5]),
            (values[6], values[7], values[8]),
        )
    if len(values) >= 3:
        return ((values[0], 0.0, 0.0), (0.0, values[1], 0.0), (0.0, 0.0, values[2]))
    if values:
        return ((values[0], 0.0, 0.0), (0.0, values[0], 0.0), (0.0, 0.0, values[0]))
    return uax_identity_scale_shear()


def uax_scale_shear_diag(matrix: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]) -> Tuple[float, float, float]:
    return matrix[0][0], matrix[1][1], matrix[2][2]


def uax_decode_scale_shear_curve(curve: Optional[GrannyCurveInfo]) -> List[Tuple[float, Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]]:
    if curve is None:
        return []
    name = curve.type_name
    if "DaIdentity" in name:
        return [(0.0, uax_identity_scale_shear())]
    controls = uax_numeric_list(uax_curve_field(curve, "Controls"))
    if "Constant32f" in name and controls:
        return [(0.0, uax_scale_shear_matrix(controls))]
    if "DaK32fC32f" in name:
        knots = uax_numeric_list(uax_curve_field(curve, "Knots"))
        controls = uax_numeric_list(uax_curve_field(curve, "Controls"))
        dimension = len(controls) // len(knots) if knots else 0
        if dimension <= 0:
            return []
        return [
            (float(knots[index]), uax_scale_shear_matrix(controls[index * dimension:index * dimension + dimension]))
            for index in range(min(len(knots), len(controls) // dimension))
        ]
    raw_values = uax_numeric_list(uax_curve_field(curve, "KnotsControls"))
    packed = "K32fC32f" not in name
    max_value = 255.0 if "K8uC8u" in name else 65535.0
    if "D9I1" in name:
        knots, controls = uax_split_packed_knots_controls(raw_values, 1)
        times = uax_knot_times(knots, uax_knot_scale(curve), packed)
        scale = float(uax_curve_field(curve, "ControlScale", 1.0) or 1.0)
        offset = float(uax_curve_field(curve, "ControlOffset", 0.0) or 0.0)
        return [
            (time, uax_scale_shear_matrix([offset + scale * uax_control_scale(controls[index], max_value, packed)]))
            for index, time in enumerate(times)
            if index < len(controls)
        ]
    if "D9I3" in name:
        knots, controls = uax_split_packed_knots_controls(raw_values, 3)
        times = uax_knot_times(knots, uax_knot_scale(curve), packed)
        scales = uax_numeric_list(uax_curve_field(curve, "ControlScales"))
        offsets = uax_numeric_list(uax_curve_field(curve, "ControlOffsets"))
        out = []
        for index, time in enumerate(times):
            raw = controls[index * 3:index * 3 + 3]
            if len(raw) < 3:
                break
            value = uax_scaled_vector(raw, scales, offsets, max_value, packed)
            out.append((time, uax_scale_shear_matrix([value[0], value[1], value[2]])))
        return out
    if "DaK" in name:
        scale_offsets = uax_numeric_list(uax_curve_field(curve, "ControlScaleOffsets"))
        dimension = len(scale_offsets) // 2
        if dimension <= 0:
            return []
        knots, controls = uax_split_packed_knots_controls(raw_values, dimension)
        times = uax_knot_times(knots, uax_knot_scale(curve), packed)
        scales = scale_offsets[:dimension]
        offsets = scale_offsets[dimension:dimension * 2]
        out = []
        for index, time in enumerate(times):
            raw = controls[index * dimension:index * dimension + dimension]
            if len(raw) < dimension:
                break
            out.append((time, uax_scale_shear_matrix(list(uax_scaled_vector(raw, scales, offsets, max_value, packed)))))
        return out
    return []


def uax_decode_scale_curve(curve: Optional[GrannyCurveInfo]) -> List[Tuple[float, Tuple[float, float, float]]]:
    return [(time, uax_scale_shear_diag(matrix)) for time, matrix in uax_decode_scale_shear_curve(curve)]


def decode_granny_transform_track(data: bytes, offset: int, track_index: int) -> GrannyTransformTrackInfo:
    orientation = grx_curve_info(data, offset + 12)
    position = grx_curve_info(data, offset + 28)
    scale_shear = grx_curve_info(data, offset + 44)
    track = GrannyTransformTrackInfo(
        grx_cstring(data, grx_u64(data, offset)),
        struct.unpack_from("<I", data, offset + 8)[0],
        orientation.type_name,
        position.type_name,
        scale_shear.type_name,
        orientation,
        position,
        scale_shear,
    )
    track.rotation_keys = uax_decode_d4_curve(orientation)
    track.position_keys = uax_decode_d3_curve(position)
    track.scale_keys = uax_decode_scale_curve(scale_shear)
    track.scale_shear_keys = uax_decode_scale_shear_curve(scale_shear)
    return track


def read_grx_transform(data: bytes, offset: int) -> GrannyTransform:
    if offset < 0 or offset + GRX_TRANSFORM_SIZE > len(data):
        return GrannyTransform()
    scale = tuple(struct.unpack_from("<3f", data, offset + 32 + row * 12) for row in range(3))
    return GrannyTransform(
        struct.unpack_from("<I", data, offset)[0],
        struct.unpack_from("<3f", data, offset + 4),
        struct.unpack_from("<4f", data, offset + 16),
        scale,
    )


def parse_granny_vector_tracks(data: bytes, count: int, array_offset: int) -> List[GrannyVectorTrackInfo]:
    tracks: List[GrannyVectorTrackInfo] = []
    if count <= 0 or array_offset <= 0 or array_offset + count * GRX_VECTOR_TRACK_SIZE > len(data):
        return tracks
    for track_index in range(count):
        offset = array_offset + track_index * GRX_VECTOR_TRACK_SIZE
        tracks.append(GrannyVectorTrackInfo(
            grx_cstring(data, grx_u64(data, offset)) or f"vector_track_{track_index}",
            struct.unpack_from("<I", data, offset + 8)[0],
            struct.unpack_from("<i", data, offset + 12)[0],
            grx_curve_info(data, offset + 16),
        ))
    return tracks


def parse_granny_transform_lod_errors(data: bytes, count: int, array_offset: int) -> List[float]:
    if count <= 0 or array_offset <= 0 or array_offset + count * 4 > len(data):
        return []
    return [struct.unpack_from("<f", data, array_offset + index * 4)[0] for index in range(count)]


def parse_granny_text_tracks(data: bytes, count: int, array_offset: int) -> List[GrannyTextTrackInfo]:
    tracks: List[GrannyTextTrackInfo] = []
    if count <= 0 or array_offset <= 0 or array_offset + count * GRX_TEXT_TRACK_SIZE > len(data):
        return tracks
    for track_index in range(count):
        offset = array_offset + track_index * GRX_TEXT_TRACK_SIZE
        entries: List[GrannyTextTrackEntryInfo] = []
        entry_count, entry_offset = grx_array_header(data, offset + 8)
        if entry_count > 0 and entry_offset > 0 and entry_offset + entry_count * GRX_TEXT_TRACK_ENTRY_SIZE <= len(data):
            for entry_index in range(entry_count):
                entry_base = entry_offset + entry_index * GRX_TEXT_TRACK_ENTRY_SIZE
                entries.append(GrannyTextTrackEntryInfo(
                    struct.unpack_from("<f", data, entry_base)[0],
                    grx_cstring(data, grx_u64(data, entry_base + 4)),
                ))
        tracks.append(GrannyTextTrackInfo(
            grx_cstring(data, grx_u64(data, offset)) or f"text_track_{track_index}",
            entries,
        ))
    return tracks


def parse_granny_track_groups(data: bytes, decode_animation_curves=True) -> List[GrannyTrackGroupInfo]:
    groups: List[GrannyTrackGroupInfo] = []
    group_offsets = list(grx_ref_array_from_header(data, GRX_FILE_INFO_TRACK_GROUPS))
    for animation_offset in grx_ref_array_from_header(data, GRX_FILE_INFO_ANIMATIONS):
        if 0 < animation_offset and animation_offset + 32 <= len(data):
            for group_offset in grx_ref_array(data, *grx_array_header(data, animation_offset + 20)):
                if group_offset not in group_offsets:
                    group_offsets.append(group_offset)
    for group_index, group_offset in enumerate(group_offsets):
        if group_offset <= 0 or group_offset + GRX_TRACK_GROUP_SIZE > len(data):
            continue
        vector_count, vector_offset = grx_array_header(data, group_offset + 8)
        transform_count, transform_offset = grx_array_header(data, group_offset + 20)
        lod_count, lod_offset = grx_array_header(data, group_offset + 32)
        text_count, text_offset = grx_array_header(data, group_offset + 44)
        tracks: List[GrannyTransformTrackInfo] = []
        if transform_count > 0 and transform_offset > 0 and transform_offset + transform_count * GRX_TRANSFORM_TRACK_SIZE <= len(data):
            for track_index in range(transform_count):
                offset = transform_offset + track_index * GRX_TRANSFORM_TRACK_SIZE
                if decode_animation_curves:
                    tracks.append(decode_granny_transform_track(data, offset, track_index))
                else:
                    tracks.append(GrannyTransformTrackInfo(grx_cstring(data, grx_u64(data, offset)),
                                                          struct.unpack_from('<I', data, offset + 8)[0]))
        groups.append(GrannyTrackGroupInfo(
            name=grx_cstring(data, grx_u64(data, group_offset)),
            offset=group_offset,
            flags=struct.unpack_from("<I", data, group_offset + 124)[0],
            loop_translation=struct.unpack_from("<3f", data, group_offset + 128),
            periodic_loop_offset=grx_u64(data, group_offset + 140),
            extended_data_type_offset=grx_u64(data, group_offset + 148),
            extended_data_object_offset=grx_u64(data, group_offset + 156),
            initial_placement=read_grx_transform(data, group_offset + 56),
            vector_tracks=parse_granny_vector_tracks(data, vector_count, vector_offset) if decode_animation_curves else [],
            transform_tracks=tracks,
            transform_lod_errors=parse_granny_transform_lod_errors(data, lod_count, lod_offset),
            text_tracks=parse_granny_text_tracks(data, text_count, text_offset),
        ))
    return groups


def parse_granny_animations(data: bytes, track_groups: List[GrannyTrackGroupInfo]) -> List[GrannyAnimationInfo]:
    groups_by_offset = {group.offset: group for group in track_groups}
    root_groups = grx_ref_array_from_header(data, GRX_FILE_INFO_TRACK_GROUPS)
    accumulation_flags = (struct.unpack_from('<I', data, root_groups[0] + 124)[0]
                          if root_groups and 0 < root_groups[0] and root_groups[0] + GRX_TRACK_GROUP_SIZE <= len(data) else 0)
    animations: List[GrannyAnimationInfo] = []
    for animation_index, animation_offset in enumerate(grx_ref_array_from_header(data, GRX_FILE_INFO_ANIMATIONS)):
        if animation_offset <= 0 or animation_offset + 56 > len(data):
            continue
        group_refs = grx_ref_array(data, *grx_array_header(data, animation_offset + 20))
        group_names: List[str] = []
        track_names: List[str] = []
        tracks: List[GrannyTransformTrackInfo] = []
        animation_groups: List[GrannyTrackGroupInfo] = []
        for group_offset in group_refs:
            group = groups_by_offset.get(group_offset)
            if group is None and group_offset > 0 and group_offset + 56 <= len(data):
                group = GrannyTrackGroupInfo(name=grx_cstring(data, grx_u64(data, group_offset)), offset=group_offset)
            if group is None:
                continue
            group_names.append(group.name)
            animation_groups.append(copy.copy(group))
            track_names.extend(track.name for track in group.transform_tracks)
            tracks.extend(group.transform_tracks)
        name = grx_cstring(data, struct.unpack_from("<Q", data, animation_offset)[0]) or f"animation_{animation_index}"
        animation = GrannyAnimationInfo(
            name,
            struct.unpack_from("<f", data, animation_offset + 8)[0],
            struct.unpack_from("<f", data, animation_offset + 12)[0],
            struct.unpack_from("<f", data, animation_offset + 16)[0],
            group_names,
            track_names,
            tracks,
            animation_groups,
            accumulation_flags,
            struct.unpack_from('<i', data, animation_offset + 32)[0],
            struct.unpack_from('<i', data, animation_offset + 36)[0],
        )
        for index, track in enumerate(animation.transform_tracks):
            curves = (track.orientation_curve_info, track.position_curve_info, track.scale_shear_curve_info)
            if not any(curve is not None and "DaKeyframes32f" in curve.type_name for curve in curves):
                continue
            track = copy.deepcopy(track)
            animation.transform_tracks[index] = track
            step = animation.time_step if animation.time_step > 0 else 1.0 / 30.0
            for curve, channel in ((track.orientation_curve_info, "rotation"), (track.position_curve_info, "position"), (track.scale_shear_curve_info, "scale")):
                if curve is None or "DaKeyframes32f" not in curve.type_name:
                    continue
                controls = uax_numeric_list(uax_curve_field(curve, "Controls"))
                dimension = int(uax_curve_field(curve, "Dimension", 0))
                if dimension <= 0 or len(controls) % dimension:
                    raise UgxError(f"Animation {animation.name}, track {track.name}: invalid keyframe curve dimension.")
                values = [(sample * step, controls[sample * dimension:(sample + 1) * dimension]) for sample in range(len(controls) // dimension)]
                if channel == "rotation":
                    track.rotation_keys = [(time, uax_blender_quaternion_xyzw(tuple(value))) for time, value in values if len(value) == 4]
                elif channel == "position":
                    track.position_keys = [(time, tuple(value)) for time, value in values if len(value) == 3]
                else:
                    track.scale_shear_keys = [(time, uax_scale_shear_matrix(value)) for time, value in values]
                    track.scale_keys = [(time, uax_scale_shear_diag(value)) for time, value in track.scale_shear_keys]
        first_track = 0
        for group in animation.track_groups:
            count = len(group.transform_tracks)
            group.transform_tracks = animation.transform_tracks[first_track:first_track + count]
            first_track += count
        animations.append(animation)
    return animations


def parse_granny_info(data: Optional[bytes], decode_animation_curves=True) -> GrannyInfo:
    info = GrannyInfo()
    if not data or len(data) < GRX_ROOT_SIZE:
        return info
    try:
        info.from_file_name = grx_cstring(data, struct.unpack_from("<Q", data, 16)[0]) or "gr2ugx"
        skeleton_refs = grx_ref_array_from_header(data, GRX_FILE_INFO_SKELETONS)
        mesh_refs = grx_ref_array_from_header(data, GRX_FILE_INFO_MESHES)
        model_refs = grx_ref_array_from_header(data, GRX_FILE_INFO_MODELS)
        skeleton_offset = grx_u64(data, model_refs[0] + 8) if model_refs else (skeleton_refs[0] if skeleton_refs else 0)
        if skeleton_offset:
            if 0 < skeleton_offset + 24 <= len(data):
                info.skeleton_name = grx_cstring(data, struct.unpack_from("<Q", data, skeleton_offset)[0])
                info.skeleton_lod_type = struct.unpack_from("<i", data, skeleton_offset + 20)[0]
                bone_count, bone_offset = grx_array_header(data, skeleton_offset + 8)
                for index in range(bone_count):
                    offset = bone_offset + index * GRX_BONE_SIZE
                    if offset + GRX_BONE_SIZE > len(data):
                        break
                    inverse = [list(struct.unpack_from("<4f", data, offset + 80 + row * 16)) for row in range(4)]
                    info.bones.append(GrannyBoneInfo(
                        grx_cstring(data, struct.unpack_from("<Q", data, offset)[0]),
                        struct.unpack_from("<i", data, offset + 8)[0],
                        read_grx_transform(data, offset + 12),
                        inverse,
                        struct.unpack_from("<f", data, offset + 144)[0],
                    ))
        mesh_indices_by_offset = {}
        for mesh_offset in mesh_refs:
            if mesh_offset <= 0 or mesh_offset + GRX_MESH_SIZE > len(data):
                continue
            binding_count, binding_offset = grx_array_header(data, mesh_offset + 48)
            bindings = []
            for binding_index in range(binding_count):
                offset = binding_offset + binding_index * GRX_BONE_BINDING_SIZE
                if offset + GRX_BONE_BINDING_SIZE > len(data):
                    break
                bindings.append(GrannyBoneBindingInfo(
                    grx_cstring(data, struct.unpack_from("<Q", data, offset)[0]),
                    struct.unpack_from("<3f", data, offset + 8),
                    struct.unpack_from("<3f", data, offset + 20),
                ))
            mesh_indices_by_offset.setdefault(mesh_offset, len(info.meshes))
            info.meshes.append(GrannyMeshInfo(
                grx_cstring(data, struct.unpack_from("<Q", data, mesh_offset)[0]),
                bindings,
            ))
        for model_offset in model_refs:
            if model_offset <= 0 or model_offset + GRX_MODEL_SIZE > len(data):
                continue
            binding_count, binding_offset = grx_array_header(data, model_offset + 84)
            mesh_names = []
            mesh_indices = []
            for mesh_ref in grx_ref_array(data, binding_count, binding_offset):
                if 0 < mesh_ref + GRX_MESH_SIZE <= len(data):
                    mesh_names.append(grx_cstring(data, struct.unpack_from("<Q", data, mesh_ref)[0]))
                    mesh_indices.append(mesh_indices_by_offset.get(mesh_ref, -1))
            info.models.append(GrannyModelInfo(
                grx_cstring(data, struct.unpack_from("<Q", data, model_offset)[0]),
                mesh_names,
                read_grx_transform(data, model_offset + 16),
                mesh_indices,
            ))
        info.track_groups = parse_granny_track_groups(data, decode_animation_curves)
        info.animations = parse_granny_animations(data, info.track_groups)
        info.parsed = bool(info.bones or info.meshes or info.models or info.track_groups or info.animations)
    except (struct.error, ValueError):
        return GrannyInfo()
    return info


def granny_mesh_for_model_index(info: GrannyInfo, index: int) -> Optional[GrannyMeshInfo]:
    if not info.models:
        return info.meshes[index] if 0 <= index < len(info.meshes) else None
    model = info.models[0]
    if not 0 <= index < len(model.mesh_names):
        return None
    if index < len(model.mesh_indices):
        source = model.mesh_indices[index]
        return info.meshes[source] if 0 <= source < len(info.meshes) else None
    name = model.mesh_names[index]
    if index < len(info.meshes) and info.meshes[index].name == name:
        return info.meshes[index]
    matches = [mesh for mesh in info.meshes if mesh.name == name]
    return matches[0] if len(matches) == 1 else None


def safe_grx_name(value: str, fallback: str) -> str:
    text = value or fallback
    if '\0' in text:
        raise UgxError(f'Granny name contains a null character: {text!r}')
    return text


def write_grx_array_header(data: bytearray, offset: int, length: int, array_offset: int):
    struct.pack_into("<IQ", data, offset, int(length), int(array_offset))


def write_grx_transform(data: bytearray, offset: int, transform: GrannyTransform):
    struct.pack_into("<I3f4f", data, offset, int(transform.flags), *transform.position, *transform.orientation)
    for row, values in enumerate(transform.scale_shear):
        struct.pack_into("<3f", data, offset + 32 + row * 12, *values)


def identity_grx_transform() -> GrannyTransform:
    return GrannyTransform()


class GrxWriter:
    def __init__(self):
        self.data = bytearray()
        self.string_offsets: Dict[str, int] = {}
        self.string_patches: List[Tuple[int, str]] = []

    def align(self, value: int = 16):
        while len(self.data) % value:
            self.data.append(0)

    def reserve(self, size: int, alignment: int = 16) -> int:
        self.align(alignment)
        offset = len(self.data)
        self.data.extend(b"\0" * size)
        return offset

    def string_ref(self, offset: int, value: str):
        if '\0' in value:
            raise UgxError(f'Granny string contains a null character: {value!r}')
        self.string_patches.append((offset, value))

    def finish_strings(self):
        self.align(16)
        for _, value in self.string_patches:
            if value not in self.string_offsets:
                self.string_offsets[value] = len(self.data)
                self.data.extend(value.encode("utf-8") + b"\0")
                self.align(16)
        for offset, value in self.string_patches:
            struct.pack_into("<Q", self.data, offset, self.string_offsets[value])


def write_grx_type_schema(
    writer: GrxWriter,
    extra_defs: Optional[Dict[str, List[Tuple[int, str, str, int]]]] = None,
) -> Dict[str, int]:
    schema_defs = dict(GRX_SCHEMA_DEFS)
    if extra_defs:
        schema_defs.update(extra_defs)
    writer.align(16)
    offsets: Dict[str, int] = {}
    for name, fields in schema_defs.items():
        offsets[name] = writer.reserve((len(fields) + 1) * GRX_TYPE_RECORD_SIZE, 16)
    for name, fields in schema_defs.items():
        base = offsets[name]
        for index, (member_type, field_name, ref_name, width) in enumerate(fields):
            offset = base + index * GRX_TYPE_RECORD_SIZE
            struct.pack_into("<I", writer.data, offset, member_type)
            writer.string_ref(offset + 4, field_name)
            struct.pack_into("<Q", writer.data, offset + 12, offsets.get(ref_name, 0))
            struct.pack_into("<i3iQ", writer.data, offset + 20, int(width), 0, 0, 0, 0)
    return offsets
