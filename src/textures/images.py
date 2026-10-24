"""Load native textures into Blender and retain editable export sources."""

from typing import Dict, List
import bpy
import hashlib
import json
import math
import os
import struct
import tempfile
from ..errors import UgxError
from .dds import (
    decode_float_dds_surface,
    downsample_rgba2x,
    downsample_rgba_floats2x,
    encode_dds_rgba_copy,
    encode_float_dds_surface,
    find_tolerant_fourcc,
    flip_rgba_rows_bytes,
    flip_rgba_rows_floats,
    float_dds_layout,
    read_tolerant_dx10_format,
    tolerant_dds_dimension,
)
from .ddx import decode_ddx
from .header import float_to_u8, fourcc, normalize_texture_format, plausible_dimension

try:
    import numpy as np
except ImportError:
    np = None








__all__ = ('image_rgba_bytes', 'image_rgba_floats', 'blender_image_dds_bytes', 'can_rebuild_packed_dds', 'dds_image_bytes', 'write_dds_image', 'set_non_color_image', 'set_ugx_image_source', 'pack_ugx_texture_previews_before_save', 'image_float_hash', 'encode_float_dds_image', 'image_from_float_dds', 'image_from_ddx', 'missing_texture_image', 'image_from_native_dds')

def image_rgba_bytes(image: bpy.types.Image) -> bytes:
    width, height = int(image.size[0]), int(image.size[1])
    if width <= 0 or height <= 0:
        raise UgxError("Image has invalid dimensions.")
    if np is not None:
        pixels = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        rgba = np.empty(pixels.size, dtype=np.uint8)
        # Bound conversion scratch memory for large textures without changing
        # float_to_u8's half-up rounding or float32 tie handling.
        for offset in range(0, pixels.size, 256 * 1024):
            values = pixels[offset:offset + 256 * 1024]
            values[~np.isfinite(values)] = 0.0
            np.clip(values, 0.0, 1.0, out=values)
            scaled = values.astype(np.float64)
            scaled *= 255.0
            scaled += 0.5
            rgba[offset:offset + len(values)] = scaled
        return rgba.tobytes()
    pixels = [0.0] * (width * height * 4)
    image.pixels.foreach_get(pixels)
    return bytes(float_to_u8(value) for value in pixels)


def image_rgba_floats(image: bpy.types.Image) -> List[float]:
    width, height = int(image.size[0]), int(image.size[1])
    if width <= 0 or height <= 0:
        raise UgxError("Image has invalid dimensions.")
    pixels = [0.0] * (width * height * 4)
    image.pixels.foreach_get(pixels)
    return pixels


def blender_image_dds_bytes(image: bpy.types.Image) -> bytes:
    width, height = int(image.size[0]), int(image.size[1])
    rgba = flip_rgba_rows_bytes(image_rgba_bytes(image), width, height)
    mip_count = int(math.floor(math.log2(max(width, height)))) + 1
    header = bytearray(128)
    header[:4] = b"DDS "
    struct.pack_into("<7I", header, 4, 124, 0x2100F, height, width, width * 4, 0, mip_count)
    struct.pack_into("<I", header, 76, 32)
    struct.pack_into("<7I", header, 80, 0x41, 0, 32, 0xFF, 0xFF00, 0xFF0000, 0xFF000000)
    struct.pack_into("<I", header, 108, 0x401008)
    compressed = bool(image.get('anders_authored_texture'))
    if compressed:
        struct.pack_into('<I', header, 8, 0xA1007)
        struct.pack_into('<I', header, 20, ((width + 3) // 4) * ((height + 3) // 4) * 16)
        struct.pack_into('<7I', header, 80, 4, fourcc('DXT5'), 0, 0, 0, 0, 0)
    payload = bytearray()
    level_width, level_height, level_rgba = width, height, rgba
    for _ in range(mip_count):
        if compressed:
            from .compression import encode_compressed_from_rgba
            payload.extend(encode_compressed_from_rgba(level_rgba, level_width, level_height, fourcc('DXT5')))
        else:
            payload.extend(level_rgba)
        if level_width > 1 or level_height > 1:
            level_width, level_height, level_rgba = downsample_rgba2x(level_rgba, level_width, level_height)
    return bytes(header + payload)


def can_rebuild_packed_dds(image: bpy.types.Image) -> bool:
    if not getattr(image, 'packed_file', None) or not image.get('hwek_ugx_decoded_preview'):
        return False
    # Packed images load lazily after opening a project. Inspect their size
    # before has_data, which otherwise reports false for valid packed pixels.
    size = tuple(image.size)
    floating = image.is_float
    return (min(size) > 0 and image.has_data and not floating
            and not image.get('hwek_ugx_missing_texture') and not image.get('hwek_ugx_dds_surface_layout')
            and image.get('hwek_ugx_dds_is_2d', True))


def _dds_is_single_surface(data: bytes) -> bool:
    if len(data) < 128 or data[:4] != b'DDS ':
        return False
    if struct.unpack_from('<I', data, 112)[0] & (0x200 | 0x200000):
        return False
    if find_tolerant_fourcc(data) == fourcc('DX10'):
        if len(data) < 148:
            return False
        dimension, flags, count = struct.unpack_from('<3I', data, 132)
        return dimension == 3 and not (flags & 4) and count == 1
    return struct.unpack_from('<I', data, 24)[0] <= 1


def dds_image_bytes(image: bpy.types.Image, *, allow_packed_preview=False, source_cache=None, encoded_cache=None) -> bytes:
    source_path = image.get("hwek_ugx_ddx_path") or getattr(image, "filepath_raw", "") or getattr(image, "filepath", "")
    source_path = os.path.abspath(bpy.path.abspath(str(source_path))) if source_path else ""
    original = b""
    if source_path:
        if source_cache is not None and source_path in source_cache:
            original = source_cache[source_path]
        else:
            if os.path.isfile(source_path):
                with open(source_path, "rb") as handle:
                    original = handle.read()
            if source_cache is not None:
                source_cache[source_path] = original
    if original[:4] != b"DDS " and not image.get("hwek_ugx_ddx_path"):
        return blender_image_dds_bytes(image)
    if not original:
        if allow_packed_preview and can_rebuild_packed_dds(image):
            return blender_image_dds_bytes(image)
        raise UgxError(f"Texture source file is missing: {source_path or image.name}.")
    if image.get("hwek_ugx_source_float_hash"):
        current_float_hash = image_float_hash(image)
        if current_float_hash == image["hwek_ugx_source_float_hash"]:
            return original
        key = (source_path, tuple(image.size), current_float_hash, 'float')
        if encoded_cache is not None and key in encoded_cache:
            return encoded_cache[key]
        encoded = encode_float_dds_image(original, image)
        if encoded_cache is not None:
            encoded_cache[key] = encoded
        return encoded
    width, height = int(image.size[0]), int(image.size[1])
    blender_rgba = image_rgba_bytes(image)
    current_image_hash = hashlib.sha256(blender_rgba).hexdigest()
    if image.get("hwek_ugx_source_rgba_hash") == current_image_hash:
        encoded = original
    else:
        texture_format = find_tolerant_fourcc(original)
        if texture_format == fourcc("DX10"):
            texture_format = read_tolerant_dx10_format(original)
        # Distinct Blender images can carry identical edits to one native map.
        # Half-float data needs its full precision, not the quantized hash.
        key = (source_path, width, height, current_image_hash) if normalize_texture_format(texture_format) != 113 else None
        if encoded_cache is not None and key is not None and key in encoded_cache:
            return encoded_cache[key]
        rgba = flip_rgba_rows_bytes(blender_rgba, width, height)
        source_texture = None
        if not image.get("hwek_ugx_source_rgba_hash"):
            try:
                source_texture = decode_ddx(source_path)
            except Exception:
                source_texture = None
        if source_texture and source_texture.width == width and source_texture.height == height and source_texture.rgba == rgba:
            encoded = original
        else:
            dds_floats = None
            if normalize_texture_format(texture_format) == 113:
                dds_floats = flip_rgba_rows_floats(image_rgba_floats(image), width, height)
            encoded = encode_dds_rgba_copy(original, rgba, dds_floats)
        if encoded_cache is not None and key is not None:
            encoded_cache[key] = encoded
    return encoded


def write_dds_image(image: bpy.types.Image, output_path: str):
    encoded = dds_image_bytes(image)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as handle:
        handle.write(encoded)


def set_non_color_image(image: bpy.types.Image):
    if hasattr(image, "colorspace_settings"):
        try:
            image.colorspace_settings.name = "Non-Color"
        except TypeError:
            pass


def set_ugx_image_source(image: bpy.types.Image, path: str, pack_preview=True):
    if os.path.splitext(path)[1].lower() in (".ddx", ".dds", ".xpr"):
        image["hwek_ugx_ddx_path"] = os.path.abspath(path)
    else:
        image["hwek_ugx_image_path"] = os.path.abspath(path)
    if image.get("hwek_ugx_decoded_preview"):
        # Keep decoded pixels as a generated Blender image. Pointing this back
        # at .ddx makes Blender 5 try to reopen it as a normal file and display
        # magenta when its image loader/GPU backend rejects that extension or
        # compression format.
        restore_pixels = None
        if image.filepath_raw or image.filepath:
            restore_pixels = [0.0] * (int(image.size[0]) * int(image.size[1]) * 4)
            image.pixels.foreach_get(restore_pixels)
        if image.filepath_raw:
            image.filepath_raw = ""
        if image.filepath:
            image.filepath = ""
        if restore_pixels is not None:
            image.pixels.foreach_set(restore_pixels)
    else:
        image.filepath_raw = path
        image.filepath = path
    image.update()
    if pack_preview and image.get("hwek_ugx_decoded_preview"):
        image.pack()


@bpy.app.handlers.persistent
def pack_ugx_texture_previews_before_save(_unused):
    native_images = set()
    for mat in bpy.data.materials:
        if mat.get("hwek_ugx_native_material") and mat.node_tree is not None:
            native_images.update(node.image for node in mat.node_tree.nodes if node.type == "TEX_IMAGE" and node.image is not None)
    for image in bpy.data.images:
        imported_preview = image.get("hwek_ugx_decoded_preview") and image.get("hwek_ugx_ddx_path")
        generated_material_image = image in native_images
        if (imported_preview or generated_material_image) and image.has_data and (image.is_dirty or image.packed_file is None):
            pixels = image_rgba_floats(image)
            if image.packed_file is not None:
                image.unpack(method="REMOVE")
            image.pixels.foreach_set(pixels)
            image.update()
            image.pack()


def image_float_hash(image: bpy.types.Image) -> str:
    count = int(image.size[0]) * int(image.size[1]) * 4
    if np is not None:
        values = np.empty(count, dtype="<f4")
        image.pixels.foreach_get(values)
        return hashlib.sha256(values.tobytes()).hexdigest()
    values = image_rgba_floats(image)
    return hashlib.sha256(struct.pack(f"<{count}f", *values)).hexdigest()


def encode_float_dds_image(original: bytes, image: bpy.types.Image) -> bytes:
    layout = float_dds_layout(original)
    if layout is None:
        raise UgxError("Imported floating-point DDS no longer has a readable source header.")
    width, height, layers = layout["width"], layout["height"], layout["layers"]
    if tuple(image.size) != (width, height * layers):
        raise UgxError(f"DDS surface atlas must remain {width} by {height * layers} pixels.")
    pixels = flip_rgba_rows_floats(image_rgba_floats(image), width, height * layers)
    payload = bytearray(original)
    for layer in range(layers):
        offset = layout["payload_offset"] + layer * layout["surface_size"]
        values = pixels[layer * width * height * 4:(layer + 1) * width * height * 4]
        source = decode_float_dds_surface(original, offset, width * height, layout)
        if all(left == right or (math.isnan(left) and math.isnan(right)) for left, right in zip(source, values)):
            continue
        encoded = bytearray()
        level_width, level_height = width, height
        for _ in range(layout["mip_count"]):
            encoded.extend(encode_float_dds_surface(values, layout))
            level_width, level_height, values = downsample_rgba_floats2x(values, level_width, level_height)
        payload[offset:offset + layout["surface_size"]] = encoded
    return bytes(payload)


def image_from_float_dds(path: str, data: bytes, layout: Dict[str, int], pack_preview=True) -> bpy.types.Image:
    width, height, layers = layout["width"], layout["height"], layout["layers"]
    surfaces = [decode_float_dds_surface(data, layout["payload_offset"] + layer * layout["surface_size"], width * height, layout) for layer in range(layers)]
    if np is not None:
        pixels = np.concatenate(surfaces).reshape(height * layers, width, 4)[::-1].copy().reshape(-1)
    else:
        pixels = flip_rgba_rows_floats([value for surface in surfaces for value in surface], width, height * layers)
    image = bpy.data.images.new(os.path.splitext(os.path.basename(path))[0], width, height * layers, alpha=True, float_buffer=True)
    set_non_color_image(image)
    image.file_format = "OPEN_EXR"
    image.pixels.foreach_set(pixels)
    image["hwek_ugx_decoded_preview"] = True
    image["hwek_ugx_dds_rows_to_blender"] = True
    image["hwek_ugx_dds_surface_layout"] = json.dumps(layout)
    set_ugx_image_source(image, path, pack_preview)
    image["hwek_ugx_source_float_hash"] = image_float_hash(image)
    return image


def image_from_ddx(path: str, non_color: bool = False, pack_preview=True) -> bpy.types.Image:
    if os.path.splitext(path)[1].lower() not in (".ddx", ".dds", ".xpr"):
        image = bpy.data.images.load(path, check_existing=False)
        if min(image.size) <= 0:
            bpy.data.images.remove(image)
            raise UgxError(f"Blender could not decode texture {path}.")
        if non_color:
            set_non_color_image(image)
        set_ugx_image_source(image, path, pack_preview)
        if pack_preview:
            image.pack()
        return image
    try:
        with open(path, "rb") as handle:
            header = handle.read(148)
            native_dds = header[:4] == b"DDS "
            if native_dds and len(header) <= 128:
                raise UgxError(f"DDS texture {path} contains no pixel data.")
            if native_dds:
                width, height = tolerant_dds_dimension(header, 16), tolerant_dds_dimension(header, 12)
                if not plausible_dimension(width) or not plausible_dimension(height):
                    raise UgxError(f"DDS texture {path} has invalid dimensions: {width} by {height}.")
            float_layout = None
            header_format = find_tolerant_fourcc(header) if native_dds else 0
            floating = header_format in (111, 112, 113, 114, 115, 116) or (
                header_format == fourcc("DX10") and read_tolerant_dx10_format(header) in (2, 10, 16, 26, 34, 41, 54))
            if floating:
                handle.seek(0)
                data = handle.read()
                float_layout = float_dds_layout(data)
                if float_layout is not None:
                    return image_from_float_dds(path, data, float_layout, pack_preview)
    except OSError:
        native_dds = False
    if native_dds:
        try:
            return image_from_native_dds(path, UgxError("Native DDS preview failed"), non_color, pack_preview)
        except Exception:
            # The pure-Python decoder remains the compatibility fallback for
            # DDS variants Blender cannot load natively.
            pass
    try:
        texture = decode_ddx(path)
    except Exception as original:
        with open(path, "rb") as handle:
            if handle.read(4) != b"DDS ":
                raise
        return image_from_native_dds(path, original, non_color, pack_preview)
    image = bpy.data.images.new(os.path.splitext(os.path.basename(path))[0], texture.width, texture.height, alpha=True)
    if non_color:
        set_non_color_image(image)
    image["hwek_ugx_decoded_preview"] = True
    image['hwek_ugx_dds_is_2d'] = native_dds and _dds_is_single_surface(header)
    blender_rgba = flip_rgba_rows_bytes(texture.rgba, texture.width, texture.height)
    if np is not None:
        pixels = np.frombuffer(blender_rgba, dtype=np.uint8).astype(np.float32) / 255.0
        image.pixels.foreach_set(pixels)
    else:
        image.pixels.foreach_set([component / 255.0 for component in blender_rgba])
    set_ugx_image_source(image, texture.path, pack_preview)
    # Keep the decoded source hash before uploading bytes as Blender floats.
    # Reading the full float pixel buffer back here made normal imports spend
    # most of their time quantizing millions of values a second time.
    image["hwek_ugx_source_rgba_hash"] = hashlib.sha256(blender_rgba).hexdigest()
    image["hwek_ugx_dds_rows_to_blender"] = True
    return image


def missing_texture_image(texture_path: str, usage: str) -> bpy.types.Image:
    name = os.path.splitext(os.path.basename(texture_path.replace("\\", "/")))[0] or f"missing_{usage}"
    image = bpy.data.images.new(name + "_missing", 1, 1, alpha=True)
    image.pixels.foreach_set([1.0, 0.0, 1.0, 1.0])
    image["hwek_ugx_texture_path"] = texture_path
    image["hwek_ugx_missing_texture"] = True
    image.update()
    return image


def image_from_native_dds(path: str, original: Exception, non_color: bool = False, pack_preview=True) -> bpy.types.Image:
    temp_path = ""
    original_bytes = b""
    source_image = None
    try:
        load_path = path
        with open(path, "rb") as source:
            original_bytes = source.read()
        if os.path.splitext(path)[1].lower() != ".dds":
            fd, temp_path = tempfile.mkstemp(prefix="hwek_ugx_", suffix=".dds")
            with os.fdopen(fd, "wb") as temp:
                temp.write(original_bytes)
            load_path = temp_path
        source_image = bpy.data.images.load(load_path, check_existing=False)
        if source_image.size[0] <= 0 or source_image.size[1] <= 0:
            raise UgxError("Native DDS loader returned an empty image.")
        if non_color:
            set_non_color_image(source_image)

        # Blender can decode formats such as BC7 on the CPU but may be unable
        # to upload the compressed DDS to the active GPU backend. Copying the
        # decoded pixels into a generated image guarantees an uncompressed,
        # editable preview instead of leaving the material black.
        width, height = int(source_image.size[0]), int(source_image.size[1])
        if np is not None:
            pixels = np.empty(width * height * 4, dtype=np.float32)
            source_image.pixels.foreach_get(pixels)
        else:
            pixels = [0.0] * (width * height * 4)
            source_image.pixels.foreach_get(pixels)
        image = bpy.data.images.new(os.path.splitext(os.path.basename(path))[0], width, height, alpha=True)
        if non_color:
            set_non_color_image(image)
        image.pixels.foreach_set(pixels)
        image["hwek_ugx_decoded_preview"] = True
        image['hwek_ugx_dds_is_2d'] = _dds_is_single_surface(original_bytes)
        image["hwek_ugx_native_dds"] = True
        image["hwek_ugx_dds_rows_to_blender"] = True
        image["hwek_ugx_uncompressed_preview"] = True
        set_ugx_image_source(image, path, pack_preview)
        image["hwek_ugx_source_rgba_hash"] = hashlib.sha256(image_rgba_bytes(image)).hexdigest()
        return image
    except Exception as native:
        raise UgxError(f"{original}; native DDS load failed: {native}") from native
    finally:
        if source_image is not None and source_image.users == 0:
            bpy.data.images.remove(source_image)
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
