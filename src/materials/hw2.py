"""Build HW2 material records from retained data and Blender edits."""

from typing import Any, Dict, List, Optional
import bpy
import json
import struct
from ..errors import UgxError
from ..ugx.data import UgxModel
from ..ugx.read import read_materials
from ..ugx.scene import ugx_armature_for, ugx_used_material_slots
from .context import ugx_material_index_for_slot
from .data import (
    HW2_SKINNED_MATERIAL_DEFAULT,
    HW2_STATIC_MATERIAL_DEFAULT,
    HW2_MATERIAL_FIELD_NAMES,
    Material,
    build_v6_material_bdt_from_specs,
    hw2_material_spec_from_parsed,
    hogan_texture_usage,
    parse_bool_value,
    parse_ufx_shader_hash,
    hw2_buffer_storage_flags,
    ufx_buffer_bytes_from_hex,
)
from .patch import material_ufx_buffer_target_bytes
from .properties import ensure_ufx_shader_hash_string_props, ufx_attr_property_name, parse_bdt_direct_edit_value
from .color_inputs import HW2_COLOR_SHADER_INPUTS
from .skin_pairs import HW2_SHADER_SKIN_PAIRS, HW2_SKINNED_SHADER_INPUTS
from .ufx import ufx_shader_path_for_name
from .native import load_native_material, retain_native_materials
from .hw1 import v4_material_attrs_from_blender, v4_material_slots_from_blender
from ..formats.ufx import ufx_vertex_input_signatures

__all__ = ('read_hw2_vertex_color_inputs', 'material_current_ufx_value', 'material_current_ufx_buffer_bytes', 'material_current_texture_paths', 'hw2_material_spec_from_blender', 'hw2_material_spec_for_authoring', 'build_v6_material_bdt', 'build_authored_v6_material_bdt', 'v6_export_materials_for_model', 'v6_material_specs_for_model', 'v6_material_table_needs_rebuild', 'parsed_v6_material_from_blender')

def material_current_ufx_value(mat: Optional[bpy.types.Material], attrs: Dict[str, Any], name: str, default: Any) -> Any:
    if mat is not None:
        prop = ufx_attr_property_name(name)
        if prop in mat:
            return mat.get(prop)
    info = attrs.get(name, {})
    if isinstance(info, dict) and "value" in info:
        return info.get("value")
    return default


def material_current_ufx_buffer_bytes(mat: Optional[bpy.types.Material], buffers: List[Dict[str, Any]], name: str, default: bytes) -> bytes:
    for index, buffer in enumerate(buffers):
        if str(buffer.get("name", "")) != name:
            continue
        if mat is not None:
            target, _ = material_ufx_buffer_target_bytes(mat, index, buffer)
            if target is not None:
                return target
        return ufx_buffer_bytes_from_hex(str(buffer.get("hex", "")))
    return default


def material_current_texture_paths(mat: Optional[bpy.types.Material]) -> List[str]:
    if mat is None:
        return []
    paths: List[str] = []
    try:
        slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
    except Exception:
        slots = []
    for index, slot in enumerate(slots):
        key = f"hwek_ugx_texture_slot_{index}"
        value = str(mat.get(key, slot.get("path", ""))).strip()
        texture_index = int(slot.get("hogan_index", index))
        while len(paths) <= texture_index:
            paths.append("")
        paths[texture_index] = value
    if paths or "hwek_ugx_texture_slots" in mat:
        # An explicit empty array stays empty instead of reviving alias paths.
        return [path or "0" for path in paths]
    for usage in ("diffuse", "normal", "packedmask", "specular", "emissive"):
        value = str(mat.get(f"hwek_ugx_texture_{usage}", "")).strip()
        if value:
            paths.append(value)
    return paths


def hw2_material_spec_from_blender(mat: Optional[bpy.types.Material], skinned: bool) -> Dict[str, Any]:
    defaults = HW2_SKINNED_MATERIAL_DEFAULT if skinned else HW2_STATIC_MATERIAL_DEFAULT
    try:
        attrs = json.loads(mat.get("hwek_ugx_ufx_attrs", "{}")) if mat is not None else {}
    except Exception:
        attrs = {}
    try:
        buffers = json.loads(mat.get("hwek_ugx_ufx_constant_buffers", "[]")) if mat is not None else []
    except Exception:
        buffers = []
    ensure_ufx_shader_hash_string_props(mat, attrs)

    shader_names = []
    shader_hashes = []
    for index in range(4):
        shader_names.append(str(material_current_ufx_value(mat, attrs, f"shaderPermutationName{index}", defaults["shader_names"][index]) or ""))
        shader_hashes.append(parse_ufx_shader_hash(
            material_current_ufx_value(mat, attrs, f"shaderPermutationHash{index}", defaults["shader_hashes"][index]),
            f"Shader pass {index} UFX header hash",
        ))

    textures = material_current_texture_paths(mat)
    if not textures and (mat is None or "hwek_ugx_texture_slots" not in mat):
        textures = list(defaults["textures"])

    native_attrs={}
    known={f'shaderPermutation{field}{index}' for field in ('Name','Hash') for index in range(4)}
    known.update(('ufxVersion','blendMode','shadowRequiresConsts','skinned','terrainBlending'))
    for name,info in attrs.items():
        if name in known:
            continue
        current=material_current_ufx_value(mat,attrs,name,info.get('value',''))
        item=dict(info,value=current)
        if info.get('direct') and info.get('kind') in ('bool','int','float'):
            item['raw_value']=parse_bdt_direct_edit_value(info,current)
        native_attrs[name]=item

    spec = {
        "shader_names": tuple(shader_names),
        "shader_hashes": tuple(shader_hashes),
        "ufx_version": int(material_current_ufx_value(mat, attrs, "ufxVersion", defaults["ufx_version"])),
        "blend_mode": int(material_current_ufx_value(mat, attrs, "blendMode", defaults["blend_mode"])),
        "shadow_requires_consts": parse_bool_value(material_current_ufx_value(mat, attrs, "shadowRequiresConsts", defaults["shadow_requires_consts"])),
        "skinned": parse_bool_value(material_current_ufx_value(mat, attrs, "skinned", defaults["skinned"]), bool(defaults["skinned"])),
        "terrain_blending": parse_bool_value(material_current_ufx_value(mat, attrs, "terrainBlending", defaults["terrain_blending"])),
        "vscb": material_current_ufx_buffer_bytes(mat, buffers, "VSCBData", defaults["vscb"]),
        "pscb": material_current_ufx_buffer_bytes(mat, buffers, "PSCBData", defaults["pscb"]),
        "textures": tuple(textures),
        "buffer_names": [str(buffer["name"]) for buffer in buffers] if mat is not None and "hwek_ugx_ufx_constant_buffers" in mat else ["VSCBData", "PSCBData"],
        "native_attrs": native_attrs,
    }
    storage=hw2_buffer_storage_flags(buffers)
    if storage:
        spec['buffer_flags']=storage
    if (mat is not None and int(mat.get('hwek_ugx_cache_version',0))==6 and mat.get('hwek_ugx_source_path')
            and not mat.get('hwek_ugx_has_hogan_material',not ('hwek_ugx_v4_attrs' in mat and not attrs))):
        spec['legacy']={'name':str(mat.get('hwek_ugx_export_name',mat.get('hwek_ugx_source_name',mat.name))),
                        'slots':v4_material_slots_from_blender(mat),'attrs':v4_material_attrs_from_blender(mat)}
        if 'hwek_ugx_legacy_containers' in mat:
            containers=json.loads(mat['hwek_ugx_legacy_containers'])
            if set(containers)!={'NameValues','Maps'}:
                spec['legacy']['containers']=containers
    elif (mat is not None and mat.get('hwek_ugx_source_path') and 'hwek_ugx_ufx_attrs' in mat
          and not all(name in attrs for name in HW2_MATERIAL_FIELD_NAMES if name!='basePermutationName')):
        spec['field_names']=list(attrs)
    return spec


_COLOR_STATIC_NAMES = tuple(f'{name.rsplit("_",1)[0]}_{int(name.rsplit("_",1)[1],16)|(1<<41):016X}'
                            for name in HW2_STATIC_MATERIAL_DEFAULT['shader_names'])
_COLOR_STATIC_HASHES = (16876447, 3515513694, 1431920515, 1509218813)
_COLOR_SKINNED_NAMES = tuple(HW2_SHADER_SKIN_PAIRS[name][1] for name in _COLOR_STATIC_NAMES)
_COLOR_SKINNED_HASHES = tuple(HW2_SHADER_SKIN_PAIRS[name][2] for name in _COLOR_STATIC_NAMES)


def hw2_material_spec_for_authoring(mat: Optional[bpy.types.Material], skinned: bool, colored=False, promote_color=False, adapt_skin=False) -> Dict[str, Any]:
    spec = hw2_material_spec_from_blender(mat, skinned)
    if mat is not None and mat.get('hwek_ugx_source_path') and not (promote_color or adapt_skin):
        return spec
    variants = [(defaults['shader_names'], defaults['shader_hashes'], False)
                for defaults in (HW2_STATIC_MATERIAL_DEFAULT, HW2_SKINNED_MATERIAL_DEFAULT)]
    variants.extend(((_COLOR_STATIC_NAMES, _COLOR_STATIC_HASHES, True), (_COLOR_SKINNED_NAMES, _COLOR_SKINNED_HASHES, True)))
    for names, hashes, source_colored in variants:
        if spec['shader_names'] != names or spec['shader_hashes'] != hashes:
            continue
        if source_colored or (colored and (promote_color or mat is None or not mat.get('hwek_ugx_source_path'))):
            spec['shader_names'] = _COLOR_SKINNED_NAMES if skinned else _COLOR_STATIC_NAMES
            spec['shader_hashes'] = _COLOR_SKINNED_HASHES if skinned else _COLOR_STATIC_HASHES
        else:
            target = HW2_SKINNED_MATERIAL_DEFAULT if skinned else HW2_STATIC_MATERIAL_DEFAULT
            spec['shader_names'] = target['shader_names']
            spec['shader_hashes'] = target['shader_hashes']
        spec['skinned'] = skinned
        break
    return spec


def read_hw2_vertex_color_inputs(mat, skinned, colored=False, *, spec=None):
    if spec is None:
        spec = hw2_material_spec_for_authoring(mat, skinned, colored, adapt_skin=True)
    has_color, prefix = False, False
    for name, header_hash in zip(spec['shader_names'], spec['shader_hashes']):
        if not name:
            continue
        color = HW2_COLOR_SHADER_INPUTS.get(name.upper())
        if color is not None and color[0] == header_hash:
            has_color = True
            prefix |= color[1]
            continue
        contract = HW2_SKINNED_SHADER_INPUTS.get(name.upper())
        static = HW2_SHADER_SKIN_PAIRS.get(name.upper())
        if (contract is not None and contract[0] == header_hash) or (static is not None and static[0] == header_hash):
            continue
        path = ufx_shader_path_for_name(str(mat.get('hwek_ugx_source_path', '')) if mat else '', name)
        if not path:
            if colored and skinned:
                raise UgxError(f'Material {mat.name if mat else name}: the UFX file is needed to place vertex colours around its bone weights.')
            continue
        with open(path, 'rb') as handle:
            data = handle.read()
        if len(data) < 12 or struct.unpack_from('<I', data, 8)[0] != header_hash:
            raise UgxError(f'Shader {path}: its UFX header hash does not match the material.')
        inputs = ufx_vertex_input_signatures(data)
        if not inputs:
            raise UgxError(f'Shader {path}: no vertex input contract is available.')
        for signature in inputs:
            semantics = [item[0] for item in signature]
            if 'COLOR' in semantics:
                has_color = True
                prefix |= 'BLENDINDICES' in semantics and semantics.index('COLOR') < semantics.index('BLENDINDICES')
    return has_color, prefix


def build_v6_material_bdt(materials: List[Optional[bpy.types.Material]], skinned: bool) -> bytes:
    specs = [hw2_material_spec_from_blender(material, skinned) for material in materials]
    return build_authored_v6_material_bdt(materials, specs)


def _retain_source_spec(material, spec, source):
    if source is None:
        return spec
    if source.native_name is not None:
        spec['name'] = source.native_name
    if material is not None and spec['textures']:
        paths=list(spec['textures'])
        for register,path in enumerate(source.source_texture_paths):
            if hogan_texture_usage(path):
                continue
            while len(paths)<=register:
                paths.append('0')
            paths[register]=path
        spec['textures']=tuple(paths)
    spec['has_textures']=source.has_texture_array or bool(spec['textures'])
    if 'legacy' in spec:
        spec['legacy']['name']=source.native_name
        if (material is not None and 'hwek_ugx_legacy_containers' not in material
                and not source.has_hogan_material and source.legacy_containers is not None):
            spec['legacy']['containers']=source.legacy_containers
    if material is not None:
        name=material.get('hwek_ugx_export_name',material.get('hwek_ugx_source_name',material.name))
        if name==material.get('hwek_ugx_source_name','') and material.name!=material.get('hwek_ugx_blender_name',material.name):
            name=material.name
        if str(name)!=source.name:
            spec['name']=str(name)
            if 'legacy' in spec:
                spec['legacy']['name']=str(name)
    return spec


def build_authored_v6_material_bdt(materials, specs):
    if len(materials) != len(specs):
        raise UgxError('HW2 material definitions do not match their Blender material table.')
    sources = [load_native_material(material, 6) for material in materials]
    specs = [_retain_source_spec(material, spec, source) for material, spec, source in zip(materials, specs, sources)]
    return retain_native_materials(build_v6_material_bdt_from_specs(specs), sources)


def v6_export_materials_for_model(obj: bpy.types.Object, model: UgxModel) -> List[Optional[bpy.types.Material]]:
    materials: List[Optional[bpy.types.Material]] = [None] * len(model.materials)
    for slot_index, material in ugx_used_material_slots(obj):
        if material is None:
            continue
        target_index = ugx_material_index_for_slot(obj, slot_index, model)
        if target_index < 0 or target_index >= len(materials):
            raise UgxError(f"Material slot {slot_index} maps outside the UGX material table.")
        if materials[target_index] is not None and materials[target_index] is not material:
            raise UgxError(f"Multiple Blender materials map to UGX material {target_index}.")
        materials[target_index] = material
    return materials


def v6_material_specs_for_model(obj: bpy.types.Object, model: UgxModel) -> List[Dict[str, Any]]:
    skinned = bool(ugx_armature_for(obj))
    materials = v6_export_materials_for_model(obj, model)
    specs=[]
    for index,material in enumerate(materials):
        source=model.materials[index]
        spec=hw2_material_spec_from_blender(material,skinned) if material is not None else hw2_material_spec_from_parsed(source,skinned)
        specs.append(_retain_source_spec(material,spec,source))
    return specs



def v6_material_table_needs_rebuild(obj: bpy.types.Object, model: UgxModel) -> bool:
    if model.cache_version != 6:
        return False
    skinned = bool(ugx_armature_for(obj))
    specs = v6_material_specs_for_model(obj, model)
    materials = v6_export_materials_for_model(obj, model)
    fields = {
        "ufx_version": "ufxVersion", "blend_mode": "blendMode",
        "shadow_requires_consts": "shadowRequiresConsts", "skinned": "skinned",
        "terrain_blending": "terrainBlending",
    }
    for index, source in enumerate(model.materials):
        material = materials[index]
        if material is None:
            continue
        if int(material.get('hwek_ugx_name_offset',source.name_offset))!=source.name_offset:
            return True
        if 'name' in specs[index] and not source.name_capacity:
            return True
        saved_attrs = json.loads(material.get('hwek_ugx_ufx_attrs','{}'))
        if set(saved_attrs)!=set(source.ufx_attrs):
            return True
        if 'field_names' in specs[index] and set(specs[index]['field_names'])!=set(source.ufx_attrs):
            return True
        saved_slots = json.loads(material.get('hwek_ugx_texture_slots','[]'))
        saved_buffers = json.loads(material.get('hwek_ugx_ufx_constant_buffers','[]'))
        if any(any(saved_attrs.get(name,{}).get(key,0) != info.get(key,0) for key in ('file_offset','entry_offset'))
               for name,info in source.ufx_attrs.items()):
            return True
        if len(saved_slots) != len(source.slots) or any(before.get('offset',0) != after.get('offset',0)
                for before,after in zip(source.slots,saved_slots)):
            return True
        if len(saved_buffers) != len(source.ufx_constant_buffers) or any(before.get('value_file_offset',0) != after.get('value_file_offset',0)
                for before,after in zip(source.ufx_constant_buffers,saved_buffers)):
            return True
        target = specs[index]
        before = hw2_material_spec_from_parsed(source, skinned)
        requested=set(target['field_names']) if 'field_names' in target else None
        if target['buffer_names'] != before['buffer_names']:
            return True
        if len(target['textures']) != len(before['textures']):
            return True
        if target['textures'] != before['textures'] and any(not slot.get('offset') for slot in source.slots):
            return True
        changed = {name for field, name in fields.items() if target[field] != before[field] and (requested is None or name in requested)}
        for slot in range(4):
            if target['shader_names'][slot] != before['shader_names'][slot] and (requested is None or f'shaderPermutationName{slot}' in requested):
                changed.add(f'shaderPermutationName{slot}')
            if target['shader_hashes'][slot] != before['shader_hashes'][slot] and (requested is None or f'shaderPermutationHash{slot}' in requested):
                changed.add(f'shaderPermutationHash{slot}')
        for name in changed:
            info = source.ufx_attrs.get(name, {})
            if info.get('kind') == 'string':
                if not info.get('file_offset'):
                    return True
            elif not info.get('direct') or not info.get('entry_offset'):
                return True
        for field, name in (('vscb', 'VSCBData'), ('pscb', 'PSCBData')):
            if target[field] == before[field]:
                continue
            buffer = next((value for value in source.ufx_constant_buffers if value['name'] == name), {})
            if len(target[field]) != int(buffer.get('value_byte_size',0)) or not buffer.get('value_file_offset'):
                return True
    return False



def parsed_v6_material_from_blender(mat: Optional[bpy.types.Material], skinned: bool) -> Material:
    payload = build_v6_material_bdt([mat], skinned)
    parsed = read_materials(payload, 1)
    if not parsed or parsed[0].bdt_index<0:
        raise UgxError("Could not build a readable Halo Wars 2 material entry.")
    return parsed[0]
