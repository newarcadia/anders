"""Overlay authored material records on retained native trees."""
from ..errors import UgxError
from ..formats.bdt_edit import (BdtRecord, bdt_entry_payload, bdt_record_name,
                                copy_bdt_entry, read_bdt_document, write_bdt_document)
from .data import HW2_MATERIAL_FIELD_NAMES, V4_MATERIAL_MAP_USAGES, v4_material_usage
from ..textures.paths import disabled_texture_path

__all__ = ('rebuild_material_tree',)


def _name(document, index):
    node=document.nodes[index]
    return bdt_record_name(document,node.entries[0]) if node is not None and node.entries else ''


def _children(document, parent):
    return [index for index,node in enumerate(document.nodes) if node is not None and node.parent==parent]


def _materials(document):
    return [index for root in _children(document,0xffff) if _name(document,root)=='Materials'
            for index in _children(document,root) if _name(document,index)=='Material']


def _remove(document, index):
    for child in _children(document,index):
        _remove(document,child)
    document.nodes[index]=None


def _clone(destination, source, index, parent):
    before=source.nodes[index]
    same_source=destination.source==source.source
    entries=list(before.entries) if same_source else [copy_bdt_entry(destination,source,entry) for entry in before.entries]
    result=len(destination.nodes)
    destination.nodes.append(BdtRecord(parent,entries,before.source_index if same_source else -1,True))
    for child in _children(source,index):
        _clone(destination,source,child,result)
    return result


def _same_value(destination, before, patch, after):
    if ((before[2]^after[2])&0xfe)!=0:
        return False
    old=bdt_entry_payload(destination,before)
    new=bdt_entry_payload(patch,after)
    if old is None or new is None:
        if (before[2]>>2)&7==1:
            return bool(before[0])==bool(after[0])
        return before[0]==after[0]
    if (before[2]>>2)&7==4:
        if bdt_record_name(destination,before)=='textures':
            def strings(payload):
                values=payload.split(b'\0')
                return values[:values.index(b'')] if b'' in values else values
            return strings(old)==strings(new)
        return old.split(b'\0',1)[0]==new.split(b'\0',1)[0]
    return old==new


def _overlay(destination, index, patch, patch_index):
    node=destination.nodes[index]
    target=patch.nodes[patch_index]
    if _name(destination,index)=='HoganMaterial':
        missing=set(HW2_MATERIAL_FIELD_NAMES)-{bdt_record_name(patch,entry) for entry in target.entries}
        node.entries=[entry for entry in node.entries if bdt_record_name(destination,entry) not in missing]
    for entry_index,entry in enumerate(target.entries):
        name=bdt_record_name(patch,entry)
        field=0 if entry_index==0 and node.entries else next((i for i,value in enumerate(node.entries) if bdt_record_name(destination,value)==name),None)
        if field is None:
            node.entries.append(copy_bdt_entry(destination,patch,entry))
        elif not _same_value(destination,node.entries[field],patch,entry):
            value=copy_bdt_entry(destination,patch,entry)
            node.entries[field]=(value[0],node.entries[field][1],value[2])
    children=_children(destination,index)
    targets=_children(patch,patch_index)
    parent_name=_name(destination,index)
    def child_name(document, child):
        name=_name(document,child)
        return v4_material_usage(name) if parent_name=='Maps' else name
    target_names={child_name(patch,child) for child in targets}
    removable={'VSCBData','PSCBData'} if parent_name=='HoganMaterial' else {'Map'} if v4_material_usage(parent_name) in V4_MATERIAL_MAP_USAGES else set()
    for child in children:
        if child_name(destination,child) in removable-target_names:
            if _name(destination,child)=='Map':
                name=next((entry for entry in destination.nodes[child].entries if bdt_record_name(destination,entry)=='Name'),None)
                payload=bdt_entry_payload(destination,name) if name is not None else None
                path=payload.split(b'\0',1)[0].decode('utf-8','replace') if payload else ''
                if not path or disabled_texture_path(path):
                    continue
            _remove(destination,child)
    for child in targets:
        name=child_name(patch,child)
        source_child=next((old for old in children if destination.nodes[old] is not None and child_name(destination,old)==name),None)
        if source_child is None:
            _clone(destination,patch,child,index)
        else:
            _overlay(destination,source_child,patch,child)


def rebuild_material_tree(source, target, materials):
    """Apply the authored schema while retaining unexposed native records."""
    destination=read_bdt_document(source)
    patch=read_bdt_document(target)
    patches=_materials(patch)
    roots=[index for index in _children(destination,0xffff) if _name(destination,index)=='Materials']
    if len(patches)!=len(materials) or len(roots)>1:
        raise UgxError('Authored material table does not match its native definitions.')
    if not roots:
        root=len(destination.nodes)
        destination.nodes.append(BdtRecord(0xffff,[(0,'Materials',256)]))
    else:
        root=roots[0]
    originals=_materials(destination)
    documents={source:read_bdt_document(source)}
    used=set()
    ordered=[]
    for index,material in enumerate(materials):
        blob=material.bdt_data if material is not None else b''
        origin=material.bdt_index if material is not None else -1
        if blob and blob not in documents:
            documents[blob]=read_bdt_document(blob)
        source_document=documents.get(blob) if blob else None
        candidates=_materials(source_document) if source_document is not None else []
        if 0<=origin<len(candidates):
            if blob==source and origin<len(originals) and originals[origin] not in used:
                record=originals[origin]
                used.add(record)
            else:
                record=_clone(destination,source_document,candidates[origin],root)
                if blob!=source:
                    for section in source_document.user_sections:
                        if section not in destination.user_sections:
                            destination.user_sections.append(section)
            _overlay(destination,record,patch,patches[index])
        else:
            record=_clone(destination,patch,patches[index],root)
        ordered.append(record)
    for index in originals:
        if index not in used:
            _remove(destination,index)
    # Native material indices follow their sibling order. Move records in the
    # graph without changing opaque entries or sharing mutable subtrees.
    material_slots=sorted(ordered)
    replacement=dict(zip(material_slots,ordered))
    new_nodes=list(destination.nodes)
    old_to_new={index:index for index in range(len(new_nodes))}
    for slot,old in replacement.items():
        new_nodes[slot]=destination.nodes[old]
        old_to_new[old]=slot
    for node in new_nodes:
        if node is not None and node.parent!=0xffff:
            node.parent=old_to_new[node.parent]
    destination.nodes=new_nodes
    return write_bdt_document(destination)
