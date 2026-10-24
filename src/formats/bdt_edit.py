"""Edit native tree records while retaining their source storage."""
from dataclasses import dataclass
import struct

from ..errors import UgxError
from .bdt import bdt_node_topology, refresh_bdt_crc_in_place, validate_bdt_node_topology

__all__ = ('BdtRecord', 'BdtDocument', 'read_bdt_document', 'write_bdt_document',
           'bdt_record_name', 'bdt_entry_payload', 'copy_bdt_entry')


@dataclass
class BdtRecord:
    parent: int
    entries: list
    source_index: int = -1
    private_entries: bool = False


@dataclass
class BdtDocument:
    source: bytes
    nodes: list
    names: bytearray
    values: bytearray
    user_sections: list


def read_bdt_document(data):
    header,nodes,_,_=bdt_node_topology(data)
    if header!=28:
        raise UgxError('Editing material trees requires the native seven-word header.')
    node_bytes,entry_bytes,name_bytes,value_bytes=struct.unpack_from('<4I',data,12)
    entries=header+data[3]*12+node_bytes
    names=entries+entry_bytes
    padding=names+name_bytes
    values=(padding+15)&~15 if value_bytes else padding
    if entries+entry_bytes>len(data) or values+value_bytes>len(data) or entry_bytes%8:
        raise UgxError('Material tree tables extend outside their source chunk.')
    records=[]
    for index,(parent,_,first,count,_) in enumerate(nodes):
        if (first+count)*8>entry_bytes:
            raise UgxError(f'Material tree node {index} has invalid value entries.')
        fields=[struct.unpack_from('<IHH',data,entries+entry*8) for entry in range(first,first+count)]
        records.append(BdtRecord(parent,fields,index))
    sections=[]
    for index in range(data[3]):
        kind,size,offset=struct.unpack_from('<III',data,header+index*12)
        if offset>len(data) or size>len(data)-offset:
            raise UgxError('Material user section extends outside its source chunk.')
        sections.append((kind,bytes(data[offset:offset+size])))
    # Retain old padding as unreferenced name storage when the tables move.
    return BdtDocument(bytes(data),records,bytearray(data[names:values]),
                       bytearray(data[values:values+value_bytes]),sections)


def bdt_record_name(document, entry):
    name=entry[1]
    if isinstance(name,str):
        return name
    if isinstance(name,bytes):
        return name.decode('utf-8','replace')
    if not 0<=name<len(document.names):
        raise UgxError('Material entry references an invalid name.')
    end=document.names.find(0,name)
    if end<0:
        raise UgxError('Material entry name is not terminated.')
    return bytes(document.names[name:end]).decode('utf-8','replace')


def bdt_entry_payload(document, entry):
    raw,_,flags=entry
    if flags&2 or (flags>>2)&7 in (0,1):
        return None
    size=(flags>>9)&127
    if size==127:
        if raw<4 or raw>len(document.values):
            raise UgxError('Material entry has an invalid indirect length.')
        size=struct.unpack_from('<I',document.values,raw-4)[0]
    if raw<=len(document.values) and size>len(document.values)-raw and (flags>>2)&7==4:
        # Some edited assets retain an older string capacity after shortening
        # the final value. Keep the terminated text within the actual pool.
        tail=document.values[raw:]
        if 0 in tail:
            return bytes(tail)
    if raw>len(document.values) or size>len(document.values)-raw:
        raise UgxError('Material entry extends outside its value storage.')
    payload=bytes(document.values[raw:raw+size])
    if (flags>>2)&7==4 and payload and b'\0' not in payload and bdt_record_name(document,entry)!='textures':
        end=document.values.find(0,raw)
        if end>=0:
            return bytes(document.values[raw:end+1])
    return payload


def copy_bdt_entry(destination, source, entry):
    raw,name,flags=entry
    if source is destination:
        return entry
    if isinstance(name,int):
        end=source.names.find(0,name)
        if not 0<=name<len(source.names) or end<0:
            raise UgxError('Material entry has an invalid source name.')
        name=bytes(source.names[name:end])
    payload=bdt_entry_payload(source,entry)
    if payload is not None:
        size=len(payload)
        flags=(flags&511)|(min(size,127)<<9)
        alignment=max(1<<((flags>>5)&7),4 if size>=127 else 1)
        prefix=4 if size>=127 else 0
        destination.values.extend(bytes((-(len(destination.values)+prefix))%alignment))
        if prefix:
            destination.values.extend(struct.pack('<I',size))
        raw=len(destination.values)
        destination.values.extend(payload)
    return raw,name,flags


def write_bdt_document(document):
    original=read_bdt_document(document.source)
    if document.nodes==original.nodes and document.names==original.names and document.values==original.values and document.user_sections==original.user_sections:
        return document.source
    source=document.source
    header=28
    old_node_bytes,old_entry_bytes,old_name_bytes,old_value_bytes=struct.unpack_from('<4I',source,12)
    old_entries=header+source[3]*12+old_node_bytes
    old_names=old_entries+old_entry_bytes
    old_values=(old_names+old_name_bytes+15)&~15 if old_value_bytes else old_names+old_name_bytes
    old_end=old_values+old_value_bytes
    declared_end=header+struct.unpack_from('<I',source,8)[0]
    if not old_end<=declared_end<=len(source):
        raise UgxError('Material tree has an invalid declared data extent.')
    if len(document.user_sections)>255:
        raise UgxError('Material tree exceeds its native user-section limit.')
    active={index for index,node in enumerate(document.nodes) if node is not None}
    roots=[]
    children={index:[] for index in active}
    for index in sorted(active):
        parent=document.nodes[index].parent
        if parent==0xffff:
            roots.append(index)
        elif parent in active:
            children[parent].append(index)
        else:
            raise UgxError(f'Material node {index} references a removed or missing parent.')
    order=list(roots)
    visited=set(roots)
    for index in order:
        for child in children[index]:
            if child in visited:
                raise UgxError('Material tree has a repeated node link.')
            visited.add(child)
            order.append(child)
    if visited!=active or len(order)>65535:
        raise UgxError('Material tree is cyclic or exceeds its native node limit.')
    indices={old:new for new,old in enumerate(order)}
    names=bytearray(document.names)
    name_offsets={}
    entries=bytearray(source[old_entries:old_entries+old_entry_bytes])
    records=bytearray()
    for index in order:
        node=document.nodes[index]
        before=original.nodes[node.source_index] if 0<=node.source_index<len(original.nodes) else None
        same_fields=before is not None and before.entries==node.entries
        encoded=bytearray()
        for field,(raw,name,flags) in enumerate(node.entries):
            if not isinstance(name,int):
                key=name.encode('utf-8') if isinstance(name,str) else name
                if b'\0' in key:
                    raise UgxError('Material field name contains a null byte.')
                if key not in name_offsets:
                    name_offsets[key]=len(names)
                    names.extend(key+b'\0')
                name=name_offsets[key]
            if not 0<=name<=65535 or not 0<=raw<=0xffffffff:
                raise UgxError('Material entry exceeds its native offset limits.')
            if not same_fields:
                flags=(flags&~256)|(256 if field==len(node.entries)-1 else 0)
            encoded.extend(struct.pack('<IHH',raw,name,flags))
        if same_fields and not node.private_entries:
            first=struct.unpack_from('<H',source,header+source[3]*12+node.source_index*8+4)[0]
        else:
            first=len(entries)//8
            entries.extend(encoded)
        if first>65535 or len(node.entries)>255 or len(children[index])>255:
            raise UgxError('Material tree exceeds its native field or child limits.')
        child_indices=[indices[child] for child in children[index]]
        parent=0xffff if node.parent==0xffff else indices[node.parent]
        records.extend(struct.pack('<HHHBB',parent,child_indices[0] if child_indices else 0xffff,
                                   first,len(node.entries),len(child_indices)))
    result=bytearray(source[:header])
    result[3]=len(document.user_sections)
    result.extend(bytes(len(document.user_sections)*12))
    result.extend(records)
    result.extend(entries)
    result.extend(names)
    if document.values:
        result.extend(bytes((-len(result))%16))
    result.extend(document.values)
    result.extend(source[old_end:declared_end])
    for index,(kind,payload) in enumerate(document.user_sections):
        offset=len(result)
        result.extend(payload)
        struct.pack_into('<III',result,header+index*12,kind,len(payload),offset)
    data_size=len(result)-header
    result.extend(source[declared_end:])
    sizes=(data_size,len(records),len(entries),len(names),len(document.values))
    if max(sizes)>0xffffffff:
        raise UgxError('Material tree exceeds its native data size limit.')
    struct.pack_into('<5I',result,8,*sizes)
    refresh_bdt_crc_in_place(result,0,len(result))
    validate_bdt_node_topology(bytes(result))
    return bytes(result)
