"""HW2 render masks and static damage-stage previews."""
import json
import math
import struct

__all__ = ('ACTION_TYPES', 'WEAPON_TYPES', 'health_region_node', 'health_stage', 'geometry_masks', 'preview_state')


ACTION_TYPES = (
    ('swappart', 'Replace Part', ''), ('throwpart', 'Detach Part', ''),
    ('togglepart', 'Show / Hide Part', ''), ('attachparticle', 'Particle Effect', ''),
    ('playsound', 'Sound', ''), ('camerashake', 'Camera Shake', ''),
    ('explosivedeath', 'Explosive Death', ''), ('swaplightstate', 'Change Light State', ''),
    ('dissolvedeath', 'Dissolve Death', ''), ('puredelay', 'Delay', ''),
    ('toggleparticle', 'Show / Hide Particle', ''), ('environmentforce', 'Environment Force', ''),
    ('CUSTOM', 'Custom', ''),
)

WEAPON_TYPES = (('Light', 'Light', ''), ('Medium', 'Medium', ''), ('Heavy', 'Heavy', ''),
                ('None', 'Health Only', 'Health changes without a weapon hit'))


def native_float(value):
    return struct.unpack('<f', struct.pack('<f', float(value)))[0]


def health_region_node(root):
    return next((node for node in root if isinstance(node.tag, str) and node.tag.casefold() == 'percentagebased'), None)


def health_stage(health, count):
    if count < 2:
        return 1 if count else 0
    # The retail tracker multiplies float32 health by (event count - 1)
    # before ceilf, then addresses events in reverse health order.
    fraction = native_float(max(0., min(100., health)) / 100.)
    scaled = native_float(fraction * (count - 1))
    return count - math.ceil(scaled)


def geometry_masks(names):
    has_intact = any('intact' in name for name in names)
    intact, damaged, fractured, dissolved = {}, {}, set(), set()
    for name in names:
        suffix = '_' + name.rsplit('_', 1)[1] if '_' in name else ''
        fracture = suffix == '_fract'
        dissolve = suffix == '_dissolve'
        intact[name] = 'intact' in name
        damaged[name] = not any(token in name for token in ('intact', 'damage')) and not fracture and not dissolve
        if fracture:
            fractured.add(name)
        if dissolve:
            dissolved.add(name)
        if 'optional' in name or name.casefold() == 'spartandriver':
            intact[name] = damaged[name] = False
        if 'neverhide' in name:
            intact[name] = damaged[name] = True
        if 'effect' in name:
            intact[name] = damaged[name] = False
    return (intact if has_intact else damaged.copy()), damaged, fractured, dissolved


def preview_state(root, names, health, regions, elements, weapon, timed=False, elapsed=0.):
    intact, damaged, fractured, dissolved = geometry_masks(names)
    threshold = next((native_float(node.text) for node in root
                      if isinstance(node.tag, str) and node.tag.casefold() == 'intacttofracturedthreshold'), native_float(.66))
    fraction = native_float(health / 100.)
    visibility = (damaged if fraction <= threshold else intact).copy()
    lookup = {name.casefold(): name for name in names}
    detached = set()
    for region in regions:
        selected = list(region.events)[1:region.active_event + 1]
        for event_index, event in enumerate(selected):
            node = elements[tuple(json.loads(event.path))]
            actions = sorted(node.findall('action'), key=lambda action: float(action.get('delay', '0')))
            for action in actions:
                filter_name = action.get('weaponFilter', 'All').casefold()
                if filter_name not in ('all', weapon.casefold()):
                    continue
                if fraction > native_float(action.get('hpThreshold', '1')):
                    continue
                if timed and event_index == len(selected) - 1 and float(action.get('delay', '0')) > elapsed:
                    continue
                parts = [lookup.get(name.strip().casefold(), name.strip()) for name in (action.text or '').split(',')]
                kind = action.get('type', '').casefold()
                if kind == 'swappart' and len(parts) == 2:
                    visibility[parts[0]], visibility[parts[1]] = False, True
                    detached.discard(parts[1])
                elif kind in ('throwpart', 'togglepart'):
                    for name in parts:
                        visibility[name] = kind == 'togglepart' and action.get('show', 'true').casefold() == 'true'
                        if kind == 'throwpart':
                            detached.add(name)
                        else:
                            detached.discard(name)
                elif kind == 'explosivedeath':
                    visibility = dict.fromkeys(names, False)
                    detached.update(fractured)
                elif kind == 'dissolvedeath':
                    visibility = {name: name in dissolved for name in names}
                    detached.clear()
    return {name.casefold(): visible for name, visible in visibility.items()}, {name.casefold() for name in detached}
