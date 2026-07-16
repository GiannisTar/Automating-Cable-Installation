#!/usr/bin/env python3
"""
Wrap cable geoms into a parent <body> so the whole cable can be moved by changing the body's `pos`/`quat`.

The script looks for geoms with a matching `rgba` (default orange "1 0.5 0 1") and moves them under
a new body named `cable_group` with a given `pos` (pivot). All `fromto` and `pos` coordinates are converted
into the new body's local frame by subtracting the body `pos`.

Usage:
  python3 scripts/wrap_cable.py -f models/empty_scene.xml.bak2 -o models/empty_scene_wrapped.xml
  python3 scripts/wrap_cable.py -f models/empty_scene.xml.bak2 -o models/empty_scene_wrapped.xml --pos -0.2 0 0.74

Options:
  --rgba : rgba to match (4 floats as string). If none provided, defaults to "1 0.5 0 1".
  --pos  : position for the new parent body (3 floats). If omitted, the script uses the first anchor's pos
           or the average of matched geom positions as a fallback.

Notes:
- This only changes geometry placement; it does not attempt to scale the geoms. To change scale, use
  the earlier `transform_cable.py` script or update `size`/`fromto` values.
- After wrapping, move the entire cable by editing the new body's `pos` attribute.
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import sys


def parse_floats(s):
    return [float(x) for x in s.strip().split() if x != '']


def fmt(v):
    return ' '.join(f"{x:.6f}" for x in v)


def find_rgba_match(elem, target_rgba):
    rgba = elem.get('rgba')
    if rgba is None:
        return False
    # compare floats with simple string normalized spacing
    return ' '.join(rgba.split()) == ' '.join(target_rgba.split())


def subtract_vec(v, ref):
    return [v[i] - ref[i] for i in range(3)]


def main():
    p = argparse.ArgumentParser(description='Wrap cable geoms into a parent body')
    p.add_argument('-f','--file', required=True, help='input XML')
    p.add_argument('-o','--out', required=True, help='output XML')
    p.add_argument('--rgba', default='1 0.5 0 1', help='rgba string to match cable geoms (default orange)')
    p.add_argument('--pos', nargs=3, type=float, help='pos for the new parent body (x y z)')
    args = p.parse_args()

    infile = Path(args.file)
    if not infile.exists():
        print('ERROR: input file not found', infile, file=sys.stderr)
        sys.exit(2)

    tree = ET.parse(str(infile))
    root = tree.getroot()

    target_rgba = args.rgba

    # Collect matching geom elements (anywhere in worldbody)
    world = root.find('worldbody')
    if world is None:
        print('ERROR: no <worldbody> found', file=sys.stderr)
        sys.exit(2)

    matched = []
    # iterate with parent tracking so we can remove from parent later
    for parent in list(world.iter()):
        for child in list(parent):
            if child.tag == 'geom' and find_rgba_match(child, target_rgba):
                matched.append((parent, child))

    if not matched:
        print('No geoms matched rgba', target_rgba)
        sys.exit(0)

    # Determine pivot pos for the new parent body
    if args.pos:
        pivot = list(args.pos)
    else:
        # try to find an anchor sphere among matched geoms (type sphere with pos)
        pivot = None
        for parent, g in matched:
            if g.get('type') == 'sphere' and 'pos' in g.attrib:
                pivot = parse_floats(g.get('pos'))[:3]
                break
        if pivot is None:
            # fallback: average of all matched geom positions (for fromto, average endpoints)
            pts = []
            for parent, g in matched:
                if 'pos' in g.attrib:
                    pts.append(parse_floats(g.get('pos'))[:3])
                elif 'fromto' in g.attrib:
                    v = parse_floats(g.get('fromto'))
                    p1, p2 = v[:3], v[3:6]
                    pts.append([(p1[i]+p2[i])/2.0 for i in range(3)])
            if pts:
                pivot = [sum(col)/len(col) for col in zip(*pts)]
            else:
                pivot = [0.0,0.0,0.0]

    # Create new body element
    cable_body = ET.Element('body', {'name':'cable_group', 'pos':fmt(pivot)})

    # Move matched geoms under new body, adjusting coordinates to be local to the body's pos
    moved_count = 0
    for parent, g in matched:
        # detach from parent
        parent.remove(g)
        # adjust coordinates
        if 'pos' in g.attrib:
            old = parse_floats(g.get('pos'))[:3]
            new = subtract_vec(old, pivot)
            g.set('pos', fmt(new))
        if 'fromto' in g.attrib:
            v = parse_floats(g.get('fromto'))
            p1 = v[:3]
            p2 = v[3:6]
            np1 = subtract_vec(p1, pivot)
            np2 = subtract_vec(p2, pivot)
            g.set('fromto', fmt(np1 + np2))
        # append to cable body
        cable_body.append(g)
        moved_count += 1

    # Insert cable_body into worldbody (append at end)
    world.append(cable_body)

    # Write output file
    outpath = Path(args.out)
    tree.write(str(outpath), encoding='utf-8', xml_declaration=True)

    print(f'Wrote {outpath} — moved {moved_count} geoms into <body name="cable_group"> with pos={fmt(pivot)}')

if __name__ == '__main__':
    main()
