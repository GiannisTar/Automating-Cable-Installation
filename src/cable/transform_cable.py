#!/usr/bin/env python3
"""
Scale and translate cable geoms in a MuJoCo XML scene.
Handles:
 - `geom type="capsule" fromto="x1 y1 z1 x2 y2 z2"`
 - `geom` with `pos="x y z"` (e.g. anchor spheres)
 - `size` attributes (radius) are multiplied by `--size-scale`

Usage examples:
 python3 scripts/transform_cable.py --file models/empty_scene.xml.bak2 --out models/empty_scene_cable_scaled.xml --scale 1.2 --size-scale 1.2 --translate 0.02 0 0
 python3 scripts/transform_cable.py -f models/empty_scene.xml.bak2 -o models/empty_scene_shifted.xml --translate 0 0 -0.05

"""
import argparse
import xml.etree.ElementTree as ET
import sys
from pathlib import Path


def parse_floats(s):
    return [float(x) for x in s.strip().split() if x != '']


def format_vec(v):
    return ' '.join(f"{x:.6f}" for x in v)


def transform_point(p, pivot, scale, translate):
    return [pivot[i] + scale * (p[i] - pivot[i]) + translate[i] for i in range(3)]


def main():
    p = argparse.ArgumentParser(description="Scale/translate MuJoCo cable geoms (fromto/pos/size)")
    p.add_argument('-f', '--file', required=True, help='Input XML file (e.g. models/empty_scene.xml.bak2)')
    p.add_argument('-o', '--out', required=True, help='Output XML file')
    p.add_argument('--scale', type=float, default=1.0, help='Uniform scale applied to coordinates around pivot (default 1.0)')
    p.add_argument('--size-scale', type=float, default=1.0, help='Scale applied to `size` attributes (radius)')
    p.add_argument('--translate', nargs=3, type=float, default=[0.0,0.0,0.0], help='Translate vector (x y z) applied after scale')
    p.add_argument('--pivot', nargs=3, type=float, default=[0.0,0.0,0.0], help='Pivot point for scaling (default world origin)')

    args = p.parse_args()
    infile = Path(args.file)
    outfile = Path(args.out)

    if not infile.exists():
        print(f"ERROR: input file not found: {infile}", file=sys.stderr)
        sys.exit(2)

    tree = ET.parse(str(infile))
    root = tree.getroot()

    scale = float(args.scale)
    size_scale = float(args.size_scale)
    translate = [float(x) for x in args.translate]
    pivot = [float(x) for x in args.pivot]

    changed = 0

    # Find all geom elements anywhere under root
    for geom in root.findall('.//geom'):
        gtype = geom.get('type', '').lower()
        # update fromto if present
        if 'fromto' in geom.attrib:
            vals = parse_floats(geom.get('fromto'))
            if len(vals) >= 6:
                p1 = vals[0:3]
                p2 = vals[3:6]
                np1 = transform_point(p1, pivot, scale, translate)
                np2 = transform_point(p2, pivot, scale, translate)
                geom.set('fromto', format_vec(np1 + np2))
                # scale radius/size if present
                if 'size' in geom.attrib:
                    try:
                        old = float(geom.get('size'))
                        geom.set('size', f"{old * size_scale:.6f}")
                    except Exception:
                        pass
                changed += 1
        # otherwise transform pos attribute if present (e.g. anchors)
        elif 'pos' in geom.attrib:
            vals = parse_floats(geom.get('pos'))
            if len(vals) >= 3:
                npv = transform_point(vals[0:3], pivot, scale, translate)
                geom.set('pos', format_vec(npv))
                if 'size' in geom.attrib:
                    try:
                        old = float(geom.get('size'))
                        geom.set('size', f"{old * size_scale:.6f}")
                    except Exception:
                        pass
                changed += 1
        else:
            # Some geoms may be defined with center/other attributes; skip
            continue

    # Write output
    tree.write(str(outfile), encoding='utf-8', xml_declaration=True)
    print(f"Wrote {outfile} — transformed {changed} geom elements (scale={scale}, size_scale={size_scale}, translate={translate}, pivot={pivot})")


if __name__ == '__main__':
    main()
