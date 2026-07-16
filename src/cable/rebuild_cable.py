#!/usr/bin/env python3
"""
Rebuild cable from existing orange capsule chain so all capsules align consistently.

Algorithm:
- Find cable geoms by rgba (default "1 0.5 0 1").
- Identify anchor points (sphere geoms with same rgba and pos) for endpoints.
- For capsule geoms with `fromto`, compute midpoints and sort them along the axis between anchors.
- Smooth the midpoint polyline with a small moving-average window.
- Resample the smoothed polyline to `n_segments` points (default = original segment count) along curve length.
- Create new capsules between successive resampled points with consistent from->to directions.
- Preserve other worldbody content; replace the matched cable geoms with the rebuilt ones.

Usage examples:
  python3 scripts/rebuild_cable.py -f models/empty_scene.xml.bak2 -o models/empty_scene_rebuilt.xml
  python3 scripts/rebuild_cable.py -f models/empty_scene_wrapped.xml -o models/empty_scene_rebuilt.xml --segments 60 --size 0.0025

"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import math
import sys


def parse_floats(s):
    return [float(x) for x in s.strip().split() if x != '']


def fmt(v):
    return ' '.join(f"{x:.6f}" for x in v)


def add(v,w):
    return [v[i]+w[i] for i in range(3)]

def sub(v,w):
    return [v[i]-w[i] for i in range(3)]

def mul(v,s):
    return [v[i]*s for i in range(3)]

def length(v):
    return math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])


def cumulative_lengths(pts):
    d=[0.0]
    for i in range(1,len(pts)):
        d.append(d[-1]+length(sub(pts[i], pts[i-1])))
    return d


def resample_polyline(pts, n):
    if n <= 1:
        return pts[:]
    d = cumulative_lengths(pts)
    total = d[-1]
    if total == 0:
        return [pts[0]]*n
    out = []
    for k in range(n+1):
        t = (k/ n) * total
        # find segment
        for i in range(1,len(d)):
            if d[i] >= t:
                break
        else:
            i = len(d)-1
        if d[i] == d[i-1]:
            out.append(pts[i])
            continue
        ratio = (t - d[i-1]) / (d[i] - d[i-1])
        p = [pts[i-1][j] + ratio * (pts[i][j]-pts[i-1][j]) for j in range(3)]
        out.append(p)
    return out


def smooth_points(pts, window=3):
    if window <= 1 or len(pts) < window:
        return pts[:]
    half = window//2
    out = []
    n = len(pts)
    for i in range(n):
        acc = [0.0,0.0,0.0]
        cnt = 0
        for j in range(max(0,i-half), min(n, i+half+1)):
            acc = add(acc, pts[j])
            cnt += 1
        out.append([acc[k]/cnt for k in range(3)])
    return out


def main():
    p = argparse.ArgumentParser(description='Rebuild cable segments to align capsules cleanly')
    p.add_argument('-f','--file', required=True, help='input XML')
    p.add_argument('-o','--out', required=True, help='output XML')
    p.add_argument('--rgba', default='1 0.5 0 1', help='rgba string to match cable geoms')
    p.add_argument('--segments', type=int, default=0, help='number of segments to generate (default = original count)')
    p.add_argument('--size', type=float, default=None, help='capsule size (radius) to set on new geoms; default=use existing sizes or 0.0025')
    p.add_argument('--smooth', type=int, default=3, help='moving-average smoothing window for midpoints (default 3)')
    args = p.parse_args()

    infile = Path(args.file)
    if not infile.exists():
        print('ERROR: input file not found', infile, file=sys.stderr)
        sys.exit(2)

    tree = ET.parse(str(infile))
    root = tree.getroot()

    world = root.find('worldbody')
    if world is None:
        print('ERROR: no <worldbody>', file=sys.stderr)
        sys.exit(2)

    target_rgba = ' '.join(args.rgba.split())

    matched = []  # list of (parent, elem)
    anchor_points = []
    sizes = []

    # scan for geoms to rebuild
    for parent in list(world.iter()):
        for child in list(parent):
            if child.tag == 'geom':
                rgba = child.get('rgba')
                if rgba and ' '.join(rgba.split()) == target_rgba:
                    # candidate
                    matched.append((parent, child))
                    if child.get('type') == 'sphere' and 'pos' in child.attrib:
                        anchor_points.append(parse_floats(child.get('pos'))[:3])
                    if 'size' in child.attrib:
                        try:
                            sizes.append(float(child.get('size')))
                        except Exception:
                            pass

    if not matched:
        print('No cable geoms found for rgba', target_rgba)
        sys.exit(0)

    # Extract midpoints for fromto geoms (capsules). Also collect any pos-only geoms
    midpoints = []
    fromto_geoms = []
    for parent, g in matched:
        if 'fromto' in g.attrib:
            v = parse_floats(g.get('fromto'))
            p1 = v[:3]
            p2 = v[3:6]
            mid = [(p1[i]+p2[i])/2.0 for i in range(3)]
            midpoints.append(mid)
            fromto_geoms.append((parent,g))
    if not midpoints:
        print('No capsule fromto geoms to rebuild', file=sys.stderr)
        sys.exit(0)

    # Determine anchor endpoints: prefer explicit anchor spheres, else estimate from min/max projection
    if len(anchor_points) >= 2:
        left_anchor = anchor_points[0]
        right_anchor = anchor_points[1]
    else:
        # approximate axis by bounding box on midpoints
        xs = sorted(midpoints, key=lambda p: p[0])
        left_anchor = xs[0]
        right_anchor = xs[-1]

    axis = sub(right_anchor, left_anchor)
    axis_len = length(axis)
    if axis_len == 0:
        # fallback: use simple ordering along x
        axis = [1.0,0.0,0.0]
        axis_len = 1.0

    # Sort midpoints by projection onto axis (from left to right)
    def proj_val(p):
        v = sub(p, left_anchor)
        return (v[0]*axis[0] + v[1]*axis[1] + v[2]*axis[2]) / axis_len

    ordered = sorted(midpoints, key=proj_val)

    # Smooth
    smoothed = smooth_points(ordered, window=args.smooth)

    # Decide segment count
    orig_count = len(ordered)
    n_segments = args.segments if args.segments > 0 else orig_count

    # Build resampled polyline with n_segments+1 points
    resampled = resample_polyline(smoothed, n_segments)

    # Build new geoms: capsules between consecutive resampled points
    new_geoms = []
    cap_size = args.size if args.size is not None else (sizes[0] if sizes else 0.0025)
    for i in range(len(resampled)-1):
        a = resampled[i]
        b = resampled[i+1]
        attrib = {
            'type':'capsule',
            'fromto': fmt(a + b),
            'size': f"{cap_size:.6f}",
            'rgba': target_rgba,
            'contype': '0',
            'conaffinity': '0'
        }
        new_geoms.append(ET.Element('geom', attrib))

    # Remove matched geoms from their parents
    removed = 0
    for parent, g in matched:
        try:
            parent.remove(g)
            removed += 1
        except Exception:
            pass

    # Insert anchors (sphere) at left/right anchor positions (if not present)
    # We'll append anchors before the new capsules
    anchor_elems = []
    anchor_elems.append(ET.Element('geom', {'type':'sphere', 'pos':fmt(left_anchor), 'size':f"{cap_size*3:.6f}", 'rgba':target_rgba, 'contype':'0','conaffinity':'0'}))
    anchor_elems.append(ET.Element('geom', {'type':'sphere', 'pos':fmt(right_anchor), 'size':f"{cap_size*3:.6f}", 'rgba':target_rgba, 'contype':'0','conaffinity':'0'}))

    # Append new elements to worldbody
    # Place anchors then capsules
    for e in anchor_elems:
        world.append(e)
    for e in new_geoms:
        world.append(e)

    outpath = Path(args.out)
    tree.write(str(outpath), encoding='utf-8', xml_declaration=True)

    print(f'Wrote {outpath}: removed {removed} old geoms, added {len(new_geoms)} capsules (segments={n_segments}), size={cap_size:.6f}')

if __name__ == '__main__':
    main()
