#!/usr/bin/env pvpython
# -*- coding: utf-8 -*-
"""
Generate multi-view RGB + camera poses from a VTI volume for 3DGS/NeRF training.
✅ 兼容 ParaView 6.x：不再调用 RemoveAllPoints/AddPoint，而是直接设置 Points。
Usage:
  pvpython scripts/gen_3dgs_dataset_from_vti.py \
    --vti out/supernova.vti \
    --out out/3dgs/supernova \
    --num 60 --w 1600 --h 1200 --vfov 45 --elev 15 --radius_scale 1.4
"""

import os, json, math, argparse
import numpy as np
from paraview.simple import *

# ---------- helpers ----------
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def c2w_from_lookat(eye, center, up_world=(0,1,0)):
    eye, center, up_world = np.asarray(eye,float), np.asarray(center,float), _unit(up_world)
    forward = _unit(center - eye)
    right   = _unit(np.cross(forward, up_world))
    up      = _unit(np.cross(right, forward))
    c2w = np.eye(4, dtype=float)
    c2w[0:3,0] = right
    c2w[0:3,1] = up
    c2w[0:3,2] = -forward
    c2w[0:3,3] = eye
    return c2w

def intrinsics_from_vfov(vfov_deg, w, h):
    vfov = math.radians(vfov_deg)
    fy = (h/2.0) / math.tan(vfov/2.0)
    fx = fy * (w/float(h))
    cx, cy = (w-1)/2.0, (h-1)/2.0
    return fx, fy, cx, cy

def parse_vec3(s, default):
    if s is None: return np.array(default, float)
    parts = [p for p in s.replace(',', ' ').split() if p]
    if len(parts) != 3: return np.array(default, float)
    return np.array([float(x) for x in parts], float)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vti", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=60)
    ap.add_argument("--w", type=int, default=1600)
    ap.add_argument("--h", type=int, default=1200)
    ap.add_argument("--vfov", type=float, default=45.0)
    ap.add_argument("--elev", type=float, default=15.0)
    ap.add_argument("--radius_scale", type=float, default=1.4)
    ap.add_argument("--up", type=str, default=None, help='e.g. "0,1,0" or "0 0 1"')
    ap.add_argument("--range", type=str, default=None, help='override scalar range "min,max"')
    ap.add_argument("--bg", type=str, default="0,0,0")
    args = ap.parse_args()

    out_img_dir = os.path.join(args.out, "images")
    os.makedirs(out_img_dir, exist_ok=True)

    up_world = parse_vec3(args.up, (0,1,0))
    bg = parse_vec3(args.bg, (0,0,0))
    override_range = None
    if args.range:
        ss = [t for t in args.range.replace(',', ' ').split() if t]
        if len(ss) == 2: override_range = (float(ss[0]), float(ss[1]))

    # --- load ---
    src = OpenDataFile(args.vti)
    if src is None: raise RuntimeError("无法打开 VTI：" + args.vti)

    rv = GetActiveViewOrCreate('RenderView')
    rv.ViewSize = [int(args.w), int(args.h)]
    try: rv.UseColorPaletteForBackground = 0
    except: pass
    rv.Background = [float(bg[0]), float(bg[1]), float(bg[2])]
    rv.CameraParallelProjection = 0
    rv.CameraViewAngle = float(args.vfov)

    Show(src, rv)
    UpdatePipeline()

    # 关掉 Ray Tracing（若存在）
    try: rv.EnableRayTracing = 0
    except: pass

    disp = GetDisplayProperties(src, view=rv)
    disp.Representation = 'Volume'

    # --- pick scalar array (prefer PointData), 6.x via DataInformation ---
    di  = src.GetDataInformation()
    pdi = di.GetPointDataInformation()
    array_name, assoc, used_c2p = None, 'POINTS', False

    if pdi and pdi.GetNumberOfArrays() > 0:
        array_name = pdi.GetArrayInformation(0).GetName()
    else:
        cdi = di.GetCellDataInformation()
        if cdi and cdi.GetNumberOfArrays() > 0:
            array_name = cdi.GetArrayInformation(0).GetName()
            c2p = CellDatatoPointData(Input=src)
            Hide(src, rv); Show(c2p, rv)
            src = c2p
            UpdatePipeline()
            disp = GetDisplayProperties(src, view=rv)
            disp.Representation = 'Volume'
            assoc, used_c2p = 'POINTS', True
        else:
            raise RuntimeError("没有可用的 Point/Cell 标量数组。")

    print("[info] use array:", assoc, array_name, "(CellData->PointData)" if used_c2p else "")
    ColorBy(disp, (assoc, array_name))

    # --- transfer functions ---
    lut = GetColorTransferFunction(array_name)
    pwf = GetOpacityTransferFunction(array_name)

    # 取范围（可被 --range 覆盖）
    if override_range is not None:
        arr_min, arr_max = override_range
    else:
        di  = src.GetDataInformation()
        pdi = di.GetPointDataInformation()
        ai  = None
        if pdi and pdi.GetNumberOfArrays() > 0:
            for i in range(pdi.GetNumberOfArrays()):
                if pdi.GetArrayInformation(i).GetName() == array_name:
                    ai = pdi.GetArrayInformation(i); break
        if ai is not None:
            try: arr_min, arr_max = ai.GetRange()
            except: arr_min, arr_max = ai.GetComponentRange(0)
        else:
            arr_min, arr_max = 0.0, 255.0

    if not np.isfinite([arr_min, arr_max]).all() or arr_min == arr_max:
        arr_min, arr_max = 0.0, 255.0

    # 色彩：先把 LUT 重标到范围；（可选也能改 RGBPoints）
    lut.RescaleTransferFunction(float(arr_min), float(arr_max))

    # 不透明度：6.x 没有 RemoveAllPoints/AddPoint，用 Points 属性（x,y,midpoint,sharpness）
    span = float(arr_max - arr_min) if arr_max > arr_min else 1.0
    def V(t): return float(arr_min + t*span)  # t in [0,1]
    # 这组对 8-bit/一般数据是个“能看见内部”的起步方案
    opacity_points = [
        V(0.00), 0.00, 0.5, 0.0,
        V(0.12), 0.00, 0.5, 0.0,
        V(0.31), 0.03, 0.5, 0.0,
        V(0.55), 0.08, 0.5, 0.0,
        V(0.78), 0.16, 0.5, 0.0,
    ]
    pwf.Points = opacity_points
    try: pwf.AllowDuplicateScalars = 0
    except: pass

    # 体渲染参数
    disp.Shade = 1
    try: disp.ScalarOpacityUnitDistance = 1.0
    except: pass

    UpdatePipeline()
    info = src.GetDataInformation()
    b = info.GetBounds()
    if not b or not np.isfinite(b).all():
        raise RuntimeError("无法获取数据包围盒。")
    center = np.array([(b[0]+b[1])/2.0, (b[2]+b[3])/2.0, (b[4]+b[5])/2.0], float)
    ext = np.array([b[1]-b[0], b[3]-b[2], b[5]-b[4]], float)
    R = float(args.radius_scale) * 0.5 * float(np.linalg.norm(ext))
    elev_rad = math.radians(float(args.elev))
    up_world = _unit(up_world)

    ResetCamera(rv)
    Render()

    # intrinsics
    fx, fy, cx, cy = intrinsics_from_vfov(float(args.vfov), int(args.w), int(args.h))

    frames, N = [], int(args.num)
    print("[info] bounds:", b, "center:", center.tolist(), "radius:", R)

    for i in range(N):
        theta = 2.0*math.pi * (i/float(N))
        x = center[0] + R*math.cos(theta)
        y = center[1] + R*math.sin(theta)
        z = center[2] + R*math.sin(elev_rad)
        eye, focal = [x,y,z], center.tolist()

        rv.CameraPosition = eye
        rv.CameraFocalPoint = focal
        rv.CameraViewUp = up_world.tolist()
        rv.CameraViewAngle = float(args.vfov)
        Render()

        fname = "frame_%03d.png" % i
        fpath = os.path.join(out_img_dir, fname)
        SaveScreenshot(fpath, rv, ImageResolution=[int(args.w), int(args.h)])
        print("Saved", fpath)

        c2w = c2w_from_lookat(eye, focal, up_world)
        frames.append({"file_path": "images/"+fname, "transform_matrix": c2w.tolist()})

    meta = {
        "w": int(args.w), "h": int(args.h),
        "fl_x": float(fx), "fl_y": float(fy), "cx": float(cx), "cy": float(cy),
        "camera_model": "OPENCV",
        "frames": frames
    }
    with open(os.path.join(args.out, "transforms_nerf.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("Wrote", os.path.join(args.out, "transforms_nerf.json"))

if __name__ == "__main__":
    main()
