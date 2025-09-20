#!/usr/bin/env pvpython
# -*- coding: utf-8 -*-
"""
Generate multi-view RGB + camera poses from a VTI volume for 3DGS/NeRF training.
增强版：
  - 多套 colormap（--cmaps），支持 per-cmap/flat 两种输出组织
  - 整球 Fibonacci 均匀采样 + FOV 自适配距离，保证 bbox 完整入画
  - 稳定的 ViewUp，避免极区翻滚；渲染循环重置裁剪范围
  - 尝试启用 NVIDIA IndeX（--index），失败自动回退
  - 体渲染质量细节：FXAA、可选梯度不透明度、统一 LUT 重标、可选 PNG 压缩
  - 兼容 ParaView 6.x（Opacity TF 用 Points 属性）
  
pvpython scripts/gen_3dgs_dataset_from_vti.py \
--vti outputs/supernova.vti \
--out outputs/3dgs_n/supernova \
--num 64 --w 1600 --h 1200 --vfov 45 --radius_scale 1.2 \
--cmaps "Viridis (matplotlib), Inferno (matplotlib), Turbo" \
--opacity_preset band_pass --opacity_scale 0.9 --shading 0 --opaque_unit 2.0 \
--fxaa 1 --use_gradient 0 --png_compress 3 \
--index --out_mode per-cmap

"""

import os, re, json, math, argparse
import numpy as np
from paraview.simple import *

# 顶部：保持这句
LoadPlugin('/home/lqb/softwares/ParaView-5.12.1-MPI-Linux-Python3.10-x86_64/lib/paraview-5.12/plugins/pvNVIDIAIndeX/pvNVIDIAIndeX.so')
print("NVIDIA IndeX plugin loaded.")

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

def parse_cmaps(s):
    if not s: return []
    items = [t.strip() for t in re.split(r'[,\n]+', s) if t.strip()]
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it); out.append(it)
    return out

def sanitize_name(s):
    s = s.strip()
    s = re.sub(r'[\s/\\:;,\(\)]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s or "cmap"

def try_apply_preset(lut, name):
    ok = False
    try: ok = lut.ApplyPreset(name, True)
    except: pass
    if not ok:
        print("[warn] ApplyPreset 失败或未找到预设：", name)

def try_enable_nvidia_index(view, disp):
    # 不再按名字二次加载——前面已用绝对路径加载成功
    # 先试图打开视图级开关（如果该属性存在就设为 1，不存在就略过）
    for attr in ["EnableNVIDIAIndeX", "EnableNVINDEX", "UseNVIDIAIndeX", "EnableIndeX"]:
        if hasattr(view, attr):
            try:
                setattr(view, attr, 1)
                print("[info] 视图属性已开启：", attr)
                break
            except:
                pass

    # 切换渲染表示法为 IndeX（关键步骤）
    try:
        # 方式A：更鲁棒
        disp.SetRepresentationType('NVIDIA IndeX')
        print("[info] 已切换 Representation: NVIDIA IndeX")
        return True
    except:
        # 方式B：回退（部分旧版只支持直接赋值）
        try:
            disp.Representation = 'NVIDIA IndeX'
            print("[info] 已切换 Representation: NVIDIA IndeX (fallback)")
            return True
        except:
            pass

    # 再试个备选枚举名（有的包叫这个）
    for rep in ['NVIDIA IndeX Volume', 'Volume (NVIDIA IndeX)']:
        try:
            disp.SetRepresentationType(rep)
            print("[info] 已切换 Representation:", rep)
            return True
        except:
            pass

    return False


def safe_viewup(forward, up_hint):
    f = np.asarray(forward, float); f /= (np.linalg.norm(f)+1e-12)
    u = np.asarray(up_hint, float);  u /= (np.linalg.norm(u)+1e-12)
    if abs(np.dot(f, u)) > 0.98:
        alt = np.array([0,0,1.0]) if abs(f[2]) < 0.9 else np.array([0,1.0,0])
        u = alt - np.dot(alt, f) * f
    u = u - np.dot(u, f) * f
    u /= (np.linalg.norm(u)+1e-12)
    return u

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vti", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=60)
    ap.add_argument("--w", type=int, default=1600)
    ap.add_argument("--h", type=int, default=1200)
    ap.add_argument("--vfov", type=float, default=45.0)
    ap.add_argument("--elev", type=float, default=15.0)   # 保留但不参与整球采样，仅做向上偏置时可用
    ap.add_argument("--radius_scale", type=float, default=1.2)
    ap.add_argument("--up", type=str, default=None, help='e.g. "0,1,0" or "0 0 1"')
    ap.add_argument("--range", type=str, default=None, help='override scalar range "min,max"')
    ap.add_argument("--bg", type=str, default="0,0,0")

    # 多 colormap 与输出模式
    ap.add_argument("--cmaps", type=str, default="", help='多个 colormap 名称，用逗号分隔')
    ap.add_argument("--out_mode", type=str, choices=["per-cmap","flat"], default="per-cmap",
                    help='per-cmap: 每个 colormap 一个子目录；flat: 同目录平铺，文件名前缀为 colormap')

    # NVIDIA IndeX
    ap.add_argument("--index", action="store_true", help="尝试使用 NVIDIA IndeX 做体渲染（插件可用时）")

    # Opacity 预设
    ap.add_argument("--opacity_preset", type=str, default="band_pass",
                    choices=["thin_shell","band_pass","core_focus","flat"],
                    help="不透明度传递函数预设")
    ap.add_argument("--opacity_scale", type=float, default=0.9,
                    help="整体不透明度缩放（<1 更透明，>1 更实）")
    ap.add_argument("--shading", type=int, default=0,
                    help="0=关闭阴影更通透, 1=开启有体感")
    ap.add_argument("--opaque_unit", type=float, default=2.0,
                    help="ScalarOpacityUnitDistance，越大越透明")

    # 质量细节
    ap.add_argument("--fxaa", type=int, default=1, help="1=启用 FXAA")
    ap.add_argument("--use_gradient", type=int, default=0, help="1=启用梯度不透明度（若版本支持）")
    ap.add_argument("--grad_opacity", type=float, default=0.2, help="梯度不透明度权重(0.1~0.4 常用)")
    ap.add_argument("--png_compress", type=int, default=0, help="PNG 压缩等级(0~5)，0=关闭")

    args = ap.parse_args()

    # 解析 cmaps；若未提供则给一组默认
    cmaps = parse_cmaps(args.cmaps)
    if not cmaps:
        cmaps = ["Viridis (matplotlib)", "Inferno (matplotlib)", "Plasma (matplotlib)", "Magma (matplotlib)", "Turbo"]
        print("[info] 未指定 --cmaps，使用默认集合：", ", ".join(cmaps))

    # 主输出
    os.makedirs(args.out, exist_ok=True)
    if args.out_mode == "flat":
        out_img_dir_flat = os.path.join(args.out, "images")
        os.makedirs(out_img_dir_flat, exist_ok=True)

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
    if args.fxaa:
        try: rv.UseFXAA = 1
        except: pass

    Show(src, rv)
    UpdatePipeline()

    try: rv.EnableRayTracing = 0
    except: pass

    disp = GetDisplayProperties(src, view=rv)
    disp.Representation = 'Volume'

    # --- pick scalar array ---
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

    # 色彩范围重标
    lut.RescaleTransferFunction(float(arr_min), float(arr_max))

    # 不透明度：6.x 使用 Points 属性（x,y,midpoint,sharpness）
    span = float(arr_max - arr_min) if arr_max > arr_min else 1.0
    def V(t): return float(arr_min + t*span)  # t in [0,1]

    preset = args.opacity_preset
    if preset == "thin_shell":
        opacity_points = [
            V(0.00), 0.00, 0.5, 0.0,
            V(0.20), 0.00, 0.5, 0.0,
            V(0.45), 0.06, 0.5, 0.0,
            V(0.60), 0.10, 0.5, 0.0,
            V(0.80), 0.18, 0.5, 0.0,
            V(1.00), 0.25, 0.5, 0.0,
        ]
    elif preset == "band_pass":
        opacity_points = [
            V(0.00), 0.00, 0.5, 0.0,
            V(0.25), 0.01, 0.5, 0.0,
            V(0.40), 0.05, 0.5, 0.0,
            V(0.55), 0.12, 0.5, 0.0,
            V(0.70), 0.06, 0.5, 0.0,
            V(0.85), 0.14, 0.5, 0.0,
            V(1.00), 0.20, 0.5, 0.0,
        ]
    elif preset == "core_focus":
        opacity_points = [
            V(0.00), 0.00, 0.5, 0.0,
            V(0.35), 0.00, 0.5, 0.0,
            V(0.55), 0.02, 0.5, 0.0,
            V(0.75), 0.07, 0.5, 0.0,
            V(0.90), 0.16, 0.5, 0.0,
            V(1.00), 0.28, 0.5, 0.0,
        ]
    else:  # flat
        opacity_points = [
            V(0.00), 0.00, 0.5, 0.0,
            V(0.50), 0.04, 0.5, 0.0,
            V(1.00), 0.08, 0.5, 0.0,
        ]

    # 全局透明度缩放
    for k in range(1, len(opacity_points), 4):
        opacity_points[k] = float(opacity_points[k]) * float(args.opacity_scale)

    pwf.Points = opacity_points
    try: pwf.AllowDuplicateScalars = 0
    except: pass

    # 体渲染参数
    disp.Shade = int(args.shading)
    try: disp.ScalarOpacityUnitDistance = float(args.opaque_unit)
    except: pass

    # 可选：梯度不透明度
    if args.use_gradient:
        try:
            disp.UseGradientOpacity = 1
            disp.GradientOpacity = float(args.grad_opacity)
        except:
            print("[warn] 此版本不支持 GradientOpacity，已忽略。")

    UpdatePipeline()
    info = src.GetDataInformation()
    b = info.GetBounds()
    if not b or not np.isfinite(b).all():
        raise RuntimeError("无法获取数据包围盒。")
    center = np.array([(b[0]+b[1])/2.0, (b[2]+b[3])/2.0, (b[4]+b[5])/2.0], float)
    ext = np.array([b[1]-b[0], b[3]-b[2], b[5]-b[4]], float)
    up_world = _unit(up_world)

    ResetCamera(rv)
    Render()

    # 尝试启用 NVIDIA IndeX（如果加了 --index）
    print("available repr:", disp.Representation)
    index_on = False
    if args.index:
        try:
            index_on = try_enable_nvidia_index(rv, disp)
            if index_on:
                try: rv.LockBounds = 1
                except: pass
            else:
                print("[warn] 未能启用 NVIDIA IndeX，继续使用内置体渲染。")
        except Exception as e:
            print("[warn] 启用 NVIDIA IndeX 出错：", e)

    # ----- intrinsics
    fx, fy, cx, cy = intrinsics_from_vfov(float(args.vfov), int(args.w), int(args.h))

    # ----- 根据 bbox + FOV 计算相机距离，保证整块体积可见
    vfov = math.radians(float(args.vfov))
    hfov = 2.0 * math.atan((float(args.w)/float(args.h)) * math.tan(vfov/2.0))

    dx, dy, dz = map(float, ext.tolist())            # 修正写法
    r_sphere = 0.5 * float(np.linalg.norm([dx, dy, dz]))  # 外接球半径

    eps = 1e-6
    d_v  = r_sphere / max(math.sin(vfov/2.0), eps)
    d_h  = r_sphere / max(math.sin(hfov/2.0), eps)
    d_fit = max(d_v, d_h)
    d = float(args.radius_scale) * d_fit

    print(f"[info] fit distance d_fit={d_fit:.4f}, use d={d:.4f} (radius_scale={args.radius_scale})")

    # ----- 整球 Fibonacci 均匀采样方向
    N = int(args.num)
    phi = (1.0 + 5**0.5) / 2.0
    ga  = 2.0 * math.pi * (1.0 - 1.0/phi)
    poses = []

    for i in range(N):
        u = (i + 0.5) / N
        z = 2.0*u - 1.0
        r_xy = max(0.0, 1.0 - z*z) ** 0.5
        theta = i * ga

        dirx = r_xy * math.cos(theta)
        diry = r_xy * math.sin(theta)
        dirz = z

        eye = [center[0] + d*dirx, center[1] + d*diry, center[2] + d*dirz]
        focal = center.tolist()

        forward = np.array(focal) - np.array(eye)
        vup = safe_viewup(forward, up_world)

        poses.append((eye, focal, vup.tolist()))

    print("[info] bounds:", b, "center:", center.tolist(), "bbox diag radius:", r_sphere)
    print("[info] cmaps:", ", ".join(cmaps))
    print("[info] 输出模式:", args.out_mode)

    # 针对每个 colormap 生成一套图片 + transforms
    for cmap in cmaps:
        cmap_tag = sanitize_name(cmap)
        if args.out_mode == "per-cmap":
            out_dir = os.path.join(args.out, cmap_tag)
            out_img_dir = os.path.join(out_dir, "images")
        else:
            out_dir = args.out
            out_img_dir = os.path.join(args.out, "images")
        os.makedirs(out_img_dir, exist_ok=True)

        # 应用 colormap 预设 + 统一范围
        try_apply_preset(lut, cmap)
        lut.RescaleTransferFunction(float(arr_min), float(arr_max))
        Render()

        frames = []
        for i, (eye, focal, vup) in enumerate(poses):
            rv.CameraPosition = eye
            rv.CameraFocalPoint = focal
            rv.CameraViewUp = vup
            rv.CameraViewAngle = float(args.vfov)
            try: rv.ResetCameraClippingRange()
            except: pass
            Render()

            if args.out_mode == "flat":
                fname = f"{cmap_tag}_frame_{i:03d}.png"
            else:
                fname = f"frame_{i:03d}.png"

            fpath = os.path.join(out_img_dir, fname)

            # 带压缩的保存（若支持）；失败则回退
            saved = False
            if args.png_compress and int(args.png_compress) > 0:
                try:
                    SaveScreenshot(fpath, rv,
                                   ImageResolution=[int(args.w), int(args.h)],
                                   CompressionLevel=int(args.png_compress))
                    saved = True
                except:
                    pass
            if not saved:
                SaveScreenshot(fpath, rv, ImageResolution=[int(args.w), int(args.h)])

            print("Saved", fpath)

            c2w = c2w_from_lookat(eye, focal, vup)
            rel_path = os.path.relpath(fpath, out_dir).replace("\\", "/")
            frames.append({"file_path": rel_path, "transform_matrix": c2w.tolist()})

        meta = {
            "w": int(args.w), "h": int(args.h),
            "fl_x": float(fx), "fl_y": float(fy), "cx": float(cx), "cy": float(cy),
            "camera_model": "OPENCV",
            "frames": frames
        }
        tf_path = os.path.join(out_dir, "transforms_nerf.json")
        with open(tf_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print("Wrote", tf_path)

if __name__ == "__main__":
    main()
