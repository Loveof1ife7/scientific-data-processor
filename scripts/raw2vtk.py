"""
raw data to vtk file
"""

import sys
import os
from pathlib import Path

from config.config_loader import ConfigLoader
from core.data_loader import VolumeDataLoader
from core.processor import VolumeProcessor
from visualization.visualizer import VolumeVisualizer

import vtk
import numpy as np
from vtk.util import numpy_support as nps

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def numpy_to_vtk(
    volume: np.ndarray,
    out_path: str,
    *,
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    axis_order="zyx",            # 你的 loader 若返回 volume[z, y, x]，就用默认 'zyx'
    scalar_name="Scalars",
    compressor="zlib"            # 'zlib' 或 'lz4'
) -> str:   

    assert volume.ndim in (3, 4), f"volume.ndim={volume.ndim}, 需为3或4 Z,Y,X[,C]"
    if not np.isfinite(volume).all():
        raise ValueError("volume 存在非有限值NaN/Inf 请先清洗或转换。")

    # 调试输出（很重要，能立刻看出“全0”等问题）
    vmin = float(np.min(volume))
    vmax = float(np.max(volume))
    print(f"[numpy_to_vti] dtype={volume.dtype}, shape={volume.shape}, min={vmin}, max={vmax}")
    if vmax == vmin:
        raise ValueError(f"数组所有值相同(min=max={vmin})，写出 .vti 也只会是一块常数体。")
    
    # axis_order can be "xyz" or "zxy" or list/tuple (2,1,0)
    if isinstance(axis_order, str):
        axis_order = axis_order.lower()
        if volume.ndim == 3:
            map3 = { "zyx": (0,1,2), "xyz": (2,1,0), "xzy": (2,0,1), "yxz": (1,2,0), "yzx": (1,0,2), "zxy": (0,2,1) }
            if axis_order not in map3:
                raise ValueError(f"不支持的 axis_order={axis_order}")
            perm = map3[axis_order]
            arr = volume.transpose(perm)  # 结果应为 (Z, Y, X)
        else:  # ndim==4, 最后一维为通道
            map4 = { "zyx": (0,1,2,3), "xyz": (2,1,0,3), "xzy": (2,0,1,3), "yxz": (1,2,0,3), "yzx": (1,0,2,3), "zxy": (0,2,1,3) }
            if axis_order not in map4:
                raise ValueError(f"不支持的 axis_order={axis_order}")
            perm = map4[axis_order]
            arr = volume.transpose(perm)
    else:
        # list or tuple
        perm = tuple(axis_order) + (() if volume.ndim == len(axis_order) else (volume.ndim-1,))
        arr = volume.transpose(perm)

    # 现在 arr 应为 (Z, Y, X) 或 (Z, Y, X, C)
    if arr.ndim == 3:
        nz, ny, nx = arr.shape
        flat = arr.ravel(order="C")
        vtk_arr = nps.numpy_to_vtk(num_array=flat, deep=True)  # dtype 自动匹配
    else:
        nz, ny, nx, nc = arr.shape
        flat = arr.reshape(-1, nc)  # N x C
        vtk_arr = nps.numpy_to_vtk(num_array=flat, deep=True)

    img = vtk.vtkImageData()
    img.SetDimensions(int(nx), int(ny), int(nz))
    img.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))
    img.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    img.GetPointData().SetScalars(vtk_arr)

    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(img)
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOff()

    if compressor.lower() == "lz4" and hasattr(writer, "SetCompressorTypeToLZ4"):
        writer.SetCompressorTypeToLZ4()
    else:
        writer.SetCompressorTypeToZLib()

    ok = writer.Write()
    if ok != 1:
        raise RuntimeError(f"写 VTI 失败：{out_path}")
    print(f"[numpy_to_vti] wrote: {out_path}")
    return str(out_path)


def raw2vtk():
    config_loader = ConfigLoader("config/datasets.json")
    dataset_config = config_loader.get_dataset_config("Supernova")
    data_loader = VolumeDataLoader(dataset_config)
    volume = data_loader.load_volume("Scalar", 1)
    stats = {
            'min': float(np.min(volume)),
            'max': float(np.max(volume)),
            'mean': float(np.mean(volume)),
            'std': float(np.std(volume)),
            'shape': volume.shape,
            'dtype': str(volume.dtype),
            'size_mb': volume.nbytes / (1024 * 1024)
        }
    
    print(stats)

    # 从配置拿 spacing/origin（如无则默认 1,1,1 和 0,0,0）
    spacing = tuple(dataset_config.get("spacing", (1.0, 1.0, 1.0)))
    origin  = tuple(dataset_config.get("origin",  (0.0, 0.0, 0.0)))
    axis_order = dataset_config.get("axis_order", "zyx")  # 若 loader 返回 xyz 顺序，改成 "xyz"

    # 输出路径（可放到 config 里）
    out_path = dataset_config.get("out_vti", "out/supernova.vti")

    # === 写出 .vti ===
    numpy_to_vtk(
        volume,
        out_path,
        spacing=spacing,
        origin=origin,
        axis_order=axis_order,
        scalar_name=dataset_config.get("scalar_name", "Scalars"),
        compressor=dataset_config.get("compressor", "zlib"),
    )

if __name__ == "__main__":
    raw2vtk()