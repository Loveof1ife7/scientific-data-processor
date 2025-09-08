# core/wavelet_analysis.py
"""
Wavelet Analyzer for 3D volumetric data.

提供体数据的小波分解/重建、能量分析、阈值去噪与可视化。
风格对齐 core/fourier_analyzer.py 便于在 usage 中互换对比。
"""

from __future__ import annotations
import numpy as np
import pywt
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union


# =========================
# 数据结构说明
# -------------------------
# 对于 wavedecn 得到的系数:
# coeffs: List[Dict[str, np.ndarray] | np.ndarray]
#   coeffs[0]            -> cA（最高层近似系数，ndarray）
#   coeffs[1..L]         -> 每层细节字典，如 {'ddd': arr, 'dda': arr, ...}
#   key 含义：d/a 表示该轴是 detail/approx 的方向组合（xyz）
#
# 对于 swtn 得到的系数:
# swt_coeffs: List[Dict[str, np.ndarray]]
#   每层都是完整字典，包括 'aaa'（近似）与细节键（'aad','ada','daa',...,'ddd'）
#   与 wavedecn 不同，SWT 每层与原尺寸一致，便于逐尺度可视化。
# =========================


class WaveletAnalyzer:
    """3D 体数据小波分析器（离散与平稳小波）。"""

    def __init__(self,
                 wavelet: str = "db2",
                 mode: str = "periodization"):
        """
        Args:
            wavelet: 小波基名称（例如 'db2','coif1','sym4','bior4.4' 等）
            mode: 边界延拓模式（'periodization' 推荐用于体数据循环边界）
        """
        self.wavelet = wavelet
        self.mode = mode

        # 保存最近一次分析的结果（便于后续可视化/统计）
        self.volume_shape: Optional[Tuple[int, int, int]] = None
        self.dwt_coeffs: Optional[List[Union[Dict[str, np.ndarray], np.ndarray]]] = None
        self.swt_coeffs: Optional[List[Dict[str, np.ndarray]]] = None
        self.levels: Optional[int] = None

    # -------- 基本接口（DWT） --------
    def dwt3d(self, volume: np.ndarray, level: Optional[int] = None
              ) -> List[Union[Dict[str, np.ndarray], np.ndarray]]:
        """
        3D 多层离散小波分解（m-D DWT）。

        Args:
            volume: 3D 体数据（numpy ndarray）
            level: 分解层数（None 时自动取最大层）

        Returns:
            coeffs: wavedecn 系数列表（cA + 多层 detail dict）
        """
        self._ensure_float(volume)
        self.volume_shape = tuple(volume.shape)
        max_level = pywt.dwtn_max_level(self.volume_shape, self.wavelet)
        self.levels = level if level is not None else max_level
        self.levels = max(1, min(self.levels, max_level))

        coeffs = pywt.wavedecn(volume, wavelet=self.wavelet,
                               mode=self.mode, level=self.levels,
                               axes=(0, 1, 2))
        self.dwt_coeffs = coeffs
        return coeffs

    def idwt3d(self, coeffs: Optional[List[Union[Dict[str, np.ndarray], np.ndarray]]] = None
               ) -> np.ndarray:
        """
        从 DWT 系数重建 3D 体数据。
        """
        if coeffs is None:
            if self.dwt_coeffs is None:
                raise ValueError("尚未进行 DWT 分解。")
            coeffs = self.dwt_coeffs
        rec = pywt.waverecn(coeffs, wavelet=self.wavelet, mode=self.mode, axes=(0, 1, 2))
        return rec

    # -------- 平稳小波（SWT，尺寸不变，适合可视化） --------
    def swt3d(self, volume: np.ndarray, level: int = 1
              ) -> List[Dict[str, np.ndarray]]:
        """
        3D 平稳小波分解（SWT），每层各子带与原图同尺寸。

        Args:
            volume: 3D 体数据
            level: 分解层数（SWT 可取多层）

        Returns:
            swt_coeffs: List[Dict]，每层含 'aaa'（近似）与细节键。
        """
        self._ensure_float(volume)
        self.volume_shape = tuple(volume.shape)
        self.levels = int(level)
        coeffs = pywt.swtn(volume, wavelet=self.wavelet, level=self.levels, axes=(0, 1, 2))
        self.swt_coeffs = coeffs
        return coeffs

    def iswt3d(self, swt_coeffs: Optional[List[Dict[str, np.ndarray]]] = None) -> np.ndarray:
        """
        SWT 系数逆变换。
        """
        if swt_coeffs is None:
            if self.swt_coeffs is None:
                raise ValueError("尚未进行 SWT 分解。")
            swt_coeffs = self.swt_coeffs
        rec = pywt.iswtn(swt_coeffs, wavelet=self.wavelet, axes=(0, 1, 2))
        return rec

    # -------- 子带选择/屏蔽与带通样式处理 --------
    def mask_coeffs(self,
                    coeffs: List[Union[Dict[str, np.ndarray], np.ndarray]],
                    keep_keys_per_level: Optional[List[Optional[List[str]]]] = None,
                    zero_approx: bool = False) -> List[Union[Dict[str, np.ndarray], np.ndarray]]:
        """
        对 DWT 系数进行子带选择/屏蔽。

        Args:
            coeffs: wavedecn 系数（cA + detail dict）
            keep_keys_per_level: 每层保留的细节键列表（例如 [['ddd','dda'], ['ddd']]）
                                 None 或空列表 => 该层全部置零
            zero_approx: 是否将最高层近似 cA 置零（用于高通）

        Returns:
            new_coeffs: 屏蔽后的系数（可用于重建）
        """
        if not isinstance(coeffs, list):
            raise ValueError("coeffs 必须是 wavedecn 的系数列表。")
        new_coeffs = []
        # cA
        cA = coeffs[0].copy()
        if zero_approx:
            cA[:] = 0
        new_coeffs.append(cA)

        # detail levels
        for li, d in enumerate(coeffs[1:], start=1):
            if not isinstance(d, dict):
                raise ValueError("细节层应为字典。")
            keep = None if keep_keys_per_level is None else (
                keep_keys_per_level[li-1] if li-1 < len(keep_keys_per_level) else None
            )
            d_new = {}
            for k, arr in d.items():
                if keep is None or (isinstance(keep, list) and k not in keep):
                    d_new[k] = np.zeros_like(arr)
                else:
                    d_new[k] = arr.copy()
            new_coeffs.append(d_new)
        return new_coeffs

    # -------- 阈值去噪（适用于 DWT 与 SWT） --------
    def threshold_denoise_dwt(self,
                              coeffs: Optional[List[Union[Dict[str, np.ndarray], np.ndarray]]] = None,
                              method: str = "soft",
                              sigma: Optional[float] = None,
                              universal: bool = True) -> List[Union[Dict[str, np.ndarray], np.ndarray]]:
        """
        对 DWT 细节系数进行阈值去噪。

        Args:
            coeffs: wavedecn 系数（默认用 self.dwt_coeffs）
            method: 'soft' 或 'hard'
            sigma: 噪声标准差，None 时用 MAD 估计
            universal: True 使用普适阈值 T = sigma * sqrt(2*log(N))

        Returns:
            new_coeffs: 阈值处理后的系数
        """
        if coeffs is None:
            if self.dwt_coeffs is None:
                raise ValueError("尚未进行 DWT 分解。")
            coeffs = self.dwt_coeffs

        # 噪声估计：用最细层的 3D 细节系数联合估计 sigma
        if sigma is None:
            finest = coeffs[-1]
            stack = np.concatenate([v.ravel() for k, v in finest.items()])
            # Median Absolute Deviation -> sigma ~ MAD/0.6745
            mad = np.median(np.abs(stack - np.median(stack)))
            sigma = mad / 0.6745 if mad > 0 else np.std(stack)

        # 普适阈值
        N = int(np.prod(self.volume_shape)) if self.volume_shape else stack.size
        T = sigma * np.sqrt(2 * np.log(N)) if universal else sigma

        def _thr(x):
            if method == "soft":
                return np.sign(x) * np.maximum(np.abs(x) - T, 0.0)
            elif method == "hard":
                y = x.copy()
                y[np.abs(y) < T] = 0.0
                return y
            else:
                raise ValueError("method 仅支持 'soft' 或 'hard'")

        new_coeffs = [coeffs[0].copy()]  # cA 保留
        for d in coeffs[1:]:
            d_new = {k: _thr(v) for k, v in d.items()}
            new_coeffs.append(d_new)
        return new_coeffs

    def threshold_denoise_swt(self,
                              swt_coeffs: Optional[List[Dict[str, np.ndarray]]] = None,
                              method: str = "soft",
                              sigma: Optional[float] = None,
                              universal: bool = True) -> List[Dict[str, np.ndarray]]:
        """
        SWT 阈值去噪（对每层细节键阈值化，'aaa' 近似保留）。
        """
        if swt_coeffs is None:
            if self.swt_coeffs is None:
                raise ValueError("尚未进行 SWT 分解。")
            swt_coeffs = self.swt_coeffs

        # 估计噪声
        if sigma is None:
            finest = swt_coeffs[-1]
            stack = np.concatenate([v.ravel() for k, v in finest.items() if k != 'aaa'])
            mad = np.median(np.abs(stack - np.median(stack)))
            sigma = mad / 0.6745 if mad > 0 else np.std(stack)

        N = int(np.prod(self.volume_shape)) if self.volume_shape else stack.size
        T = sigma * np.sqrt(2 * np.log(N)) if universal else sigma

        def _thr(x):
            if method == "soft":
                return np.sign(x) * np.maximum(np.abs(x) - T, 0.0)
            elif method == "hard":
                y = x.copy()
                y[np.abs(y) < T] = 0.0
                return y
            else:
                raise ValueError("method 仅支持 'soft' 或 'hard'")

        new_coeffs: List[Dict[str, np.ndarray]] = []
        for level_dict in swt_coeffs:
            nd = {}
            for k, v in level_dict.items():
                if k == 'aaa':  # 近似不过阈
                    nd[k] = v.copy()
                else:
                    nd[k] = _thr(v)
            new_coeffs.append(nd)
        return new_coeffs

    # -------- 统计：各子带能量/占比 --------
    @staticmethod
    def subband_energy(coeffs: List[Union[Dict[str, np.ndarray], np.ndarray]]
                       ) -> Dict[str, float]:
        """
        统计 DWT 各子带能量（平方和），返回 {name: energy}。
        命名规则：
          L0: cA
          L{li}_{key}: 第 li 层的 detail 键（li=1 是最细层）
        """
        energies: Dict[str, float] = {}
        energies["L0_cA"] = float(np.sum(np.square(coeffs[0])))
        for li, d in enumerate(coeffs[1:], start=1):
            for k, v in d.items():
                energies[f"L{li}_{k}"] = float(np.sum(np.square(v)))
        return energies

    @staticmethod
    def energy_ratios(energies: Dict[str, float]) -> Dict[str, float]:
        total = sum(energies.values())
        if total <= 0:
            return {k: 0.0 for k in energies}
        return {k: (v / total) for k, v in energies.items()}

    # -------- 辅助 --------
    @staticmethod
    def _ensure_float(volume: np.ndarray) -> None:
        if not isinstance(volume, np.ndarray):
            raise TypeError("volume 必须是 numpy ndarray")
        if volume.dtype not in (np.float32, np.float64):
            # 转 float32 足够
            volume[:] = volume.astype(np.float32)


class WaveletVisualizer:
    """小波结果可视化器（切片/能量柱状图等）。"""

    @staticmethod
    def _center_indices(shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return tuple(s // 2 for s in shape)

    @staticmethod
    def plot_dwt_slices(coeffs: List[Union[Dict[str, np.ndarray], np.ndarray]],
                        title: str = "DWT Slices (center planes)") -> None:
        """
        显示 DWT 最高层近似与若干细节子带的中心切片。
        """
        cA = coeffs[0]
        cx, cy, cz = WaveletVisualizer._center_indices(cA.shape)

        # 选取最细层（coeffs[1]）中的常见方向
        details = coeffs[1] if len(coeffs) > 1 else {}
        pick_keys = [k for k in ['ddd', 'dda', 'dad', 'add'] if k in details]
        n_cols = 1 + len(pick_keys)  # cA + 若干 detail

        fig, axes = plt.subplots(3, n_cols, figsize=(5*n_cols, 12))
        fig.suptitle(title)

        def _imshow(ax, vol, name):
            im = ax.imshow(vol, cmap='viridis', origin='lower')
            ax.set_title(name)
            plt.colorbar(im, ax=ax, shrink=0.8)

        # cA
        _imshow(axes[0, 0], cA[:, :, cz], f"cA (Z={cz})")
        _imshow(axes[1, 0], cA[:, cy, :], f"cA (Y={cy})")
        _imshow(axes[2, 0], cA[cx, :, :], f"cA (X={cx})")

        # details
        for j, k in enumerate(pick_keys, start=1):
            d = details[k]
            _imshow(axes[0, j], d[:, :, cz], f"{k} (Z={cz})")
            _imshow(axes[1, j], d[:, cy, :], f"{k} (Y={cy})")
            _imshow(axes[2, j], d[cx, :, :], f"{k} (X={cx})")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_energy_bars(energies: Dict[str, float],
                         title: str = "Wavelet Subband Energies") -> None:
        """
        子带能量柱状图（可配合能量占比使用）。
        """
        keys = list(energies.keys())
        vals = np.array([energies[k] for k in keys], dtype=np.float64)
        order = np.argsort(vals)[::-1]
        keys = [keys[i] for i in order]
        vals = vals[order]

        plt.figure(figsize=(max(8, 0.4*len(keys)), 6))
        plt.bar(keys, vals)
        plt.xticks(rotation=60, ha='right')
        plt.ylabel("Energy (sum of squares)")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_original_vs_recon(original: np.ndarray,
                                  recon: np.ndarray,
                                  title: str = "Original vs Reconstructed") -> None:
        """
        原数据与重建结果的切片与差异。
        """
        assert original.shape == recon.shape
        cx, cy, cz = WaveletVisualizer._center_indices(original.shape)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(title)

        im1 = axes[0, 0].imshow(original[:, :, cz], cmap='viridis', origin='lower')
        axes[0, 0].set_title("Original (Z-center)"); plt.colorbar(im1, ax=axes[0, 0])

        im2 = axes[0, 1].imshow(recon[:, :, cz], cmap='viridis', origin='lower')
        axes[0, 1].set_title("Reconstructed (Z-center)"); plt.colorbar(im2, ax=axes[0, 1])

        diff = original - recon
        im3 = axes[0, 2].imshow(diff[:, :, cz], cmap='RdBu_r', origin='lower')
        axes[0, 2].set_title("Difference (Z-center)"); plt.colorbar(im3, ax=axes[0, 2])

        # 统计比较
        axes[1, 0].hist(original.ravel(), bins=50, alpha=0.7, density=True, label='Original')
        axes[1, 0].hist(recon.ravel(), bins=50, alpha=0.7, density=True, label='Reconstructed')
        axes[1, 0].legend(); axes[1, 0].set_title("Value Distribution")

        e_ori = float(np.sum(original**2))
        e_rec = float(np.sum(recon**2))
        axes[1, 1].bar(['Original', 'Recon'], [e_ori, e_rec])
        axes[1, 1].set_title(f"Energy Compare (ratio={e_rec/e_ori:.3f})")

        # 空一格供扩展
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()
