# core/wavelet_processor.py
# -*- coding: utf-8 -*-

import sys
import os
import time
import numpy as np
from typing import Optional, Dict, Any

# 项目内引用（保持与你的 FourierProcessor 相同的用法）
from core.wavelet_analyzer import WaveletAnalyzer, WaveletVisualizer
from volume_project import VolumeDataProject   # 你的项目入口类（和 FourierProcessor 一致）


class WaveletProcessor:
    """Volumes 数据的小波分析处理器（对标 FourierProcessor 的用法）"""

    def __init__(self, project: VolumeDataProject,
                 wavelet: str = "db2",
                 mode: str = "periodization",
                 level: Optional[int] = None,
                 use_swt: bool = False):
        """
        Args:
            project: VolumeDataProject 实例
            wavelet: 小波基（如 'db2', 'sym4', 'coif1', 'bior4.4'）
            mode: 边界延拓（'periodization' 对体数据很稳）
            level: 分解层数（None 自动最大层）
            use_swt: True 使用平稳小波（SWT），False 使用离散小波（DWT）
        """
        self.project = project
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        self.use_swt = use_swt

        self.analyzer = WaveletAnalyzer(wavelet=wavelet, mode=mode)
        self.viz = WaveletVisualizer()

    # ------- 只计时：DWT 或 SWT -------
    def time_wavelet(self,
                     dataset_name: str = "Supernova",
                     variable: str = "Scalar",
                     sample_id: int = 1) -> None:
        """加载一个 volume，执行一次 3D 小波分解并打印耗时。"""
        if self.project.data_loader is None:
            self.project.load_dataset(dataset_name)

        volume = self.project.data_loader.load_volume(variable, sample_id)
        print(f"数据加载完成: {volume.shape}, dtype={volume.dtype}")

        t0 = time.time()
        if self.use_swt:
            # SWT：尺寸不变，适合可视化
            _ = self.analyzer.swt3d(volume, level=self.level or 1)
            tag = f"SWT(level={self.level or 1})"
        else:
            # DWT：系数逐层降采样，更适合压缩/分析
            _ = self.analyzer.dwt3d(volume, level=self.level)
            tag = f"DWT(level={self.analyzer.levels})"
        dt = time.time() - t0
        print(f"{tag} 计算完成, 耗时: {dt:.4f} 秒")

    # ------- 完整分析：DWT/SWT + 统计 + 去噪 + 可视化 -------
    def full_wavelet_analysis(self,
                              dataset_name: str = "Supernova",
                              variable: str = "Scalar",
                              sample_id: int = 1,
                              denoise: bool = True,
                              threshold_method: str = "soft") -> Dict[str, Any]:
        """
        执行完整的小波分析流程，返回结果字典。

        Args:
            denoise: 是否做阈值去噪
            threshold_method: 'soft' 或 'hard'
        """
        print("=== Supernova 小波分析 ===")
        if self.project.data_loader is None:
            self.project.load_dataset(dataset_name)

        volume = self.project.data_loader.load_volume(variable, sample_id)
        if volume.dtype not in (np.float32, np.float64):
            volume = volume.astype(np.float32)
        print(f"数据加载完成: {volume.shape}, dtype={volume.dtype}")

        # 1) 分解
        if self.use_swt:
            # 平稳小波：每层与原尺寸一致，便于做 per-level 可视化
            t0 = time.time()
            swt_coeffs = self.analyzer.swt3d(volume, level=self.level or 1)
            print(f"SWT(level={self.level or 1}) 计算完成, 耗时: {time.time()-t0:.4f} 秒")

            # 2) 去噪（可选）+ 重建
            if denoise:
                swt_denoised = self.analyzer.threshold_denoise_swt(
                    swt_coeffs, method=threshold_method
                )
                recon = self.analyzer.iswt3d(swt_denoised)
            else:
                recon = self.analyzer.iswt3d(swt_coeffs)

            # 3) 可视化（对比）
            self.viz.compare_original_vs_recon(volume, recon, title="SWT Denoise Recon")

            # 注意：SWT 的“能量/子带统计”通常对每层的 detail 做汇总；这里示例不展开
            results = {
                "mode": "SWT",
                "volume": volume,
                "swt_coeffs": swt_coeffs,
                "reconstructed": recon,
            }
            return results

        else:
            # 离散小波：wavedecn/waverecn
            t0 = time.time()
            coeffs = self.analyzer.dwt3d(volume, level=self.level)
            print(f"DWT(level={self.analyzer.levels}) 计算完成, 耗时: {time.time()-t0:.4f} 秒")

            # 2) 子带能量统计
            energies = self.analyzer.subband_energy(coeffs)
            ratios = self.analyzer.energy_ratios(energies)
            print("子带能量统计(Top 8):")
            topk = sorted(energies.items(), key=lambda kv: kv[1], reverse=True)[:8]
            for k, v in topk:
                print(f"  {k:<10s}: {v:.6e}")

            # 3) 去噪（可选）
            if denoise:
                coeffs_dn = self.analyzer.threshold_denoise_dwt(
                    coeffs, method=threshold_method
                )
                recon = self.analyzer.idwt3d(coeffs_dn)
            else:
                recon = self.analyzer.idwt3d(coeffs)

            # 4) 选择性保留子带（示例：仅保留最细层 ddd，且置零近似 -> 类高通）
            masked = self.analyzer.mask_coeffs(
                coeffs,
                keep_keys_per_level=[['ddd']],  # 只保留 L1 的 'ddd'
                zero_approx=True
            )
            recon_highpass_like = self.analyzer.idwt3d(masked)

            # 5) 可视化
            self.viz.plot_dwt_slices(coeffs, title="DWT Slices (center planes)")
            self.viz.plot_energy_bars(energies, title="Wavelet Subband Energies")
            self.viz.compare_original_vs_recon(volume, recon, title="DWT Denoise Recon")

            results = {
                "mode": "DWT",
                "volume": volume,
                "coeffs": coeffs,
                "energies": energies,
                "energy_ratios": ratios,
                "reconstructed": recon,
                "highpass_like": recon_highpass_like,
            }
            return results


def supernova_time_dwt():
    """只计时：离散小波（默认自动最大层）"""
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    project = VolumeDataProject()
    proc = WaveletProcessor(project, wavelet="db2", mode="periodization", level=None, use_swt=False)
    try:
        proc.time_wavelet(dataset_name="Supernova", variable="Scalar", sample_id=1)
    except Exception as e:
        print(f"时间DWT过程中出现错误: {e}")


def supernova_full_wavelet_analysis():
    """完整小波分析：默认 DWT，可自行切换 use_swt=True 测 SWT"""
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    project = VolumeDataProject()
    # DWT：
    proc = WaveletProcessor(project, wavelet="db2", mode="periodization", level=None, use_swt=False)
    # 如果想用 SWT：把 use_swt=True，并指定 level（例如 2 或 3）
    # proc = WaveletProcessor(project, wavelet="db2", mode="periodization", level=2, use_swt=True)

    try:
        results = proc.full_wavelet_analysis(
            dataset_name="Supernova", variable="Scalar", sample_id=1,
            denoise=True, threshold_method="soft"
        )
        print("\n分析完成！结果字典键：", list(results.keys()))

        # 示例：保存子带能量
        if results.get("energies") is not None:
            np.save("supernova_wavelet_energies.npy", results["energies"])
            print("子带能量已保存为 'supernova_wavelet_energies.npy'")

    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        print("请检查数据路径和配置是否正确")


if __name__ == "__main__":
    supernova_time_dwt()
    # 或：
    supernova_full_wavelet_analysis()
