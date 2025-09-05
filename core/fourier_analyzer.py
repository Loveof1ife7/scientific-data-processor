
"""
傅立叶变换分析器
专门用于处理体积数据的频域分析
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, fftfreq, fftshift
from typing import Tuple, Dict, Optional, List
import warnings


class FourierAnalyzer:
    """体积数据傅立叶变换分析器"""
    
    def __init__(self):
        self.fft_data = None
        self.frequencies = None
        self.power_spectrum = None
        self.original_shape = None
    
    def compute_3d_fft(self, volume: np.ndarray, 
                       center_dc: bool = True) -> np.ndarray:
        """
        计算3D傅立叶变换
        
        Args:
            volume: 输入3D体积数据
            center_dc: 是否将DC分量移到中心
            
        Returns:
            复数FFT结果
        """
        print(f"开始计算3D FFT，数据形状: {volume.shape}")
        
        # 确保数据是float类型
        if volume.dtype not in [np.float32, np.float64]:
            volume = volume.astype(np.float32)
        
        # 计算3D FFT
        fft_result = fftn(volume)
        
        if center_dc:
            fft_result = fftshift(fft_result)
        
        self.fft_data = fft_result
        self.original_shape = volume.shape
        
        # 计算频率轴
        self._compute_frequency_axes()
        
        print("3D FFT计算完成")
        return fft_result
    
    def _compute_frequency_axes(self):
        """计算频率轴"""
        if self.original_shape is None:
            raise ValueError("必须先计算FFT")
        
        self.frequencies = {}
        for i, dim in enumerate(['x', 'y', 'z']):
            freq = fftfreq(self.original_shape[i])
            if self.fft_data is not None:
                freq = fftshift(freq)
            self.frequencies[dim] = freq
    
    def compute_power_spectrum(self) -> np.ndarray:
        """
        计算功率谱
        
        Returns:
            功率谱密度
        """
        if self.fft_data is None:
            raise ValueError("必须先计算FFT")
        
        # 计算功率谱：|FFT|^2
        self.power_spectrum = np.abs(self.fft_data) ** 2
        
        return self.power_spectrum
    
    def compute_radial_average(self, center: Optional[Tuple[int, int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算径向平均功率谱
        
        Args:
            center: 中心点坐标，默认为数组中心
            
        Returns:
            (径向距离, 径向平均功率)
        """
        if self.power_spectrum is None:
            self.compute_power_spectrum()
        
        if center is None:
            center = tuple(s // 2 for s in self.original_shape)
        
        # 创建距离数组
        coords = np.ogrid[:self.original_shape[0], 
                         :self.original_shape[1], 
                         :self.original_shape[2]]
        
        distances = np.sqrt((coords[0] - center[0])**2 + 
                           (coords[1] - center[1])**2 + 
                           (coords[2] - center[2])**2)
        
        # 计算径向平均
        max_radius = int(np.max(distances))
        radial_profile = np.zeros(max_radius)
        
        for r in range(max_radius):
            mask = (distances >= r) & (distances < r + 1)
            if np.any(mask):
                radial_profile[r] = np.mean(self.power_spectrum[mask])
        
        radial_distances = np.arange(max_radius)
        
        return radial_distances, radial_profile
    
    def analyze_dominant_frequencies(self, top_k: int = 10) -> List[Dict]:
        """
        分析主导频率
        
        Args:
            top_k: 返回前k个最强的频率分量
            
        Returns:
            主导频率信息列表
        """
        if self.power_spectrum is None:
            self.compute_power_spectrum()
        
        # 找到最强的频率分量
        flat_power = self.power_spectrum.flatten()
        top_indices = np.argpartition(flat_power, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(flat_power[top_indices])[::-1]]
        
        # 转换为3D坐标
        dominant_freqs = []
        for idx in top_indices:
            coord_3d = np.unravel_index(idx, self.original_shape)
            
            freq_info = {
                'coordinate': coord_3d,
                'power': flat_power[idx],
                'frequency_x': self.frequencies['x'][coord_3d[0]],
                'frequency_y': self.frequencies['y'][coord_3d[1]], 
                'frequency_z': self.frequencies['z'][coord_3d[2]],
                'magnitude': np.sqrt(sum(f**2 for f in [
                    self.frequencies['x'][coord_3d[0]],
                    self.frequencies['y'][coord_3d[1]],
                    self.frequencies['z'][coord_3d[2]]
                ]))
            }
            dominant_freqs.append(freq_info)
        
        return dominant_freqs
    
    def filter_frequency_domain(self, low_pass: Optional[float] = None,
                               high_pass: Optional[float] = None,
                               band_pass: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        在频域进行滤波
        
        Args:
            low_pass: 低通滤波截止频率
            high_pass: 高通滤波截止频率
            band_pass: 带通滤波频率范围 (low, high)
            
        Returns:
            滤波后的空域数据
        """
        if self.fft_data is None:
            raise ValueError("必须先计算FFT")
        
        filtered_fft = self.fft_data.copy()
        
        # 创建频率掩码
        center = tuple(s // 2 for s in self.original_shape)
        coords = np.ogrid[:self.original_shape[0], 
                         :self.original_shape[1], 
                         :self.original_shape[2]]
        
        distances = np.sqrt((coords[0] - center[0])**2 + 
                           (coords[1] - center[1])**2 + 
                           (coords[2] - center[2])**2)
        
        # 归一化距离到[0, 0.5]范围
        max_dist = np.sqrt(sum((s//2)**2 for s in self.original_shape))
        normalized_dist = distances / (2 * max_dist)
        
        # 应用滤波器
        if low_pass is not None:
            mask = normalized_dist <= low_pass
            filtered_fft[~mask] = 0
            
        if high_pass is not None:
            mask = normalized_dist >= high_pass
            filtered_fft[~mask] = 0
            
        if band_pass is not None:
            low, high = band_pass
            mask = (normalized_dist >= low) & (normalized_dist <= high)
            filtered_fft[~mask] = 0
        
        # 逆变换回空域
        filtered_volume = np.real(np.fft.ifftn(np.fft.ifftshift(filtered_fft)))
        
        return filtered_volume
    
    def compute_spectral_statistics(self) -> Dict[str, float]:
        """
        计算频谱统计信息
        
        Returns:
            频谱统计字典
        """
        if self.power_spectrum is None:
            self.compute_power_spectrum()
        
        # 计算各种统计量
        total_power = np.sum(self.power_spectrum)
        dc_power = self.power_spectrum[tuple(s//2 for s in self.original_shape)]
        
        # 去除DC分量后的功率
        power_no_dc = self.power_spectrum.copy()
        power_no_dc[tuple(s//2 for s in self.original_shape)] = 0
        ac_power = np.sum(power_no_dc)
        
        # 频谱重心（质心频率）
        center = tuple(s // 2 for s in self.original_shape)
        coords = np.ogrid[:self.original_shape[0], 
                         :self.original_shape[1], 
                         :self.original_shape[2]]
        
        distances = np.sqrt((coords[0] - center[0])**2 + 
                           (coords[1] - center[1])**2 + 
                           (coords[2] - center[2])**2)
        
        spectral_centroid = np.sum(distances * self.power_spectrum) / total_power
        
        stats = {
            'total_power': float(total_power),
            'dc_power': float(dc_power),
            'ac_power': float(ac_power),
            'dc_ratio': float(dc_power / total_power),
            'spectral_centroid': float(spectral_centroid),
            'max_power': float(np.max(self.power_spectrum)),
            'mean_power': float(np.mean(self.power_spectrum)),
            'std_power': float(np.std(self.power_spectrum))
        }
        
        return stats


class FourierVisualizer:
    """傅立叶变换结果可视化器"""
    
    @staticmethod
    def plot_power_spectrum_slices(power_spectrum: np.ndarray, 
                                  title: str = "Power Spectrum Slices"):
        """绘制功率谱的2D切片"""
        center_slices = [s // 2 for s in power_spectrum.shape]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title)
        
        # 对数刻度显示功率谱
        log_power = np.log10(power_spectrum + 1e-10)
        
        # XY平面 (Z=center)
        im1 = axes[0].imshow(log_power[:, :, center_slices[2]], 
                            cmap='hot', origin='lower')
        axes[0].set_title(f"XY Plane (Z={center_slices[2]})")
        axes[0].set_xlabel("Y")
        axes[0].set_ylabel("X")
        plt.colorbar(im1, ax=axes[0], label='log10(Power)')
        
        # XZ平面 (Y=center)
        im2 = axes[1].imshow(log_power[:, center_slices[1], :], 
                            cmap='hot', origin='lower')
        axes[1].set_title(f"XZ Plane (Y={center_slices[1]})")
        axes[1].set_xlabel("Z")
        axes[1].set_ylabel("X")
        plt.colorbar(im2, ax=axes[1], label='log10(Power)')
        
        # YZ平面 (X=center)
        im3 = axes[2].imshow(log_power[center_slices[0], :, :], 
                            cmap='hot', origin='lower')
        axes[2].set_title(f"YZ Plane (X={center_slices[0]})")
        axes[2].set_xlabel("Z")
        axes[2].set_ylabel("Y")
        plt.colorbar(im3, ax=axes[2], label='log10(Power)')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_radial_spectrum(radial_distances: np.ndarray, 
                           radial_profile: np.ndarray,
                           title: str = "Radial Power Spectrum"):
        """绘制径向功率谱"""
        plt.figure(figsize=(10, 6))
        plt.loglog(radial_distances[1:], radial_profile[1:], 'b-', linewidth=2)
        plt.xlabel('Radial Distance (pixels)')
        plt.ylabel('Average Power')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # 添加理论参考线
        # k^-3 slope (Kolmogorov spectrum)
        k_ref = radial_distances[10:50]
        kolmogorov_ref = radial_profile[10] * (k_ref / radial_distances[10])**(-3)
        plt.loglog(k_ref, kolmogorov_ref, 'r--', alpha=0.7, 
                  label='k^-3 (Kolmogorov)')
        
        plt.legend()
        plt.show()
    
    @staticmethod
    def plot_frequency_comparison(original: np.ndarray, 
                                filtered: np.ndarray,
                                filter_type: str = "Filtered"):
        """比较原始数据和滤波后数据"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Original vs {filter_type}")
        
        center_slice = original.shape[2] // 2
        
        # 原始数据
        im1 = axes[0, 0].imshow(original[:, :, center_slice], 
                               cmap='viridis', origin='lower')
        axes[0, 0].set_title("Original Data")
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 滤波后数据
        im2 = axes[1, 0].imshow(filtered[:, :, center_slice], 
                               cmap='viridis', origin='lower')
        axes[1, 0].set_title(f"{filter_type} Data")
        plt.colorbar(im2, ax=axes[1, 0])
        
        # 差异
        diff = original - filtered
        im3 = axes[0, 1].imshow(diff[:, :, center_slice], 
                               cmap='RdBu_r', origin='lower')
        axes[0, 1].set_title("Difference")
        plt.colorbar(im3, ax=axes[0, 1])
        
        # 统计比较
        axes[0, 2].hist(original.flatten(), bins=50, alpha=0.7, 
                       label='Original', density=True)
        axes[0, 2].hist(filtered.flatten(), bins=50, alpha=0.7, 
                       label=filter_type, density=True)
        axes[0, 2].set_title("Value Distribution")
        axes[0, 2].legend()
        
        # 功率谱比较
        analyzer_orig = FourierAnalyzer()
        analyzer_orig.compute_3d_fft(original)
        analyzer_orig.compute_power_spectrum()
        rad_dist_orig, rad_prof_orig = analyzer_orig.compute_radial_average()
        
        analyzer_filt = FourierAnalyzer()
        analyzer_filt.compute_3d_fft(filtered)
        analyzer_filt.compute_power_spectrum()
        rad_dist_filt, rad_prof_filt = analyzer_filt.compute_radial_average()
        
        axes[1, 1].loglog(rad_dist_orig[1:], rad_prof_orig[1:], 
                         'b-', label='Original')
        axes[1, 1].loglog(rad_dist_filt[1:], rad_prof_filt[1:], 
                         'r-', label=filter_type)
        axes[1, 1].set_xlabel('Radial Distance')
        axes[1, 1].set_ylabel('Power')
        axes[1, 1].set_title('Radial Power Spectrum')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 能量保持
        energy_orig = np.sum(original**2)
        energy_filt = np.sum(filtered**2)
        energy_ratio = energy_filt / energy_orig
        
        axes[1, 2].bar(['Original', filter_type], 
                      [energy_orig, energy_filt])
        axes[1, 2].set_title(f'Energy Comparison\nRatio: {energy_ratio:.3f}')
        axes[1, 2].set_ylabel('Total Energy')
        
        plt.tight_layout()
        plt.show()
