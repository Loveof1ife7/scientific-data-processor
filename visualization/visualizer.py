
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from typing import Optional, Tuple, List, Dict
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly未安装，3D可视化功能将不可用")


class VolumeVisualizer:
    """体积数据可视化器"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
    
    def plot_slice(self, volume: np.ndarray, slice_axis: int = 2, 
                   slice_idx: Optional[int] = None, title: str = "Volume Slice"):
        """
        绘制体积的2D切片
        
        Args:
            volume: 3D体积数据
            slice_axis: 切片轴 (0=x, 1=y, 2=z)
            slice_idx: 切片索引，默认为中间切片
            title: 图像标题
        """
        if slice_idx is None:
            slice_idx = volume.shape[slice_axis] // 2
        
        if slice_axis == 0:
            slice_data = volume[slice_idx, :, :]
        elif slice_axis == 1:
            slice_data = volume[:, slice_idx, :]
        else:  # slice_axis == 2
            slice_data = volume[:, :, slice_idx]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(slice_data, cmap='viridis', origin='lower')
        plt.colorbar(label='Value')
        plt.title(f"{title} - Axis {slice_axis}, Slice {slice_idx}")
        plt.xlabel(f"{'Y' if slice_axis == 0 else 'X'}")
        plt.ylabel(f"{'Z' if slice_axis == 2 else 'Y'}")
        plt.show()
    
    def interactive_slice_viewer(self, volume: np.ndarray, title: str = "Interactive Volume Viewer"):
        """
        交互式切片查看器
        
        Args:
            volume: 3D体积数据
            title: 图像标题
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title)
        
        # 初始切片索引
        mid_x, mid_y, mid_z = [s // 2 for s in volume.shape]
        
        # 显示三个方向的切片
        im1 = axes[0].imshow(volume[mid_x, :, :], cmap='viridis', origin='lower')
        axes[0].set_title(f"YZ Plane (X={mid_x})")
        axes[0].set_xlabel("Z")
        axes[0].set_ylabel("Y")
        
        im2 = axes[1].imshow(volume[:, mid_y, :], cmap='viridis', origin='lower')
        axes[1].set_title(f"XZ Plane (Y={mid_y})")
        axes[1].set_xlabel("Z")
        axes[1].set_ylabel("X")
        
        im3 = axes[2].imshow(volume[:, :, mid_z], cmap='viridis', origin='lower')
        axes[2].set_title(f"XY Plane (Z={mid_z})")
        axes[2].set_xlabel("Y")
        axes[2].set_ylabel("X")
        
        # 添加颜色条
        for ax, im in zip(axes, [im1, im2, im3]):
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()
    
    def plot_volume_3d(self, volume: np.ndarray, threshold: Optional[float] = None,
                      title: str = "3D Volume Visualization"):
        """
        3D体积可视化
        
        Args:
            volume: 3D体积数据
            threshold: 阈值，只显示大于阈值的值
            title: 图像标题
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly未安装，无法进行3D可视化")
            return
        
        if threshold is None:
            threshold = np.percentile(volume, 95)  # 使用95%分位数作为阈值
        
        # 创建体积的等值面
        x, y, z = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]
        
        fig = go.Figure(data=go.Volume(
            x=x.flatten(),
            y=y.flatten(), 
            z=z.flatten(),
            value=volume.flatten(),
            isomin=threshold,
            isomax=volume.max(),
            opacity=0.1,
            surface_count=15,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        fig.show()
    
    def plot_statistics(self, volumes: List[np.ndarray], labels: List[str]):
        """
        绘制多个体积的统计信息对比
        
        Args:
            volumes: 体积数据列表
            labels: 对应的标签列表
        """
        stats = []
        for volume, label in zip(volumes, labels):
            stats.append({
                'label': label,
                'min': np.min(volume),
                'max': np.max(volume),
                'mean': np.mean(volume),
                'std': np.std(volume)
            })
        
        # 绘制统计信息
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['min', 'max', 'mean', 'std']
        titles = ['Minimum Values', 'Maximum Values', 'Mean Values', 'Standard Deviation']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i // 2, i % 2]
            values = [stat[metric] for stat in stats]
            ax.bar(labels, values)
            ax.set_title(title)
            ax.set_ylabel('Value')
            
        plt.tight_layout()
        plt.show()
    
    def plot_multiscale_comparison(self, pyramid: Dict[int, np.ndarray], 
                                  slice_axis: int = 2):
        """
        绘制多尺度金字塔对比
        
        Args:
            pyramid: 多尺度体积字典
            slice_axis: 切片轴
        """
        scales = sorted(pyramid.keys())
        n_scales = len(scales)
        
        fig, axes = plt.subplots(1, n_scales, figsize=(4*n_scales, 4))
        if n_scales == 1:
            axes = [axes]
        
        for i, scale in enumerate(scales):
            volume = pyramid[scale]
            slice_idx = volume.shape[slice_axis] // 2
            
            if slice_axis == 0:
                slice_data = volume[slice_idx, :, :]
            elif slice_axis == 1:
                slice_data = volume[:, slice_idx, :]
            else:
                slice_data = volume[:, :, slice_idx]
            
            im = axes[i].imshow(slice_data, cmap='viridis', origin='lower')
            axes[i].set_title(f"Scale 1:{scale}\nShape: {volume.shape}")
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.show()
