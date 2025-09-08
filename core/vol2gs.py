"""
Volume数据到3D Gaussian Splatting数据集转换器
生成体渲染图像和NeRF格式的transforms.json文件
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
import cv2
from tqdm import tqdm
import warnings

class VolumeRenderer:
    def __init__(self, volume: np.ndarray,
                 desity_threshold: float = 0.1,
                 opacity_scale: float = 1.0):
        """
        初始化体积渲染器
        
        Args:
            volume: 3D体积数据
            density_threshold: 密度阈值
            opacity_scale: 透明度缩放因子
        """
        self.volume = volume.astype(np.float32)
        self.density_threshold = desity_threshold
        self.opacity_scale = opacity_scale
        self.shape = self.volume.shape
        
        self.volume_min = np.min(self.volume)
        self.volume_max = np.max(self.volume)
        self.normalized_volume = (self.volume - self.volume_min) / (self.volume_max - self.volume_min)
        
    def set_colormap(self, colormap: str = "viridis")
        import matplotlib.cm as cm
        self.colormap = cm.get_cmap(colormap)
        
    def ray_casting(self, origin: np.ndarray, direction: np.ndarray, 
                   image_width: int, image_height: int,
                   step_size: float = 0.5) -> np.ndarray:
        """
        光线投射渲染
        
        Args:
            origin: 光线起点 (3,)
            direction: 光线方向向量 (H, W, 3)
            image_width: 图像宽度
            image_height: 图像高度
            step_size: 采样步长
            
        Returns:
            渲染的RGB图像 (H, W, 3)
        """
        # 计算光线参数
        max_steps = int(np.sqrt(sum(s**2 for s in self.shape)) / step_size)
        
        # 初始化图像
        image = np.zeros((image_height, image_width, 3), dtype=np.float32)
        alpha_composite = np.zeros((image_height, image_width), dtype=np.float32)
        
        # 创建3D网格插值器
        x_coords = np.linspace(0, self.shape[0]-1, self.shape[0])
        y_coords = np.linspace(0, self.shape[1]-1, self.shape[1])
        z_coords = np.linspace(0, self.shape[2]-1, self.shape[2])
        
        interpolator = RegularGridInterpolator(
            (x_coords, y_coords, z_coords), 
            self.normalized_volume,
            bounds_error=False, 
            fill_value=0.0
        )
        
        # 对每个像素进行光线投射
        for step in range(max_steps):
            t = step * step_size
            
            # 计算当前采样点
            sample_points = origin[None, None, :] + t * direction
            
            # 检查边界
            valid_mask = (
                (sample_points[:, :, 0] >= 0) & (sample_points[:, :, 0] < self.shape[0]) &
                (sample_points[:, :, 1] >= 0) & (sample_points[:, :, 1] < self.shape[1]) &
                (sample_points[:, :, 2] >= 0) & (sample_points[:, :, 2] < self.shape[2])
            )
            
            if not np.any(valid_mask):
                continue
            
            # 采样体积数据
            flat_points = sample_points[valid_mask]
            if len(flat_points) == 0:
                continue
                
            sampled_values = interpolator(flat_points)
            
            # 计算透明度和颜色
            alpha = np.clip(sampled_values * self.opacity_scale, 0, 1)
            colors = self.colormap(sampled_values)[:, :3]  # 移除alpha通道
            
            # 体积渲染合成
            full_alpha = np.zeros((image_height, image_width))
            full_colors = np.zeros((image_height, image_width, 3))
            
            full_alpha[valid_mask] = alpha
            full_colors[valid_mask] = colors
            
            # Alpha blending
            transmittance = 1.0 - alpha_composite
            contribution = full_alpha * transmittance
            
            for c in range(3):
                image[:, :, c] += contribution * full_colors[:, :, c]
            
            alpha_composite += contribution
            
            # 早期终止（如果透明度接近1）
            if np.mean(alpha_composite) > 0.99:
                break
        
        # 确保值在[0,1]范围内
        image = np.clip(image, 0, 1)
        
        return image
    
    def volume_projection(self, view_matrix: np.ndarray, 
                         projection_matrix: np.ndarray,
                         image_width: int = 512, 
                         image_height: int = 512) -> np.ndarray:
        """
        基于投影矩阵的体积渲染
        
        Args:
            view_matrix: 视图矩阵 (4, 4)
            projection_matrix: 投影矩阵 (4, 4)
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            渲染图像 (H, W, 3)
        """
        # 创建图像平面的射线
        i, j = np.meshgrid(
            np.linspace(-1, 1, image_width), 
            np.linspace(-1, 1, image_height), 
            indexing='xy'
        )
        
        # 射线方向（从相机到像素）
        directions = np.stack([i, j, -np.ones_like(i)], axis=-1)
        directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
        
        # 变换到世界坐标系
        inv_view = np.linalg.inv(view_matrix)
        camera_pos = inv_view[:3, 3]
        
        # 变换方向向量
        world_directions = np.zeros_like(directions)
        for y in range(image_height):
            for x in range(image_width):
                dir_homo = np.append(directions[y, x], 0)
                world_dir = inv_view @ dir_homo
                world_directions[y, x] = world_dir[:3]
        
        # 归一化方向向量
        world_directions = world_directions / np.linalg.norm(world_directions, axis=-1, keepdims=True)
        
        # 将世界坐标映射到体积坐标
        volume_center = np.array(self.shape) / 2
        scale = max(self.shape)
        
        volume_origin = (camera_pos + volume_center) / scale * max(self.shape)
        volume_directions = world_directions / scale * max(self.shape)
        
        # 执行光线投射
        image = self.ray_casting(volume_origin, volume_directions, 
                               image_width, image_height)
        
        return image
    
class CameraTrajectoryGenerator:
"""相机轨迹生成器"""

    def __init__(self, volume_shape: Tuple[int, int, int]):
        """
        初始化相机轨迹生成器

        Args:
            volume_shape: 体积数据形状
        """
        
        self.volume_shape = volume_shape
        self.volume_center = np.array(volume_shape) / 2
        self.volume_scale = max(volume_shape)
        
    def generate_spherical_trajectory(self, num_views: int = 100,
                                      radius: float = 3.0,
                                      elevation_range: Tuple[float, float] = (-30, 30),
                                      full_rotation: bool = True
                                      ) -> List[np.ndarray]:
        """
        生成球面轨迹的相机姿态
        
        Args:
            num_views: 视角数量
            radius: 相机距离中心的半径
            elevation_range: 仰角范围（度）
            full_rotation: 是否完整旋转360度
            
        Returns:
            相机变换矩阵列表
        """
        transforms = []
        
        return transforms
        
          @staticmethod
    def look_at_matrix(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
        """
        构建look-at变换矩阵
        
        Args:
            eye: 相机位置
            center: 目标点
            up: 上向量
            
        Returns:
            4x4变换矩阵
        """
        f = center - eye
        f = f / np.linalg.norm(f)
        
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        
        u = np.cross(s, f)
        
        view_matrix = np.array([
            [s[0], s[1], s[2], -np.dot(s, eye)],
            [u[0], u[1], u[2], -np.dot(u, eye)],
            [-f[0], -f[1], -f[2], np.dot(f, eye)],
            [0, 0, 0, 1]
        ])
        
        return view_matrix
    
class Volume3DGSDatasetGenerator:
    """Volume到3DGS数据集生成器"""
    def __init__(self, volume: np.ndarray, output_dir: str):
        """
        初始化3DGS数据集生成器
            
        Args:
            volume: 输入体积数据
            output_dir: 输出目录
        """
        self.volume = volume
        self.output_dir = output_dir
        self.trajectory_generator = CameraTrajectoryGenerator(volume.shape)
        self.volume_renderer = VolumeRenderer(volume)   
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        
    def generate_dataset(self, 
                        num_views: int = 100,
                        image_width: int = 512,
                        image_height: int = 512,
                        trajectory_type: str = "spherical",
                        test_split: float = 0.1,
                        camera_params: Optional[Dict] = None) -> Dict:
        
        """
        生成完整的3DGS数据集
        
        Args:
            num_views: 视角数量
            image_width: 图像宽度
            image_height: 图像高度
            trajectory_type: 轨迹类型 ("spherical", "circular", "random")
            test_split: 测试集比例
            camera_params: 相机参数
            
        Returns:
            数据集信息字典
        """
        
        print(f"开始生成3DGS数据集，总共 {num_views} 个视角...")
    
        