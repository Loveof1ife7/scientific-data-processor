
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict, Any, Optional
import warnings


class VolumeProcessor:
    """体积数据处理器"""
    
    def __init__(self, block_dims: Tuple[int, int, int] = (8, 8, 8)):
        self.block_dims = block_dims
        self.processed_data = {}
    
    def downsample_volume(self, volume: np.ndarray, scale_factor: int = 2, 
                         apply_gaussian: bool = True) -> np.ndarray:
        """
        下采样体积数据
        
        Args:
            volume: 输入体积
            scale_factor: 缩放因子
            apply_gaussian: 是否应用高斯滤波
            
        Returns:
            下采样后的体积
        """
        if apply_gaussian:
            # 应用高斯滤波减少锯齿效应
            sigma = scale_factor / 4.0
            volume = gaussian_filter(volume, sigma=sigma)
        
        # 下采样
        downsampled = volume[::scale_factor, ::scale_factor, ::scale_factor]
        return downsampled
    
    def create_multiscale_pyramid(self, volume: np.ndarray, 
                                 scales: List[int]) -> Dict[int, np.ndarray]:
        """
        创建多尺度金字塔
        
        Args:
            volume: 原始体积
            scales: 尺度列表
            
        Returns:
            多尺度体积字典
        """
        pyramid = {1: volume.copy()}
        
        for scale in sorted(scales):
            if scale == 1:
                continue
            
            current_volume = volume.copy()
            current_scale = scale
            
            while current_scale > 1:
                current_volume = self.downsample_volume(current_volume, 2)
                current_scale //= 2
            
            pyramid[scale] = current_volume
        
        return pyramid
    
    def extract_blocks(self, volume: np.ndarray, 
                      stride: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        从体积中提取块
        
        Args:
            volume: 输入体积
            stride: 步长，默认等于块尺寸
            
        Returns:
            块数组 (n_blocks, block_size)
        """
        if stride is None:
            stride = self.block_dims
        
        # 计算需要的填充
        padding = self._calculate_padding(volume.shape, stride)
        
        # 添加填充
        padded_volume = np.pad(volume, padding, mode='constant')
        
        # 提取块
        blocks = []
        d, h, w = padded_volume.shape
        bd, bh, bw = self.block_dims
        sd, sh, sw = stride
        
        for z in range(0, d - bd + 1, sd):
            for y in range(0, h - bh + 1, sh):
                for x in range(0, w - bw + 1, sw):
                    block = padded_volume[z:z+bd, y:y+bh, x:x+bw]
                    blocks.append(block.flatten())
        
        return np.array(blocks)
    
    def _calculate_padding(self, volume_shape: Tuple[int, int, int], 
                          stride: Tuple[int, int, int]) -> List[Tuple[int, int]]:
        """计算填充大小"""
        padding = []
        for vol_dim, block_dim, step in zip(volume_shape, self.block_dims, stride):
            if vol_dim < block_dim:
                total_padding = block_dim - vol_dim
            else:
                total_padding = (step - ((vol_dim - block_dim) % step)) % step
            
            pad_before = total_padding // 2
            pad_after = total_padding - pad_before
            padding.append((pad_before, pad_after))
        
        return padding
    
    def cluster_blocks(self, blocks: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        对数据块进行聚类
        
        Args:
            blocks: 数据块数组
            n_clusters: 聚类数量
            
        Returns:
            聚类标签和聚类中心
        """
        if len(blocks) < n_clusters:
            warnings.warn(f"块数量({len(blocks)})少于聚类数量({n_clusters})")
            n_clusters = len(blocks)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(blocks)
        
        return labels, kmeans.cluster_centers_
    
    def filter_blocks_by_variance(self, blocks: np.ndarray, 
                                 threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据方差过滤数据块
        
        Args:
            blocks: 数据块数组
            threshold: 方差阈值
            
        Returns:
            过滤后的块和对应的掩码
        """
        variances = np.var(blocks, axis=1)
        mask = variances > threshold
        filtered_blocks = blocks[mask]
        
        return filtered_blocks, mask
    
    def normalize_blocks(self, blocks: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        标准化数据块
        
        Args:
            blocks: 输入数据块
            
        Returns:
            标准化后的块和归一化参数
        """
        block_mins = np.min(blocks, axis=1, keepdims=True)
        block_maxs = np.max(blocks, axis=1, keepdims=True)
        
        # 避免除零
        range_vals = block_maxs - block_mins
        range_vals[range_vals == 0] = 1.0
        
        # 标准化到[-1, 1]
        normalized_blocks = 2.0 * ((blocks - block_mins) / range_vals) - 1.0
        
        # 处理NaN值
        normalized_blocks[np.isnan(normalized_blocks)] = 0.0
        
        normalization_params = {
            'mins': block_mins,
            'maxs': block_maxs
        }
        
        return normalized_blocks, normalization_params
    
    def denormalize_blocks(self, normalized_blocks: np.ndarray, 
                          normalization_params: Dict[str, np.ndarray]) -> np.ndarray:
        """逆标准化"""
        mins = normalization_params['mins']
        maxs = normalization_params['maxs']
        
        # 从[-1, 1]恢复到原始范围
        original_blocks = (normalized_blocks + 1.0) / 2.0 * (maxs - mins) + mins
        
        return original_blocks
