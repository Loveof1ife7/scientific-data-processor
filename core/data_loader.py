
import numpy as np
import os
from typing import Tuple, List, Optional, Dict, Any
from scipy.ndimage import gaussian_filter
import warnings


class VolumeDataLoader:
    """体积数据加载器"""
    
    def __init__(self, dataset_config: Dict[str, Any]):
        self.config = dataset_config
        self.dimensions = dataset_config["dim"]
        self.variables = dataset_config["vars"]
        self.data_paths = dataset_config["data_path"]
        self.total_samples = dataset_config["total_samples"]
        
    def read_raw_file(self, filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
        """
        读取RAW格式文件
        
        Args:
            filepath: 文件路径
            dtype: 数据类型
            
        Returns:
            一维数组数据
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        try:
            data = np.fromfile(filepath, dtype=dtype)
            expected_size = np.prod(self.dimensions)
            
            if len(data) != expected_size:
                warnings.warn(
                    f"数据大小不匹配: 期望 {expected_size}, 实际 {len(data)}"
                )
            
            return data
        except Exception as e:
            raise RuntimeError(f"读取文件失败 {filepath}: {e}")
    
    def load_volume(self, variable: str, sample_id: int, 
                   reshape: bool = True) -> np.ndarray:
        """
        加载指定变量和样本的体积数据
        
        Args:
            variable: 变量名称
            sample_id: 样本ID
            reshape: 是否重塑为3D数组
            
        Returns:
            体积数据
        """
        if variable not in self.variables:
            raise ValueError(f"变量 '{variable}' 不在配置的变量列表中")
        
        data_path_template = self.data_paths[variable]
        filepath = f"{data_path_template}{sample_id:04d}.raw"
        
        # 读取RAW数据
        raw_data = self.read_raw_file(filepath)
        
        if reshape:
            # 重塑为3D数组并转置到正确的维度顺序
            volume = raw_data.reshape(
                self.dimensions[2], self.dimensions[1], self.dimensions[0]
            ).transpose()
            return volume
        
        return raw_data
    
    def load_time_series(self, variable: str, time_steps: List[int]) -> List[np.ndarray]:
        """
        加载时间序列数据
        
        Args:
            variable: 变量名称
            time_steps: 时间步列表
            
        Returns:
            时间序列体积数据列表
        """
        volumes = []
        for timestep in time_steps:
            try:
                volume = self.load_volume(variable, timestep)
                volumes.append(volume)
            except Exception as e:
                print(f"加载时间步 {timestep} 失败: {e}")
                continue
        
        return volumes
    
    def get_data_statistics(self, variable: str, sample_id: int) -> Dict[str, float]:
        """
        获取数据统计信息
        
        Args:
            variable: 变量名称
            sample_id: 样本ID
            
        Returns:
            统计信息字典
        """
        volume = self.load_volume(variable, sample_id)
        
        stats = {
            'min': float(np.min(volume)),
            'max': float(np.max(volume)),
            'mean': float(np.mean(volume)),
            'std': float(np.std(volume)),
            'shape': volume.shape,
            'dtype': str(volume.dtype),
            'size_mb': volume.nbytes / (1024 * 1024)
        }
        
        return stats

