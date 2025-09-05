
import json
import os
from typing import Dict, Any, List
from pathlib import Path

class ConfigLoader:
    """配置文件加载器"""
    
    def __init__(self, config_path: str = "config/datasets.json"):
        self.config_path = config_path
        self.datasets_config = None
        self.load_config()
    
    def load_config(self) -> None:
        """加载JSON配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.datasets_config = json.load(f)
            print(f"配置文件加载成功: {self.config_path}")
        except FileNotFoundError:
            print(f"配置文件未找到: {self.config_path}")
            self.create_default_config()
        except json.JSONDecodeError as e:
            print(f"配置文件格式错误: {e}")
            raise
    
    def create_default_config(self) -> None:
        """创建默认配置文件"""
        default_config = {
            "Supernova": {
                "dim": [432, 432, 432],
                "total_samples": 1,
                "vars": ["Scalar"],
                "data_path": {
                    "Scalar": "~/datasets/supernova/E_"
                },
                "_comment": "available total_time_samples is 60"
            },
            "Tangaroa": {
                "dim": [300, 180, 120],
                "total_samples": 150,
                "vars": ["VTM"],
                "data_path": {
                    "VTM": "~/datasets/tangaroa/tangaroa-"
                },
                "_comment": "available total_time_samples is 100"
            }
        }
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        self.datasets_config = default_config
        print(f"默认配置文件已创建: {self.config_path}")
    
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        if self.datasets_config is None:
            raise RuntimeError("Configuration has not been loaded.")
        """获取指定数据集的配置"""
        if dataset_name not in self.datasets_config:
            raise ValueError(f"数据集 '{dataset_name}' 不存在于配置文件中")
        return self.datasets_config[dataset_name]
    
    def get_available_datasets(self) -> List[str]:
        """获取所有可用的数据集名称"""
        return list(self.datasets_config.keys())
    
    def validate_dataset_paths(self, dataset_name: str) -> Dict[str, bool]:
        """验证数据集路径是否存在"""
        config = self.get_dataset_config(dataset_name)
        validation_results = {}
        
        for var_name, data_path in config["data_path"].items():
            path_dir = os.path.dirname(data_path)
            validation_results[var_name] = os.path.exists(path_dir)
        
        return validation_results