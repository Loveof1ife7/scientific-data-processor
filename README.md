
"""
# 体积数据处理项目

这是一个专门用于处理RAW格式体积数据的Python项目，支持多尺度处理、数据可视化和批量处理。

## 功能特性

- 🔧 **灵活的配置系统**: 通过JSON文件配置不同的数据集
- ⚡ **数据处理**: 支持创建数据金字塔和多分辨率分析、块分割、聚类、标准化等处理功能
- 🎨 **可视化**: 2D切片查看、3D体积渲染、交互式浏览
- 🔄 **批量处理**: 支持时间序列数据的批量处理

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 配置数据集

编辑 `config/datasets.json` 文件:

```json
{
  "YourDataset": {
    "dim": [256, 256, 256],
    "total_samples": 100,
    "vars": ["Scalar"],
    "data_path": {
      "Scalar": "./Data/your_data/data_"
    },
    "_comment": "Your dataset description"
  }
}
```

### 2. 列出可用数据集

```bash
python main.py --list
```

### 3. 验证数据集

```bash
python main.py --validate YourDataset
```

### 4. 处理单个体积

```bash
python main.py --dataset Supernova --variable Scalar --sample 0
```

### 5. 批量处理

```bash
python main.py --dataset Supernova --variable Scalar --batch 0,10 --output results/
```

## 项目结构

```
volume_data_processor/
├── config/
│   ├── config_loader.py    # 配置加载器
│   └── datasets.json       # 数据集配置
├── core/
│   ├── data_loader.py      # 数据加载器
│   ├── processor.py        # 数据处理器
│   └── utils.py           # 工具函数
├── visualization/
│   └── visualizer.py      # 可视化工具
├── main.py                # 主程序
└── requirements.txt       # 依赖包
```

## 使用示例

### Python API 使用

```python
from config.config_loader import ConfigLoader
from core.data_loader import VolumeDataLoader
from core.processor import VolumeProcessor
from visualization.visualizer import VolumeVisualizer

# 加载配置
config_loader = ConfigLoader()
dataset_config = config_loader.get_dataset_config("Supernova")

# 加载数据
data_loader = VolumeDataLoader(dataset_config)
volume = data_loader.load_volume("Scalar", 0)

# 处理数据
processor = VolumeProcessor()
pyramid = processor.create_multiscale_pyramid(volume, [1, 2, 4, 8])
blocks = processor.extract_blocks(volume)

# 可视化
visualizer = VolumeVisualizer()
visualizer.interactive_slice_viewer(volume)
visualizer.plot_multiscale_comparison(pyramid)
```

## 许可证

MIT License
"""