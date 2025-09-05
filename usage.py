# examples/usage_examples.py

"""
体积数据处理项目使用示例
展示如何使用项目的各种功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import ConfigLoader
from core.data_loader import VolumeDataLoader
from core.processor import VolumeProcessor
from visualization.visualizer import VolumeVisualizer
import numpy as np


def example_1_basic_usage():
    """示例1: 基本使用流程"""
    print("=== 示例1: 基本使用流程 ===")
    
    # 1. 加载配置
    config_loader = ConfigLoader("config/datasets.json")
    
    # 2. 列出可用数据集
    print("可用数据集:", config_loader.get_available_datasets())
    
    # 3. 获取特定数据集配置
    try:
        supernova_config = config_loader.get_dataset_config("Supernova")
        print("Supernova配置:", supernova_config)
    except ValueError as e:
        print(f"配置错误: {e}")
        return
    
    # 4. 验证数据路径
    validation_results = config_loader.validate_dataset_paths("Supernova")
    print("路径验证结果:", validation_results)


def example_2_load_and_process_data():
    """示例2: 加载和处理数据"""
    print("\n=== 示例2: 加载和处理数据 ===")
    
    try:
        # 加载配置和数据
        config_loader = ConfigLoader("config/datasets.json")
        dataset_config = config_loader.get_dataset_config("Supernova")
        data_loader = VolumeDataLoader(dataset_config)
        
        # 加载单个体积
        volume = data_loader.load_volume("Scalar", 1)
        print(f"加载的体积形状: {volume.shape}")
        
        # 获取数据统计信息
        stats = data_loader.get_data_statistics("Scalar", 1)
        print("数据统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 处理数据
        processor = VolumeProcessor(block_dims=(8, 8, 8))
        
        # 创建多尺度金字塔
        pyramid = processor.create_multiscale_pyramid(volume, [1, 2, 4])
        print(f"多尺度金字塔尺度: {list(pyramid.keys())}")
        for scale, vol in pyramid.items():
            print(f"  尺度 {scale}: {vol.shape}")
        
        # 提取数据块
        blocks = processor.extract_blocks(volume)
        print(f"提取的数据块: {blocks.shape}")
        
        # 过滤低方差块
        filtered_blocks, mask = processor.filter_blocks_by_variance(blocks, threshold=0.01)
        print(f"过滤后的数据块: {filtered_blocks.shape}")
        
        # 数据标准化
        normalized_blocks, norm_params = processor.normalize_blocks(filtered_blocks)
        print(f"标准化后的数据范围: {normalized_blocks.min():.3f} ~ {normalized_blocks.max():.3f}")
        
        return volume, pyramid, normalized_blocks
        
    except Exception as e:
        print(f"处理错误: {e}")
        return None, None, None


def example_3_visualization():
    """示例3: 数据可视化"""
    print("\n=== 示例3: 数据可视化 ===")
    
    # 生成示例数据（如果无法加载真实数据）
    volume = create_sample_volume()
    
    visualizer = VolumeVisualizer()
    
    # 基本切片可视化
    print("显示基本切片...")
    visualizer.plot_slice(volume, slice_axis=2, title="Sample Volume")
    
    # 交互式切片查看器
    print("显示交互式切片查看器...")
    visualizer.interactive_slice_viewer(volume, "Sample Volume - Interactive")
    
    # 多尺度对比
    processor = VolumeProcessor()
    pyramid = processor.create_multiscale_pyramid(volume, [1, 2, 4])
    print("显示多尺度对比...")
    visualizer.plot_multiscale_comparison(pyramid)


def example_4_batch_processing():
    """示例4: 批量处理"""
    print("\n=== 示例4: 批量处理演示 ===")
    
    # 创建示例数据文件（模拟）
    output_dir = "temp_data"
    create_sample_dataset(output_dir)
    
    # 创建临时配置
    temp_config = {
        "SampleDataset": {
            "dim": [64, 64, 64],
            "total_samples": 5,
            "vars": ["Data"],
            "data_path": {
                "Data": f"{output_dir}/sample_"
            },
            "_comment": "Temporary sample dataset"
        }
    }
    
    # 保存临时配置
    import json
    temp_config_path = "temp_config.json"
    with open(temp_config_path, 'w') as f:
        json.dump(temp_config, f, indent=2)
    
    try:
        # 使用临时配置进行批量处理
        config_loader = ConfigLoader(temp_config_path)
        dataset_config = config_loader.get_dataset_config("SampleDataset")
        data_loader = VolumeDataLoader(dataset_config)
        processor = VolumeProcessor()
        
        # 批量加载和处理
        time_steps = [0, 1, 2, 3, 4]
        results = []
        
        for timestep in time_steps:
            print(f"处理时间步 {timestep}...")
            
            volume = data_loader.load_volume("Data", timestep)
            blocks = processor.extract_blocks(volume)
            filtered_blocks, _ = processor.filter_blocks_by_variance(blocks)
            normalized_blocks, _ = processor.normalize_blocks(filtered_blocks)
            
            results.append({
                'timestep': timestep,
                'volume_shape': volume.shape,
                'num_blocks': len(normalized_blocks),
                'data_range': (normalized_blocks.min(), normalized_blocks.max())
            })
        
        print("批量处理结果:")
        for result in results:
            print(f"  时间步 {result['timestep']}: "
                  f"形状 {result['volume_shape']}, "
                  f"块数量 {result['num_blocks']}, "
                  f"数据范围 [{result['data_range'][0]:.3f}, {result['data_range'][1]:.3f}]")
    
    finally:
        # 清理临时文件
        cleanup_temp_files(output_dir, temp_config_path)


def create_sample_volume(shape=(64, 64, 64)):
    """创建示例体积数据"""
    x, y, z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    
    # 创建一个有趣的3D函数
    center_x, center_y, center_z = [s//2 for s in shape]
    
    # 球形渐变
    sphere = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
    sphere_data = np.exp(-sphere / 10.0)
    
    # 添加一些噪声和结构
    noise = np.random.normal(0, 0.1, shape)
    wave = np.sin(x/5) * np.sin(y/5) * np.sin(z/5)
    
    volume = sphere_data + 0.3 * wave + noise
    return volume.astype(np.float32)


def create_sample_dataset(output_dir):
    """创建示例数据集文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(5):
        volume = create_sample_volume()
        
        # 添加时间变化
        time_factor = np.sin(i * 0.5)
        volume = volume * (1 + 0.2 * time_factor)
        
        # 保存为RAW文件
        filepath = f"{output_dir}/sample_{i:04d}.raw"
        volume.tofile(filepath)
        print(f"创建示例文件: {filepath}")


def cleanup_temp_files(output_dir, config_path):
    """清理临时文件"""
    import shutil
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"删除临时目录: {output_dir}")
    
    if os.path.exists(config_path):
        os.remove(config_path)
        print(f"删除临时配置: {config_path}")


def example_5_advanced_features():
    """示例5: 高级功能演示"""
    print("\n=== 示例5: 高级功能演示 ===")
    
    # 创建示例数据
    volume = create_sample_volume((32, 32, 32))
    processor = VolumeProcessor(block_dims=(4, 4, 4))
    
    # 提取块并进行聚类
    blocks = processor.extract_blocks(volume)
    print(f"原始块数量: {blocks.shape[0]}")
    
    # 聚类分析
    n_clusters = min(10, blocks.shape[0])
    labels, centers = processor.cluster_blocks(blocks, n_clusters)
    
    print(f"聚类结果: {n_clusters} 个聚类")
    print(f"每个聚类的块数量:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  聚类 {label}: {count} 个块")
    
    # 分析不同聚类的特征
    print("\n聚类特征分析:")
    for i in range(n_clusters):
        cluster_blocks = blocks[labels == i]
        if len(cluster_blocks) > 0:
            mean_val = np.mean(cluster_blocks)
            std_val = np.std(cluster_blocks)
            print(f"  聚类 {i}: 均值={mean_val:.3f}, 标准差={std_val:.3f}")


def main():
    """运行所有示例"""
    print("体积数据处理项目使用示例")
    print("=" * 50)
    
    # # 运行示例
    # example_1_basic_usage()
    
    volume, pyramid, blocks = example_2_load_and_process_data()
    
    # if volume is not None:
    #     # 只有在成功加载数据时才运行可视化
    #     example_3_visualization()
    # else:
    #     print("跳过可视化示例（数据加载失败）")
    
    # example_4_batch_processing()
    # example_5_advanced_features()
    
    # print("\n" + "=" * 50)
    # print("所有示例运行完成！")


if __name__ == "__main__":
    main()