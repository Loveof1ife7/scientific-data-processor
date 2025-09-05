
import argparse
import os
from config.config_loader import ConfigLoader
from core.data_loader import VolumeDataLoader
from core.processor import VolumeProcessor
from visualization.visualizer import VolumeVisualizer


class VolumeDataProject:
    """体积数据处理项目主类"""
    
    def __init__(self, config_path: str = "config/datasets.json"):
        self.config_loader = ConfigLoader(config_path)
        self.data_loader = None
        self.processor = VolumeProcessor()
        self.visualizer = VolumeVisualizer()
    
    def list_datasets(self):
        """列出所有可用的数据集"""
        datasets = self.config_loader.get_available_datasets()
        print("可用的数据集:")
        for i, dataset in enumerate(datasets, 1):
            config = self.config_loader.get_dataset_config(dataset)
            print(f"{i}. {dataset}")
            print(f"   维度: {config['dim']}")
            print(f"   变量: {config['vars']}")
            print(f"   样本数: {config['total_samples']}")
            print()
    
    def validate_dataset(self, dataset_name: str):
        """验证数据集配置和路径"""
        try:
            config = self.config_loader.get_dataset_config(dataset_name)
            validation_results = self.config_loader.validate_dataset_paths(dataset_name)
            
            print(f"数据集 '{dataset_name}' 验证结果:")
            print(f"配置: ✓")
            
            all_paths_valid = True
            for var_name, is_valid in validation_results.items():
                status = "✓" if is_valid else "✗"
                print(f"路径 {var_name}: {status}")
                if not is_valid:
                    all_paths_valid = False
            
            return all_paths_valid
            
        except Exception as e:
            print(f"验证失败: {e}")
            return False
    
    def load_dataset(self, dataset_name: str):
        """加载数据集"""
        config = self.config_loader.get_dataset_config(dataset_name)
        self.data_loader = VolumeDataLoader(config)
        print(f"数据集 '{dataset_name}' 加载成功")
    
    def process_single_volume(self, dataset_name: str, variable: str, 
                            sample_id: int, visualize: bool = True):
        """处理单个体积数据"""
        if self.data_loader is None:
            self.load_dataset(dataset_name)
        
        print(f"处理 {dataset_name} - {variable} - 样本 {sample_id}")
        
        # 加载数据
        volume = self.data_loader.load_volume(variable, sample_id)
        print(f"原始体积形状: {volume.shape}")
        
        # 获取统计信息
        stats = self.data_loader.get_data_statistics(variable, sample_id)
        print("数据统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 创建多尺度金字塔
        scales = [1, 2, 4, 8]
        pyramid = self.processor.create_multiscale_pyramid(volume, scales)
        print(f"多尺度金字塔创建完成，尺度: {list(pyramid.keys())}")
        
        # 提取和处理块
        blocks = self.processor.extract_blocks(volume)
        print(f"提取的块数量: {blocks.shape[0]}, 块大小: {blocks.shape[1]}")
        
        # 过滤低方差块
        filtered_blocks, mask = self.processor.filter_blocks_by_variance(blocks)
        print(f"过滤后的块数量: {filtered_blocks.shape[0]}")
        
        # 标准化
        normalized_blocks, norm_params = self.processor.normalize_blocks(filtered_blocks)
        print(f"数据标准化完成")
        
        if visualize:
            # 可视化
            print("开始可视化...")
            self.visualizer.interactive_slice_viewer(volume, f"{dataset_name} - {variable}")
            self.visualizer.plot_multiscale_comparison(pyramid)
        
        return {
            'volume': volume,
            'pyramid': pyramid,
            'blocks': normalized_blocks,
            'norm_params': norm_params,
            'stats': stats
        }
    
    def batch_process(self, dataset_name: str, variable: str, 
                     time_steps: list, output_dir: str = "output"):
        """批量处理多个时间步"""
        if self.data_loader is None:
            self.load_dataset(dataset_name)
        
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        print(f"批量处理 {len(time_steps)} 个时间步...")
        
        for i, timestep in enumerate(time_steps):
            print(f"处理进度: {i+1}/{len(time_steps)} (时间步 {timestep})")
            
            try:
                result = self.process_single_volume(
                    dataset_name, variable, timestep, visualize=False
                )
                results.append({
                    'timestep': timestep,
                    'result': result
                })
                
                # 保存处理结果的统计信息
                output_file = os.path.join(output_dir, f"stats_{timestep:04d}.txt")
                with open(output_file, 'w') as f:
                    for key, value in result['stats'].items():
                        f.write(f"{key}: {value}\n")
                
            except Exception as e:
                print(f"处理时间步 {timestep} 失败: {e}")
                continue
        
        print(f"批量处理完成，成功处理 {len(results)} 个时间步")
        return results


def main():
    parser = argparse.ArgumentParser(description="体积数据处理项目")
    parser.add_argument("--config", default="config/datasets.json", 
                       help="配置文件路径")
    parser.add_argument("--list", action="store_true", 
                       help="列出所有可用数据集")
    parser.add_argument("--validate", type=str, 
                       help="验证指定数据集")
    parser.add_argument("--dataset", type=str, 
                       help="要处理的数据集名称")
    parser.add_argument("--variable", type=str, 
                       help="要处理的变量名称")
    parser.add_argument("--sample", type=int, default=0, 
                       help="要处理的样本ID")
    parser.add_argument("--batch", type=str, 
                       help="批量处理时间步，格式: start,end 或 start,end,step")
    parser.add_argument("--output", type=str, default="output", 
                       help="输出目录")
    parser.add_argument("--no-viz", action="store_true", 
                       help="禁用可视化")
    
    args = parser.parse_args()
    
    # 创建项目实例
    project = VolumeDataProject(args.config)
    
    # 列出数据集
    if args.list:
        project.list_datasets()
        return
    
    # 验证数据集
    if args.validate:
        project.validate_dataset(args.validate)
        return
    
    # 处理数据
    if args.dataset and args.variable:
        if args.batch:
            # 批量处理
            parts = args.batch.split(',')
            if len(parts) == 2:
                start, end = map(int, parts)
                time_steps = list(range(start, end + 1))
            elif len(parts) == 3:
                start, end, step = map(int, parts)
                time_steps = list(range(start, end + 1, step))
            else:
                print("批量处理格式错误，应为: start,end 或 start,end,step")
                return
            
            project.batch_process(args.dataset, args.variable, time_steps, args.output)
        else:
            # 单个处理
            project.process_single_volume(
                args.dataset, args.variable, args.sample, 
                visualize=not args.no_viz
            )
    else:
        print("请指定数据集和变量名称")
        parser.print_help()


if __name__ == "__main__":
    main()


# =============================================================================
# requirements.txt
# =============================================================================

"""
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
scikit-learn>=1.0.0
plotly>=5.0.0
tqdm>=4.60.0
"""