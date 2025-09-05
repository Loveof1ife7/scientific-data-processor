    # 导入必要的模块
import sys
import os
from core.fourier_analyzer import FourierAnalyzer, FourierVisualizer

class SuppernovaFourierProcessor:
    """Supernova数据的傅立叶分析处理器"""
    
    def __init__(self, project):
        self.project = project
        self.fourier_analyzer = FourierAnalyzer()
        self.visualizer = FourierVisualizer()
    
    def full_fourier_analysis(self, dataset_name: str = "Supernova", 
                             variable: str = "Scalar", sample_id: int = 1):
        """对Supernova数据进行完整的傅立叶分析"""
        
        print("=== Supernova 傅立叶分析 ===")
        
        # 1. 加载数据
        if self.project.data_loader is None:
            self.project.load_dataset(dataset_name)
        
        volume = self.project.data_loader.load_volume(variable, sample_id)
        print(f"数据加载完成: {volume.shape}")
        
        # 2. 计算FFT
        fft_result = self.fourier_analyzer.compute_3d_fft(volume)
        print("FFT计算完成")
        
        # 3. 计算功率谱
        power_spectrum = self.fourier_analyzer.compute_power_spectrum()
        print("功率谱计算完成")
        
        # 4. 径向平均
        radial_dist, radial_prof = self.fourier_analyzer.compute_radial_average()
        print("径向平均计算完成")
        
        # 5. 主导频率分析
        dominant_freqs = self.fourier_analyzer.analyze_dominant_frequencies(top_k=10)
        print("主导频率分析完成")
        
        # 6. 频谱统计
        spectral_stats = self.fourier_analyzer.compute_spectral_statistics()
        print("频谱统计计算完成")
        
        # 7. 可视化
        print("开始可视化...")
        
        # 原始数据切片
        from visualization.visualizer import VolumeVisualizer
        vol_viz = VolumeVisualizer()
        vol_viz.interactive_slice_viewer(volume, "Supernova Original Data")
        
        # 功率谱切片
        self.visualizer.plot_power_spectrum_slices(power_spectrum, 
                                                  "Supernova Power Spectrum")
        
        # 径向谱
        self.visualizer.plot_radial_spectrum(radial_dist, radial_prof,
                                           "Supernova Radial Spectrum")
        
        # 8. 频域滤波示例
        print("执行频域滤波...")
        
        # 低通滤波
        low_pass_data = self.fourier_analyzer.filter_frequency_domain(low_pass=0.1)
        self.visualizer.plot_frequency_comparison(volume, low_pass_data, 
                                                 "Low-pass Filtered")
        
        # 高通滤波
        high_pass_data = self.fourier_analyzer.filter_frequency_domain(high_pass=0.05)
        self.visualizer.plot_frequency_comparison(volume, high_pass_data, 
                                                 "High-pass Filtered")
        
        # 9. 打印分析结果
        print("\n=== 分析结果 ===")
        print("频谱统计:")
        for key, value in spectral_stats.items():
            print(f"  {key}: {value:.6e}")
        
        print(f"\n主导频率 (前5个):")
        for i, freq_info in enumerate(dominant_freqs[:5]):
            print(f"  {i+1}. 坐标: {freq_info['coordinate']}")
            print(f"     功率: {freq_info['power']:.6e}")
            print(f"     频率幅值: {freq_info['magnitude']:.6f}")
        
        return {
            'volume': volume,
            'fft_result': fft_result,
            'power_spectrum': power_spectrum,
            'radial_profile': (radial_dist, radial_prof),
            'dominant_frequencies': dominant_freqs,
            'spectral_statistics': spectral_stats,
            'filtered_data': {
                'low_pass': low_pass_data,
                'high_pass': high_pass_data
            }
        }


# 使用示例脚本
def supernova_fourier_example():
    """Supernova傅立叶分析使用示例"""
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from main import VolumeDataProject
    
    # 创建项目实例
    project = VolumeDataProject()
    
    # 创建傅立叶处理器
    fourier_processor = SuppernovaFourierProcessor(project)
    
    # 执行完整分析
    try:
        results = fourier_processor.full_fourier_analysis()
        print("\n分析完成！结果已保存在返回的字典中。")
        
        # 可以进一步处理结果
        # 例如保存关键数据
        import numpy as np
        np.save('supernova_power_spectrum.npy', results['power_spectrum'])
        print("功率谱已保存为 'supernova_power_spectrum.npy'")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        print("请检查数据路径和配置是否正确")


if __name__ == "__main__":
    supernova_fourier_example()