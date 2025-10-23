import numpy as np
from typing import List, Tuple

def calculate_seanet_receptive_field(
    ratios: List[int] = [8, 5, 4, 2],
    kernel_size: int = 7,
    residual_kernel_size: int = 3,
    last_kernel_size: int = 7,
    n_residual_layers: int = 1,
    dilation_base: int = 2,
    compress: int = 2,
    sample_rate: int = 32000
) -> Tuple[int, float, float]:
    """
    计算SEANet编码器的有效感受野
    
    Args:
        ratios: 各级下采样比例 [8, 5, 4, 2]
        kernel_size: 主卷积核大小
        residual_kernel_size: 残差块卷积核大小
        last_kernel_size: 最后一层卷积核大小
        n_residual_layers: 每个下采样块的残差层数
        dilation_base: 膨胀率基数
        compress: 压缩因子
        sample_rate: 采样率
    
    Returns:
        (总感受野样本数, 等效窗口时长ms, 覆盖token数)
    """
    
    print(f"=== SEANet感受野计算 ===")
    print(f"配置参数:")
    print(f"  ratios: {ratios}")
    print(f"  kernel_size: {kernel_size}")
    print(f"  residual_kernel_size: {residual_kernel_size}")
    print(f"  n_residual_layers: {n_residual_layers}")
    print(f"  dilation_base: {dilation_base}")
    print(f"  sample_rate: {sample_rate}")
    print()
    
    # 初始感受野
    rf = 1
    cumulative_stride = 1
    
    print("逐层计算:")
    
    # 初始卷积层
    rf_increment = (kernel_size - 1) * cumulative_stride
    rf += rf_increment
    print(f"  初始卷积 (kernel={kernel_size}): RF += {rf_increment}, 总RF = {rf}")
    
    # 遍历每个下采样块
    for i, ratio in enumerate(ratios):
        print(f"\n  下采样块 {i+1} (stride={ratio}):")
        
        # 该块内的残差层
        for res_layer in range(n_residual_layers):
            # 膨胀率按指数增长: 1, 2, 4, 8, ...
            dilation = dilation_base ** res_layer
            
            # 残差层的两个卷积
            for conv_idx in range(compress):  # compress=2表示每个残差块有2个卷积
                rf_increment = (residual_kernel_size - 1) * dilation * cumulative_stride
                rf += rf_increment
                print(f"    残差层{res_layer+1}-卷积{conv_idx+1} (kernel={residual_kernel_size}, dilation={dilation}): RF += {rf_increment}, 总RF = {rf}")
        
        # 下采样卷积
        rf_increment = (kernel_size - 1) * cumulative_stride
        rf += rf_increment
        print(f"    下采样卷积 (kernel={kernel_size}): RF += {rf_increment}, 总RF = {rf}")
        
        # 更新累计步幅
        cumulative_stride *= ratio
        print(f"    累计步幅更新为: {cumulative_stride}")
    
    # 最后一层卷积
    rf_increment = (last_kernel_size - 1) * cumulative_stride
    rf += rf_increment
    print(f"\n  最后卷积 (kernel={last_kernel_size}): RF += {rf_increment}, 总RF = {rf}")
    
    # 计算等效窗口时长和token覆盖数
    total_stride = np.prod(ratios)  # 总下采样倍数
    window_duration_ms = (rf / sample_rate) * 1000
    hop_length = total_stride  # 对于32kHz: 8*5*4*2 = 320 或 8*5*4*4 = 640
    tokens_covered = rf / hop_length
    
    print(f"\n=== 计算结果 ===")
    print(f"总感受野: {rf} 样本点")
    print(f"总下采样倍数: {total_stride}")
    print(f"等效窗口时长: {window_duration_ms:.2f} ms")
    print(f"覆盖token数: {tokens_covered:.2f} 个token")
    print(f"Frame rate: {sample_rate / total_stride:.1f} Hz")
    
    return rf, window_duration_ms, tokens_covered

def analyze_musicgen_config():
    """分析MusicGen 32kHz配置"""
    print("\n" + "="*50)
    print("MusicGen 32kHz 配置分析")
    print("="*50)
    
    # 默认配置 (ratios=[8,5,4,2], stride=320)
    print("\n1. 默认配置 (stride=320):")
    rf1, duration1, tokens1 = calculate_seanet_receptive_field(
        ratios=[8, 5, 4, 2],
        sample_rate=32000
    )
    
    print("\n" + "-"*50)
    
    # MusicGen配置 (ratios=[8,5,4,4], stride=640)
    print("\n2. MusicGen配置 (stride=640):")
    rf2, duration2, tokens2 = calculate_seanet_receptive_field(
        ratios=[8, 5, 4, 4],
        sample_rate=32000
    )
    
    print("\n" + "="*50)
    print("对比分析:")
    print(f"默认配置: RF={rf1}样本, {duration1:.2f}ms, {tokens1:.2f}token")
    print(f"MusicGen配置: RF={rf2}样本, {duration2:.2f}ms, {tokens2:.2f}token")
    print(f"\n这解释了为什么在token 250注入噪声时，")
    print(f"loss会在前面约{tokens2:.0f}个token(即token {250-int(tokens2):.0f}左右)开始上升!")

if __name__ == "__main__":
    analyze_musicgen_config()