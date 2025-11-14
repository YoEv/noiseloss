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
    print(f"=== SEANet Receptive Field Calculation ===")
    print(f"Configuration:")
    print(f"  ratios: {ratios}")
    print(f"  kernel_size: {kernel_size}")
    print(f"  residual_kernel_size: {residual_kernel_size}")
    print(f"  n_residual_layers: {n_residual_layers}")
    print(f"  dilation_base: {dilation_base}")
    print(f"  sample_rate: {sample_rate}")
    print()
    
    rf = 1
    cumulative_stride = 1
    
    print("Layer-by-layer computation:")
    
    rf_increment = (kernel_size - 1) * cumulative_stride
    rf += rf_increment
    print(f"  Initial convolution (kernel={kernel_size}): RF += {rf_increment}, Total RF = {rf}")
    
    for i, ratio in enumerate(ratios):
        print(f"\n  Downsampling block {i+1} (stride={ratio}):")
        
        for res_layer in range(n_residual_layers):
            dilation = dilation_base ** res_layer
            
            for conv_idx in range(compress):
                rf_increment = (residual_kernel_size - 1) * dilation * cumulative_stride
                rf += rf_increment
                print(f"    Residual layer {res_layer+1} - conv {conv_idx+1} (kernel={residual_kernel_size}, dilation={dilation}): RF += {rf_increment}, Total RF = {rf}")
        
        rf_increment = (kernel_size - 1) * cumulative_stride
        rf += rf_increment
        print(f"    Downsampling convolution (kernel={kernel_size}): RF += {rf_increment}, Total RF = {rf}")
        
        cumulative_stride *= ratio
        print(f"    Cumulative stride updated to: {cumulative_stride}")
    
    rf_increment = (last_kernel_size - 1) * cumulative_stride
    rf += rf_increment
    print(f"\n  Final convolution (kernel={last_kernel_size}): RF += {rf_increment}, Total RF = {rf}")
    
    total_stride = np.prod(ratios)
    window_duration_ms = (rf / sample_rate) * 1000
    hop_length = total_stride
    tokens_covered = rf / hop_length
    
    print(f"\n=== Results ===")
    print(f"Total receptive field: {rf} samples")
    print(f"Total downsampling factor: {total_stride}")
    print(f"Equivalent window duration: {window_duration_ms:.2f} ms")
    print(f"Tokens covered: {tokens_covered:.2f} tokens")
    print(f"Frame rate: {sample_rate / total_stride:.1f} Hz")
    
    return rf, window_duration_ms, tokens_covered

def analyze_musicgen_config():
    """Analyze MusicGen 32kHz configuration"""
    print("\n" + "="*50)
    print("MusicGen 32kHz Configuration Analysis")
    print("="*50)
    
    print("\n1. Default config (stride=320):")
    rf1, duration1, tokens1 = calculate_seanet_receptive_field(
        ratios=[8, 5, 4, 2],
        sample_rate=32000
    )
    
    print("\n" + "-"*50)
    
    print("\n2. MusicGen config (stride=640):")
    rf2, duration2, tokens2 = calculate_seanet_receptive_field(
        ratios=[8, 5, 4, 4],
        sample_rate=32000
    )
    
    print("\n" + "="*50)
    print("Comparison:")
    print(f"Default config: RF={rf1} samples, {duration1:.2f}ms, {tokens1:.2f} tokens")
    print(f"MusicGen config: RF={rf2} samples, {duration2:.2f}ms, {tokens2:.2f} tokens")
    print(f"\nThis explains why when injecting noise at token 250,")
    print(f"the loss begins to rise around {tokens2:.0f} tokens earlier (around token {250-int(tokens2):.0f}).")

if __name__ == "__main__":
    analyze_musicgen_config()