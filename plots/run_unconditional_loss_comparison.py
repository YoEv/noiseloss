#!/usr/bin/env python3

import subprocess
import sys
import os

def run_unconditional_loss_comparison():
    """
    Run unconditional loss comparisons across different token lengths.
    """
    # Define token configurations: suffix -> noise_end
    token_configs = {
        'tk5': 254,
        'tk10': 259,
        'tk20': 269,
        'tk50': 299,
        'tk100': 349,
        'tk200': 399
    }
    
    # Base parameters
    base_dir1 = "Unconditional_loss_time_ins_ori"
    base_dir2_prefix = "Unconditional_loss_time_ins"
    output_base = "Loss_Plot/Phase4_3/loss_diff_plot_unconditional"
    
    # Fixed parameters
    noise_start = 250
    plot_start = 200
    plot_end = 750
    y_min = -8
    y_max = 12
    
    print("Starting unconditional loss comparison across token lengths...")
    print("=" * 60)
    
    success_count = 0
    total_count = len(token_configs)
    
    for token_suffix, noise_end in token_configs.items():
        dir2 = f"{base_dir2_prefix}_{token_suffix}"
        output_dir = f"{output_base}_{token_suffix}"
        
        print(f"Processing {token_suffix} (noise_end: {noise_end})...")
        print(f"  Input dir1: {base_dir1}")
        print(f"  Input dir2: {dir2}")
        print(f"  Output dir: {output_dir}")
        
        # Construct the command
        cmd = [
            'python', 'Plot/plot_loss_diff_time_in_separate.py',
            '--dir1', base_dir1,
            '--dir2', dir2,
            '--output_dir', output_dir,
            '--noise_start', str(noise_start),
            '--noise_end', str(noise_end),
            '--plot_start', str(plot_start),
            '--plot_end', str(plot_end),
            '--y_min', str(y_min),
            '--y_max', str(y_max)
        ]
        
        try:
            # Run the command
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"  ✓ Successfully completed {token_suffix}")
            success_count += 1
            
            # Print any output from the script
            if result.stdout:
                print(f"  Output: {result.stdout.decode('utf-8').strip()}")
                
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to process {token_suffix}")
            print(f"  Error: {e.stderr.decode('utf-8').strip() if e.stderr else str(e)}")
        except FileNotFoundError:
            print(f"  ✗ Error: plot_loss_diff_time_in_separate.py not found")
            print(f"  Make sure the script exists in the Plot/ directory")
        
        print("-" * 50)
    
    print(f"\nComparison completed!")
    print(f"Successfully processed: {success_count}/{total_count} configurations")
    print(f"Results stored in: {output_base}_*")
    
    return success_count == total_count

if __name__ == "__main__":
    success = run_unconditional_loss_comparison()
    sys.exit(0 if success else 1)