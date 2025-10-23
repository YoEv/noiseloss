import numpy as np
import pandas as pd
import os
import glob
import argparse

def convert_npy_to_csv(token_output_dir, output_csv_file):
    """
    Convert all .npy files in token_output_dir to a single CSV file
    """
    npy_files = glob.glob(os.path.join(token_output_dir, "*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {token_output_dir}")
        return
    
    all_data = []
    
    for npy_file in npy_files:
        # Load the numpy array
        data = np.load(npy_file)
        
        # Get filename without extension
        filename = os.path.splitext(os.path.basename(npy_file))[0]
        # Remove '_tokens' suffix if present
        if filename.endswith('_tokens'):
            filename = filename[:-7]
        
        # Flatten the array and create a record for each token
        if data.ndim == 2:  # Shape: (K, T) - codebooks x time
            K, T = data.shape
            for k in range(K):
                for t in range(T):
                    all_data.append({
                        'filename': filename,
                        'codebook': k,
                        'token_position': t,
                        'loss_value': data[k, t]
                    })
        elif data.ndim == 3:  # Shape: (B, K, T) - batch x codebooks x time
            B, K, T = data.shape
            for b in range(B):
                for k in range(K):
                    for t in range(T):
                        all_data.append({
                            'filename': filename,
                            'batch': b,
                            'codebook': k,
                            'token_position': t,
                            'loss_value': data[b, k, t]
                        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv_file, index=False)
    print(f"Converted {len(npy_files)} .npy files to {output_csv_file}")
    print(f"Total records: {len(df)}")

def convert_npy_to_summary_csv(token_output_dir, output_csv_file):
    """
    Convert .npy files to a summary CSV with statistics per file
    """
    npy_files = glob.glob(os.path.join(token_output_dir, "*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {token_output_dir}")
        return
    
    summary_data = []
    
    for npy_file in npy_files:
        data = np.load(npy_file)
        
        filename = os.path.splitext(os.path.basename(npy_file))[0]
        if filename.endswith('_tokens'):
            filename = filename[:-7]
        
        summary_data.append({
            'filename': filename,
            'mean_loss': np.mean(data),
            'std_loss': np.std(data),
            'min_loss': np.min(data),
            'max_loss': np.max(data),
            'total_tokens': data.size,
            'shape': str(data.shape)
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_csv_file, index=False)
    print(f"Created summary CSV with {len(df)} files: {output_csv_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert .npy token files to CSV")
    parser.add_argument("--token_dir", type=str, required=True, help="Directory containing .npy token files")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV file path")
    parser.add_argument("--summary_only", action="store_true", help="Create summary CSV instead of detailed CSV")
    
    args = parser.parse_args()
    
    if args.summary_only:
        convert_npy_to_summary_csv(args.token_dir, args.output_csv)
    else:
        convert_npy_to_csv(args.token_dir, args.output_csv)

if __name__ == "__main__":
    main()