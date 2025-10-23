import os
import shutil
import librosa
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def analyze_tempo(audio_path):
    """
    Analyze the tempo of an audio file using librosa.
    
    Args:
        audio_path (str): Path to the audio file
        
    Returns:
        float: Estimated tempo in BPM
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract tempo using beat tracking
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        return float(tempo)
    except Exception as e:
        print(f"Error analyzing {audio_path}: {e}")
        return None

def categorize_tempo(tempo):
    """
    Categorize tempo into 5 groups based on musical tempo markings.
    
    Args:
        tempo (float): Tempo in BPM
        
    Returns:
        tuple: (category_name, category_number)
    """
    if tempo < 60:
        return "very_slow", 1  # Largo, Grave
    elif tempo < 80:
        return "slow", 2       # Adagio, Lento
    elif tempo < 120:
        return "moderate", 3   # Andante, Moderato
    elif tempo < 160:
        return "fast", 4       # Allegro
    else:
        return "very_fast", 5  # Presto, Prestissimo

def create_output_directories(base_output_dir):
    """
    Create output directories for each tempo category.
    
    Args:
        base_output_dir (str): Base directory for output
        
    Returns:
        dict: Mapping of category names to directory paths
    """
    categories = {
        "very_slow": "1_very_slow_under60",
        "slow": "2_slow_60-80", 
        "moderate": "3_moderate_80-120",
        "fast": "4_fast_120-160",
        "very_fast": "5_very_fast_over160"
    }
    
    output_dirs = {}
    for category, dir_name in categories.items():
        dir_path = os.path.join(base_output_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        output_dirs[category] = dir_path
        
    return output_dirs

def process_audio_files(input_dir, output_dir):
    """
    Process all audio files in the input directory, analyze tempo, and organize into categories.
    
    Args:
        input_dir (str): Directory containing audio files
        output_dir (str): Base output directory for categorized files
    """
    # Get all audio files
    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Create output directories
    output_dirs = create_output_directories(output_dir)
    
    # Store tempo analysis results
    tempo_results = []
    category_counts = {"very_slow": 0, "slow": 0, "moderate": 0, "fast": 0, "very_fast": 0}
    
    # Process each audio file
    for filename in tqdm(audio_files, desc="Analyzing tempo"):
        input_path = os.path.join(input_dir, filename)
        
        # Analyze tempo
        tempo = analyze_tempo(input_path)
        
        if tempo is not None:
            # Categorize tempo
            category, category_num = categorize_tempo(tempo)
            
            # Store results
            tempo_results.append({
                'filename': filename,
                'tempo': tempo,
                'category': category,
                'category_num': category_num
            })
            
            # Copy file to appropriate category directory
            output_path = os.path.join(output_dirs[category], filename)
            shutil.copy2(input_path, output_path)
            
            category_counts[category] += 1
            
            print(f"{filename}: {tempo:.1f} BPM -> {category}")
        else:
            print(f"Failed to analyze tempo for {filename}")
    
    # Sort results by tempo
    tempo_results.sort(key=lambda x: x['tempo'])
    
    # Print summary
    print("\n" + "="*60)
    print("TEMPO ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nTotal files processed: {len(tempo_results)}")
    print(f"\nTempo distribution:")
    for category, count in category_counts.items():
        percentage = (count / len(tempo_results)) * 100 if tempo_results else 0
        print(f"  {category.replace('_', ' ').title()}: {count} files ({percentage:.1f}%)")
    
    # Print detailed results
    print(f"\nDetailed results (sorted by tempo):")
    for result in tempo_results:
        print(f"  {result['filename']}: {result['tempo']:.1f} BPM ({result['category']})")
    
    # Save results to file
    results_file = os.path.join(output_dir, "tempo_analysis_results.txt")
    with open(results_file, 'w') as f:
        f.write("Tempo Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total files processed: {len(tempo_results)}\n\n")
        
        f.write("Tempo distribution:\n")
        for category, count in category_counts.items():
            percentage = (count / len(tempo_results)) * 100 if tempo_results else 0
            f.write(f"  {category.replace('_', ' ').title()}: {count} files ({percentage:.1f}%)\n")
        
        f.write("\nDetailed results (sorted by tempo):\n")
        for result in tempo_results:
            f.write(f"  {result['filename']}: {result['tempo']:.1f} BPM ({result['category']})\n")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Files organized in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze tempo of audio files and organize them into 5 tempo categories')
    parser.add_argument('--input_dir', type=str, default='../asap_100', 
                       help='Input directory containing audio files (default: ../asap_100)')
    parser.add_argument('--output_dir', type=str, default='../tempo_groups', 
                       help='Output directory for categorized files (default: ../tempo_groups)')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(script_dir, args.input_dir))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    # Validate input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process files
    process_audio_files(input_dir, output_dir)

if __name__ == "__main__":
    main()