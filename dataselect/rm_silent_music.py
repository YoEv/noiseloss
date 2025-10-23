import os
import shutil
import librosa
import numpy as np
from pathlib import Path
import soundfile as sf
from tqdm import tqdm

def detect_silence_segments(audio, sr, silence_threshold_db=-60, min_silence_duration=2.0):
    """
    Detect silence segments in audio.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        silence_threshold_db: Threshold in dB below which audio is considered silent
        min_silence_duration: Minimum duration in seconds for a segment to be considered silence
    
    Returns:
        List of tuples (start_time, end_time) for silence segments longer than min_silence_duration
    """
    # Convert to dB using RMS in windows
    hop_length = 512
    frame_length = 2048
    
    # Calculate RMS energy for each frame
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Convert RMS to dB
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Find silent frames
    silent_frames = rms_db < silence_threshold_db
    
    # Convert frame indices to time
    times = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=hop_length)
    
    # Find continuous silence segments
    silence_segments = []
    in_silence = False
    silence_start = 0
    
    for i, is_silent in enumerate(silent_frames):
        current_time = times[i]
        
        if is_silent and not in_silence:
            # Start of silence
            in_silence = True
            silence_start = current_time
        elif not is_silent and in_silence:
            # End of silence
            in_silence = False
            silence_duration = current_time - silence_start
            if silence_duration >= min_silence_duration:
                silence_segments.append((silence_start, current_time))
    
    # Handle case where audio ends in silence
    if in_silence and len(times) > 0:
        silence_duration = times[-1] - silence_start
        if silence_duration >= min_silence_duration:
            silence_segments.append((silence_start, times[-1]))
    
    return silence_segments

def detect_low_volume_segments(audio, sr, volume_threshold_db=-60, min_duration=1.0):
    """
    Detect continuous low volume segments longer than min_duration.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        volume_threshold_db: Volume threshold in dB
        min_duration: Minimum duration in seconds for a segment to be considered problematic
    
    Returns:
        List of tuples (start_time, end_time) for low volume segments longer than min_duration
    """
    # Use same windowing as silence detection for consistency
    hop_length = 512
    frame_length = 2048
    
    # Calculate RMS energy for each frame
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Convert RMS to dB
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Find low volume frames
    low_volume_frames = rms_db < volume_threshold_db
    
    # Convert frame indices to time
    times = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=hop_length)
    
    # Find continuous low volume segments
    low_volume_segments = []
    in_low_volume = False
    low_volume_start = 0
    
    for i, is_low_volume in enumerate(low_volume_frames):
        current_time = times[i]
        
        if is_low_volume and not in_low_volume:
            # Start of low volume segment
            in_low_volume = True
            low_volume_start = current_time
        elif not is_low_volume and in_low_volume:
            # End of low volume segment
            in_low_volume = False
            segment_duration = current_time - low_volume_start
            if segment_duration >= min_duration:
                low_volume_segments.append((low_volume_start, current_time))
    
    # Handle case where audio ends in low volume
    if in_low_volume and len(times) > 0:
        segment_duration = times[-1] - low_volume_start
        if segment_duration >= min_duration:
            low_volume_segments.append((low_volume_start, times[-1]))
    
    return low_volume_segments

def analyze_audio_file(file_path):
    """
    Analyze an audio file for the filtering criteria.
    
    Args:
        file_path: Path to the audio file
    
    Returns:
        Dictionary with analysis results
    """
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=None)
        duration = len(audio) / sr
        
        # Check duration (must be >= 15 seconds)
        if duration < 15.0:
            return {
                'valid': False,
                'reason': f'Too short: {duration:.2f}s < 15s',
                'duration': duration
            }
        
        # Check for long silence segments (> 2 seconds at -60dB)
        silence_segments = detect_silence_segments(audio, sr, silence_threshold_db=-60, min_silence_duration=2.0)
        if silence_segments:
            return {
                'valid': False,
                'reason': f'Has silence segments > 2s: {len(silence_segments)} segments',
                'duration': duration,
                'silence_segments': silence_segments
            }
        
        # Check for low volume segments (> 1 second at < -60dB)
        low_volume_segments = detect_low_volume_segments(audio, sr, volume_threshold_db=-60, min_duration=1.0)
        if low_volume_segments:
            return {
                'valid': False,
                'reason': f'Has low volume segments > 1s: {len(low_volume_segments)} segments below -60dB',
                'duration': duration,
                'low_volume_segments': low_volume_segments
            }
        
        return {
            'valid': True,
            'reason': 'Passed all criteria',
            'duration': duration
        }
        
    except Exception as e:
        return {
            'valid': False,
            'reason': f'Error loading file: {str(e)}',
            'duration': 0
        }

def filter_audio_files(source_dir, target_dir):
    """
    Filter audio files based on the specified criteria.
    
    Args:
        source_dir: Source directory containing audio files
        target_dir: Target directory for selected files
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Supported audio extensions
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    
    # Find all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(source_path.glob(f'**/*{ext}'))
        audio_files.extend(source_path.glob(f'**/*{ext.upper()}'))
    
    print(f"Found {len(audio_files)} audio files in {source_dir}")
    
    # Statistics
    stats = {
        'total': len(audio_files),
        'selected': 0,
        'rejected_short': 0,
        'rejected_silence': 0,
        'rejected_low_volume': 0,
        'rejected_error': 0
    }
    
    # Process each file
    selected_files = []
    rejected_files = []
    
    for file_path in tqdm(audio_files, desc="Analyzing audio files"):
        result = analyze_audio_file(file_path)
        
        if result['valid']:
            # Copy file to target directory
            relative_path = file_path.relative_to(source_path)
            target_file_path = target_path / relative_path
            
            # Create subdirectories if needed
            target_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(file_path, target_file_path)
            
            selected_files.append({
                'file': str(relative_path),
                'duration': result['duration'],
                'reason': result['reason']
            })
            stats['selected'] += 1
            
        else:
            rejected_files.append({
                'file': str(file_path.relative_to(source_path)),
                'duration': result.get('duration', 0),
                'reason': result['reason']
            })
            
            # Update rejection statistics
            if 'Too short' in result['reason']:
                stats['rejected_short'] += 1
            elif 'silence segments' in result['reason']:
                stats['rejected_silence'] += 1
            elif 'low volume segments' in result['reason']:
                stats['rejected_low_volume'] += 1
            else:
                stats['rejected_error'] += 1
    
    # Print statistics
    print("\n" + "="*50)
    print("FILTERING RESULTS")
    print("="*50)
    print(f"Total files processed: {stats['total']}")
    print(f"Selected files: {stats['selected']}")
    print(f"Rejected - Too short (<15s): {stats['rejected_short']}")
    print(f"Rejected - Long silence (>2s): {stats['rejected_silence']}")
    print(f"Rejected - Low volume segments (>1s at <-60dB): {stats['rejected_low_volume']}")
    print(f"Rejected - Processing errors: {stats['rejected_error']}")
    print(f"\nSelection rate: {stats['selected']/stats['total']*100:.1f}%")
    
    # Save detailed report
    report_file = target_path / 'filtering_report.txt'
    with open(report_file, 'w') as f:
        f.write("AUDIO FILTERING REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("FILTERING CRITERIA:\n")
        f.write("-"*20 + "\n")
        f.write("1. Remove files with silence segments > 2 seconds (at -60dB threshold)\n")
        f.write("2. Remove files shorter than 15 seconds\n")
        f.write("3. Remove files with continuous low volume segments > 1 second (below -60dB)\n\n")
        
        f.write("STATISTICS:\n")
        f.write("-"*20 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nSELECTED FILES:\n")
        f.write("-"*20 + "\n")
        for file_info in selected_files:
            f.write(f"{file_info['file']} ({file_info['duration']:.2f}s) - {file_info['reason']}\n")
        
        f.write("\nREJECTED FILES:\n")
        f.write("-"*20 + "\n")
        for file_info in rejected_files:
            f.write(f"{file_info['file']} ({file_info['duration']:.2f}s) - {file_info['reason']}\n")
    
    print(f"\nDetailed report saved to: {report_file}")
    print(f"Selected files copied to: {target_dir}")

def main():
    """
    Main function to run the audio filtering process.
    """
    # Define source and target directories
    source_dir = '/home/evev/asap-dataset/asap_tempo_sele_100'
    target_dir = '/home/evev/asap-dataset/asap_silent_sele'
    
    print("Audio File Filtering Tool")
    print("="*30)
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    print("\nFiltering criteria:")
    print("1. Remove files with silence segments > 2 seconds (at -60dB)")
    print("2. Remove files shorter than 15 seconds")
    print("3. Remove files with continuous low volume segments > 1 second (below -60dB)")
    print()
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist!")
        return
    
    # Start filtering
    filter_audio_files(source_dir, target_dir)
    
    print("\nFiltering completed successfully!")

if __name__ == "__main__":
    main()