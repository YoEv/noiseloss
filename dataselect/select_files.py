import os
import shutil
import re
from pathlib import Path

def find_and_copy_selected_rhythm_files(source_dir, target_dir, patterns):
	Path(target_dir).mkdir(parents=True, exist_ok=True)

	pattern_str = "|".join([re.escape(p) for p in patterns])
	regex = re.compile(f'.*({pattern_str})$')

	file_count = 0
	for root, dirs, files in os.walk(source_dir):
		for file in files:
			if regex.match(file):
				src_path = os.path.join(root, file)
				dst_path = os.path.join(target_dir, file)

				counter = 1
				while os.path.exists(dst_path):
					name, ext = os.path.splitext(file)
					dst_path = os.path.join(target_dir, f"{name}_{counter}{ext}")
					counter += 1
				shutil.copy2(src_path, dst_path)
				file_count += 1

	print(f"\n copied! {file_count} files to {target_dir}")

if __name__ == "__main__":
	source_directory = os.path.join("selected_midis_cropped") # "experiment_results_rhythm_test4_loss_report","rhythm_processed"
	target_directory = os.path.join("selected_rhythm_files")

	target_patterns = ["midi_score.mid"]
		# "rhythm1_4_step2_r40.mid",
		# "rhythm1_4_step2_r30.mid",
		# "rhythm1_4_step1_r40.mid",
		# "rhythm1_4_step1_r30.mid"
	if not os.path.exists(source_directory):
		print(f"wornings: wrong source directory {source_directory}")

	find_and_copy_selected_rhythm_files(source_directory, target_directory, target_patterns)
