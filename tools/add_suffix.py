import os
import re

def process_file(input_file, output_file=None):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if output_file is None:
        with open(input_file, 'r') as f:
            lines = f.readlines()

        with open(input_file, 'w') as f:
            for line in lines:
                line = line.strip()
                if not line:
                    f.write('\n')
                    continue

                parts = line.split(':')
                if len(parts) != 2:
                    f.write(line + '\n')
                    continue

                filename = parts[0].strip()
                loss_value = parts[1].strip()

                new_filename = re.sub(r'(.+)(\.(wav|mp3))', r'\1_ori\2', filename)

                f.write(f"{new_filename}: {loss_value}\n")
    else:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    f_out.write('\n')
                    continue

                parts = line.split(':')
                if len(parts) != 2:
                    f_out.write(line + '\n')
                    continue

                filename = parts[0].strip()
                loss_value = parts[1].strip()

                new_filename = re.sub(r'(.+)(\.(wav|mp3))', r'\1_ori\2', filename)

                f_out.write(f"{new_filename}: {loss_value}\n")

script_dir = os.path.dirname(os.path.abspath(__file__))
input_filename = os.path.join(script_dir, 'loss_Shutter_medium.txt')

print(f"Processing file: {input_filename}")

process_file(input_filename)
print(f"Completed. File updated: {input_filename}")