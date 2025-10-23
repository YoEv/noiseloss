import os
import re

def process_file(input_file, output_file=None):
    """
    处理文本文件，在文件名的 .wav 或 .mp3 扩展名之前添加 _ori 后缀

    参数:
    - input_file: 输入文件路径
    - output_file: 输出文件路径，如果为 None，则直接修改输入文件
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件未找到: {input_file}")

    # 如果没有指定输出文件，则直接修改输入文件
    if output_file is None:
        # 读取整个文件内容
        with open(input_file, 'r') as f:
            lines = f.readlines()

        # 处理每一行并直接写回原文件
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

                # 在 .wav 或 .mp3 扩展名之前添加 _ori
                new_filename = re.sub(r'(.+)(\.(wav|mp3))', r'\1_ori\2', filename)

                # 写入修改后的行
                f.write(f"{new_filename}: {loss_value}\n")
    else:
        # 确保输出文件的目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 处理文件并写入新文件
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

                # 在 .wav 或 .mp3 扩展名之前添加 _ori
                new_filename = re.sub(r'(.+)(\.(wav|mp3))', r'\1_ori\2', filename)

                # 写入修改后的行
                f_out.write(f"{new_filename}: {loss_value}\n")

# 设置输入文件路径
script_dir = os.path.dirname(os.path.abspath(__file__))
input_filename = os.path.join(script_dir, 'loss_Shutter_medium.txt')

print(f"正在处理文件: {input_filename}")

# 直接修改原文件
process_file(input_filename)
print(f"处理完成。文件已更新: {input_filename}")