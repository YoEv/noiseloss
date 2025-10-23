import os
import glob
import re
import argparse

def process_file(input_file, output_file):
    """Process a single file to extract MIDI filename and overall_mse"""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # 匹配所有处理类型（pitch/rhythm等）的文件路径
            match = re.search(
                r'.*_processed/.*?/(.*?\.mid):.*?overall_mse: (\d+\.\d+)',
                line.strip()
            )
            if match:
                midi_name = match.group(1)
                overall_mse = match.group(2)
                f_out.write(f"{midi_name}: {overall_mse}\n")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='提取MIDI文件名和MSE值')
    parser.add_argument('--input', required=True,
                       help='输入文件或目录路径（支持通配符如*.txt）')
    parser.add_argument('--output_dir', default='simplified_results',
                       help='输出目录路径（默认为./simplified_results）')
    args = parser.parse_args()

    # 获取输入文件列表
    if os.path.isfile(args.input):
        input_files = [args.input]
    else:
        input_files = glob.glob(args.input)

    if not input_files:
        print(f"错误：未找到输入文件 {args.input}")
        return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"将处理 {len(input_files)} 个文件...")

    # 处理每个文件
    for input_file in input_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(args.output_dir, f"loss_cal_{filename}")

        try:
            process_file(input_file, output_file)
            print(f"成功处理：{input_file} -> {output_file}")
        except Exception as e:
            print(f"处理失败 {input_file}: {str(e)}")

    print(f"\n处理完成！结果保存在：{os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()