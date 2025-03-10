def clean_fasta(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        write_line = True  # 仅写入非数字行
        for line in infile:
            if line.startswith('>'):
                write_line = True  # 标识符行保留
                outfile.write(line)
            elif line.strip().isalpha():  # 只保留纯氨基酸序列（A-Z字符）
                outfile.write(line)
            else:
                write_line = False  # 跳过包含数字的行

# 用法示例
input_fasta = "D:/bishedata/数据2.fasta"  # 替换为你的输入文件
output_fasta = "D:/bishedata/数据.fasta"  # 处理后的输出文件
clean_fasta(input_fasta, output_fasta)