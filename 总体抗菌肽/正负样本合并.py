# 定义两个输入 FASTA 文件路径
file1 = r'D:\bishedata2\grampacdhit.fasta'
file2 = r'D:\bishedata2\yinxingyangbencdhit.fasta'

# 定义合并后的输出路径
output_file = r'D:\bishedata2\combined_sequences.fasta'

# 读取两个文件内容并写入合并文件
with open(output_file, 'w') as outfile:
    for fasta_file in [file1, file2]:
        with open(fasta_file, 'r') as infile:
            outfile.write(infile.read())
            outfile.write('\n')  # 添加换行以确保格式整洁

print("合并完成，文件已保存为 D:\\bishedata2\\总体正负样本合并sequences.fasta")
