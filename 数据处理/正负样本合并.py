from Bio import SeqIO

# 定义输入文件路径和输出文件路径
file1 = "D:/bishedata/yinxingyangbencdhit.fasta"
file2 = "D:/bishedata/tonglvcdhit_筛选.fasta"
output_file = "D:/bishedata/tonglvcdhit+阴性样本.fasta"

# 打开输出文件
with open(output_file, "w") as out_handle:
    # 处理第一个文件
    with open(file1, "r") as f1:
        for record in SeqIO.parse(f1, "fasta"):
            # 写入序列编号和氨基酸序列
            out_handle.write(f">{record.id}\n{record.seq}\n")

    # 处理第二个文件
    with open(file2, "r") as f2:
        for record in SeqIO.parse(f2, "fasta"):
            # 写入序列编号和氨基酸序列
            out_handle.write(f">{record.id}\n{record.seq}\n")

print("合并完成！")
