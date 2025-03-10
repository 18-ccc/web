from Bio import SeqIO

# 输入和输出文件路径
input_file = r'D:\bishedata\tonglvcdhit.fasta'
output_file = r'D:\bishedata\tonglvcdhit_筛选.fasta'

# 筛选条件：序列长度在7到39之间
min_length = 7
max_length = 43

# 读取原始 FASTA 文件并筛选
selected_records = []
total_records = 0  # 总的序列数量
discarded_records = 0  # 筛选掉的序列数量

# 统计总序列数量，并筛选符合条件的序列
for record in SeqIO.parse(input_file, "fasta"):
    total_records += 1
    if min_length <= len(record.seq) <= max_length:
        selected_records.append(record)
    else:
        discarded_records += 1

# 将筛选后的序列写入新的 FASTA 文件
SeqIO.write(selected_records, output_file, "fasta")

# 显示筛选信息
print(f"筛选出 {len(selected_records)} 条氨基酸序列。")
print(f"筛选掉 {discarded_records} 条氨基酸序列。")
print(f"剩余 {len(selected_records)} 条氨基酸序列。")
print(f"原始文件中总共有 {total_records} 条氨基酸序列。")
