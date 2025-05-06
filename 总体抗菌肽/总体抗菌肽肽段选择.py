import pandas as pd

# 读取 Excel 文件
file_path = r'D:\bishedata2\grampa.xls'
df = pd.read_excel(file_path)

# 计算 sequence 长度并筛选：长度 >10 且 <100
df['seq_len'] = df['sequence'].astype(str).apply(len)
filtered_df = df[(df['seq_len'] > 10) & (df['seq_len'] < 100)]

# 打印筛选结果数量和预览
print(f"筛选后剩余 {len(filtered_df)} 条记录")
print(filtered_df[['sequence', 'seq_len']].head())

# 保存为 Excel 文件（可选）
filtered_df.to_excel(r'D:\bishedata2\grampa_filtered.xlsx', index=False)

# ----------------- 保存为 FASTA 文件 -----------------
# 创建 FASTA 格式文本
fasta_lines = []
for i, row in filtered_df.iterrows():
    identifier = f"peptide_{i}"  # 或使用其他唯一列：row['url_source'] 等
    sequence = row['sequence']
    fasta_lines.append(f">{identifier}")
    fasta_lines.append(sequence)

# 写入 .fasta 文件
with open(r'D:\bishedata2\grampa_filtered.fasta', 'w') as fasta_file:
    fasta_file.write('\n'.join(fasta_lines))

print("FASTA 文件已保存到 D:\\bishedata2\\grampa_filtered.fasta")
