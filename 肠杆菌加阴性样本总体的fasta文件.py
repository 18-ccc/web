import pandas as pd

# 读取 Excel 文件
file_path = 'D:/bishedate/changgancdhit+阴性样本.xls'
df = pd.read_excel(file_path, sheet_name=0)  # 假设数据在第一个工作表中

# 假设第一列是序列 ID，第二列是序列
with open('D:/bishedate/changgancdhit+阴性样本.fasta', 'w') as fasta_file:
    for index, row in df.iterrows():
        sequence_id = row[0]  # 第一列是 ID
        sequence = row[11]     # 第二列是序列
        fasta_file.write(f">{sequence_id}\n{sequence}\n")