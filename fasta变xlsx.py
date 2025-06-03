import os
from Bio import SeqIO
import pandas as pd


def fasta_to_excel(fasta_file, output_file):
    """
    将FASTA文件转换为Excel文件。

    参数:
    fasta_file (str): 输入的FASTA文件路径。
    output_file (str): 输出的Excel文件路径。
    """
    # 读取FASTA文件
    records = list(SeqIO.parse(fasta_file, "fasta"))

    # 提取序列信息
    data = []
    for record in records:
        data.append({
            "ID": record.id,
            "Description": record.description,
            "Sequence": str(record.seq)
        })

    # 将数据转换为DataFrame
    df = pd.DataFrame(data)

    # 保存为Excel文件
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"数据已成功保存到 {output_file}")


# 输入文件路径
input_fasta = r"D:\bishedata2\肠杆菌_cdhit.fasta"
# 输出文件路径
output_excel = r"D:\bishedata2\肠杆菌_cdhit.xlsx"

# 调用函数
fasta_to_excel(input_fasta, output_excel)