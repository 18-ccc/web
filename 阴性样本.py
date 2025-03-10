import pandas as pd

def fasta_to_csv(fasta_file, csv_file):
    """
    将FASTA文件转换为CSV文件。

    参数:
        fasta_file (str): 输入的FASTA文件路径。
        csv_file (str): 输出的CSV文件路径。
    """
    # 初始化列表存储FASTA内容
    fasta_data = {"id": [], "sequence": []}

    # 读取FASTA文件
    with open(fasta_file, "r") as fasta_handle:
        lines = fasta_handle.readlines()

    # 解析FASTA内容
    for i in range(0, len(lines), 2):
        seq_id = lines[i].strip().lstrip(">")
        sequence = lines[i + 1].strip()
        fasta_data["id"].append(seq_id)
        fasta_data["sequence"].append(sequence)

    # 转换为DataFrame
    fasta_df = pd.DataFrame(fasta_data)

    # 保存为CSV文件
    fasta_df.to_csv(csv_file, index=False)
    print(f"转换完成！CSV文件已保存到：{csv_file}")

# 指定文件路径
fasta_file = "D:/bishedate/yinxingyangbencdhit.fasta"  # 输入的FASTA文件
csv_file = "D:/bishedate/yinxingyangbencdhit.csv"  # 输出的CSV文件

# 调用函数进行转换
fasta_to_csv(fasta_file, csv_file)