import pandas as pd

def xls_to_fasta(xls_file, fasta_file):
    """
    将XLS/XLSX文件中的id列和mic列转换为FASTA格式的文件。

    参数:
        xls_file (str): 输入的XLS/XLSX文件路径，包含id和mic列。
        fasta_file (str): 输出的FASTA文件路径。
    """
    # 读取XLS/XLSX文件
    df = pd.read_excel(xls_file)

    # 检查是否包含id和mic列
    if "id" not in df.columns or "mic" not in df.columns:
        raise ValueError("XLS文件必须包含'id'和'mic'两列。")

    # 打开FASTA文件进行写入
    with open(fasta_file, "w") as fasta_handle:
        for index, row in df.iterrows():
            seq_id = row["id"]
            mic_value = row["mic"]

            # 写入FASTA格式的内容
            fasta_handle.write(f">{seq_id}\n")  # 序列标识符
            fasta_handle.write(f"{mic_value}\n")  # 序列内容（MIC值）

    print(f"转换完成！FASTA文件已保存到：{fasta_file}")
    return df  # 返回DataFrame以便后续处理

def fasta_to_csv(fasta_file, csv_file):
    """
    将FASTA文件转换为CSV文件。

    参数:
        fasta_file (str): 输入的FASTA文件路径。
        csv_file (str): 输出的CSV文件路径。
    """
    # 初始化列表存储FASTA内容
    fasta_data = {"id": [], "mic": []}

    # 读取FASTA文件
    with open(fasta_file, "r") as fasta_handle:
        lines = fasta_handle.readlines()

    # 解析FASTA内容
    for i in range(0, len(lines), 2):
        seq_id = lines[i].strip().lstrip(">")
        mic_value = lines[i + 1].strip()
        fasta_data["id"].append(seq_id)
        fasta_data["mic"].append(mic_value)

    # 转换为DataFrame
    fasta_df = pd.DataFrame(fasta_data)

    # 保存为CSV文件
    fasta_df.to_csv(csv_file, index=False)
    print(f"转换完成！CSV文件已保存到：{csv_file}")

# 指定文件路径
xls_file = "D:/bishedate/鲍曼不动杆菌.xlsx"  # 输入的XLS/XLSX文件
fasta_file = "D:/bishedate/鲍曼不动杆菌mic_values.fasta"  # 输出的FASTA文件
csv_file = "D:/bishedate/鲍曼不动杆菌mic_values.csv"  # 输出的CSV文件

# 调用函数进行转换
xls_df = xls_to_fasta(xls_file, fasta_file)  # 从XLS转换为FASTA
fasta_to_csv(fasta_file, csv_file)  # 从FASTA转换为CSV