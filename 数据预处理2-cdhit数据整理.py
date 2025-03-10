import csv


def fasta_to_csv(input_fasta, output_csv):
    """
    将 FASTA 文件中的 ID 和序列提取到 CSV 文件中。
    :param input_fasta: 输入的 FASTA 文件路径
    :param output_csv: 输出的 CSV 文件路径
    """
    with open(input_fasta, 'r') as fasta_file, open(output_csv, 'w', newline='') as csv_file:
        # 创建 CSV 写入器
        csv_writer = csv.writer(csv_file)
        # 写入表头
        csv_writer.writerow(["ID", "Sequence"])

        # 初始化变量
        sequence_id = None
        sequence = []

        # 逐行读取 FASTA 文件
        for line in fasta_file:
            line = line.strip()
            if line.startswith(">"):  # 新序列的起始行
                # 如果之前有序列，先写入 CSV
                if sequence_id:
                    csv_writer.writerow([sequence_id, ''.join(sequence)])
                # 更新新的序列 ID
                sequence_id = line[1:]  # 去掉 '>' 符号
                sequence = []  # 重置序列
            else:
                sequence.append(line)  # 添加序列行

        # 写入最后一个序列
        if sequence_id:
            csv_writer.writerow([sequence_id, ''.join(sequence)])


# 示例用法
input_fasta = "D:/bishedate/tonglvcdhit.fasta"  # 输入的 FASTA 文件
output_csv = "D:/bishedate/outputtonglvcdhit.csv"  # 输出的 CSV 文件
fasta_to_csv(input_fasta, output_csv)