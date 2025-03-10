from Bio import SeqIO


def filter_fasta(input_fasta, output_fasta):
    """
    从FASTA文件中删除包含非标准氨基酸的序列，并将符合条件的序列保存到新的文件中。
    同时打印出包含非标准氨基酸的序列ID和非标准氨基酸的详细信息。

    参数:
        input_fasta (str): 输入的FASTA文件路径。
        output_fasta (str): 输出的FASTA文件路径，仅包含标准氨基酸的序列。
    """
    # 定义标准氨基酸字符集
    standard_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

    # 初始化计数器
    total_sequences = 0
    non_standard_count = 0

    # 打开输入文件和输出文件
    with open(input_fasta, "r") as input_handle, open(output_fasta, "w") as output_handle:
        # 遍历输入文件中的每个序列
        for record in SeqIO.parse(input_handle, "fasta"):
            total_sequences += 1
            # 检查序列是否只包含标准氨基酸
            if set(record.seq).issubset(standard_amino_acids):
                # 如果是标准氨基酸序列，则写入输出文件
                SeqIO.write(record, output_handle, "fasta")
            else:
                non_standard_count += 1
                # 找出非标准氨基酸
                non_standard_aas = set(record.seq) - standard_amino_acids
                print(f"Sequence {record.id} contains non-standard amino acids ({non_standard_aas}) and was removed.")

    # 打印总结信息
    print(f"Total sequences processed: {total_sequences}")
    print(f"Sequences removed due to non-standard amino acids: {non_standard_count}")
    print(f"Filtered sequences saved to: {output_fasta}")


# 指定输入文件和输出文件路径
input_fasta = "D:/bishedate/baomancdhit.fasta"  # 输入文件路径
output_fasta = "D:/bishedate/标准baomancdhit.fasta"  # 输出文件路径

# 调用函数处理文件
filter_fasta(input_fasta, output_fasta)