from Bio import SeqIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 指定多个FASTA文件
fasta_files = {
    "Changgan": r"D:\bishedata\changgancdhit.fasta",
    "Baoman": r"D:\bishedata\baomancdhit.fasta",
    "Tonglv": r"D:\bishedata\tonglvcdhit.fasta"
}

# 存储所有数据
all_seq_lengths = []  # 存储所有文件的序列长度
all_aa_counts = Counter()  # 统计所有文件的氨基酸出现次数

# 分别分析每个文件
for dataset_name, fasta_path in fasta_files.items():
    sequences = [record.seq for record in SeqIO.parse(fasta_path, "fasta")]

    # 计算序列长度
    seq_lengths = [len(seq) for seq in sequences]
    all_seq_lengths.extend(seq_lengths)  # 记录所有长度数据

    # 统计氨基酸组成
    aa_counts = Counter("".join(map(str, sequences)))
    all_aa_counts.update(aa_counts)  # 累计统计所有氨基酸

    # 转换氨基酸统计为 DataFrame
    aa_df = pd.DataFrame.from_dict(aa_counts, orient='index', columns=['Count']).sort_values(by='Count',
                                                                                             ascending=False)

    # 绘制当前文件的序列长度分布
    plt.figure(figsize=(8, 5))
    sns.histplot(seq_lengths, bins=30, kde=True, color="blue")
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.title(f"Sequence Length Distribution ({dataset_name})")
    plt.show()

    # 绘制当前文件的氨基酸组成柱状图
    plt.figure(figsize=(10, 5))
    sns.barplot(x=aa_df.index, y=aa_df["Count"], palette="viridis")
    plt.xlabel("Amino Acid")
    plt.ylabel("Count")
    plt.title(f"Amino Acid Composition ({dataset_name})")
    plt.show()

# -------- 合并分析 --------
# 绘制整体序列长度分布
plt.figure(figsize=(8, 5))
sns.histplot(all_seq_lengths, bins=30, kde=True, color="red")
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.title("Overall Sequence Length Distribution")
plt.show()

# 绘制整体氨基酸组成
all_aa_df = pd.DataFrame.from_dict(all_aa_counts, orient='index', columns=['Count']).sort_values(by='Count',
                                                                                                 ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=all_aa_df.index, y=all_aa_df["Count"], palette="magma")
plt.xlabel("Amino Acid")
plt.ylabel("Count")
plt.title("Overall Amino Acid Composition")
plt.show()
