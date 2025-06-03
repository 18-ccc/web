from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter
import pandas as pd
import math

# 标准氨基酸
amino_acids = "ACDEFGHIKLMNPQRSTVWY"

# 输入文件路径
file_path = r"D:\bishedata2\正负样本sequence.fasta"

# 保存所有序列特征
features = []

# Shannon 熵计算函数
def shannon_entropy(sequence):
    total_len = len(sequence)
    freq = Counter(sequence)
    entropy = -sum((count / total_len) * math.log2(count / total_len) for count in freq.values())
    return entropy

# 遍历序列
for record in SeqIO.parse(file_path, "fasta"):
    seq = str(record.seq).upper()
    total_len = len(seq)

    # 不再跳过包含非标准氨基酸的序列

    try:
        analysis = ProteinAnalysis(seq)
        physico_features = {
            "Length": total_len,
            "MolecularWeight": analysis.molecular_weight(),
            "IsoelectricPoint": analysis.isoelectric_point(),
            "Aromaticity": analysis.aromaticity(),
            "InstabilityIndex": analysis.instability_index(),
            "Gravy": analysis.gravy()
        }
    except Exception as e:
        # 遇到无法计算的序列，用NaN或默认值填充
        physico_features = {
            "Length": total_len,
            "MolecularWeight": float('nan'),
            "IsoelectricPoint": float('nan'),
            "Aromaticity": float('nan'),
            "InstabilityIndex": float('nan'),
            "Gravy": float('nan')
        }

    # Shannon 熵（序列复杂度）
    entropy = shannon_entropy(seq)
    complexity_feature = {"ShannonEntropy": entropy}

    # 合并所有特征
    all_features = {**physico_features, **complexity_feature}
    all_features["ID"] = record.id
    features.append(all_features)

# 创建 DataFrame
df = pd.DataFrame(features)
df.set_index("ID", inplace=True)

# 保存结果
output_path = "biopython.csv"
df.to_csv(output_path)

print(f"提取完成，保存为 {output_path}")
