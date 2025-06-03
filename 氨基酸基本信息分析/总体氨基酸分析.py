import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from collections import Counter
import matplotlib
from matplotlib.font_manager import FontProperties

# ========== 0. 全局设置 ==========

# 设置宋体（SimSun）为全局中文字体
font_path = r"C:\Windows\Fonts\simsun.ttc"  # 请确保路径正确
font_prop = FontProperties(fname=font_path, size=12)

matplotlib.rcParams['font.family'] = font_prop.get_name()
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12

# seaborn 样式美化
sns.set(style="whitegrid")

# 保存图片路径
save_dir = r"D:\bishedata2\正负样本特点可视化文件夹"
os.makedirs(save_dir, exist_ok=True)

# ========== 1. 定义输入文件 ==========

pos_files = [r"D:\bishedata2\grampacdhit.fasta"]  # 正样本
neg_file = r"D:\bishedata2\yinxingyangbencdhit.fasta"  # 负样本
aa_list = list('ACDEFGHIKLMNPQRSTVWY')  # 20种氨基酸

# ========== 2. 读取序列并统计 ==========

def read_sequences(file_list):
    lengths = []
    aa_counter = Counter()
    for file in file_list:
        for record in SeqIO.parse(file, "fasta"):
            seq = str(record.seq).upper()
            lengths.append(len(seq))
            aa_counter.update(seq)
    return lengths, aa_counter

pos_lengths, pos_aa_counts = read_sequences(pos_files)
neg_lengths, neg_aa_counts = read_sequences([neg_file])

print(f"正样本数量: {len(pos_lengths)}")
print(f"负样本数量: {len(neg_lengths)}")

# ========== 3. 样本长度分布绘图 ==========

plt.figure(figsize=(8, 6))
sns.kdeplot(pos_lengths, label="正样本", fill=True, color="#5B9BD5", linewidth=2)
sns.kdeplot(neg_lengths, label="负样本", fill=True, color="#ED7D31", linewidth=2)

plt.xlabel("序列长度", fontproperties=font_prop)
plt.ylabel("密度", fontproperties=font_prop)
plt.legend(prop=font_prop)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "图3-1_正负样本序列长度分布.png"), dpi=600, transparent=True)
plt.show()

# ========== 4. 氨基酸组成频率分布 ==========

# 计算频率
pos_total = sum(pos_aa_counts[aa] for aa in aa_list)
neg_total = sum(neg_aa_counts[aa] for aa in aa_list)

pos_freq = {aa: pos_aa_counts.get(aa, 0) / pos_total for aa in aa_list}
neg_freq = {aa: neg_aa_counts.get(aa, 0) / neg_total for aa in aa_list}

# 构建 DataFrame
df = pd.DataFrame({
    "Amino Acid": aa_list,
    "正样本": [pos_freq[aa] for aa in aa_list],
    "负样本": [neg_freq[aa] for aa in aa_list]
})

df_melt = df.melt(id_vars="Amino Acid", var_name="样本类型", value_name="频率")

plt.figure(figsize=(10, 6))
sns.barplot(data=df_melt, x="Amino Acid", y="频率", hue="样本类型", palette=["#5B9BD5", "#ED7D31"])

plt.xlabel("氨基酸", fontproperties=font_prop)
plt.ylabel("频率", fontproperties=font_prop)
plt.legend(title="样本类型", prop=font_prop, title_fontproperties=font_prop)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "图3-2_正负样本氨基酸组成频率分布.png"), dpi=600, transparent=True)
plt.show()

# ========== 5. 正负样本数量比例饼图 ==========

labels = ['正样本', '负样本']
sizes = [len(pos_lengths), len(neg_lengths)]
colors = ['#5B9BD5', '#ED7D31']

plt.figure(figsize=(6, 6))
plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    colors=colors,
    startangle=140,
    textprops={'fontproperties': font_prop}
)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "图3-3_正负样本数量比例分布.png"), dpi=600, transparent=True)
plt.show()
