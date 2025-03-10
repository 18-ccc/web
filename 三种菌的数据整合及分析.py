import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# 读取Excel文件
file_path = r'D:\HuaweiMoveData\Users\陈雯静.LAPTOP-CJOIH1UC\Desktop\毕设\数据\肠杆菌.xls'
df = pd.read_excel(file_path)

# 计算氨基酸序列的长度
df['length'] = df['sequence'].apply(len)

# 筛选出长度在 7 到 40 之间的氨基酸序列
filtered_df = df[(df['length'] >= 6) & (df['length'] <= 46)]

# 统计化学修饰的数量
all_modifications = []
for mods in filtered_df['modifications']:
    if isinstance(mods, list):
        all_modifications.extend(mods)
    else:
        # 如果不是列表，直接添加
        all_modifications.append(mods)

modification_counts = Counter(all_modifications)

# 找出排名前五的修饰
top_five_modifications = modification_counts.most_common(5)

# 打印结果
print("筛选后的氨基酸长度分布：")
print(filtered_df['length'].describe())

print("\n化学修饰统计：")
for mod, count in modification_counts.items():
    print(f"{mod}: {count}")

print("\n排名前五的修饰：")
for mod, count in top_five_modifications:
    print(f"{mod}: {count}")

# 绘制多肽序列长度的直方图
plt.figure(figsize=(10, 6))
plt.hist(filtered_df['length'], bins=20, color='blue', alpha=0.7, edgecolor='black')
plt.title('Distribution of Peptide Lengths (Filtered)')  # 标题更新为已筛选数据
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.grid(True)

# 计算均值、标准差以及95%分位数范围
mean = np.mean(filtered_df['length'])
std_dev = np.std(filtered_df['length'])
lower_percentile = np.percentile(filtered_df['length'], 2.5)  # 2.5% 分位数
upper_percentile = np.percentile(filtered_df['length'], 97.5) # 97.5% 分位数

# 在图上标注均值、标准差和95%范围
plt.axvline(x=mean, color='red', linestyle='dashed', linewidth=1)
plt.axvline(x=mean - std_dev, color='green', linestyle='dashed', linewidth=1)
plt.axvline(x=mean + std_dev, color='green', linestyle='dashed', linewidth=1)

# 95% 范围标注
plt.axvline(x=lower_percentile, color='purple', linestyle='dashed', linewidth=1)
plt.axvline(x=upper_percentile, color='purple', linestyle='dashed', linewidth=1)

# 显示标注文本
plt.text(mean, plt.ylim()[1]*0.9, f'Mean: {mean:.2f}', color='red')
plt.text(mean - std_dev, plt.ylim()[1]*0.8, f'Std Dev: {std_dev:.2f}', color='green')
plt.text(lower_percentile, plt.ylim()[1]*0.7, f'2.5%: {lower_percentile:.2f}', color='purple')
plt.text(upper_percentile, plt.ylim()[1]*0.7, f'97.5%: {upper_percentile:.2f}', color='purple')

# 输出 95% 范围的氨基酸长度
print(f"\n95% 范围的氨基酸序列长度：{lower_percentile:.2f} - {upper_percentile:.2f}")

# 保存筛选后的数据到新的 Excel 文件
output_file = r'D:\HuaweiMoveData\Users\陈雯静.LAPTOP-CJOIH1UC\Desktop\毕设\数据\肠杆菌1.xlsx'
filtered_df.to_excel(output_file, index=False)

# 显示图表
plt.show()
