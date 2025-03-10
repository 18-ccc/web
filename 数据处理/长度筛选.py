import pandas as pd

# 读取CSV文件
file_path = "D:/bishedata/cdhit的xls格式 changgancdhit.csv"
df = pd.read_csv(file_path)

# 筛选Sequence列中长度在6到46之间的序列
filtered_df = df[df['Sequence'].apply(lambda x: 6 <= len(str(x)) <= 46)]

# 计算被筛选掉的个数
filtered_out_count = len(df) - len(filtered_df)

# 输出被筛选掉的行数
print(f"被筛选掉的行数: {filtered_out_count}")

# 将筛选后的结果保存到新的CSV文件
filtered_df.to_csv("D:/bishedata/changgancdhit_filtered.csv", index=False)
