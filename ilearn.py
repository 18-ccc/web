import pandas as pd
import numpy as np

# 导入数据
file_path = r'D:\HuaweiMoveData\Users\陈雯静.LAPTOP-CJOIH1UC\Desktop\毕设\阳性样本.xls'
data = pd.read_excel(file_path)

# 查看数据的前几行
print(data.head())

# 计算肽段长度并添加新列
data['peptide_length'] = data['sequence'].apply(len)

# 删除长度超过50或小于5的肽段
cleaned_data = data[(data['peptide_length'] <= 50) & (data['peptide_length'] >= 5)]

# 定义几何平均函数
def geometric_mean(series):
    return np.exp(np.log(series).mean())

# 计算相同 sequence 的 mic 值的几何平均
final_results = cleaned_data.groupby('sequence')['mic'].apply(geometric_mean).reset_index()

# 添加 log10 运算
final_results['Log10_Geometric_Mean_MIC'] = np.log10(final_results['mic'])

# 重命名列
final_results.columns = ['sequence', 'Geometric_Mean_MIC', 'Log10_Geometric_Mean_MIC']

# 查看最终结果
print(final_results)

# 可选择将结果保存为新的 Excel 文件
output_path = r'D:\HuaweiMoveData\Users\陈雯静.LAPTOP-CJOIH1UC\Desktop\毕设\清洗后的阳性样本.xlsx'
final_results.to_excel(output_path, index=False)
