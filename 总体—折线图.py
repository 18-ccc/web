import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 中文字体设置
font_path = 'C:/Windows/Fonts/simsun.ttc'
font_prop = fm.FontProperties(fname=font_path)

# 你的数据
data = {
    '特征数量': [50, 60, 70, 80, 90, 100, 110],
    'Accuracy': [0.9028, 0.9012, 0.9012, 0.9049, 0.9095, 0.9100, 0.9095],
    'Recall': [0.9028, 0.9012, 0.9012, 0.9049, 0.9095, 0.9100, 0.9095],
    'F1 Score': [0.9027, 0.9009, 0.9009, 0.9047, 0.9094, 0.9097, 0.9094],
    'AUC': [0.9562, 0.9568, 0.9579, 0.9589, 0.9588, 0.9592, 0.9589]
}

df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))

# 画折线图
for metric in ['Accuracy', 'Recall', 'F1 Score', 'AUC']:
    plt.plot(df['特征数量'], df[metric], marker='o', label=metric)

# 删除标题，注释或删除下一行即可
# plt.title('指标随特征数量变化趋势', fontproperties=font_prop, fontsize=16, fontweight='bold')

plt.xlabel('特征数量', fontproperties=font_prop, fontsize=14)
plt.ylabel('指标值', fontproperties=font_prop, fontsize=14)
plt.xticks(df['特征数量'])
plt.ylim(0.88, 1.0)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(prop=font_prop, fontsize=12)

plt.tight_layout()
plt.savefig("D:/bishedata2/拼接图/指标变化折线图.png")
plt.show()
