import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

# 设置中文字体和负号显示
matplotlib.rcParams['font.sans-serif'] = ['SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

# 指标和角度
metrics = ['Accuracy', 'Recall', 'F1 Score', 'AUC']
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

# 模型数据
baumannii = {
    'ESM2+Transformer': [0.9794, 0.7143, 0.7500, 0.9913],
    'ESM2+GCN': [0.9962, 0.9978, 0.9962, 0.9999]
}
pseudomonas = {
    'ESM2+Transformer': [0.9404, 0.8314, 0.8594, 0.9572],
    'ESM2+GCN': [0.9667, 0.9602, 0.9665, 0.9923]
}
enterobacter = {
    'ESM2+Transformer': [0.9290, 0.8688, 0.8908, 0.9606],
    'ESM2+GCN': [0.9414, 0.9290, 0.9407, 0.9819]
}

# 颜色
colors = {
    'ESM2+Transformer': '#ED7D31',
    'ESM2+GCN': '#5B9BD5',
}

# 输出目录
output_dir = r'D:\bishedata2\雷达图_GCN对比Transformer'
os.makedirs(output_dir, exist_ok=True)

# 绘图函数
def plot_radar(data, title, filename):
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=26, fontweight='bold', fontname='SimSun')
    ax.set_ylim(0.70, 1.00)
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_yticklabels(['0.75', '0.80', '0.85', '0.90', '0.95', '1.00'], fontsize=18, fontweight='bold', fontname='SimSun')

    for model, scores in data.items():
        stats = scores + scores[:1]
        ax.plot(angles, stats, label=model, color=colors[model], linewidth=3)
        ax.fill(angles, stats, color=colors[model], alpha=0.25)

    ax.set_title(title, size=26, weight='bold', loc='left', pad=40, fontname='SimSun')
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.15, 0.25),
        prop={'family': 'SimSun', 'size': 20, 'weight': 'bold'},
        borderaxespad=0.3,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# 绘图
plot_radar(baumannii, 'A. 鲍曼不动杆菌：GCN优于Transformer', 'A_鲍曼不动杆菌_对比.png')
plot_radar(pseudomonas, 'B. 铜绿假单胞菌：GCN优于Transformer', 'B_铜绿假单胞菌_对比.png')
plot_radar(enterobacter, 'C. 肠杆菌科：GCN优于Transformer', 'C_肠杆菌科_对比.png')
