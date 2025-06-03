import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

# 设置中文宋体显示
matplotlib.rcParams['font.sans-serif'] = ['SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

# 指标和角度设置
metrics = ['Accuracy', 'Recall', 'F1 Score', 'AUC']
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # 闭合圆

# 模型数据（新数据）
baumannii = {
    'GCN特征': [0.9957, 1.0000, 0.9957, 0.9999],
    '拼接特征': [0.9962, 0.9978, 0.9962, 0.9999]
}
pseudomonas = {
    'GCN特征': [0.9565, 0.9484, 0.9561, 0.9872],
    '拼接特征': [0.9667, 0.9602, 0.9665, 0.9923]
}
enterobacter = {
    'GCN特征': [0.9204, 0.9043, 0.9191, 0.9722],
    '拼接特征': [0.9414, 0.9290, 0.9407, 0.9819]
}

# 颜色定义
colors = {
    'GCN特征': '#5B9BD5',
    '拼接特征': '#ED7D31',
}

# 输出目录
output_dir = r'D:\bishedata2\拼接图'
os.makedirs(output_dir, exist_ok=True)

# 雷达图绘图函数
def plot_radar(data, title, filename):
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=26, fontweight='bold', fontname='SimSun')
    ax.set_ylim(0.90, 1.00)  # 设置合理的上下限
    ax.set_yticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
    ax.set_yticklabels(['0.90', '0.92', '0.94', '0.96', '0.98', '1.00'], fontsize=18, fontweight='bold', fontname='SimSun')

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
plot_radar(baumannii, 'A. 鲍曼不动杆菌', 'A_鲍曼不动杆菌_新数据.png')
plot_radar(enterobacter, 'B. 肠杆菌科', 'B_肠杆菌科_新数据.png')
plot_radar(pseudomonas, 'C. 铜绿假单胞菌', 'C_铜绿假单胞菌_新数据.png')
