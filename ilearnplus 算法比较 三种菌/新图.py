import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.cm import get_cmap

# 设置中文黑体显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# 定义绘制雷达图的函数
def plot_radar(data_dict, title):
    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['Accuracy', 'Recall', 'F1 Score', 'AUC'])

    labels = df.columns.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

    colors = get_cmap('Set2').colors
    color_cycle = colors[:len(df)]

    for i, (feature_name, row) in enumerate(df.iterrows()):
        stats = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, stats, label=feature_name, linewidth=2, color=color_cycle[i])
        ax.fill(angles, stats, alpha=0.25, color=color_cycle[i])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=24, fontweight='bold')
    ax.set_ylim(0.7, 1.0)
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    ax.set_yticklabels(['0.75', '0.80', '0.85', '0.90', '0.95', '1.00'], fontsize=24, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold', y=1.1)

    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.1, 0.2),
        frameon=True,
        prop={'family': 'SimHei', 'size': 20, 'weight': 'bold'}
    )

    plt.tight_layout()
    plt.show()


# 鲍曼不动杆菌数据
bowman_data = {
    'AAC': [0.9753, 0.9753, 0.9733, 0.9566],
    'APAAC': [0.9678, 0.9678, 0.966, 0.962],
    'CKSAAP': [0.9492, 0.9492, 0.9529, 0.8313],
    'CTriad': [0.8875, 0.8875, 0.9098, 0.802],
    'ESM2+图结构+GCN': [0.9667, 0.9602, 0.9665, 0.9923]
}

# 肠杆菌数据
enterobacter_data = {
    'AAC': [0.9020, 0.9020, 0.9019, 0.9558],
    'APAAC': [0.9058, 0.9058, 0.9055, 0.9510],
    'CKSAAP': [0.8430, 0.8430, 0.8426, 0.8683],
    'CTriad': [0.7502, 0.7502, 0.7539, 0.8049],
    'ESM2+图结构+GCN': [0.9667, 0.9602, 0.9665, 0.9923]
}

# 铜绿假单胞菌数据
pseudomonas_data = {
    'AAC': [0.9348, 0.9348, 0.9353, 0.9660],
    'APAAC': [0.9214, 0.9214, 0.9218, 0.9606],
    'CKSAAP': [0.8563, 0.8563, 0.8555, 0.8358],
    'CTriad': [0.7727, 0.7727, 0.7841, 0.8199],
    'ESM2+图结构+GCN': [0.9667, 0.9602, 0.9665, 0.9923]
}

# 绘制三张雷达图
plot_radar(bowman_data, "鲍曼不动杆菌不同特征性能比较")
plot_radar(enterobacter_data, "肠杆菌不同特征性能比较")
plot_radar(pseudomonas_data, "铜绿假单胞菌不同特征性能比较")
