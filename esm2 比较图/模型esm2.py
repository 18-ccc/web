import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.cm import get_cmap

# 设置中文黑体显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.weight'] = 'bold'

def plot_radar(df, title):
    labels = df.columns.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    colors = get_cmap('Set2').colors
    color_cycle = colors[:len(df)]

    # 自动确定半径范围
    all_values = df.values.flatten()
    min_val = max(0.0, np.floor((all_values.min() - 0.02) * 100) / 100)
    max_val = min(1.0, np.ceil((all_values.max() + 0.02) * 100) / 100)

    for i, (method, row) in enumerate(df.iterrows()):
        stats = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, stats, label=method, linewidth=2, color=color_cycle[i])
        ax.fill(angles, stats, alpha=0.25, color=color_cycle[i])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=24, fontweight='bold')

    # 自动设置半径范围
    ax.set_ylim(min_val, max_val)
    yticks = np.linspace(min_val, max_val, 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{tick:.2f}" for tick in yticks], fontsize=26, fontweight='bold')

    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.0, 0.2),
        frameon=True,
        prop={'family': 'SimHei', 'size': 20, 'weight': 'bold'}
    )

    plt.title(title, fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.show()

# 三种菌的 ESM2 模型性能数据
data_dict = {
    '肠杆菌': {
        'ESM2-Model1': [0.9247,0.8645,0.8844,0.9594],
        'ESM2-Model2': [0.9962,0.9978,0.9962,0.9999]
    },
    '鲍曼不动杆菌': {
        'ESM2-Model1': [0.9773,0.6190,0.7027,0.9893],
        'ESM2-Model2': [0.9274,0.9108,0.9262,0.9748]
    },
    '铜绿假单胞菌': {
        'ESM2-Model1': [0.942,0.8352,0.8633,0.9568],
        'ESM2-Model2': [0.9667,0.9602,0.9665,0.9893]
    }
}

columns = ['Accuracy', 'Recall', 'F1 Score', 'AUC']

# 逐个绘制雷达图
for bacteria, methods_data in data_dict.items():
    df = pd.DataFrame(methods_data, index=columns).T
    plot_radar(df, f"{bacteria}：ESM2 两种模型性能对比")
