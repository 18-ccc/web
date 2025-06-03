import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

# 中文字体设置，加载宋体
font_path = 'C:/Windows/Fonts/simsun.ttc'
font_prop = fm.FontProperties(fname=font_path)

data = {
    '特征组合': ['apaac+paac+qso+aac'] * 7,
    '特征数量': [50, 60, 70, 80, 90, 100, 110],
    'Accuracy': [0.9028, 0.9012, 0.9012, 0.9049, 0.9095, 0.9100, 0.9095],
    'Recall':   [0.9028, 0.9012, 0.9012, 0.9049, 0.9095, 0.9100, 0.9095],
    'F1 Score': [0.9027, 0.9009, 0.9009, 0.9047, 0.9094, 0.9097, 0.9094],
    'AUC':      [0.9562, 0.9568, 0.9579, 0.9589, 0.9588, 0.9592, 0.9589]
}

df = pd.DataFrame(data)

def get_pivot(df, metric):
    return df.pivot(index='特征组合', columns='特征数量', values=metric)

metrics = ['Accuracy', 'Recall', 'F1 Score', 'AUC']
labels = ['A', 'B', 'C', 'D']

# 自定义颜色渐变，从白色到黄色到红色
colors = ["#ffffff", "#ffff00", "#ff0000"]
custom_cmap = LinearSegmentedColormap.from_list("white_yellow_red", colors)

fig, axes = plt.subplots(2, 2, figsize=(24, 18))

# 为每个指标设定合适的vmin和vmax，参考你的数据范围调整
vmins = [0.89, 0.89, 0.89, 0.95]
vmaxs = [0.915, 0.915, 0.915, 0.96]

for ax, metric, label, vmin, vmax in zip(axes.flat, metrics, labels, vmins, vmaxs):
    pivot = get_pivot(df, metric)
    heatmap = sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap=custom_cmap,
        ax=ax,
        linewidths=0.2,
        linecolor='#cccccc',
        square=True,
        cbar_kws={'label': metric}
    )
    ax.set_xlabel('特征数量', fontproperties=font_prop, fontsize=20, fontweight='bold')
    ax.set_ylabel('特征组合', fontproperties=font_prop, fontsize=20, fontweight='bold')

    ax.set_xticklabels(
        ax.get_xticklabels(),
        fontproperties=font_prop,
        fontsize=18,
        fontweight='bold',
        rotation=0
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        fontproperties=font_prop,
        fontsize=18,
        fontweight='bold',
        rotation=0
    )
    ax.tick_params(axis='x', pad=10)
    ax.tick_params(axis='y', pad=10)

    # 设置 colorbar 字体
    cbar = heatmap.collections[0].colorbar
    cbar.set_label(metric, fontsize=20, fontweight='bold', family=font_prop.get_name())
    cbar.ax.tick_params(labelsize=16)
    for label_cbar in cbar.ax.get_yticklabels():
        label_cbar.set_fontweight('bold')
        label_cbar.set_fontname(font_prop.get_name())

    # 添加子图标签 A, B, C, D
    ax.text(-0.12, 1.08, label, transform=ax.transAxes,
            fontsize=30, fontweight='bold', va='top', ha='left')

plt.tight_layout()
plt.show()
