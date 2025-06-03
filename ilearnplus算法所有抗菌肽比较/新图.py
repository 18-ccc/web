import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.cm import get_cmap

# 设置中文黑体显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 特征名称与指标数据
metrics = {
    'AAC':      [0.8689, 0.8689, 0.8685, 0.9335],
    'APAAC':    [0.9054, 0.9054, 0.9052, 0.9579],
    'CKSAAP':   [0.8941, 0.8941, 0.8929, 0.9496],
    'CTriad':   [0.8375, 0.8375, 0.8370, 0.9040],
    'PAAC':     [0.9008, 0.9008, 0.9006, 0.9575],
    'PseAAC':   [0.8689, 0.8689, 0.8685, 0.9335],
    'QSOrder':  [0.9003, 0.9003, 0.9002, 0.9535]
}

# 转为 DataFrame
df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Accuracy', 'Recall', 'F1 Score', 'AUC'])

# 设置角度
labels = df.columns.tolist()
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合

# 创建图形
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# 设置颜色（柔和+区分度）
colors = get_cmap('Set2').colors  # 可选 'tab10', 'Pastel1', 'Set2'
color_cycle = colors[:len(df)]

# 绘制雷达图
for i, (feature_name, row) in enumerate(df.iterrows()):
    stats = row.tolist() + [row.tolist()[0]]
    ax.plot(angles, stats, label=feature_name, linewidth=2, color=color_cycle[i])
    ax.fill(angles, stats, alpha=0.25, color=color_cycle[i])

# 样式调整
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=20, fontweight='bold')
ax.set_ylim(0.8, 1.0)
ax.set_yticks([0.85, 0.9, 0.95, 1.0])
ax.set_yticklabels(['0.85', '0.90', '0.95', '1.00'], fontsize=16, fontweight='bold')

# 图例设置（位置下移且左移，字体为黑体）
ax.legend(
    loc='center left',
    bbox_to_anchor=(1.1, 0.2),  # 左移一点，下移一点
    frameon=True,
    prop={'family': 'SimHei', 'size': 14, 'weight': 'bold'}
)

plt.tight_layout()
plt.show()
