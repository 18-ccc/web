import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.cm import get_cmap

# 设置中文字体和加粗显示
matplotlib.rcParams['font.sans-serif'] = ['SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.weight'] = 'bold'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.cm import get_cmap

# 设置中文黑体显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 模型性能数据
data = {
    'RF':              [0.8932, 0.8932, 0.8921, 0.9574],
    'LR':              [0.8882, 0.8882, 0.8881, 0.9400],
    'KNN':             [0.8853, 0.8853, 0.8847, 0.9399],
    'CatBoost':        [0.9016, 0.9016, 0.9014, 0.9596],
    'LSTM':            [0.8668, 0.8668, 0.8662, 0.9280],
    'SVM':             [0.9095, 0.9095, 0.9094, 0.9589],
}

# 转为 DataFrame
df = pd.DataFrame.from_dict(data, orient='index', columns=['Accuracy', 'Recall', 'F1 Score', 'AUC'])

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
for i, (model_name, row) in enumerate(df.iterrows()):
    stats = row.tolist() + [row.tolist()[0]]
    ax.plot(angles, stats, label=model_name, linewidth=2, color=color_cycle[i])
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


# 创建 DataFrame
df = pd.DataFrame.from_dict(data, orient='index', columns=['Accuracy', 'Recall', 'F1 Score', 'AUC'])

# 设置角度
labels = df.columns.tolist()
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# 设置颜色（可选 cmap: 'Set2' / 'tab10' / 'Dark2'）
colors = get_cmap('Set2').colors
color_cycle = colors[:len(df)]

# 绘制雷达图
for i, (model_name, row) in enumerate(df.iterrows()):
    stats = row.tolist() + [row.tolist()[0]]
    ax.plot(angles, stats, label=model_name, linewidth=2, color=color_cycle[i])
    ax.fill(angles, stats, alpha=0.25, color=color_cycle[i])

# 设置角度、刻度标签
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=18, fontweight='bold')

# 设置半径范围和标签
ax.set_ylim(0.85, 1.0)
ax.set_yticks([0.85, 0.90, 0.95, 1.00])
ax.set_yticklabels(['0.85', '0.90', '0.95', '1.00'], fontsize=14, fontweight='bold')

# 设置图例
ax.legend(
    loc='center left',
    bbox_to_anchor=(1.2, 0.5),
    frameon=True,
    prop={'family': 'SimSun', 'size': 14, 'weight': 'bold'}
)

# 布局优化
plt.tight_layout()
plt.show()
