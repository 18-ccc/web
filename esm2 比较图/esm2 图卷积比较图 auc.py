import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文宋体显示
matplotlib.rcParams['font.sans-serif'] = ['SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

# 特征名称
features = ['ESM2', 'GCN', 'ESM2+GCN']

# AUC 数据
auc_data = {
    '鲍曼不动杆菌': [0.9872, 0.9999, 0.9999],
    '肠杆菌科': [0.9722, 0.9999, 0.9819],
    '铜绿假单胞菌': [0.9922, 0.9999, 0.9923]
}

# 设置角度
labels = features
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合

# 颜色定义
colors = {
    '鲍曼不动杆菌': '#5B9BD5',
    '肠杆菌科': '#ED7D31',
    '铜绿假单胞菌': '#A5A5A5'
}

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

# 绘制雷达图
for bacteria, values in auc_data.items():
    stats = values + values[:1]
    ax.plot(angles, stats, label=bacteria, color=colors[bacteria], linewidth=2)
    ax.fill(angles, stats, color=colors[bacteria], alpha=0.25)

# 设置雷达图样式
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0.9, 1.0)  # 根据具体数值微调

ax.set_title('不同特征下三种细菌的AUC表现对比（雷达图）', fontsize=14)

# 设置图例
ax.legend(loc='lower right', bbox_to_anchor=(1.05, 0.0), frameon=True, prop={'family': 'SimSun', 'size': 10})

plt.tight_layout()
plt.show()
