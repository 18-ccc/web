import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文宋体显示
matplotlib.rcParams['font.sans-serif'] = ['SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

# 特征名称
features = ['AAC', 'APAAC', 'CKSAAP', 'CTriad', 'PAAC', 'PSEAAC', 'QSO']

# Accuracy 数据
accuracy_data = {
    '鲍曼不动杆菌': [0.9753, 0.9678, 0.9492, 0.8875, 0.9746, 0.9753, 0.9712],
    '肠杆菌科':   [0.9020, 0.9058, 0.8430, 0.7502, 0.9077, 0.9020, 0.8892],
    '铜绿假单胞菌': [0.9348, 0.9214, 0.8563, 0.7727, 0.9242, 0.9348, 0.9114]
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
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))  # 加大图形尺寸

# 绘制雷达图
for bacteria, values in accuracy_data.items():
    stats = values + values[:1]
    ax.plot(angles, stats, label=bacteria, color=colors[bacteria], linewidth=2)
    ax.fill(angles, stats, color=colors[bacteria], alpha=0.25)

# 设置雷达图样式
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 设置角度标签字体放大加粗
ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=20, fontweight='bold')

# 设置Y轴范围
ax.set_ylim(0.7, 1.0)

# 设置Y轴刻度字体大小
ax.set_yticks([0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
ax.set_yticklabels(['0.75', '0.80', '0.85', '0.90', '0.95', '1.00'], fontsize=16, fontweight='bold')

# 图例远离雷达图并加粗字体
ax.legend(
    loc='center left',
    bbox_to_anchor=(1.00, 0.25),
    frameon=True,
    prop={'family': 'SimSun', 'size': 16, 'weight': 'bold'}
)


plt.tight_layout()
plt.show()
