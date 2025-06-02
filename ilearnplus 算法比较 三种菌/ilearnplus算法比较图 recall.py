import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文宋体显示（可选，适用于中文环境）
matplotlib.rcParams['font.sans-serif'] = ['SimSun']  # 设置宋体字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 准备数据
features = ['AAC', 'APAAC', 'CKSAAP', 'CTriad', 'PAAC', 'PSEAAC', 'QSO']
recall_data = {
    '鲍曼不动杆菌': [0.9400, 0.9450, 0.8100, 0.7300, 0.9500, 0.9400, 0.9300],
    '肠杆菌科': [0.9380, 0.9350, 0.8500, 0.7800, 0.9300, 0.9380, 0.9200],
    '铜绿假单胞菌': [0.9480, 0.9430, 0.8200, 0.8000, 0.9280, 0.9400, 0.9250]
}

# 雷达图角度设置
labels = features
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

# 自定义颜色
colors = {
    '鲍曼不动杆菌': '#5B9BD5',
    '肠杆菌科': '#ED7D31',
    '铜绿假单胞菌': '#A5A5A5'
}

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

# 绘制每个细菌的雷达图
for bacteria, values in recall_data.items():
    stats = values + values[:1]  # 闭合图形
    ax.plot(angles, stats, label=bacteria, color=colors[bacteria], linewidth=2)
    ax.fill(angles, stats, color=colors[bacteria], alpha=0.25)

# 图表设置
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0.7, 1.0)  # 设置Recall范围
ax.set_title('不同特征下三种细菌的Recall表现对比（雷达图）', fontsize=14)

# 设置图例（宋体）
ax.legend(loc='lower right', bbox_to_anchor=(1.05, 0.0), frameon=True, prop={'family': 'SimSun', 'size': 10})

plt.tight_layout()
plt.show()
