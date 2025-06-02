import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置中文宋体显示
matplotlib.rcParams['font.sans-serif'] = ['SimSun']  # 设置宋体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 原始数据
features = ['AAC', 'APAAC', 'CKSAAP', 'CTriad', 'PAAC', 'PSEAAC', 'QSO']
auc_data = {
    '鲍曼不动杆菌': [0.9566, 0.9620, 0.8313, 0.7443, 0.9644, 0.9566, 0.9450],
    '肠杆菌科': [0.9558, 0.9510, 0.8683, 0.7952, 0.9452, 0.9558, 0.9417],
    '铜绿假单胞菌': [0.9660, 0.9606, 0.8358, 0.8221, 0.9503, 0.9597, 0.9518]
}

# 角度设置
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
for bacteria, values in auc_data.items():
    stats = values + values[:1]  # 闭合
    ax.plot(angles, stats, label=bacteria, color=colors[bacteria], linewidth=2)
    ax.fill(angles, stats, color=colors[bacteria], alpha=0.25)

# 图表设置
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0.7, 1.0)
ax.set_title('不同特征下三种细菌的AUC表现对比（雷达图）', fontsize=14)

# 设置图例（宋体）
ax.legend(loc='lower right', bbox_to_anchor=(1.05, 0.0), frameon=True, prop={'family': 'SimSun', 'size': 10})

plt.tight_layout()
plt.show()

