import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体显示
matplotlib.rcParams['font.sans-serif'] = ['SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

# 特征名称（9种）
features = ['AAC', 'APAAC', 'QSO', 'CKSAAP', 'CTriad', 'PAAC', 'PSEAAC', 'DDE', 'DPC']

# 每种特征提取方法下的测试集 AUC
auc_values = [0.9470, 0.9525, 0.9519, 0.8899, 0.8635, 0.9520, 0.9470, 0.9046, 0.8836]

# 设置角度
num_vars = len(features)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

# AUC 值闭合
stats = auc_values + auc_values[:1]

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

# 绘制雷达图
ax.plot(angles, stats, label='测试集 AUC', color='#5B9BD5', linewidth=2)
ax.fill(angles, stats, color='#5B9BD5', alpha=0.25)

# 设置样式
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), features, fontsize=10)
ax.set_ylim(0.85, 1.0)  # 可根据 AUC 分布微调

ax.set_title('不同特征下抗菌肽分类模型的AUC表现（雷达图）', fontsize=14)

# 图例设置
ax.legend(loc='lower right', frameon=True, prop={'family': 'SimSun', 'size': 10})

plt.tight_layout()
plt.show()
