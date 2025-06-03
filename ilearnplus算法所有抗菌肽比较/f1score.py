import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文宋体显示
matplotlib.rcParams['font.sans-serif'] = ['SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

# 特征名称（9种）
features = ['AAC', 'APAAC', 'QSO', 'CKSAAP', 'CTriad', 'PAAC', 'PSEAAC', 'DDE', 'DPC']

# 每种特征提取方法下的测试集 F1 Score
f1_values = [0.8768, 0.8877, 0.8899, 0.8236, 0.7891, 0.8865, 0.8768, 0.8340, 0.8079]

# 设置角度
num_vars = len(features)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合

# 补上首尾以闭合雷达图
stats = f1_values + f1_values[:1]

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

# 绘制雷达图
ax.plot(angles, stats, label='测试集 F1 Score', color='#ED7D31', linewidth=2)
ax.fill(angles, stats, color='#ED7D31', alpha=0.25)

# 设置雷达图样式
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), features, fontsize=10)
ax.set_ylim(0.7, 1.0)  # 可根据分布调整

ax.set_title('不同特征下抗菌肽分类模型的F1 Score表现（雷达图）', fontsize=14)

# 设置图例
ax.legend(loc='lower right', frameon=True, prop={'family': 'SimSun', 'size': 10})

plt.tight_layout()
plt.show()
