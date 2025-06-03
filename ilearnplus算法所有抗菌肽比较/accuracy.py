import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文宋体显示
matplotlib.rcParams['font.sans-serif'] = ['SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

# 特征名称（9种）
features = ['AAC', 'APAAC', 'QSO', 'CKSAAP', 'CTriad', 'PAAC', 'PSEAAC', 'DDE', 'DPC']

# 每种特征提取方法下的测试集 Accuracy
accuracy_values = [0.8777, 0.8886, 0.8907, 0.8262, 0.7894, 0.8874, 0.8777, 0.8354, 0.8103]

# 设置角度
num_vars = len(features)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合

# 补上首尾以闭合雷达图
stats = accuracy_values + accuracy_values[:1]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))  # 加大图形尺寸

# 绘制雷达图
ax.plot(angles, stats, label='测试集 Accuracy', color='#5B9BD5', linewidth=2)
ax.fill(angles, stats, color='#5B9BD5', alpha=0.25)

# 设置雷达图样式
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 设置角度标签（特征名）字体加粗放大
ax.set_thetagrids(np.degrees(angles[:-1]), features, fontsize=20, fontweight='bold')

# 设置Y轴范围与刻度
ax.set_ylim(0.7, 1.0)
ax.set_yticks([0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
ax.set_yticklabels(['0.75', '0.80', '0.85', '0.90', '0.95', '1.00'], fontsize=16, fontweight='bold')

# 设置图例位置和样式（远离雷达图）
ax.legend(
    loc='center left',
    bbox_to_anchor=(1.25, 0.5),
    frameon=True,
    prop={'family': 'SimSun', 'size': 16, 'weight': 'bold'}
)

# 设置标题加粗加大
ax.set_title('不同特征下抗菌肽分类模型的Accuracy表现（雷达图）', fontsize=18, fontweight='bold')

plt.tight_layout()
plt.show()
