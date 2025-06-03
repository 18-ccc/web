import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文宋体显示
matplotlib.rcParams['font.sans-serif'] = ['SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

# 指标标签
labels = ['Accuracy', 'Recall', 'F1 Score', 'AUC']
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合

# 不同细菌的数据（模型1 和 模型2）
data = {
    '肠杆菌': {
        'esm2-model1': [0.9247, 0.8645, 0.8844, 0.9594],
        'esm2-model2': [0.9204, 0.8516, 0.8770, 0.9599]
    },
    '鲍曼不动杆菌': {
        'esm2-model1': [0.9773, 0.6190, 0.7027, 0.9893],
        'esm2-model2': [0.9783, 0.6190, 0.7123, 0.9880]
    },
    '铜绿假单胞菌': {
        'esm2-model1': [0.9420, 0.8352, 0.8633, 0.9568],
        'esm2-model2': [0.9378, 0.8237, 0.8531, 0.9578]
    }
}

# 颜色定义
colors = {
    'esm2-model1': '#5B9BD5',  # 蓝色
    'esm2-model2': '#ED7D31'   # 橙色
}

# 开始绘图，每种细菌一张雷达图
for bacteria, model_data in data.items():
    fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))

    for model_name, values in model_data.items():
        stats = values + values[:1]  # 闭合
        ax.plot(angles, stats, label=model_name, color=colors[model_name], linewidth=2)
        ax.fill(angles, stats, color=colors[model_name], alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0.5, 1.0)

    ax.set_title(f'{bacteria}：两模型性能雷达图对比', fontsize=14)
    ax.legend(loc='lower right', bbox_to_anchor=(1.1, 0.0), frameon=True, prop={'family': 'SimSun', 'size': 10})
    plt.tight_layout()
    plt.show()
