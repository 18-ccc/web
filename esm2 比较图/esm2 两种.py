import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

# 设置中文宋体显示
matplotlib.rcParams['font.sans-serif'] = ['SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

# 性能指标
metrics = ['Accuracy', 'Recall', 'F1 Score', 'AUC']
num_vars = len(metrics)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合圆

# 模型性能数据（仅ESM2-Model1和ESM2-Model2）
baumannii = {
    'ESM2-Model1': [0.9773, 0.6190, 0.7027, 0.9893],
    'ESM2-Model2': [0.9783, 0.6190, 0.7123, 0.9880]
}
enterobacter = {
    'ESM2-Model1': [0.9247, 0.8645, 0.8844, 0.9594],
    'ESM2-Model2': [0.9204, 0.8516, 0.8770, 0.9599]
}
pseudomonas = {
    'ESM2-Model1': [0.9420, 0.8352, 0.8633, 0.9568],
    'ESM2-Model2': [0.9378, 0.8237, 0.8531, 0.9578]
}

# 颜色定义（两个模型）
colors = {
    'ESM2-Model1': '#ED7D31',
    'ESM2-Model2': '#A5A5A5'
}

# 输出目录
output_dir = r'D:\bishedata2\拼接图'
os.makedirs(output_dir, exist_ok=True)

def plot_radar(data, title, filename):
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 设置角度标签
    ax.set_thetagrids(
        np.degrees(angles[:-1]),
        metrics,
        fontsize=26,
        fontweight='bold',
        fontname='SimSun'
    )

    # 设置坐标范围和Y刻度
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(['0.6', '0.7', '0.8', '0.9', '1.0'],
                       fontsize=20, fontweight='bold', fontname='SimSun')

    # 绘图
    for model, scores in data.items():
        stats = scores + scores[:1]
        ax.plot(angles, stats, label=model, color=colors[model], linewidth=3)
        ax.fill(angles, stats, color=colors[model], alpha=0.25)

    # 设置标题加粗、宋体、26号字体
    ax.set_title(title, size=26, weight='bold', loc='left', pad=40, fontname='SimSun')

    # 设置图例加粗、宋体、20号字体，图例在图外右侧
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.15, 0.25),
        prop={'family': 'SimSun', 'size': 20, 'weight': 'bold'},
        borderaxespad=0.3,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# 生成雷达图并保存
plot_radar(baumannii, 'A. 鲍曼不动杆菌', 'A_鲍曼不动杆菌.png')
plot_radar(enterobacter, 'B. 肠杆菌科', 'B_肠杆菌科.png')
plot_radar(pseudomonas, 'C. 铜绿假单胞菌', 'C_铜绿假单胞菌.png')
