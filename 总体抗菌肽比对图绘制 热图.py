import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
font_path = 'C:/Windows/Fonts/simsun.ttc'
font_prop = fm.FontProperties(fname=font_path)

# 数据准备
data = {
    '特征组合': ['aac']*4 + ['cksaap']*4 + ['apaac']*4 + ['paac']*4 + ['ctriad']*4 + ['pseaac']*4 + ['qso']*4,
    '特征数量': ['10', '20', '30', '40'] * 7,
    'Accuracy': [0.8631, 0.8777, 0.8777, 0.8777,
                 0.7927, 0.8116, 0.8229, 0.8283,
                 0.8794, 0.8869, 0.8886, 0.8886,
                 0.8731, 0.8832, 0.8874, 0.8874,
                 0.7437, 0.7747, 0.7885, 0.7873,
                 0.8631, 0.8777, 0.8777, 0.8777,
                 0.8626, 0.8853, 0.8903, 0.8878],
    'Recall':   [0.8631, 0.8777, 0.8777, 0.8777,
                 0.7927, 0.8116, 0.8229, 0.8283,
                 0.8794, 0.8869, 0.8886, 0.8886,
                 0.8731, 0.8832, 0.8874, 0.8874,
                 0.7437, 0.7747, 0.7885, 0.7873,
                 0.8631, 0.8777, 0.8777, 0.8777,
                 0.8626, 0.8853, 0.8903, 0.8878],
    'F1 Score': [0.8623, 0.8768, 0.8768, 0.8768,
                 0.7893, 0.8101, 0.8208, 0.8258,
                 0.8790, 0.8862, 0.8877, 0.8877,
                 0.8724, 0.8823, 0.8865, 0.8865,
                 0.7435, 0.7747, 0.7883, 0.7872,
                 0.8623, 0.8768, 0.8768, 0.8768,
                 0.8618, 0.8845, 0.8895, 0.8870],
    'AUC':      [0.9380, 0.9470, 0.9470, 0.9470,
                 0.8456, 0.8742, 0.8865, 0.8919,
                 0.9363, 0.9497, 0.9525, 0.9525,
                 0.9390, 0.9492, 0.9520, 0.9520,
                 0.7984, 0.8452, 0.8616, 0.8632,
                 0.9380, 0.9470, 0.9470, 0.9470,
                 0.9272, 0.9450, 0.9511, 0.9533]
}
df = pd.DataFrame(data)

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(24, 18))
metrics = ['Accuracy', 'Recall', 'F1 Score', 'AUC']
vmins = [0.74, 0.74, 0.74, 0.79]
vmaxs = [0.91, 0.91, 0.91, 0.96]
labels = ['A', 'B', 'C', 'D']

for ax, metric, vmin, vmax, label in zip(axes.flat, metrics, vmins, vmaxs, labels):
    pivot = df.pivot(index='特征组合', columns='特征数量', values=metric)
    heatmap = sns.heatmap(
        pivot,
        cmap=sns.color_palette("YlOrBr", as_cmap=True),
        ax=ax,
        vmin=vmin, vmax=vmax,
        annot=False,
        square=True,
        cbar_kws={'label': metric},
        linewidths=0.2,
        linecolor='#cccccc'
    )

    # 坐标轴标签字体加粗且字体设为“黑体”
    ax.set_xlabel('特征数量', fontproperties=font_prop, fontsize=15, fontweight='black')
    ax.set_ylabel('特征组合', fontproperties=font_prop, fontsize=15, fontweight='black')

    # 坐标轴刻度标签字体加粗且字体设为“黑体”
    ax.set_xticklabels(
        ax.get_xticklabels(),
        fontproperties=font_prop,
        fontsize=20,
        fontweight='black',
        rotation=0
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        fontproperties=font_prop,
        fontsize=20,
        fontweight='black',
        rotation=0
    )
    ax.tick_params(axis='x', pad=12)
    ax.tick_params(axis='y', pad=12)

    # colorbar 标签字体加粗（黑体）
    cbar = heatmap.collections[0].colorbar
    cbar.set_label(metric, fontsize=22, fontweight='black', family=font_prop.get_name())

    # colorbar 刻度标签字体加粗，字体设为黑体
    cbar.ax.tick_params(labelsize=18)
    for label_cbar in cbar.ax.get_yticklabels():
        label_cbar.set_fontweight('black')
        label_cbar.set_fontname(font_prop.get_name())

    # 左上角标签保持正常字体大小和正常粗细
    ax.text(-0.12, 1.08, label, transform=ax.transAxes,
            fontsize=30, fontweight='normal', va='top', ha='left')

plt.tight_layout()
plt.savefig("D:/bishedata2/拼接图/总体随机森林热图拼接.png", dpi=900)
plt.savefig("D:/bishedata2/拼接图/总体随机森林热图拼接.svg", dpi=900)
plt.show()
