from graphviz import Digraph

def draw_model_pipeline():
    dot = Digraph(comment='抗菌肽预测模型流程', format='png')

    # 设置节点样式
    dot.attr('node', shape='box', style='filled', color='lightblue2', fontname='Microsoft YaHei')

    # 添加节点
    dot.node('A', '1. 加载与合并特征文件\n(apaac, paac, qso)')
    dot.node('B', '2. 特征预处理\n- 合并数据\n- 缺失值处理\n- 转换为数值型')
    dot.node('C', '3. 线性SVM特征选择\n- 选取前90个特征')
    dot.node('D', '4. 数据划分与标准化\n- 训练/测试集划分\n- 标准化')
    dot.node('E', '5. 定义核函数SVM模型\n- RBF核\n- 类别权重平衡')
    dot.node('F', '6. 训练集交叉验证评估\n- 10折CV\n- 计算Accuracy, Recall, F1, AUC')
    dot.node('G', '7. 测试集性能评估\n- 预测\n- 计算指标')
    dot.node('H', '8. ROC曲线绘制\n- 训练集CV ROC\n- 测试集ROC')
    dot.node('I', '9. SHAP解释性分析\n- 计算SHAP值\n- 重要特征展示')

    # 添加边连接节点
    dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH', 'HI'])

    # 生成并保存图像
    dot.render('antimicrobial_peptide_model_pipeline', view=True)

if __name__ == '__main__':
    draw_model_pipeline()
