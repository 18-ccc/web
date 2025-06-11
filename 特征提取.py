import subprocess
import os
import pandas as pd


def extract_features_from_ilearnplus(sequence):
    # 将序列保存为临时的FASTA文件
    with open('temp_sequence.fasta', 'w') as f:
        f.write(f">protein_sequence\n{sequence}\n")

    # 调用IlearnPlus进行特征提取
    # 请确保IlearnPlus的路径正确，修改为你实际的路径
    command = [
        'python', '"D:\bishe\毕业设计\ilearn-plus.py"',  # 修改为你的IlearnPlus.py的实际路径
        '--input', 'temp_sequence.fasta',  # 输入文件
        '--output', 'output.xlsx',  # 输出结果文件
        '--feature', 'apaac,paac,qso'  # 提取APAAC, PAAC, QSO特征
    ]

    # 执行命令
    subprocess.run(command, check=True)

    # 读取IlearnPlus的输出文件
    features_df = pd.read_excel('output.xlsx')

    # 删除临时文件
    os.remove('temp_sequence.fasta')

    return features_df
