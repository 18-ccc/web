import subprocess
import os

def extract_features_from_ilearnplus(sequence):
    base_dir = r"D:\\bishe\毕业设计"
    fasta_path = os.path.join(base_dir, "temp_sequence.fasta")
    output_path = os.path.join(base_dir, "output.xlsx")

    # 确保目录存在
    os.makedirs(base_dir, exist_ok=True)

    # 写入序列文件
    with open(fasta_path, "w") as f:
        f.write(f">seq1\n{sequence}\n")

    # 确保命令路径正确
    command = [
        r"D:\anacond\python.exe",  # 显式指定Python路径
        r"D:\bishe\毕业设计\ilearnplus.py",  # 使用原始字符串确保路径正确
        "--input", fasta_path,
        "--output", output_path,
        "--feature", "apaac,paac,qso"
    ]

    try:
        # 执行命令
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # 自动解码字符串
            cwd=base_dir  # 设置工作目录
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"命令失败: {e.cmd}")
        print(f"返回码: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise RuntimeError(f"特征提取失败: {e.stderr}") from e

    return "特征提取成功"
