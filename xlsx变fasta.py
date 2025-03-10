# 打开 TXT 文件和输出的 FASTA 文件
with open("D:/bishedata/数据2.txt", "r") as txt_file, \
        open("D:/bishedata/数据2.fasta", "w") as fasta_file:
    # 逐行读取 TXT 文件
    for line in txt_file:
        # 去除行首和行尾的空白字符（包括换行符）
        line = line.strip()

        # 如果是以 '>' 开头的行，直接写入 FASTA 文件
        if line.startswith(">"):
            fasta_file.write(line + "\n")  # 写入序列名
        else:
            fasta_file.write(line + "\n")  # 写入序列内容