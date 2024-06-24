def create_mapping(yolo_filename, quant_filename, output_filename):
    yolo_lines = []
    quant_lines = []

    # 读取 yolo_state_dict 文件内容
    with open(yolo_filename, 'r') as yolo_file:
        yolo_lines = yolo_file.read().strip().split('\n')

    # 读取 quant_state_dict 文件内容
    with open(quant_filename, 'r') as quant_file:
        quant_lines = quant_file.read().strip().split('\n')

    # 确保两个文件行数相同
    if len(yolo_lines) != len(quant_lines):
        raise ValueError("两个文件行数不一致，无法建立映射表。")

    # 创建映射表
    mapping = {}
    for yolo_line, quant_line in zip(yolo_lines, quant_lines):
        mapping[yolo_line] = quant_line

    # 将映射表写入 mapping.py 文件
    with open(output_filename, 'w') as output_file:
        output_file.write("mapping = {\n")
        for yolo_line, quant_line in mapping.items():
            output_file.write(f"    '{yolo_line}': '{quant_line}',\n")
        output_file.write("}\n")

# 调用函数并生成映射表文件
yolo_filename = 'yolo_state_dict.txt'  # 替换为你的文件名
quant_filename = 'quant_state_dict.txt'  # 替换为你的文件名
output_filename = 'mapping.py'
create_mapping(yolo_filename, quant_filename, output_filename)
