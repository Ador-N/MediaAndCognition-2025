import os
import re
import matplotlib.pyplot as plt

# 设置日志目录和输出图像目录
log_dir = './logs'          # 替换为你的日志目录路径
output_dir = './figures'    # 图像保存目录

os.makedirs(output_dir, exist_ok=True)

# 预定义文件顺序列表
file_order = [
    'log_train_midterm_lstm.txt',
    'log_train_transformer.txt',
    'log_train_bert.txt',
    'log_train_bert_adjusted.txt',
]

# 正则表达式定义
log_pattern = re.compile(
    r"Epoch \[(\d+)/\d+\]: Train Loss: ([\d.]+) \| Val Loss: ([\d.]+) \| Test Loss: ([\d.]+)")
best_pattern = re.compile(r"> Best model updated at epoch (\d+)")

# 收集所有日志文件并按预定义顺序排序
all_log_files = [f for f in os.listdir(log_dir) if f.endswith('.txt')]
log_files = []

# 先添加预定义顺序中存在的文件
for file_name in file_order:
    if file_name in all_log_files:
        log_files.append(file_name)

# 再添加其他未在预定义列表中的文件
# for file_name in all_log_files:
#     if file_name not in log_files:
#         log_files.append(file_name)

if not log_files:
    print("未找到日志文件")
    exit()

# 计算subplot布局
num_files = len(log_files)
cols = 2
rows = (num_files + cols - 1) // cols

# 创建图形
fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
if num_files == 1:
    axes = [axes]
elif rows == 1:
    axes = axes.reshape(1, -1)

# 遍历所有.txt日志文件
for idx, filename in enumerate(log_files):
    row = idx // cols
    col = idx % cols
    ax = axes[row, col] if rows > 1 else axes[col]
    
    filepath = os.path.join(log_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        log_text = f.read()

    # 解析日志内容
    epochs, train_losses, val_losses, test_losses = [], [], [], []
    best_epochs = set()

    for match in log_pattern.finditer(log_text):
        epoch = int(match.group(1))
        train_loss = float(match.group(2))
        val_loss = float(match.group(3))
        test_loss = float(match.group(4))
        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

    for match in best_pattern.finditer(log_text):
        best_epochs.add(int(match.group(1)))

    # 若未成功解析，跳过该文件
    if not epochs:
        print(f"[跳过] 无有效日志内容：{filename}")
        ax.text(0.5, 0.5, f"无数据\n{filename}", ha='center', va='center', transform=ax.transAxes)
        continue

    # 获取文件名作为标签
    label_name = filename.replace('.txt', '').replace("log_train_", '')
    
    # 在当前subplot上绘制损失曲线
    ax.plot(epochs, train_losses, label='Train', linestyle='-', alpha=0.7)
    ax.plot(epochs, val_losses, label='Val', linestyle='--', alpha=0.7)
    ax.plot(epochs, test_losses, label='Test', linestyle=':', alpha=0.7)

    # 标记最佳epoch
    for epoch in best_epochs:
        if epoch in epochs:
            idx = epochs.index(epoch)
            ax.scatter(epoch, val_losses[idx], color='red', s=50, alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{label_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    print(f"[完成] 已处理：{filename}")

# 隐藏多余的subplot
for idx in range(num_files, rows * cols):
    row = idx // cols
    col = idx % cols
    if rows > 1:
        axes[row, col].set_visible(False)
    else:
        axes[col].set_visible(False)

plt.tight_layout()

# 保存图像
output_path = os.path.join(output_dir, 'loss_curves_subplots.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[完成] 已保存分离图像：{output_path}")
