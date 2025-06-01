import os
import re
import matplotlib.pyplot as plt

# 设置日志目录和输出图像目录
log_dir = './logs'          # 替换为你的日志目录路径
output_dir = './figures'    # 图像保存目录

os.makedirs(output_dir, exist_ok=True)

# 正则表达式定义
log_pattern = re.compile(
    r"Epoch \[(\d+)/\d+\]: Train Loss: ([\d.]+) \| Val Loss: ([\d.]+) \| Test Loss: ([\d.]+)")
best_pattern = re.compile(r"> Best model updated at epoch (\d+)")

# 遍历所有.txt日志文件
for filename in os.listdir(log_dir):
    if not filename.endswith('.txt'):
        continue

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
        continue

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='s')
    plt.plot(epochs, test_losses, label='Test Loss', marker='^')

    for epoch in best_epochs:
        if epoch in epochs:
            idx = epochs.index(epoch)
            plt.scatter(epoch, val_losses[idx], color='red', label='Best Epoch' if epoch == min(
                best_epochs) else "")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve: {filename}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    output_path = os.path.join(output_dir, filename.replace('.txt', '.png'))
    plt.savefig(output_path)
    plt.close()

    print(f"[完成] 已保存图像：{output_path}")
