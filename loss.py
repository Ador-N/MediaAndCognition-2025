import torch
import torch.nn.functional as F


# 请在这里写出对比损失函数
def contrastive_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor, temperature=0.07):
    """
    image_embeds, text_embeds: [batch, embed_dim]
    """

    image_embeds = F.normalize(image_embeds, p=2, dim=1)
    text_embeds = F.normalize(text_embeds, p=2, dim=1)

    # 余弦相似度矩阵：[B, B]
    logits = image_embeds @ text_embeds.T / temperature  # 图像作为query
    logits = logits.float()  # Ensure logits are float for cross_entropy
    labels = torch.arange(image_embeds.size(0), device=image_embeds.device)
    l_i2t = F.cross_entropy(logits, labels)
    l_t2i = F.cross_entropy(logits.T, labels)

    return 0.5 * (l_i2t + l_t2i)
