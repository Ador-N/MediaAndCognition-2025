# query.py

import os
from tabnanny import check
import torch
from torch.utils.data import DataLoader
from models import text_encoder
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from data_loader import Flickr8kDataset
from utils import SimpleTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm

def query():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和tokenizer
    model_file = "trained_models/best_clip_model.pth"
    token_file = "Flickr8k/captions.txt"

    # 读取所有caption构建词表
    with open(token_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    captions = [line.strip().split(',')[1] for line in lines if line.strip()]

    # 构建tokenizer
    tokenizer = SimpleTokenizer(captions, min_freq=1)
    vocab_size = len(tokenizer)

    # 加载模型
    checkpoint = torch.load(model_file)
    img_encoder = ImageEncoder().to(device)
    txt_encoder = TextEncoder(vocab_size, encoder_type="transformer").to(device)
    img_encoder.load_state_dict(checkpoint['img_encoder_state_dict'])
    txt_encoder.load_state_dict(checkpoint['txt_encoder_state_dict'])

    # 加载所有测试图像
    test_dataset = Flickr8kDataset(
        root_dir="Flickr8k/images",
        captions_file="Flickr8k/test_captions.txt",
        tokenizer=tokenizer
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # 提取所有图像特征
    img_encoder.eval()
    txt_encoder.eval()

    all_image_embeds = []
    all_image_paths = []

    print("正在提取图像特征...")
    with torch.no_grad():
        for idx, (images, _) in enumerate(tqdm(test_dataloader)):
            images = images.to(device)
            image_embed = img_encoder(images)
            all_image_embeds.append(image_embed.cpu())
            # 获取当前图像的路径
            img_path = test_dataset.pairs[idx][0]
            all_image_paths.append(img_path)

    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_image_embeds = F.normalize(all_image_embeds, dim=1)

    # 交互式查询
    while True:
        query_text = input("\n请输入查询文本 (输入q退出): ")
        if query_text.lower() == 'q':
            break

        # 对输入文本进行编码
        tokens = tokenizer.encode(query_text)
        tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)

        with torch.no_grad():
            text_embed = txt_encoder(tokens)
            text_embed = F.normalize(text_embed, dim=1)

        # 计算相似度并获取Top-5结果
        sim_scores = torch.matmul(text_embed.cpu(), all_image_embeds.T)
        top_k_scores, top_k_indices = torch.topk(sim_scores[0], k=5)

        print(f"\n查询 Caption:\n\"{query_text}\"")
        print("\nTop-5 相似图像结果:")

        # 显示结果图像
        import matplotlib.pyplot as plt
        from PIL import Image

        plt.figure(figsize=(20, 4))
        for idx, (score, img_idx) in enumerate(zip(top_k_scores, top_k_indices)):
            img_path = all_image_paths[img_idx]
            img = Image.open("Flickr8k/images/" + img_path)
            plt.subplot(1, 5, idx + 1)
            plt.imshow(img)
            plt.title(f"Rank {idx + 1}\nScore: {score:.3f}")
            plt.axis('off')
            print(f"Rank {idx + 1}: {img_path}")
        plt.show()


if __name__ == "__main__":
    query()
