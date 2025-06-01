# eval.py

import torch
from torch.utils.data import DataLoader
from loss import contrastive_loss
from models.bert_encoder import BERTTextEncoder
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from data_loader import Flickr8kDataset
from utils import SimpleTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from global_config import text_encoder, image_encoder, eval_target\

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# 供同学们参考
def evaluate_top_k(img_encoder, txt_encoder, dataloader, device, topk=(1, 5, 10)):
    img_encoder.eval()
    txt_encoder.eval()

    all_image_embeds = []
    all_text_embeds = []

    with torch.no_grad():
        for images, data in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            image_embed = img_encoder(images)

            if text_encoder == "bert":
                for k in data:
                    data[k] = data[k].to(device)
                text_embed = txt_encoder(data)
            else:
                captions_ids = data.to(device)
                text_embed = txt_encoder(captions_ids)

            all_image_embeds.append(image_embed.cpu())
            all_text_embeds.append(text_embed.cpu())

    all_image_embeds = torch.cat(all_image_embeds, dim=0)  # [N, D]
    all_text_embeds = torch.cat(all_text_embeds, dim=0)    # [N, D]

    # 归一化
    all_image_embeds = F.normalize(all_image_embeds, dim=1)
    all_text_embeds = F.normalize(all_text_embeds, dim=1)

    # 文本 -> 图像检索
    sim_matrix = torch.matmul(all_text_embeds, all_image_embeds.T)  # [N, N]
    txt2img_ranks = torch.argsort(sim_matrix, dim=1, descending=True)

    # 图像 -> 文本检索
    sim_matrix_T = sim_matrix.T  # [N, N]
    img2txt_ranks = torch.argsort(sim_matrix_T, dim=1, descending=True)

    def recall_at_k(ranks, topk):
        recalls = []
        for k in topk:
            match = [i in ranks[i][:k] for i in range(len(ranks))]
            recalls.append(np.mean(match))
        return recalls

    r_txt2img = recall_at_k(txt2img_ranks, topk)
    r_img2txt = recall_at_k(img2txt_ranks, topk)

    print("\n📈 Text → Image Retrieval:")
    for i, k in enumerate(topk):
        print(f"Recall@{k}: {r_txt2img[i]*100:.2f}%")
    print("\n📈 Image → Text Retrieval:")
    for i, k in enumerate(topk):
        print(f"Recall@{k}: {r_img2txt[i]*100:.2f}%")

    return r_txt2img, r_img2txt


def evaluate_loss(img_encoder: ImageEncoder, txt_encoder: TextEncoder, dataloader: DataLoader, device):
    img_encoder.eval()
    txt_encoder.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, data in dataloader:
            images = images.to(device)
            image_embeds = img_encoder(images)

            if isinstance(txt_encoder.encoder, BERTTextEncoder):
                for k in data:
                    data[k] = data[k].to(device)
                text_embeds = txt_encoder(data)
            else:
                captions_ids = data.to(device)
                text_embeds = txt_encoder(captions_ids)   # [batch, embed_dim]
            
            loss = contrastive_loss(image_embeds, text_embeds)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file = f"trained_models/best_clip_model_{eval_target}.pth"
    token_file = "Flickr8k/captions.txt"              # 总的 captions 文件，用于构建词表
    test_token_file = "Flickr8k/test_captions.txt"    # 测试集

    # 读取所有 caption 用于构建总词表（假设以 tab 分隔，如果不是，请修改 split 参数）
    with open(token_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    captions = [line.strip().split(',')[1] for line in lines if line.strip()]

    # 构建统一的 tokenizer
    if text_encoder != "bert":
        tokenizer = SimpleTokenizer(captions, min_freq=1)
        vocab_size = len(tokenizer)
        print(f"Vocabulary size: {vocab_size}")
    else:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("D:/Projects/MediaAndRecognition/models/bert") #from_pretrained('bert-base-uncased')
        vocab_size = None
        print(f"Using BERT tokenizer with vocab size: {tokenizer.vocab_size}")

    # 构建测试集
    test_dataset = Flickr8kDataset(
        root_dir="Flickr8k/images",
        captions_file=test_token_file,    # 测试集 captions 文件
        tokenizer=tokenizer
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False)

    # load from best_clip_model.pth
    checkpoint = torch.load(model_file)
    # run model
    img_encoder = ImageEncoder(encoder_type=image_encoder).to(device)
    txt_encoder = TextEncoder(vocab_size, encoder_type=text_encoder).to(device)
    img_encoder.load_state_dict(checkpoint['img_encoder_state_dict'])
    txt_encoder.load_state_dict(checkpoint['txt_encoder_state_dict'])

    evaluate_top_k(img_encoder, txt_encoder, test_dataloader, device)

def query():
    pass


if __name__ == "__main__":
    main()