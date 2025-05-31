# train.py

import os
from pydoc import text
import torch
from torch.utils.data import DataLoader
from eval import evaluate_loss
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from loss import contrastive_loss
from data_loader import Flickr8kDataset
from utils import SimpleTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer


text_encoder = "bert"

def main():
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 文件路径，根据实际调整
    token_file = "Flickr8k/captions.txt"         # 总的 captions 文件，用于构建词表
    train_token_file = "Flickr8k/train_captions.txt"  # 训练集，格式： image,caption
    val_token_file = "Flickr8k/val_captions.txt"      # 验证集
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
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        vocab_size = None
        print(f"Using BERT tokenizer with vocab size: {tokenizer.vocab_size}")

    # 构建数据集与 DataLoader：训练集、验证集、测试集
    train_dataset = Flickr8kDataset(
        root_dir="Flickr8k/images",      # 图片所在目录
        captions_file=train_token_file,   # 训练集 captions 文件，格式： image<TAB>caption
        tokenizer=tokenizer,
        use_bert_tokenizer=text_encoder == "bert"
    )
    val_dataset = Flickr8kDataset(
        root_dir="Flickr8k/images",
        captions_file=val_token_file,     # 验证集 captions 文件
        tokenizer=tokenizer,
        use_bert_tokenizer=text_encoder == "bert"
    )
    test_dataset = Flickr8kDataset(
        root_dir="Flickr8k/images",
        captions_file=test_token_file,    # 测试集 captions 文件
        tokenizer=tokenizer,
        use_bert_tokenizer=text_encoder == "bert"
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    # 为保证评估稳定，每个 batch 使用 batch_size=1
    val_dataloader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False)

    # 构造模型（设定 embed_dim=256）
    embed_dim = 256
    img_encoder = ImageEncoder(embed_dim=embed_dim).to(device)
    txt_encoder = TextEncoder(vocab_size, embed_dim=embed_dim, encoder_type=text_encoder).to(device)

    params = list(img_encoder.parameters()) + list(txt_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=1e-6)

    best_val_loss = float('inf')
    patience = 6
    early_stop_counter = 0
    min_delta = 0.001

    epochs = 40
    for epoch in range(epochs):
        img_encoder.train()
        txt_encoder.train()
        epoch_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for images, data in pbar:
            images = images.to(device)
            image_embeds = img_encoder(images)      # [batch, embed_dim]

            if text_encoder == "bert":
                for k in data:
                    data[k] = data[k].to(device)
                text_embeds = txt_encoder(data)
            else:
                captions_ids = data.to(device)
                text_embeds = txt_encoder(captions_ids)   # [batch, embed_dim]

            loss = contrastive_loss(image_embeds, text_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_dataloader)

        # Evaluate on validation and test sets

        val_loss = evaluate_loss(img_encoder, txt_encoder, val_dataloader, device)
        test_loss = evaluate_loss(img_encoder, txt_encoder, test_dataloader, device)

        print(f"Epoch [{epoch+1}/{epochs}]: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f}")

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # 如果验证集有改善，则保存最佳模型，这里需要同学们自己选择评估标准
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            early_stop_counter = 0
            tokenizer: BertTokenizer
            checkpoint = {
                'epoch': epoch + 1,
                'img_encoder_state_dict': img_encoder.state_dict(),
                'txt_encoder_state_dict': txt_encoder.state_dict(),
                'tokenizer_vocab': tokenizer.word2idx if text_encoder != 'bert' else None,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, "best_clip_model.pth")
            print(f"    > Best model updated at epoch {epoch+1} ")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break


    # 训练完成，最终在测试集上评估
    final_test_loss = evaluate_loss(img_encoder, txt_encoder, test_dataloader, device)
    print(f"Final Test Loss: {final_test_loss:.4f}")


if __name__ == "__main__":
    main()
