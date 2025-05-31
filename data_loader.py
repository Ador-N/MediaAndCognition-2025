import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, tokenizer, transform=None, max_len=32, use_bert_tokenizer=False):
        self.root_dir = root_dir
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = self._load_pairs(captions_file)
        self.use_bert_tokenizer = use_bert_tokenizer

    def _load_pairs(self, captions_file):
        pairs = []
        with open(captions_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                image_filename = row[0].strip()
                caption = row[1].strip()
                pairs.append((image_filename, caption))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        filename, caption = self.pairs[idx]
        image_path = os.path.join(self.root_dir, filename)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        if self.use_bert_tokenizer:
            encoding = self.tokenizer(
                caption,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            return image, {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            }
        else:
            caption_ids = self.tokenizer.encode(caption)
            return image, torch.tensor(caption_ids)
