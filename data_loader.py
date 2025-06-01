import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import nlpaug.augmenter.word as naw
from models.bert_encoder import BertTokenizer
from global_config import enable_data_strengthen, image_encoder


class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, tokenizer, transform=None, max_len=32):
        self.root_dir = root_dir

        transforms = []
        if enable_data_strengthen:
            transforms = [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
            ]
        else:
            transforms += [
                T.Resize((224, 224)),
                T.ToTensor(),
            ]
        if image_encoder == "vit":
            transforms.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transform = transform or T.Compose(transforms)
            
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = self._load_pairs(captions_file)
        self.use_bert_tokenizer = isinstance(tokenizer, BertTokenizer)

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

        text_aug = naw.SynonymAug(aug_src='wordnet')
        caption = text_aug.augment(caption)

        if self.use_bert_tokenizer:
            self.tokenizer: BertTokenizer
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
            # Ensure caption is a string (augment might return a list)
            if isinstance(caption, list):
                caption = caption[0]
            caption_ids = self.tokenizer.encode(caption)
            return image, torch.tensor(caption_ids)
