import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class BERTTextEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.project = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def forward(self, captions):
        encoded_input = {k: v.to(next(self.parameters()).device)
                         for k, v in captions.items()}
        outputs = self.bert(**encoded_input)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # 取 [CLS]
        return self.project(cls_embedding)
