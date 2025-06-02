# import torch.nn as nn
# import torchvision.models as models

# class ImageEncoder(nn.Module):
#     def __init__(self, embed_dim=256):
#         super().__init__()
#         self.resnet = models.resnet18(pretrained=True)
#         self.resnet.fc = nn.Identity()  # 移除最后的分类层
#         self.fc = nn.Linear(512, embed_dim)  # 映射到共享空间

#     def forward(self, images):
#         features = self.resnet(images)  # [batch, 512]
#         embeddings = self.fc(features)  # [batch, embed_dim]
#         return embeddings
# image_encoder.py
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, encoder_type='vit', embed_dim=256):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == 'vit':
            from transformers import ViTModel
            self.backbone = ViTModel.from_pretrained(
                'D:/Projects/MediaAndRecognition/models/ViT')  # google/vit-base-patch16-224
            # freeze ViT, fine-tune output layer only
            for name, param in self.backbone.named_parameters():
                param.requires_grad = \
                    name.startswith("encoder.layer.11") \
                    or name.startswith("layernorm")
            self.out_dim = self.backbone.config.hidden_size

        elif encoder_type == 'resnet18':
            from .resnet_custom import ResNet18
            self.resnet = ResNet18()
            self.out_dim = 256

        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        self.fc = nn.Linear(self.out_dim, embed_dim)

    def forward(self, images):
        if self.encoder_type == 'vit':
            outputs = self.backbone(pixel_values=images)
            pooled_output = outputs.pooler_output
            return self.fc(pooled_output)
        else:
            features = self.resnet(images)
            return self.fc(features)
