import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cam import *

__all__ = ['CAMViT']

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x is of shape [batch_size, window_size, channels, height, width]
        batch_size, window_size, channels, height, width = x.shape
        # Merge batch and window dimensions
        x = x.view(batch_size * window_size, channels, height, width)
        x = self.projection(x)  # Apply convolution to get patch embeddings
        x = x.flatten(2).transpose(1, 2)  # Reshape to [batch * window, num_patches, emb_size]
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, emb_size=768, depth=12, n_heads=12, mlp_ratio=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=n_heads, dim_feedforward=emb_size * mlp_ratio)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x)

class CAMViT(nn.Module):
    def __init__(self, input_data, patch_size=5, in_channels=3, num_classes=1000, emb_size=768, depth=12, n_heads=12, mlp_ratio=4):
        super().__init__()
        self.img_size = (50, 65) if input_data == 'gdaps_kim' else (151, 130)
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, self.img_size)
        num_patches = (self.img_size[0] // patch_size) * (self.img_size[1] // patch_size)

        self.emb_size = emb_size
        self.patch_size = patch_size

        # Calcul du nombre de patches dans chaque dimension (H et W)
        num_patches_h, num_patches_w = self.img_size[0] // patch_size, self.img_size[1] // patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.positional_embedding = nn.Parameter(torch.zeros(1, 1 + num_patches_h * num_patches_w, emb_size))
        self.transformer_encoder = TransformerEncoder(emb_size, depth, n_heads, mlp_ratio)
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )

        # Ajout d'une couche finale pour adapter la sortie
        self.final_layer = nn.Conv2d(in_channels=emb_size, out_channels=num_classes, kernel_size=1)
        
        self.channel_attention = ChannelAttentionModule(num_channels=in_channels)

    def forward(self, x, t):
        x = self.channel_attention(x)
        batch_size, window_size, _, _, _ = x.shape
        x = self.patch_embedding(x)
        batch_window = batch_size * window_size
        cls_token = self.cls_token.repeat(batch_window, 1, 1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.positional_embedding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = x[:, 1:, :]  # Ignorer le cls_token pour la reconstruction spatiale

        x = x.view(batch_size, window_size, -1, self.emb_size)
        x = x.mean(dim=1)
        
        # Reshape pour préparer la convolution
        batch_size, _, emb_size = x.shape
        x = x.transpose(1, 2).view(batch_size, emb_size, int(self.img_size[0] / self.patch_size), int(self.img_size[1] / self.patch_size))

        # Convolution pour passer de emb_size à num_classes avec un ajustement spatial
        x = self.final_layer(x)

        # Upsample pour correspondre à la taille d'image originale si nécessaire
        x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        
        return x