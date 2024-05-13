# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
from model.cam import *
import torch.nn.functional as F


from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from model.base_swin_sys import SwinTransformerSys

logger = logging.getLogger(__name__)

__all__ = ['SwinUnet']
T = 6
NB_CLASSES = 3
CHANNELS = 12
EMBED_DIM = 96 
WINDOW_SIZE = 7 

class SwinUnet(nn.Module):
    def __init__(self, input_data = 'gdaps_kim', num_classes=NB_CLASSES, zero_head=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head

        if input_data == 'gdaps_kim':
            self.h = 50
            self.w = 65
        elif input_data == 'gdaps_um':
            self.h = 151
            self.w = 130

        # Create Channel Attention Module
        self.channel_attention = ChannelAttentionModule(num_channels=CHANNELS)

        self.swin_unet = SwinTransformerSys(img_size=224,
                                patch_size=4,
                                in_chans=CHANNELS*T,
                                num_classes=self.num_classes,
                                embed_dim=EMBED_DIM,
                                depths=[2,2,6,2],
                                num_heads=[3,6,12,24],
                                window_size=WINDOW_SIZE,
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)
        
        self.num_input_channels = 36
        self.channel_doubling_conv = nn.Conv2d(self.num_input_channels, 2*self.num_input_channels, kernel_size=(1, 1))

    def forward(self, x, t):
        #Channel attention module
        x = self.channel_attention(x)
        B,T,C,H,W = x.shape
        x = x.view(B,-1,H,W)
        # Redimensionnement à (B, C*T, img_size, img_size)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # On doit doubler le nombre de channels pour que ça fonctionne
        x = self.channel_doubling_conv(x)
        
        logits = self.swin_unet(x)
        
        # Normalisation
        tensor_min = x.view(logits.size(0), logits.size(1), -1).min(2)[0].unsqueeze(-1).unsqueeze(-1)
        tensor_max = x.view(logits.size(0), logits.size(1), -1).max(2)[0].unsqueeze(-1).unsqueeze(-1)
        normalized_tensor = (logits - tensor_min) / (tensor_max - tensor_min)
        # Redimensionnement à (B*T, 3, 50, 65)
        logits = F.interpolate(normalized_tensor, size=(self.h, self.w), mode='bilinear', align_corners=False)
        logits = logits.view(B,3,H,W)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
 

 
if __name__ == "__main__":
    input_tensor = torch.rand((5,6,12,50,65))  # Example input tensor B*T*C*H*W
    camt= SwinUnet()
    output = camt(input_tensor, 0)
    print(output.shape) #should be logits B,3,H,W