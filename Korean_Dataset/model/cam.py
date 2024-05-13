import torch
import torch.nn as nn

__all__ = ['ChannelAttentionModule']


class ChannelAttentionModule(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.num_channels = num_channels
        self.reduction_ratio = reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.GELU(),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if len(list(x.shape)) == 4:
            x = x.unsqueeze(1)
        B, T, C, H, W = x.shape
        x_tmp = x.view(-1, C, H, W)  # Merge batch and time dimensions
        avg_pool = self.avg_pool(x_tmp).view(x_tmp.size(0), -1)
        max_pool = self.max_pool(x_tmp).view(x_tmp.size(0), -1)
        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)
        if T == 1:
            out = self.sigmoid(avg_out + max_out).view(x_tmp.size(0), self.num_channels, 1, 1)
            return x_tmp * out.expand_as(x_tmp) 
        else:
            out = self.sigmoid(avg_out + max_out).view(B, T, C, 1, 1)
            return x * out.expand_as(x)
        

if __name__ == "__main__":
    input_tensor = torch.rand((10, 64, 32, 32))  # Example input tensor
    cam = ChannelAttentionModule(num_channels=64)
    output = cam(input_tensor)
    print(output.shape)  # Should have the same shape as input_tensor