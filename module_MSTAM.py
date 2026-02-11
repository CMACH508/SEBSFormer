from torch import nn
import torch
import torch.nn.functional as F

class MS_TAM(nn.Module):
    def __init__(self, T, r=4):
        super(MS_TAM, self).__init__()
        inter_channels = T // r

        self.local_att = nn.Sequential(
            nn.Conv2d(T, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, T, kernel_size=1),
            nn.BatchNorm2d(T)
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(T, inter_channels, kernel_size=1), 
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, T, kernel_size=1),
            nn.BatchNorm2d(T)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        local_feat = self.local_att(x)
        global_feat = self.global_att(x)      
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='nearest')
        fused = local_feat + global_feat
        fused = fused - fused.max(dim=1, keepdim=True)[0]
        attn_weights = F.softmax(fused, dim=1)
        x_weighted = x * attn_weights
        out = x_weighted.sum(dim=1)
        return out, attn_weights           
