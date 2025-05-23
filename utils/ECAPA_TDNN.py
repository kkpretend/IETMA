import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import librosa
import numpy as np
from glob import glob
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import logging
import time
from Diff_Transform import MultiheadDiffAttn as att
import argparse


class improved_ECAPA_TDNN(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim, seq_len, num_blocks, dilation, hidden_size, depth, num_heads, reduction=4):
        super(improved_ECAPA_TDNN, self).__init__()

        # 第一层TDNN卷积
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=1, padding=2)

        # 创建多个SE-Res2Block模块
        self.blocks = nn.ModuleList([improved_SERes2Block(hidden_dim, reduction, dilation=dilation) for _ in range(num_blocks)])

        self.sea1 = SEAttention(hidden_dim)
        self.attn = att(args1, 256, depth, num_heads)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 分类层
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads

        self.attn1 = att(args1, hidden_dim, depth, num_heads)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.fc1 = nn.Linear(64, num_classes)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        # 第一次卷积
        x = self.bn1(F.relu(self.conv1(x)))

        # SE-Res2Block模块
        sea = self.sea1(x) + x
        residual1 = x + sea
        for block in self.blocks:
            residual = x + sea
            if not torch.equal(residual, residual1):
                residual = residual1
            else:
                residual = 0
            x = block(x) + residual

        x = self.bn2(self.global_pool(x))

        # 将输出的时间维度去除，转换为 [batch_size, channels]
        x = x.squeeze(-1)

        # 分类层
        x = self.bn3(self.attn1(x.unsqueeze(1)).squeeze(1)) * x
        x = self.bn4(self.fc(x))
        return x


class improved_SERes2Block(nn.Module):
    def __init__(self, in_channels, reduction=4, dilation=2):
        super(improved_SERes2Block, self).__init__()
        
        # 两个卷积层
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(in_channels)  # BatchNorm 层
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(in_channels)  # BatchNorm 层
        self.dilated_conv = improved_Res2DilatedConv1D(in_channels, dilation)

        self.sea = SEAttention(in_channels)

        # SE 模块
        self.se = SEAttention(in_channels, reduction)  

        # 残差连接
        self.shortcut = nn.Identity() if in_channels == in_channels else nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x = x + self.sea(x) + x
        residual = x

        # 第一卷积层 + BN + ReLU
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.dilated_conv(out)
        
        # 第二卷积层 + BN
        out = self.bn2(self.conv2(out))
        
        # SE 模块
        out = self.se(out)
        
        # 加入残差连接
        out += self.shortcut(residual)
        
        # 输出激活
        return F.relu(out)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        
        # 通道压缩层
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        
    def forward(self, x):
        # 全局平均池化
        z = torch.mean(x, dim=-1)  # [batch_size, channels]
        
        # 通道注意力机制
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z)).unsqueeze(-1)  # [batch_size, channels, 1]
        
        # 将注意力应用于输入特征
        return x * z


class improved_Res2DilatedConv1D(nn.Module):
    def __init__(self, in_channels, dilation=2):
        super(improved_Res2DilatedConv1D, self).__init__()
        self.conv1 = MultiScaleConvAttentionModule(in_channels, in_channels // 2)
        self.bn1 = nn.BatchNorm1d(in_channels // 2)
        self.conv2 = nn.Conv1d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 残差连接
        return out


class SEAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * torch.sigmoid(y)


class MultiScaleConvAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConvAttentionModule, self).__init__()

        self.conv1_3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, dilation=1)
        self.conv1_5 = nn.Conv1d(in_channels, in_channels, kernel_size=5, dilation=2, padding=3)
        self.conv1_1 = nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=3)
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, dilation=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(4 * in_channels, 4 * in_channels // 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4 * in_channels // 64, 4 * in_channels, bias=False)
        )
        self.conv1_4 = nn.Conv1d(in_channels * 4, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x1 = F.relu(self.conv1_3(F.relu(self.conv2(x))))
        x2 = F.relu(self.conv1_5(F.relu(self.conv3(x))))
        x3 = F.relu(self.conv1_1(x))
        x4 = F.relu(self.conv1(self.maxpool(x)))
        x_concat = torch.cat((x1, x2, x3, x4), dim=1)
        b, c, _ = x_concat.size()
        y = self.avg_pool(x_concat).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x_att = x_concat * torch.sigmoid(y)
        return self.conv1_4(x_att)



# # 检查GPU是否可用
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# def define_args():
#     parser = argparse.ArgumentParser(description='Description of your program.')
#     parser.add_argument('--model_parallel_size', type=int, default=1, help='Size of model parallelism.')
#     parser.add_argument('--decoder_kv_attention_heads', type=int, default=None,
#                         help='Number of decoder key-value attention heads.')
#     # 添加其他需要的参数

#     return parser.parse_args()
# args1 = define_args()
# def debug(model, batch_size, input_dim, seq_len):
#     x = torch.randn(batch_size, input_dim, seq_len).to(device)
#     x = model(x)
#     labels = torch.randn(batch_size, 6).to(device)
#     criterion = nn.CrossEntropyLoss()
#     loss = criterion(x, labels)
#     loss.backward()

# if __name__ == "__main__":
#     input_dim = 13  
#     output_dim = 6
#     seq_len = 256
#     hidden_dim = 128
#     num_blocks = 3
#     reduction = 3
#     dilation = 1
#     hidden_size = 32
#     depth = 9
#     num_heads = 4
#     batch_size = 128
#     num_epochs = 1000 
#     learning_rate = 0.00019003
#     model2 = improved_ECAPA_TDNN(num_classes=output_dim,
#                         input_dim=input_dim,
#                         hidden_dim=hidden_dim,
#                         seq_len=seq_len,
#                         num_blocks=num_blocks,
#                         reduction=reduction,
#                         hidden_size=hidden_size,
#                         depth=depth,
#                         num_heads=num_heads,
#                         dilation=dilation).to(device)
#     debug(model2, batch_size, input_dim, seq_len)