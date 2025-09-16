import torch
import torch.nn.functional as F
import numpy as np

from torch import nn



class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activation='gelu'):
        super(ConvBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = ConvBlock2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = ConvBlock2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, activation='no')
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        att = self.conv1(x)
        att = self.conv2(att)
        att = self.bn(att)
        att = self.sigmoid(att)
        return x * att, att

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        x_pool = self.avg_pool(x).view(batch_size, channels)
        x_fc = F.relu(self.fc1(x_pool))
        x_att = self.sigmoid(self.fc2(x_fc)).view(batch_size, channels, 1, 1)
        return x * x_att

class CombinedAttention(nn.Module):
    def __init__(self, in_channels):
        super(CombinedAttention, self).__init__()
        self.spatial = SpatialAttention(in_channels)
        self.channel = ChannelAttention(in_channels)

    def forward(self, x):
        x = self.channel(x)
        x, spatial_att = self.spatial(x)
        return x, spatial_att

class ConvolutionalSelfAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, padding=1):
        super(ConvolutionalSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=kernel_size, padding=padding)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=kernel_size, padding=padding)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        nn.init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.query_conv.bias is not None:
            nn.init.zeros_(self.query_conv.bias)
            nn.init.zeros_(self.key_conv.bias)
            nn.init.zeros_(self.value_conv.bias)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("ConvolutionalSelfAttention input contains NaN or Inf values.")
        batch_size, C, H, W = x.size()
        q = self.query_conv(x).view(batch_size, -1, H * W)
        k = self.key_conv(x).view(batch_size, -1, H * W)
        v = self.value_conv(x).view(batch_size, C, H * W)
        energy = torch.bmm(q.transpose(1, 2), k)
        attention = self.softmax(energy)
        out = torch.bmm(v, attention.transpose(1, 2))
        out = out.view(batch_size, C, H, W)
        if torch.isnan(out).any() or torch.isinf(out).any():
            raise ValueError("ConvolutionalSelfAttention output contains NaN or Inf values.")
        out = self.gamma * out + x
        return out

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.conv11 = ConvBlock2d(in_channels=3, out_channels=32)
        self.conv12 = ConvBlock2d(in_channels=32, out_channels=32)
        self.conv13 = ConvBlock2d(in_channels=3, out_channels=32, kernel_size=1, padding=0, activation='no')

        self.conv21 = ConvBlock2d(in_channels=32, out_channels=128)
        self.conv22 = ConvBlock2d(in_channels=128, out_channels=128)
        self.conv23 = ConvBlock2d(in_channels=32, out_channels=128, kernel_size=1, padding=0, activation='no')

        self.conv31 = ConvBlock2d(in_channels=128, out_channels=256)
        self.conv32 = ConvBlock2d(in_channels=256, out_channels=144)
        self.conv33 = ConvBlock2d(in_channels=128, out_channels=144, kernel_size=1, padding=0, activation='no')

        self.attention = CombinedAttention(in_channels=144)
        
        self.pool = nn.AvgPool2d(kernel_size=2) # pooling H va W xuong con H' va W'. Chia cho 2
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x11 = self.conv11(x)
        x11 = self.conv12(x11)
        x12 = self.conv13(x)
        x1 = x11 + x12
        
        x1 = self.dropout(x1)
        x1_pool = self.pool(x1)

        x21 = self.conv21(x1_pool)
        x21 = self.conv22(x21)
        x22 = self.conv23(x1_pool)
        x2 = x21 + x22
        
        x2 = self.dropout(x2)
        x2_pool = self.pool(x2)

        x31 = self.conv31(x2_pool)
        x31 = self.conv32(x31)
        x32 = self.conv33(x2_pool)
        
        x3 = x31 + x32
        x3 = self.dropout(x3)
        
        x3, spatial_att = self.attention(x3)
        return x1, x2, x3

class CrossStreamFusion(nn.Module):
    def __init__(self, in_channels=144):
        super(CrossStreamFusion, self).__init__()
        # 4 convs để chuyển đổi thông tin giữa các streams
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, in_channels, kernel_size=1) for _ in range(4)])
        self.norms = nn.ModuleList([nn.BatchNorm2d(in_channels) for _ in range(4)])

    def forward(self, features):
        """
        features: list of 4 tensors [B, C, H, W]
        return: list of 4 tensors sau khi đã trao đổi thông tin
        """
        B, C, H, W = features[0].shape
        out_features = []
        for i in range(4):
            # residual: stream i + tổng thông tin từ các stream khác
            res = features[i]
            for j in range(4):
                if i != j:
                    res = res + self.norms[i](self.convs[j](features[j]))
            out_features.append(res)
        return out_features
    
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Define the backbone
        self.base_model = BaseModel()
        
        self.cross_stream_fusion = CrossStreamFusion(in_channels=144)
        
        # Define the reduce in fusion
        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels=144 * 4, out_channels=144, kernel_size=1, padding=0),
            nn.BatchNorm2d(144),
            nn.GELU()
        )
        
        # Define the self attention part
        self.convolutional_self_attention = ConvolutionalSelfAttention(in_channels=144)
        
        # Define the adaptive pooling for desired output size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Define the position embedding
        self.patch_pos_embed = nn.ParameterList()
        for i in range(4):
            pos = torch.zeros(1, 144)
            for d in range(144):
                if d % 2 == 0:
                    pos[0, d] = np.sin(i / (10000 ** (2 * (d // 2) / 144)))
                else:
                    pos[0, d] = np.cos(i / (10000 ** (2 * (d // 2) / 144)))
            self.patch_pos_embed.append(nn.Parameter(pos, requires_grad=True))
            
        # Define the fully connected layer for yaw
        self.yaw_class = nn.Linear(144, 66)
        
        # Define the fully connected layer for pitch
        self.pitch_class = nn.Linear(144, 66)
        
        # Define the fully connected layer for roll
        self.roll_class = nn.Linear(144, 66)

    def forward(self, x_parts):
        x_features = []
        
        # 1. Go through 4 streams
        for i in range(4):
            _, _, x3 = self.base_model(x_parts[i]) # 1.1. Go through backbone - now is (100, 144, 8, 8)
            
            pos_embed = self.patch_pos_embed[i] # Get the positional embedding - [ [ ... ] ] - ... are total 144 values - size is (1, 144)
            
            if torch.isnan(pos_embed).any() or torch.isinf(pos_embed).any():
                raise ValueError(f"Positional embedding for patch {i} contains NaN or Inf values.")
            
            
            pos_embed = pos_embed[:, :, None, None].expand(-1, -1, 8, 8) # [1, 144, 8, 8]
            # pos_embed = pos_embed.unsqueeze(2).unsqueeze(3).expand(-1, -1, 8, 8)
            
            # 1.2. Add position embedding in this stream
            x3 = x3 + pos_embed
            
            x_features.append(x3)
            
        x_features = self.cross_stream_fusion(x_features)
            
        # 2. Fusion all features in 4 stream in to 1 
        x4 = torch.cat(x_features, dim=1) # 2.1. concatenate all 4 stream to have the shape of (100, 576, 8, 8)
        
        x4 = self.reduction_conv(x4) # 2.2. Go through a convolutional layer to reduce channel size to 144
        
        x4 = self.convolutional_self_attention(x4) # 2.3. Go through self-attention
        
        # 3. Multibin classification and regression
        x5 = self.pool(x4).flatten(1) # 3.1. pool down to (100, 144, 1, 1) => 3.2. flatten it to (100, 144)
        
        yaw_class = self.yaw_class(x5) # 3.2. di qua dense layer de du doan bin trong khoang -99 den 99
        pitch_class = self.pitch_class(x5)
        roll_class = self.roll_class(x5)
        
        return yaw_class, pitch_class, roll_class