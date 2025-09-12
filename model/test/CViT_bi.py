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
    
class BiDirectionalCrossAttention(nn.Module):
    def __init__(self, dim=144, num_streams=4, num_heads=4, mlp_ratio=4.0, qkv_bias=True, 
                 drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()
        
        # Tạo cross-attention modules cho từng cặp streams (i -> j)
        self.cross_attns = nn.ModuleList()
        for i in range(num_streams):
            stream_attns = nn.ModuleList()
            for j in range(num_streams):
                if i != j:  # Chỉ tạo cross-attention giữa các streams khác nhau
                    stream_attns.append(
                        nn.Sequential(
                            nn.LayerNorm(dim),
                            Attention(
                                dim=dim,
                                num_heads=num_heads,
                                qkv_bias=qkv_bias,
                                attn_drop=attn_drop_rate,
                                proj_drop=drop_rate
                            )
                        )
                    )
                else:
                    stream_attns.append(None)
            self.cross_attns.append(stream_attns)
        
        # MLP sau mỗi cross-attention
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                Mlp(
                    in_features=dim,
                    hidden_features=int(dim * mlp_ratio),
                    drop=drop_rate
                )
            ) for _ in range(num_streams)
        ])
        
    def forward(self, streams):
        batch_size = streams[0].shape[0]
        num_streams = len(streams)
        
        # Chuyển streams về dạng tokens để dễ xử lý
        tokens = []
        for stream in streams:
            # Global average pooling
            token = stream.mean(dim=(-1, -2))  # [B, C]
            tokens.append(token)
        
        # Bi-directional cross-attention
        updated_tokens = [token.clone() for token in tokens]  # Copy tokens để lưu kết quả
        
        # Bước 1: Áp dụng cross-attention cho mỗi cặp streams
        cross_attended_tokens = []
        for i in range(num_streams):
            cross_attns_for_i = []
            for j in range(num_streams):
                if i != j:
                    # Token của stream i lấy thông tin từ stream j
                    cross_attn = self.cross_attns[i][j]
                    
                    # Tạo input cho cross-attention (target token ở đầu)
                    cross_input = torch.stack([tokens[i], tokens[j]], dim=1)  # [B, 2, C]
                    
                    # Áp dụng attention và chỉ lấy kết quả của token đầu tiên
                    cross_result = cross_attn(cross_input)[:, 0]  # [B, C]
                    cross_attns_for_i.append(cross_result)
            
            # Tổng hợp tất cả cross-attention results cho stream i
            if cross_attns_for_i:
                cross_sum = sum(cross_attns_for_i)
                cross_attended_tokens.append(tokens[i] + cross_sum)
            else:
                cross_attended_tokens.append(tokens[i])
        
        # Bước 2: Áp dụng MLP sau cross-attention
        final_tokens = []
        for i in range(num_streams):
            token_with_mlp = cross_attended_tokens[i] + self.mlps[i](cross_attended_tokens[i])
            final_tokens.append(token_with_mlp)
        
        # Chuyển tokens trở lại dạng spatial features
        enhanced_streams = []
        for i in range(num_streams):
            h, w = streams[i].shape[-2:]
            token_spatial = final_tokens[i].view(batch_size, -1, 1, 1).expand(-1, -1, h, w)
            enhanced_streams.append(streams[i] + token_spatial)
            
        return enhanced_streams


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Tính toán q, k, v riêng biệt
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Tính attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Áp dụng attention
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Giữ nguyên các thành phần khác
        self.base_model = BaseModel()
        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels=144 * 4, out_channels=144, kernel_size=1, padding=0),
            nn.BatchNorm2d(144),
            nn.GELU()
        )
        self.convolutional_self_attention = ConvolutionalSelfAttention(in_channels=144)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Thêm bi-directional cross-attention
        self.bidirectional_cross_attn = BiDirectionalCrossAttention(
            dim=144, 
            num_streams=4,
            num_heads=4
        )
        
        # Positional embeddings
        self.patch_pos_embed = nn.ParameterList()
        for i in range(4):
            pos = torch.zeros(1, 144)
            for d in range(144):
                if d % 2 == 0:
                    pos[0, d] = np.sin(i / (10000 ** (2 * (d // 2) / 144)))
                else:
                    pos[0, d] = np.cos(i / (10000 ** (2 * (d // 2) / 144)))
            self.patch_pos_embed.append(nn.Parameter(pos, requires_grad=True))
        
        # Output heads
        self.yaw_class = nn.Linear(144, 66)
        self.pitch_class = nn.Linear(144, 66)
        self.roll_class = nn.Linear(144, 66)
        
    def forward(self, x_parts):
        x_features = []
        
        # 1. Extract features from each stream
        for i in range(4):
            _, _, x3 = self.base_model(x_parts[i])
            
            # Add positional embedding
            pos_embed = self.patch_pos_embed[i]
            pos_embed = pos_embed[:, :, None, None].expand(-1, -1, 8, 8)
            x3 = x3 + pos_embed
            
            x_features.append(x3)
        
        # 2. Apply bi-directional cross-attention
        enhanced_features = self.bidirectional_cross_attn(x_features)
        
        # 3. Concatenate enhanced features
        x4 = torch.cat(enhanced_features, dim=1)
        
        # 4. Continue with the rest of the model
        x4 = self.reduction_conv(x4)
        x4 = self.convolutional_self_attention(x4)
        x5 = self.pool(x4).flatten(1)
        
        yaw_class = self.yaw_class(x5)
        pitch_class = self.pitch_class(x5)
        roll_class = self.roll_class(x5)
        
        return yaw_class, pitch_class, roll_class