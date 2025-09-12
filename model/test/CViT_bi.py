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
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion module inspired by CrossViT paper
    """
    def __init__(self, dim=144, num_heads=4, qkv_bias=True, proj_drop=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Multi-head attention components for each stream
        self.q_proj = nn.ModuleList([
            nn.Linear(dim, dim, bias=qkv_bias) for _ in range(4)
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(dim, dim, bias=qkv_bias) for _ in range(4)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(dim, dim, bias=qkv_bias) for _ in range(4)
        ])
        
        self.attn_drop = nn.Dropout(proj_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Layer norms for each stream
        self.norm1 = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(4)
        ])
        self.norm2 = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(4)
        ])
        
        # MLP blocks for each stream after attention
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
                nn.Dropout(proj_drop)
            ) for _ in range(4)
        ])
        
        # Final fusion layer
        self.fusion_norm = nn.LayerNorm(dim)
        self.fusion_proj = nn.Linear(dim * 4, dim)
        
    def forward(self, features_list):
        B = features_list[0].shape[0]
        N = features_list[0].shape[2] * features_list[0].shape[3]  # H*W
        fused_features = []
        
        # Reshape features to token format
        token_features = []
        for x in features_list:
            # Reshape from [B, C, H, W] to [B, H*W, C]
            tokens = x.flatten(2).transpose(1, 2)
            token_features.append(tokens)
        
        # Cross-attention between each pair of streams
        for i in range(4):  # For each stream as query
            tokens_i = token_features[i]
            tokens_i = self.norm1[i](tokens_i)
            
            # Compute query for current stream
            q = self.q_proj[i](tokens_i).reshape(B, N, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
            
            # Initialize accumulated attention output
            attn_out = 0
            
            # Attend to all streams (including self)
            for j in range(4):  # For each stream as key/value
                tokens_j = token_features[j]
                tokens_j = self.norm1[j](tokens_j)
                
                # Compute key and value for target stream
                k = self.k_proj[j](tokens_j).reshape(B, N, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
                v = self.v_proj[j](tokens_j).reshape(B, N, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
                
                # Compute attention
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                
                # Apply attention to values
                x_ij = (attn @ v).transpose(1, 2).reshape(B, N, self.dim)
                attn_out += x_ij / 4  # Average across all streams
            
            # Project output and apply residual connection
            attn_out = self.proj(attn_out)
            attn_out = self.proj_drop(attn_out)
            tokens_i = tokens_i + attn_out
            
            # Apply MLP and second residual
            tokens_i = tokens_i + self.mlp[i](self.norm2[i](tokens_i))
            
            # Reshape back to feature map format [B, C, H, W]
            H, W = features_list[i].shape[2], features_list[i].shape[3]
            fused_feature = tokens_i.transpose(1, 2).reshape(B, self.dim, H, W)
            fused_features.append(fused_feature)
        
        # Final fusion of all attended features
        # Convert to tokens for final fusion
        final_tokens = [f.flatten(2).transpose(1, 2) for f in fused_features]
        final_tokens = [self.fusion_norm(tokens) for tokens in final_tokens]
        final_tokens = torch.cat(final_tokens, dim=-1)  # [B, N, C*4]
        
        # Project to original dimension
        final_tokens = self.fusion_proj(final_tokens)  # [B, N, C]
        
        # Reshape back to feature map
        H, W = features_list[0].shape[2], features_list[0].shape[3]
        final_feature = final_tokens.transpose(1, 2).reshape(B, self.dim, H, W)
        
        return final_feature
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Define the backbone
        self.base_model = BaseModel()
        
        # Replace the fusion method with CrossViT-inspired cross-attention fusion
        self.cross_attention_fusion = CrossAttentionFusion(dim=144, num_heads=4)
        
        # Define the self attention part (kept from original model)
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
            _, _, x3 = self.base_model(x_parts[i]) # 1.1. Go through backbone - now is (B, 144, 8, 8)
            
            pos_embed = self.patch_pos_embed[i] # Get the positional embedding
            pos_embed = pos_embed[:, :, None, None].expand(-1, -1, 8, 8) # [1, 144, 8, 8]
            
            # 1.2. Add position embedding in this stream
            x3 = x3 + pos_embed
            
            x_features.append(x3)
        
        # 2. Use CrossViT fusion instead of concatenation + conv reduction
        x4 = self.cross_attention_fusion(x_features)
        
        # 3. Apply self-attention (kept from original model)
        x4 = self.convolutional_self_attention(x4)
        
        # 4. Multibin classification and regression
        x5 = self.pool(x4).flatten(1)
        
        yaw_class = self.yaw_class(x5)
        pitch_class = self.pitch_class(x5)
        roll_class = self.roll_class(x5)
        
        return yaw_class, pitch_class, roll_class