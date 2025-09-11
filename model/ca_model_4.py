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
import math

class ImprovedBiDirectionalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, temperature=1.0):
        super(ImprovedBiDirectionalCrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        
        # Pre-normalization for better training stability
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Projections for Query, Key, Value for both directions
        self.to_q_a = nn.Linear(dim, dim, bias=False)
        self.to_k_a = nn.Linear(dim, dim, bias=False)
        self.to_v_a = nn.Linear(dim, dim, bias=False)
        
        self.to_q_b = nn.Linear(dim, dim, bias=False)
        self.to_k_b = nn.Linear(dim, dim, bias=False)
        self.to_v_b = nn.Linear(dim, dim, bias=False)
        
        # Learnable scaling parameters
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)
        self.beta = nn.Parameter(torch.ones(1) * 0.1)
        
        # Enhanced fusion network
        self.fusion_proj = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for module in [self.to_q_a, self.to_k_a, self.to_v_a, 
                      self.to_q_b, self.to_k_b, self.to_v_b]:
            nn.init.xavier_uniform_(module.weight)
            
    def forward(self, x_a, x_b):
        # Pre-normalization
        x_a_norm = self.norm1(x_a)
        x_b_norm = self.norm2(x_b)
        
        batch_size, seq_len, _ = x_a.shape
        
        # Direction A→B: x_a attends to x_b
        q_a = self.to_q_a(x_a_norm).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_b = self.to_k_b(x_b_norm).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_b = self.to_v_b(x_b_norm).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention with temperature scaling
        attn_a_to_b = torch.matmul(q_a, k_b.transpose(-1, -2)) * self.scale / self.temperature
        attn_a_to_b = F.softmax(attn_a_to_b, dim=-1)
        attn_a_to_b = self.dropout(attn_a_to_b)
        x_a_attends_to_b = torch.matmul(attn_a_to_b, v_b).permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        # Direction B→A: x_b attends to x_a
        q_b = self.to_q_b(x_b_norm).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_a = self.to_k_a(x_a_norm).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_a = self.to_v_a(x_a_norm).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn_b_to_a = torch.matmul(q_b, k_a.transpose(-1, -2)) * self.scale / self.temperature
        attn_b_to_a = F.softmax(attn_b_to_a, dim=-1)
        attn_b_to_a = self.dropout(attn_b_to_a)
        x_b_attends_to_a = torch.matmul(attn_b_to_a, v_a).permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        # Enhanced fusion with learnable scaling
        fused_features = self.fusion_proj(torch.cat([x_a_attends_to_b, x_b_attends_to_a], dim=-1))
        x_a_enhanced = x_a + self.alpha * fused_features
        
        return x_a_enhanced

class SpatialRelationModule(nn.Module):
    def __init__(self, dim):
        super(SpatialRelationModule, self).__init__()
        self.dim = dim
        
        # Relative position encoding
        self.rel_pos_h = nn.Parameter(torch.randn(2 * 8 - 1, dim // 2))
        self.rel_pos_w = nn.Parameter(torch.randn(2 * 8 - 1, dim // 2))
        
        # Spatial transformation network
        self.spatial_transform = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim)
        )
        
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        h = w = int(math.sqrt(seq_len))
        
        # Add relative position encoding
        x = x.view(batch_size, h, w, dim)
        
        # Apply spatial transformation
        x_flat = x.view(batch_size, seq_len, dim)
        spatial_weights = torch.sigmoid(self.spatial_transform(x_flat))
        x_enhanced = x_flat * spatial_weights
        
        return x_enhanced

class MultiScaleFusionModule(nn.Module):
    def __init__(self, dim):
        super(MultiScaleFusionModule, self).__init__()
        self.dim = dim
        
        # Multi-scale 1D convolutions
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=1)
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(dim, dim, kernel_size=5, padding=2)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        x_t = x.transpose(1, 2)  # [batch_size, dim, seq_len]
        
        out1 = self.conv1(x_t).transpose(1, 2)
        out3 = self.conv3(x_t).transpose(1, 2)
        out5 = self.conv5(x_t).transpose(1, 2)
        
        # Concatenate and fuse
        multi_scale = torch.cat([out1, out3, out5], dim=-1)
        fused = self.fusion(multi_scale)
        
        # Residual connection
        return fused + x

class AdaptiveNeighborhoodFusionTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(AdaptiveNeighborhoodFusionTransformer, self).__init__()
        
        # Multi-layer cross-attention for better interaction
        self.cross_attn_layers = nn.ModuleList([
            ImprovedBiDirectionalCrossAttention(dim, num_heads, dropout) 
            for _ in range(2)
        ])
        
        # Spatial relation module
        self.spatial_relation = SpatialRelationModule(dim)
        
        # Multi-scale fusion
        self.multi_scale_fusion = MultiScaleFusionModule(dim)
        
        # Gating mechanism for adaptive fusion
        self.gate = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.Sigmoid()
        )
        
        # Learnable weights for adaptive pooling
        self.adaptive_weights = nn.Parameter(torch.ones(4) / 4)
        
        # Enhanced self-attention
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_final = nn.LayerNorm(dim)
        
        # Cross-attention between all pairs
        self.cross_attns = nn.ModuleDict()
        pairs = [(1, 2), (1, 4), (2, 3), (3, 4)]
        for i, (a, b) in enumerate(pairs):
            self.cross_attns[f'{a}_{b}'] = ImprovedBiDirectionalCrossAttention(dim, num_heads, dropout)
    
    def _reshape_to_sequence(self, x):
        """Reshape spatial features to sequence format for transformer"""
        batch_size, channels, height, width = x.shape
        return x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
    
    def _reshape_to_spatial(self, x, height, width):
        """Reshape sequence back to spatial format"""
        batch_size, seq_len, channels = x.shape
        return x.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
    
    def forward(self, features):
        f1, f2, f3, f4 = features
        batch_size, channels, height, width = f1.shape
        
        # Reshape to sequence format
        f1_seq = self._reshape_to_sequence(f1)
        f2_seq = self._reshape_to_sequence(f2)
        f3_seq = self._reshape_to_sequence(f3)
        f4_seq = self._reshape_to_sequence(f4)
        
        # Apply spatial relation module
        f1_seq = self.spatial_relation(f1_seq)
        f2_seq = self.spatial_relation(f2_seq)
        f3_seq = self.spatial_relation(f3_seq)
        f4_seq = self.spatial_relation(f4_seq)
        
        # Multi-layer cross-attention for each pair
        enhanced_features = [f1_seq, f2_seq, f3_seq, f4_seq]
        
        for layer in self.cross_attn_layers:
            # Apply cross-attention between neighboring parts
            f1_enhanced = []
            f1_enhanced.append(layer(enhanced_features[0], enhanced_features[1]))  # f1 with f2
            f1_enhanced.append(layer(enhanced_features[0], enhanced_features[3]))  # f1 with f4
            f1_final = sum(f1_enhanced) / len(f1_enhanced)
            
            f2_enhanced = []
            f2_enhanced.append(layer(enhanced_features[1], enhanced_features[0]))  # f2 with f1
            f2_enhanced.append(layer(enhanced_features[1], enhanced_features[2]))  # f2 with f3
            f2_final = sum(f2_enhanced) / len(f2_enhanced)
            
            f3_enhanced = []
            f3_enhanced.append(layer(enhanced_features[2], enhanced_features[1]))  # f3 with f2
            f3_enhanced.append(layer(enhanced_features[2], enhanced_features[3]))  # f3 with f4
            f3_final = sum(f3_enhanced) / len(f3_enhanced)
            
            f4_enhanced = []
            f4_enhanced.append(layer(enhanced_features[3], enhanced_features[0]))  # f4 with f1
            f4_enhanced.append(layer(enhanced_features[3], enhanced_features[2]))  # f4 with f3
            f4_final = sum(f4_enhanced) / len(f4_enhanced)
            
            enhanced_features = [f1_final, f2_final, f3_final, f4_final]
        
        # Apply multi-scale fusion to each enhanced feature
        enhanced_features = [self.multi_scale_fusion(feat) for feat in enhanced_features]
        
        # Adaptive weighted fusion
        weights = F.softmax(self.adaptive_weights, dim=0)
        weighted_features = [w * feat for w, feat in zip(weights, enhanced_features)]
        
        # Concatenate for gating
        all_features = torch.cat(enhanced_features, dim=-1)
        gate_weights = self.gate(all_features)
        
        # Final fusion with gating
        final_features = sum(weighted_features)
        final_features = final_features * gate_weights
        
        # Self-attention for final refinement
        final_features, _ = self.self_attn(final_features, final_features, final_features)
        final_features = self.norm_final(final_features)
        
        # Reshape back to spatial format
        output = self._reshape_to_spatial(final_features, height, width)
        
        return output
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Define the backbone
        self.base_model = BaseModel()
        
        # Define the neighborhood fusion with transformer-based bi-directional cross-attention
        self.neighborhood_fusion = AdaptiveNeighborhoodFusionTransformer(dim=144, num_heads=8, dropout=0.1)
        
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
        
        # Define the adaptive pooling for desired output size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Define the fully connected layers for angle prediction
        self.yaw_class = nn.Linear(144, 66)
        self.pitch_class = nn.Linear(144, 66)
        self.roll_class = nn.Linear(144, 66)

    def forward(self, x_parts):
        x_features = []
        
        # 1. Process each part through the backbone
        for i in range(4):
            _, _, x3 = self.base_model(x_parts[i])
            
            # Add position embedding
            pos_embed = self.patch_pos_embed[i]
            if torch.isnan(pos_embed).any() or torch.isinf(pos_embed).any():
                raise ValueError(f"Positional embedding for patch {i} contains NaN or Inf values.")
            
            pos_embed = pos_embed[:, :, None, None].expand(-1, -1, 8, 8)
            x3 = x3 + pos_embed
            
            x_features.append(x3)
        
        # 2. Apply neighborhood fusion with transformer-based bi-directional cross-attention
        x4 = self.neighborhood_fusion(x_features)
        
        # 3. Pooling and prediction
        x5 = self.pool(x4).flatten(1)
        
        yaw_class = self.yaw_class(x5)
        pitch_class = self.pitch_class(x5)
        roll_class = self.roll_class(x5)
        
        return yaw_class, pitch_class, roll_class