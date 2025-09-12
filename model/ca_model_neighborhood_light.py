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


class OptimizedBiDirectionalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(OptimizedBiDirectionalCrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Shared projections for efficiency
        self.qkv_proj = nn.Linear(dim, dim * 6)  # 6 = 3 (Q,K,V) * 2 (directions)
        
        # Learnable scaling parameters (initialized small for stability)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma_ffn = nn.Parameter(torch.zeros(1))
        
        # Efficient fusion with residual connection
        self.fusion_proj = nn.Linear(dim * 2, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Optimized FFN with GLU activation
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x_a, x_b):
        batch_size, seq_len, _ = x_a.shape
        
        # Single projection for all Q, K, V
        qkv_a = self.qkv_proj(x_a).chunk(6, dim=-1)
        qkv_b = self.qkv_proj(x_b).chunk(6, dim=-1)
        
        q_a, k_a, v_a = qkv_a[:3]
        q_b, k_b, v_b = qkv_b[:3]
        
        # Reshape for multi-head attention
        def reshape_for_attention(x):
            return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        q_a, k_a, v_a = map(reshape_for_attention, [q_a, k_a, v_a])
        q_b, k_b, v_b = map(reshape_for_attention, [q_b, k_b, v_b])
        
        # Efficient attention computation
        def compute_attention(q, k, v):
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
            return out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # Bi-directional attention
        x_a_to_b = compute_attention(q_a, k_b, v_b)
        x_b_to_a = compute_attention(q_b, k_a, v_a)
        
        # Efficient fusion with residual connection
        fused = self.fusion_proj(torch.cat([x_a_to_b, x_b_to_a], dim=-1))
        x_enhanced = x_a + torch.tanh(self.gamma) * fused
        x_enhanced = self.norm1(x_enhanced)
        
        # FFN with residual
        ffn_out = self.ffn(x_enhanced)
        x_enhanced = x_enhanced + torch.tanh(self.gamma_ffn) * ffn_out
        x_enhanced = self.norm2(x_enhanced)
        
        return x_enhanced


class OptimizedNeighborhoodFusionTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(OptimizedNeighborhoodFusionTransformer, self).__init__()
        
        # Single shared cross-attention module for efficiency
        self.cross_attn = OptimizedBiDirectionalCrossAttention(dim, num_heads, dropout)
        
        # Neighbor mapping for 2x2 grid topology
        self.neighbor_pairs = [(0, 1), (0, 3), (1, 2), (2, 3)]  # (part1-part2), (part1-part4), etc.
        
        # Learnable attention weights for neighbor aggregation
        self.neighbor_weights = nn.Parameter(torch.ones(4, 2) * 0.5)  # Each part has 2 neighbors
        
        # Efficient fusion layers
        self.pre_fusion_norm = nn.LayerNorm(dim)
        self.fusion_proj = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        
        # Self-attention for global context
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.final_norm = nn.LayerNorm(dim)
        
    def _reshape_to_sequence(self, x):
        """Optimized reshape with memory efficiency"""
        b, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(b, h * w, c)
    
    def _reshape_to_spatial(self, x, h, w):
        """Optimized reshape back to spatial"""
        b, seq_len, c = x.shape
        return x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    
    def forward(self, features):
        f1, f2, f3, f4 = features
        b, c, h, w = f1.shape
        
        # Convert to sequence format
        feature_seqs = [self._reshape_to_sequence(f) for f in features]
        
        # Compute neighbor interactions efficiently
        enhanced_features = [torch.zeros_like(seq) for seq in feature_seqs]
        
        # Process neighbor pairs
        for i, (idx1, idx2) in enumerate(self.neighbor_pairs):
            # Bi-directional cross-attention
            enhanced_1 = self.cross_attn(feature_seqs[idx1], feature_seqs[idx2])
            enhanced_2 = self.cross_attn(feature_seqs[idx2], feature_seqs[idx1])
            
            # Accumulate enhancements
            enhanced_features[idx1] = enhanced_features[idx1] + enhanced_1
            enhanced_features[idx2] = enhanced_features[idx2] + enhanced_2
        
        # Weighted combination with original features
        for i in range(4):
            neighbor_count = len([p for p in self.neighbor_pairs if i in p])
            if neighbor_count > 0:
                enhanced_features[i] = enhanced_features[i] / neighbor_count
                # Residual connection with original features
                enhanced_features[i] = feature_seqs[i] + enhanced_features[i]
            else:
                enhanced_features[i] = feature_seqs[i]
            
            enhanced_features[i] = self.pre_fusion_norm(enhanced_features[i])
        
        # Global fusion
        all_features = torch.cat(enhanced_features, dim=-1)
        fused_features = self.fusion_proj(all_features)
        
        # Self-attention for global context
        fused_features, _ = self.self_attn(fused_features, fused_features, fused_features)
        fused_features = self.final_norm(fused_features)
        
        # Reshape back to spatial
        output = self._reshape_to_spatial(fused_features, h, w)
        return output


class OptimizedModel(nn.Module):
    def __init__(self, num_classes=66):
        super(OptimizedModel, self).__init__()
        
        # Assuming BaseModel is provided
        self.base_model = BaseModel()
        
        # Optimized neighborhood fusion
        self.neighborhood_fusion = OptimizedNeighborhoodFusionTransformer(
            dim=144, num_heads=8, dropout=0.1
        )
        
        # Learnable position embeddings (more efficient than sinusoidal)
        self.patch_pos_embed = nn.Parameter(
            torch.randn(4, 144, 8, 8) * 0.02
        )
        
        # Efficient pooling and prediction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Shared prediction head with better regularization
        self.prediction_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(144, 144 // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Separate final layers for each angle
        self.yaw_head = nn.Linear(144 // 2, num_classes)
        self.pitch_head = nn.Linear(144 // 2, num_classes)
        self.roll_head = nn.Linear(144 // 2, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Improved weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x_parts):
        features = []
        
        # Process each part through backbone
        for i, x_part in enumerate(x_parts):
            _, _, x3 = self.base_model(x_part)
            
            # Add learnable position embedding
            pos_embed = self.patch_pos_embed[i:i+1]
            x3 = x3 + pos_embed
            
            features.append(x3)
        
        # Apply neighborhood fusion
        fused_features = self.neighborhood_fusion(features)
        
        # Global pooling and prediction
        pooled = self.global_pool(fused_features).flatten(1)
        
        # Shared feature extraction
        shared_features = self.prediction_head(pooled)
        
        # Individual angle predictions
        yaw_pred = self.yaw_head(shared_features)
        pitch_pred = self.pitch_head(shared_features)
        roll_pred = self.roll_head(shared_features)
        
        return yaw_pred, pitch_pred, roll_pred