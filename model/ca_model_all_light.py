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


class OptimizedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(OptimizedCrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Single projection for Q, K, V for efficiency
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        """
        Optimized cross-attention computation
        Args:
            query: [B, C, H, W]
            key_value: [B, C, H, W]
        Returns:
            attended_features: [B, C, H, W]
        """
        B, C, H, W = query.shape
        seq_len = H * W
        
        # Reshape to sequence format
        query_seq = query.flatten(2).transpose(1, 2)  # [B, H*W, C]
        kv_seq = key_value.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Compute Q from query, K,V from key_value
        q = self.qkv_proj(query_seq)[:, :, :C]  # [B, H*W, C]
        k = self.qkv_proj(kv_seq)[:, :, C:2*C]  # [B, H*W, C]
        v = self.qkv_proj(kv_seq)[:, :, 2*C:]   # [B, H*W, C]
        
        # Reshape for multi-head attention
        def reshape_for_heads(x):
            return x.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k, v = map(reshape_for_heads, [q, k, v])
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention and reshape back
        out = torch.matmul(attn, v)  # [B, num_heads, H*W, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, seq_len, C)
        
        # Reshape back to spatial format
        out = out.transpose(1, 2).view(B, C, H, W)
        
        return out


class OptimizedBidirectionalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(OptimizedBidirectionalCrossAttention, self).__init__()
        self.cross_attn = OptimizedCrossAttention(dim, num_heads, dropout)
        
        # Efficient fusion with depthwise separable conv
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, groups=dim, bias=False),  # depthwise
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),  # pointwise
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        # Learnable scaling with better initialization
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Layer normalization for better training stability
        self.norm = nn.GroupNorm(num_groups=8, num_channels=dim, eps=1e-6)
        
    def forward(self, x1, x2):
        """
        Optimized bidirectional cross-attention
        """
        # Normalize inputs
        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)
        
        # Bidirectional attention
        x1_to_x2 = self.cross_attn(x1_norm, x2_norm)
        x2_to_x1 = self.cross_attn(x2_norm, x1_norm)
        
        # Efficient fusion
        x_cat = torch.cat([x1_to_x2, x2_to_x1], dim=1)
        x_fused = self.fusion_conv(x_cat)
        
        # Scaled residual connection
        x1_out = x1 + torch.tanh(self.gamma) * x_fused
        
        return x1_out


class OptimizedCrossStreamFusion(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(OptimizedCrossStreamFusion, self).__init__()
        self.dim = dim
        self.num_streams = 4
        
        # Shared bidirectional cross-attention module for efficiency
        self.shared_cross_attn = OptimizedBidirectionalCrossAttention(dim, num_heads, dropout)
        
        # Stream-specific adaptation layers (lightweight)
        self.stream_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                nn.GroupNorm(8, dim),
                nn.GELU()
            ) for _ in range(4)
        ])
        
        # Efficient fusion with attention-based weighting
        self.fusion_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 4, dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, 4, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        self.final_fusion = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=3, padding=1, groups=dim),  # depthwise
            nn.Conv2d(dim, dim, kernel_size=1),  # pointwise
            nn.GroupNorm(8, dim),
            nn.GELU()
        )
        
        # Feed-forward network with GLU activation
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        self.gamma_ffn = nn.Parameter(torch.zeros(1))
        
    def forward(self, features):
        """
        Optimized cross-stream fusion similar to neighborhood approach
        Args:
            features: list of 4 feature maps [B, C, H, W]
        Returns:
            fused_features: [B, C, H, W]
        """
        B, C, H, W = features[0].shape
        enhanced_features = []
        
        # Process each stream with its neighbors (similar to neighborhood approach)
        for i in range(self.num_streams):
            master = features[i]
            cross_attended_features = []
            
            # Apply stream-specific adaptation
            adapted_master = self.stream_adapters[i](master)
            
            # Cross-attend with all other streams (like neighborhood with 3 neighbors)
            for j in range(self.num_streams):
                if j != i:
                    # Use shared cross-attention module
                    attended = self.shared_cross_attn(adapted_master, features[j])
                    cross_attended_features.append(attended)
            
            # Combine cross-attended features (average like in neighborhood)
            if cross_attended_features:
                master_enhanced = torch.stack(cross_attended_features).mean(dim=0)
                enhanced_features.append(master_enhanced)
            else:
                enhanced_features.append(adapted_master)
        
        # Global fusion with attention weighting
        all_features = torch.cat(enhanced_features, dim=1)  # [B, 4*C, H, W]
        
        # Compute attention weights for each stream
        attn_weights = self.fusion_attention(all_features)  # [B, 4, 1, 1]
        
        # Weighted combination
        weighted_features = []
        for i in range(4):
            weighted = enhanced_features[i] * attn_weights[:, i:i+1, :, :]
            weighted_features.append(weighted)
        
        # Final fusion
        fused = torch.cat(weighted_features, dim=1)
        fused = self.final_fusion(fused)
        
        # Apply feed-forward with residual connection
        fused = fused + torch.tanh(self.gamma_ffn) * self.ffn(fused)
        
        return fused


class OptimizedModel(nn.Module):
    def __init__(self, num_classes=66):
        super(OptimizedModel, self).__init__()
        
        # Backbone (assuming BaseModel is provided)
        self.base_model = BaseModel()
        
        # Optimized cross-stream fusion
        self.cross_stream_fusion = OptimizedCrossStreamFusion(dim=144, num_heads=8, dropout=0.1)
        
        # Self-attention (assuming ConvolutionalSelfAttention is provided)
        self.convolutional_self_attention = ConvolutionalSelfAttention(in_channels=144)
        
        # Learnable position embeddings (more efficient than sinusoidal)
        self.patch_pos_embed = nn.Parameter(
            torch.randn(4, 144, 1, 1) * 0.02  # Broadcasting-friendly shape
        )
        
        # Efficient pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Improved prediction heads with shared backbone
        self.shared_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(144, 144 // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Individual angle predictors
        self.yaw_head = nn.Linear(144 // 2, num_classes)
        self.pitch_head = nn.Linear(144 // 2, num_classes)
        self.roll_head = nn.Linear(144 // 2, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights properly"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d,)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x_parts):
        features = []
        
        # Process each stream through backbone
        for i, x_part in enumerate(x_parts):
            _, _, x3 = self.base_model(x_part)
            
            # Add learnable position embedding (broadcasting)
            pos_embed = self.patch_pos_embed[i:i+1]  # [1, 144, 1, 1]
            x3 = x3 + pos_embed.expand_as(x3)  # Broadcast to [B, 144, H, W]
            
            features.append(x3)
        
        # Apply cross-stream fusion (1 stream with 3 others, similar to neighborhood)
        fused = self.cross_stream_fusion(features)
        
        # Apply self-attention for refinement
        refined = self.convolutional_self_attention(fused)
        
        # Global pooling and prediction
        pooled = self.global_pool(refined).flatten(1)  # [B, 144]
        
        # Shared feature processing
        shared_features = self.shared_head(pooled)
        
        # Individual angle predictions
        yaw_pred = self.yaw_head(shared_features)
        pitch_pred = self.pitch_head(shared_features)
        roll_pred = self.roll_head(shared_features)
        
        return yaw_pred, pitch_pred, roll_pred