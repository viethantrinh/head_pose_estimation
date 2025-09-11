import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

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

class EnhancedBiDirectionalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(EnhancedBiDirectionalCrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Pre-normalization cho stability
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        
        # Projections với improved initialization
        self.to_q_a = nn.Linear(dim, dim, bias=False)
        self.to_k_a = nn.Linear(dim, dim, bias=False)
        self.to_v_a = nn.Linear(dim, dim, bias=False)
        
        self.to_q_b = nn.Linear(dim, dim, bias=False)
        self.to_k_b = nn.Linear(dim, dim, bias=False)
        self.to_v_b = nn.Linear(dim, dim, bias=False)
        
        # Temperature parameter for attention sharpness control
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
        # Learnable scaling parameters with better initialization
        self.alpha = nn.Parameter(torch.zeros(1))  # For cross-attention
        self.beta = nn.Parameter(torch.zeros(1))   # For FFN
        
        # Enhanced fusion network
        self.cross_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # Multi-layer FFN for better feature transformation
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_a, x_b):
        batch_size, seq_len, _ = x_a.shape
        
        # Pre-normalization
        x_a_norm = self.norm_a(x_a)
        x_b_norm = self.norm_b(x_b)
        
        # Compute Q, K, V for both directions
        q_a = self.to_q_a(x_a_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_b = self.to_k_b(x_b_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_b = self.to_v_b(x_b_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        q_b = self.to_q_b(x_b_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_a = self.to_k_a(x_a_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_a = self.to_v_a(x_a_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation với temperature scaling
        # A → B attention
        attn_a_to_b = torch.matmul(q_a, k_b.transpose(-2, -1)) * self.scale / self.temperature.clamp(min=0.01)
        attn_a_to_b = F.softmax(attn_a_to_b, dim=-1)
        attn_a_to_b = F.dropout(attn_a_to_b, p=0.1, training=self.training)
        
        # B → A attention  
        attn_b_to_a = torch.matmul(q_b, k_a.transpose(-2, -1)) * self.scale / self.temperature.clamp(min=0.01)
        attn_b_to_a = F.softmax(attn_b_to_a, dim=-1)
        attn_b_to_a = F.dropout(attn_b_to_a, p=0.1, training=self.training)
        
        # Apply attention
        x_a_attends_to_b = torch.matmul(attn_a_to_b, v_b).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        x_b_attends_to_a = torch.matmul(attn_b_to_a, v_a).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Cross-modal fusion
        cross_attended = self.cross_fusion(torch.cat([x_a_attends_to_b, x_b_attends_to_a], dim=-1))
        
        # Residual connection với learnable scaling
        x_enhanced = x_a + torch.tanh(self.alpha) * cross_attended
        
        # FFN với residual connection
        ffn_out = self.ffn(x_enhanced)
        x_enhanced = x_enhanced + torch.tanh(self.beta) * ffn_out
        
        # Final output projection
        output = self.output_proj(x_enhanced)
        
        return output

class SpatialRelationModule(nn.Module):
    """Module để học spatial relationships giữa các patches"""
    def __init__(self, dim):
        super(SpatialRelationModule, self).__init__()
        self.dim = dim
        
        # Relative position encoding
        self.relative_pos_embed = nn.Parameter(torch.randn(4, 4, dim) * 0.02)
        
        # Spatial transformation network
        self.spatial_transform = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid()
        )
        
    def forward(self, features, patch_indices):
        """
        features: List of 4 patch features [B, L, D]
        patch_indices: [0, 1, 2, 3] for patches 1, 2, 3, 4
        """
        enhanced_features = []
        
        for i, feat in enumerate(features):
            # Add relative position encoding
            pos_encoding = self.relative_pos_embed[i].unsqueeze(0).expand(feat.size(0), -1, -1)
            
            # Spatial transformation
            spatial_weight = self.spatial_transform(feat)
            
            # Apply spatial enhancement
            enhanced_feat = feat + pos_encoding * spatial_weight
            enhanced_features.append(enhanced_feat)
            
        return enhanced_features

class MultiScaleFusionModule(nn.Module):
    """Multi-scale fusion để capture different levels of information"""
    def __init__(self, dim, scales=[1, 2, 4]):
        super(MultiScaleFusionModule, self).__init__()
        self.scales = scales
        self.dim = dim
        
        # Multi-scale convolutions
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=scale, padding=scale//2, groups=dim//8),
                nn.BatchNorm1d(dim),
                nn.GELU()
            ) for scale in scales
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(dim * len(scales), dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x):
        """x: [B, L, D]"""
        # Transpose for 1D conv: [B, D, L]
        x_conv = x.transpose(1, 2)
        
        # Apply multi-scale convolutions
        scale_outputs = []
        for conv in self.scale_convs:
            scale_out = conv(x_conv).transpose(1, 2)  # Back to [B, L, D]
            scale_outputs.append(scale_out)
        
        # Concatenate and fuse
        multi_scale = torch.cat(scale_outputs, dim=-1)
        fused = self.fusion(multi_scale)
        
        return fused + x  # Residual connection

class AdaptiveNeighborhoodFusionTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, num_layers=2):
        super(AdaptiveNeighborhoodFusionTransformer, self).__init__()
        
        # Spatial relation module
        self.spatial_relation = SpatialRelationModule(dim)
        
        # Multi-scale fusion
        self.multiscale_fusion = MultiScaleFusionModule(dim)
        
        # Adaptive cross-attention layers
        self.num_layers = num_layers
        self.cross_attention_layers = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            layer = nn.ModuleDict({
                'cross_attn_1_2': EnhancedBiDirectionalCrossAttention(dim, num_heads, dropout),
                'cross_attn_1_4': EnhancedBiDirectionalCrossAttention(dim, num_heads, dropout),
                'cross_attn_2_3': EnhancedBiDirectionalCrossAttention(dim, num_heads, dropout),
                'cross_attn_3_4': EnhancedBiDirectionalCrossAttention(dim, num_heads, dropout),
                'layer_norm': nn.LayerNorm(dim),
                'gate': nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.Sigmoid()
                )
            })
            self.cross_attention_layers.append(layer)
        
        # Global fusion network
        self.global_fusion = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
        
        # Self-attention for final refinement
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(dim)
        
        # Adaptive pooling weights
        self.pool_weights = nn.Parameter(torch.ones(4) / 4)
        
    def _reshape_to_sequence(self, x):
        """Reshape spatial features to sequence format"""
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
        f_seqs = [
            self._reshape_to_sequence(f1),
            self._reshape_to_sequence(f2),
            self._reshape_to_sequence(f3),
            self._reshape_to_sequence(f4)
        ]
        
        # Apply spatial relation enhancement
        f_seqs = self.spatial_relation(f_seqs, [0, 1, 2, 3])
        
        # Apply multi-scale fusion to each patch
        for i in range(4):
            f_seqs[i] = self.multiscale_fusion(f_seqs[i])
        
        # Multi-layer cross-attention with adaptive gating
        for layer in self.cross_attention_layers:
            # Store original features for residual
            f1_orig, f2_orig, f3_orig, f4_orig = f_seqs
            
            # Cross-attention between neighbors
            f1_with_2 = layer['cross_attn_1_2'](f_seqs[0], f_seqs[1])
            f1_with_4 = layer['cross_attn_1_4'](f_seqs[0], f_seqs[3])
            
            f2_with_1 = layer['cross_attn_1_2'](f_seqs[1], f_seqs[0])
            f2_with_3 = layer['cross_attn_2_3'](f_seqs[1], f_seqs[2])
            
            f3_with_2 = layer['cross_attn_2_3'](f_seqs[2], f_seqs[1])
            f3_with_4 = layer['cross_attn_3_4'](f_seqs[2], f_seqs[3])
            
            f4_with_1 = layer['cross_attn_1_4'](f_seqs[3], f_seqs[0])
            f4_with_3 = layer['cross_attn_3_4'](f_seqs[3], f_seqs[2])
            
            # Adaptive fusion với learned weights
            f1_enhanced = (f1_with_2 + f1_with_4) / 2
            f2_enhanced = (f2_with_1 + f2_with_3) / 2
            f3_enhanced = (f3_with_2 + f3_with_4) / 2
            f4_enhanced = (f4_with_1 + f4_with_3) / 2
            
            # Apply gating mechanism
            gate1 = layer['gate'](f1_enhanced)
            gate2 = layer['gate'](f2_enhanced)
            gate3 = layer['gate'](f3_enhanced)
            gate4 = layer['gate'](f4_enhanced)
            
            # Gated residual connection
            f_seqs[0] = layer['layer_norm'](f1_orig + gate1 * f1_enhanced)
            f_seqs[1] = layer['layer_norm'](f2_orig + gate2 * f2_enhanced)
            f_seqs[2] = layer['layer_norm'](f3_orig + gate3 * f3_enhanced)
            f_seqs[3] = layer['layer_norm'](f4_orig + gate4 * f4_enhanced)
        
        # Global fusion với adaptive pooling
        pool_weights_norm = F.softmax(self.pool_weights, dim=0)
        weighted_features = []
        
        for i, (feat, weight) in enumerate(zip(f_seqs, pool_weights_norm)):
            weighted_features.append(feat * weight.view(1, 1, 1))
        
        # Concatenate all features
        all_features = torch.cat(weighted_features, dim=-1)
        global_fused = self.global_fusion(all_features)
        
        # Self-attention for final refinement
        global_fused_attn, _ = self.self_attn(global_fused, global_fused, global_fused)
        final_output = self.self_attn_norm(global_fused + global_fused_attn)
        
        # Reshape back to spatial format
        output = self._reshape_to_spatial(final_output, height, width)
        
        return output

# Updated Model với improved neighborhood fusion
class ImprovedModel(nn.Module):
    def __init__(self):
        super(ImprovedModel, self).__init__()
        
        # Giữ nguyên base model
        self.base_model = BaseModel()  # Sử dụng BaseModel gốc của bạn
        
        # Thay thế neighborhood fusion bằng phiên bản cải tiến
        self.neighborhood_fusion = AdaptiveNeighborhoodFusionTransformer(
            dim=144, 
            num_heads=8, 
            dropout=0.1, 
            num_layers=2
        )
        
        # Enhanced positional embeddings với learnable parameters
        self.patch_pos_embed = nn.ParameterList()
        for i in range(4):
            # Khởi tạo với sinusoidal nhưng cho phép học
            pos = torch.zeros(1, 144)
            for d in range(144):
                if d % 2 == 0:
                    pos[0, d] = np.sin(i / (10000 ** (2 * (d // 2) / 144)))
                else:
                    pos[0, d] = np.cos(i / (10000 ** (2 * (d // 2) / 144)))
            
            # Add learnable offset
            learnable_offset = torch.randn(1, 144) * 0.02
            self.patch_pos_embed.append(nn.Parameter(pos + learnable_offset, requires_grad=True))
        
        # Improved pooling và prediction heads
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Enhanced feature processing
        self.feature_norm = nn.LayerNorm(144)
        self.feature_dropout = nn.Dropout(0.3)
        
        # Improved prediction heads với residual connections
        hidden_dim = 192
        
        self.feature_proj = nn.Sequential(
            nn.Linear(144, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Separate heads cho từng góc với shared backbone
        self.yaw_class = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 66)
        )
        
        self.pitch_class = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 66)
        )
        
        self.roll_class = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 66)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x_parts):
        x_features = []
        
        # Process each part through base model (giữ nguyên)
        for i in range(4):
            _, _, x3 = self.base_model(x_parts[i])
            
            # Enhanced positional embedding
            pos_embed = self.patch_pos_embed[i]
            pos_embed = pos_embed[:, :, None, None].expand(-1, -1, x3.size(2), x3.size(3))
            x3_with_pos = x3 + pos_embed
            
            x_features.append(x3_with_pos)
        
        # Apply improved neighborhood fusion
        x4 = self.neighborhood_fusion(x_features)
        
        # Enhanced feature processing
        x5 = self.adaptive_pool(x4).flatten(1)
        x5 = self.feature_norm(x5)
        x5 = self.feature_dropout(x5)
        
        # Project features
        features = self.feature_proj(x5)
        
        # Angle predictions
        yaw_class = self.yaw_class(features)
        pitch_class = self.pitch_class(features)
        roll_class = self.roll_class(features)
        
        return yaw_class, pitch_class, roll_class