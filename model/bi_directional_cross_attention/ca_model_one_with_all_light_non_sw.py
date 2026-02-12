import torch
import torch.nn.functional as F
import numpy as np

from torch import nn


class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activation='gelu'):
        super(ConvBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding)
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


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.conv11 = ConvBlock2d(in_channels=3, out_channels=32)
        self.conv12 = ConvBlock2d(in_channels=32, out_channels=32)
        self.conv13 = ConvBlock2d(
            in_channels=3, out_channels=32, kernel_size=1, padding=0, activation='no')

        self.conv21 = ConvBlock2d(in_channels=32, out_channels=128)
        self.conv22 = ConvBlock2d(in_channels=128, out_channels=128)
        self.conv23 = ConvBlock2d(
            in_channels=32, out_channels=128, kernel_size=1, padding=0, activation='no')

        self.conv31 = ConvBlock2d(in_channels=128, out_channels=256)
        self.conv32 = ConvBlock2d(in_channels=256, out_channels=144)
        self.conv33 = ConvBlock2d(
            in_channels=128, out_channels=144, kernel_size=1, padding=0, activation='no')

        self.attention = CombinedAttention(in_channels=144)

        # pooling H va W xuong con H' va W'. Chia cho 2
        self.pool = nn.AvgPool2d(kernel_size=2)
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
    
class ConvolutionalMultiheadAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, kernel_size=3, padding=1):
        super(ConvolutionalMultiheadAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        # Use smaller dimension for Q and K to reduce computation
        self.qk_channels = max(in_channels // 8, num_heads)
        self.qk_channels = (self.qk_channels // num_heads) * \
            num_heads  # Ensure divisible by num_heads
        
        # Scale should match Q and K dimension, not Value dimension
        self.qk_head_dim = self.qk_channels // num_heads
        self.scale = self.qk_head_dim ** -0.5

        self.query_conv = nn.Conv2d(
            in_channels, self.qk_channels, kernel_size=kernel_size, padding=padding)
        self.key_conv = nn.Conv2d(
            in_channels, self.qk_channels, kernel_size=kernel_size, padding=padding)
        self.value_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=padding)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(0.1)

        # Initialize weights
        nn.init.kaiming_normal_(self.query_conv.weight,
                                mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.key_conv.weight,
                                mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.value_conv.weight,
                                mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.out_conv.weight,
                                mode='fan_out', nonlinearity='relu')

        if self.query_conv.bias is not None:
            nn.init.zeros_(self.query_conv.bias)
            nn.init.zeros_(self.key_conv.bias)
            nn.init.zeros_(self.value_conv.bias)
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError(
                "ConvolutionalMultiheadAttention input contains NaN or Inf values.")

        batch_size, C, H, W = x.size()

        # Generate Q, K, V
        q = self.query_conv(x)  # [B, qk_channels, H, W]
        k = self.key_conv(x)    # [B, qk_channels, H, W]
        v = self.value_conv(x)  # [B, C, H, W]

        # Reshape for multi-head attention
        q = q.view(batch_size, self.num_heads, self.qk_channels //
                   self.num_heads, H * W)  # [B, num_heads, head_dim_qk, H*W]
        k = k.view(batch_size, self.num_heads, self.qk_channels //
                   self.num_heads, H * W)  # [B, num_heads, head_dim_qk, H*W]
        v = v.view(batch_size, self.num_heads, self.head_dim,
                   H * W)  # [B, num_heads, head_dim, H*W]

        # Compute attention scores
        # [B, num_heads, H*W, H*W]
        attn_scores = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(v, attn_weights.transpose(-2, -1)
                           )  # [B, num_heads, head_dim, H*W]

        # Reshape back to spatial dimensions
        out = out.view(batch_size, C, H, W)

        # Apply output projection
        out = self.out_conv(out)

        if torch.isnan(out).any() or torch.isinf(out).any():
            raise ValueError(
                "ConvolutionalMultiheadAttention output contains NaN or Inf values.")

        # Residual connection with learnable scaling
        out = x + torch.tanh(self.gamma) * out

        return out

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.1, proj_drop=0.1):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # # Pre-Norm for training stability
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # # Regularization
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, query, key_value):
        """
        Args:
            query: [B, N_q, C] - query tokens
            key_value: [B, N_kv, C] - key/value tokens
        Returns:
            out: [B, N_q, C]
        """
        batch_size, N_q, C = query.shape
        N_kv = key_value.shape[1]
        
        # Pre-Norm for training stability
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        
        # Linear projections
        q = self.q_proj(q)    # [B, N_q, C]
        k = self.k_proj(kv)   # [B, N_kv, C]
        v = self.v_proj(kv)   # [B, N_kv, C]
        
        # Reshape for multi-head: [B, num_heads, N, head_dim]
        q = q.view(batch_size, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention: [B, num_heads, N_q, N_kv]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Weighted sum: [B, num_heads, N_q, head_dim]
        out = torch.matmul(attn, v)
        
        # Merge heads: [B, N_q, C]
        out = out.transpose(1, 2).reshape(batch_size, N_q, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        
        return out


class BidirectionalCrossAttention(nn.Module):
    """
    Efficient bidirectional cross-attention.
    Each master stream attends to ALL tokens from ALL other streams simultaneously,
    rather than pairwise attention + averaging.
    """
    def __init__(self, dim, num_heads=8, attn_drop=0.1, proj_drop=0.1):
        super(BidirectionalCrossAttention, self).__init__()
        
        # Single cross-attention that handles variable-length KV
        self.cross_attn = CrossAttention(dim, num_heads, attn_drop, proj_drop)
        
        # Learnable scaling for residual
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, master, other_streams):
        """
        Args:
            master: [B, C, H, W] - the master stream
            other_streams: list of [B, C, H, W] - all other streams
        Returns:
            enhanced_master: [B, C, H, W] - master with cross-stream info
        """
        B, C, H, W = master.shape
        
        # Flatten master: [B, N, C]
        q = master.flatten(2).transpose(1, 2)
        
        # Concatenate ALL other streams into one KV sequence: [B, K*N, C]
        # This lets the master attend to all tokens from all other streams simultaneously
        kv_list = [s.flatten(2).transpose(1, 2) for s in other_streams]
        kv = torch.cat(kv_list, dim=1)  # [B, K*N, C]
        
        # Cross-attention: master queries, all others as KV
        attn_out = self.cross_attn(q, kv)  # [B, N, C]
        
        # Reshape back to spatial: [B, C, H, W]
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        
        # Residual connection with learnable scaling
        out = master + torch.tanh(self.gamma) * attn_out
        
        return out
    
    
class CrossStreamFusion(nn.Module):
    """
    Improved cross-stream fusion with:
    1. Pre-Norm (LayerNorm before attention) for training stability
    2. Dropout for regularization  
    3. Efficient multi-stream attention (attend to all streams at once)
    4. Gate mechanism after fusion
    5. FFN with pre-norm
    """
    def __init__(self, dim, num_heads=8, attn_drop=0.1, proj_drop=0.1):
        super(CrossStreamFusion, self).__init__()
        self.dim = dim

        # Separate cross-attention module for each stream
        self.cross_attentions = nn.ModuleList([
            BidirectionalCrossAttention(dim, num_heads, attn_drop, proj_drop)
            for _ in range(4)
        ])

        # Fusion: concat 4 enhanced streams -> project to dim
        self.fusion_conv = nn.Conv2d(dim * 4, dim, kernel_size=1)
        self.fusion_bn = nn.BatchNorm2d(dim)
        self.fusion_act = nn.GELU()
        
        # # Gate mechanism after fusion
        # self.gate = nn.Sequential(
        #     nn.Conv2d(dim, dim, kernel_size=1),
        #     nn.Sigmoid()
        # )

        # FFN with Pre-Norm
        self.ffn_norm = nn.BatchNorm2d(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Conv2d(dim * 4, dim, kernel_size=1),
            nn.Dropout(proj_drop),
        )
        self.gamma_ffn = nn.Parameter(torch.zeros(1))

    def forward(self, features):
        """
        Args:
            features: list of 4 feature maps, each with shape [B, C, H, W]
        Returns:
            fused_features: tensor with shape [B, C, H, W]
        """
        enhanced_features = []

        # Each stream as master, attend to all other streams simultaneously
        for i in range(4):
            master = features[i]
            others = [features[j] for j in range(4) if j != i]
            
            # Each stream uses its own cross-attention module
            enhanced = self.cross_attentions[i](master, others)
            enhanced_features.append(enhanced)

        # Concatenate all 4 enhanced features and project
        fused = torch.cat(enhanced_features, dim=1)  # [B, 4C, H, W]
        fused = self.fusion_conv(fused)
        fused = self.fusion_bn(fused)
        fused = self.fusion_act(fused)
        
        # Gate mechanism
        # gate_values = self.gate(fused)
        # fused = fused * gate_values

        # FFN with pre-norm and residual
        ffn_out = self.ffn(self.ffn_norm(fused))
        fused = fused + torch.tanh(self.gamma_ffn) * ffn_out

        return fused

class Model(nn.Module):
    """
    VERSION T1: Enhanced with Gate mechanism after soft attention in fusion layer
    Use this model to test T1 enhancement only
    """
    def __init__(self):
        super(Model, self).__init__()

        # Define the backbone
        self.base_model = BaseModel()

        # T1: Enhanced cross-stream fusion with gating
        self.cross_stream_fusion = CrossStreamFusion(dim=144, num_heads=8)

        # Define the self attention part
        self.convolutional_multihead_attention = ConvolutionalMultiheadAttention(
            in_channels=144)

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

        # Original Linear prediction heads
        self.yaw_class = nn.Linear(144, 66)
        self.pitch_class = nn.Linear(144, 66)
        self.roll_class = nn.Linear(144, 66)

    def forward(self, x_parts):
        x_features = []

        # 1. Go through 4 streams
        for i in range(4):
            _, _, x3 = self.base_model(x_parts[i])

            # Add positional embedding
            pos_embed = self.patch_pos_embed[i].to(x3.device)
            if torch.isnan(pos_embed).any() or torch.isinf(pos_embed).any():
                raise ValueError(
                    f"Positional embedding for patch {i} contains NaN or Inf values.")

            pos_embed = pos_embed[:, :, None,
                                  None].expand(-1, -1, 8, 8)  # [1, 144, 8, 8]

            # Add position embedding to this stream
            x3 = x3 + pos_embed

            x_features.append(x3)

        # 2. Apply T1: Enhanced cross-stream fusion with gating
        x4 = self.cross_stream_fusion(x_features)

        # 3. Apply self-attention for further refinement
        x4 = self.convolutional_multihead_attention(x4)

        # 4. Pool down to (B, 144, 1, 1) and flatten to (B, 144)
        x5 = self.pool(x4).flatten(1)

        # 5. Predict angle bins using original Linear layers
        yaw_class = self.yaw_class(x5)
        pitch_class = self.pitch_class(x5)
        roll_class = self.roll_class(x5)

        return yaw_class, pitch_class, roll_class



    
