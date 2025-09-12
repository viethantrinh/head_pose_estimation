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
        self.conv1 = ConvBlock2d(
            in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = ConvBlock2d(
            in_channels=16, out_channels=1, kernel_size=3, padding=1, activation='no')
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


class CrossStreamAttentionFusion(nn.Module):
    """Advanced fusion module using cross-attention between streams"""

    def __init__(self, embed_dim=144, num_heads=8):
        super(CrossStreamAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Cross-attention components
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Stream mixing weights
        self.stream_weights = nn.Parameter(torch.ones(4) / 4)

        # Feature refinement
        self.refine_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim * 2, kernel_size=1),
            nn.BatchNorm2d(embed_dim * 2),
            nn.GELU(),
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, stream_features):
        """
        stream_features: list of 4 tensors [B, C, H, W]
        """
        B, C, H, W = stream_features[0].shape

        # Convert to sequence format for attention
        stream_seqs = []
        for feat in stream_features:
            seq = feat.view(B, C, H*W).transpose(1, 2)  # [B, H*W, C]
            stream_seqs.append(seq)

        # Cross-attention between streams
        attended_streams = []
        for i, query_stream in enumerate(stream_seqs):
            q = self.q_proj(query_stream)  # [B, H*W, C]

            # Aggregate key and value from all streams
            all_streams = torch.cat(stream_seqs, dim=1)  # [B, 4*H*W, C]
            k = self.k_proj(all_streams)
            v = self.v_proj(all_streams)

            # Multi-head attention
            q = q.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, 4*H*W, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, 4*H*W, self.num_heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / \
                (self.head_dim ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v)

            # Concatenate heads and project
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, H*W, C)
            attended = self.out_proj(attn_out)

            # Residual connection with stream-specific weighting
            attended = self.gamma * attended + \
                self.stream_weights[i] * query_stream

            # Convert back to spatial format
            attended = attended.transpose(1, 2).view(B, C, H, W)
            attended_streams.append(attended)

        # Concatenate attended streams and refine
        fused = torch.cat(attended_streams, dim=1)  # [B, 4*C, H, W]
        refined = self.refine_conv(fused)  # [B, C, H, W]

        return refined


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # Reshape to sequence format: [B, H*W, C]
        x_seq = x.view(B, C, H*W).transpose(1, 2)

        # Apply projections
        q = self.q_proj(x_seq)  # [B, H*W, C]
        k = self.k_proj(x_seq)  # [B, H*W, C]
        v = self.v_proj(x_seq)  # [B, H*W, C]

        # Reshape for multi-head attention: [B, H*W, num_heads, head_dim]
        # [B, num_heads, H*W, head_dim]
        q = q.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, num_heads, H*W, head_dim]
        k = k.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, num_heads, H*W, head_dim]
        v = v.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / \
            (self.head_dim ** 0.5)  # [B, num_heads, H*W, H*W]
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        # [B, num_heads, H*W, head_dim]
        attn_out = torch.matmul(attn_weights, v)

        # Concatenate heads: [B, H*W, C]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, H*W, C)

        # Final projection
        output = self.out_proj(attn_out)  # [B, H*W, C]

        # Reshape back to spatial format: [B, C, H, W]
        output = output.transpose(1, 2).view(B, C, H, W)

        # Residual connection with learnable scaling
        output = self.gamma * output + x

        return output


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


class Spatial2DPositionalEmbedding(nn.Module):
    """Learnable 2D positional embeddings"""

    def __init__(self, num_streams=4, dim=144, height=8, width=8):
        super().__init__()
        self.spatial_embed = nn.Parameter(torch.zeros(1, dim, height, width))
        self.stream_embeds = nn.Parameter(torch.zeros(num_streams, dim, 1, 1))

        nn.init.trunc_normal_(self.spatial_embed, std=0.02)
        nn.init.trunc_normal_(self.stream_embeds, std=0.02)

    def forward(self, features, stream_idx):
        B, C, H, W = features.shape

        # Resize spatial embedding if needed
        if self.spatial_embed.shape[-2:] != (H, W):
            spatial_embed = F.interpolate(self.spatial_embed, size=(H, W),
                                          mode='bilinear', align_corners=False)
        else:
            spatial_embed = self.spatial_embed

        spatial_embed = spatial_embed.expand(B, -1, -1, -1)
        stream_embed = self.stream_embeds[stream_idx].expand(B, -1, H, W)

        return features + spatial_embed + stream_embed


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Define the backbone
        self.base_model = BaseModel()

        # Define the advanced cross-stream fusion
        self.cross_stream_fusion = CrossStreamAttentionFusion(
            embed_dim=144, num_heads=8)

        # Define the multi-head self attention part (now handles 144 channels after fusion)
        self.multi_head_self_attention = MultiHeadSelfAttention(
            embed_dim=144, num_heads=8)

        # Define the adaptive pooling for desired output size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Define the position embedding
        self.pos_embedding = Spatial2DPositionalEmbedding(
            num_streams=4, dim=144, height=8, width=8
        )

        # Define the fully connected layer for yaw (back to 144 features after better fusion)
        self.yaw_class = nn.Linear(144, 66)

        # Define the fully connected layer for pitch
        self.pitch_class = nn.Linear(144, 66)

        # Define the fully connected layer for roll
        self.roll_class = nn.Linear(144, 66)

    def forward(self, x_parts):
        x_features = []

        # 1. Go through 4 streams
        # Extract features with improved positional embeddings
        for i in range(4):
            _, _, x3 = self.base_model(x_parts[i])
            x3 = self.pos_embedding(x3, i)
            x_features.append(x3)

        # 2. Advanced Cross-Stream Fusion with Attention
        # Use cross-attention to let streams communicate and learn from each other
        x4 = self.cross_stream_fusion(x_features)  # [B, 144, H, W]

        # 3. Self-attention for further refinement
        x4 = self.multi_head_self_attention(x4)

        # 4. Multibin classification and regression
        # pool down to (100, 144, 1, 1) => flatten it to (100, 144)
        x5 = self.pool(x4).flatten(1)

        # go through dense layer to predict bin in range -99 to 99
        yaw_class = self.yaw_class(x5)
        pitch_class = self.pitch_class(x5)
        roll_class = self.roll_class(x5)

        return yaw_class, pitch_class, roll_class
