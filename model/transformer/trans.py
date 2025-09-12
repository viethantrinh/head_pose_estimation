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

# NEW: Transformer-based Fusion Module
class TransformerFusionLayer(nn.Module):
    def __init__(self, embed_dim=144, num_heads=8, dropout=0.1):
        super(TransformerFusionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Cross-attention between streams
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        # Self-attention
        attn_output, _ = self.self_attention(query, query, query)
        query = self.norm1(query + self.dropout(attn_output))
        
        # Cross-attention
        cross_output, _ = self.cross_attention(query, key_value, key_value)
        query = self.norm2(query + self.dropout(cross_output))
        
        # Feed-forward
        ffn_output = self.ffn(query)
        query = self.norm3(query + self.dropout(ffn_output))
        
        return query

class HybridTransformerFusionModule(nn.Module):
    """
    Hybrid fusion module that converts CNN features to transformer tokens
    and applies transformer-based fusion
    """
    def __init__(self, in_channels=144, num_streams=4, num_heads=8, num_layers=3):
        super(HybridTransformerFusionModule, self).__init__()
        self.in_channels = in_channels
        self.num_streams = num_streams
        
        # Convert 2D features to 1D tokens
        self.feature_to_token = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable stream tokens
        self.stream_tokens = nn.Parameter(torch.randn(num_streams, 1, in_channels))
        
        # Positional embedding for spatial patches
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, 64, in_channels))  # 8x8=64 patches
        
        # Transformer layers for fusion
        self.fusion_layers = nn.ModuleList([
            TransformerFusionLayer(in_channels, num_heads)
            for _ in range(num_layers)
        ])
        
        # Final aggregation
        self.final_norm = nn.LayerNorm(in_channels)
        self.class_token = nn.Parameter(torch.randn(1, 1, in_channels))
        
        # Global attention for final representation
        self.global_attention = nn.MultiheadAttention(
            in_channels, num_heads, batch_first=True
        )
        
    def forward(self, stream_features):
        """
        Args:
            stream_features: List of 4 tensors, each (B, 144, 8, 8)
        Returns:
            Fused representation (B, 144)
        """
        batch_size = stream_features[0].size(0)
        
        # Convert each stream to tokens
        stream_tokens = []
        for i, features in enumerate(stream_features):
            # Convert spatial features to tokens: (B, 144, 8, 8) -> (B, 144, 64) -> (B, 64, 144)
            tokens = self.feature_to_token(features)
            tokens = tokens.flatten(2).transpose(1, 2)  # (B, 64, 144)
            
            # Add spatial positional embedding
            tokens = tokens + self.spatial_pos_embed
            
            # Add stream-specific token at the beginning
            stream_token = self.stream_tokens[i].expand(batch_size, -1, -1)
            tokens = torch.cat([stream_token, tokens], dim=1)  # (B, 65, 144)
            
            stream_tokens.append(tokens)
        
        # Bi-directional cross-attention fusion
        enhanced_streams = []
        for i in range(self.num_streams):
            query_stream = stream_tokens[i]
            
            # Create key-value from all other streams
            other_streams = [stream_tokens[j] for j in range(self.num_streams) if j != i]
            key_value = torch.cat(other_streams, dim=1)  # Concatenate all other streams
            
            # Apply transformer fusion layers
            enhanced = query_stream
            for layer in self.fusion_layers:
                enhanced = layer(enhanced, key_value)
            
            enhanced_streams.append(enhanced)
        
        # Global fusion
        # Concatenate all enhanced streams
        all_tokens = torch.cat(enhanced_streams, dim=1)  # (B, 65*4, 144)
        
        # Add global class token
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        all_tokens = torch.cat([class_tokens, all_tokens], dim=1)
        
        # Final global attention
        final_output, _ = self.global_attention(class_tokens, all_tokens, all_tokens)
        final_output = self.final_norm(final_output)
        
        # Extract class token as final representation
        return final_output.squeeze(1)  # (B, 144)

# Modified Model class with Transformer-based fusion
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Keep the original CNN backbone
        self.base_model = BaseModel()
        
        # Keep the original positional embedding
        self.patch_pos_embed = nn.ParameterList()
        for i in range(4):
            pos = torch.zeros(1, 144)
            for d in range(144):
                if d % 2 == 0:
                    pos[0, d] = np.sin(i / (10000 ** (2 * (d // 2) / 144)))
                else:
                    pos[0, d] = np.cos(i / (10000 ** (2 * (d // 2) / 144)))
            self.patch_pos_embed.append(nn.Parameter(pos, requires_grad=True))
        
        # NEW: Replace fusion with Transformer-based fusion
        self.fusion_module = HybridTransformerFusionModule(
            in_channels=144, 
            num_streams=4, 
            num_heads=8, 
            num_layers=3
        )
        
        # Keep the original multi-bin classification heads
        self.yaw_class = nn.Linear(144, 66)
        self.pitch_class = nn.Linear(144, 66)
        self.roll_class = nn.Linear(144, 66)

    def forward(self, x_parts):
        stream_features = []
        
        # 1. Process each stream through CNN backbone (unchanged)
        for i in range(4):
            _, _, x3 = self.base_model(x_parts[i])  # (B, 144, 8, 8)
            
            # Add positional embedding (unchanged)
            pos_embed = self.patch_pos_embed[i]
            if torch.isnan(pos_embed).any() or torch.isinf(pos_embed).any():
                raise ValueError(f"Positional embedding for patch {i} contains NaN or Inf values.")
            
            pos_embed = pos_embed[:, :, None, None].expand(-1, -1, 8, 8)
            x3 = x3 + pos_embed
            
            stream_features.append(x3)
        
        # 2. Apply Transformer-based fusion (NEW)
        fused_features = self.fusion_module(stream_features)  # (B, 144)
        
        # 3. Multi-bin classification (unchanged)
        yaw_class = self.yaw_class(fused_features)
        pitch_class = self.pitch_class(fused_features)
        roll_class = self.roll_class(fused_features)
        
        return yaw_class, pitch_class, roll_class
