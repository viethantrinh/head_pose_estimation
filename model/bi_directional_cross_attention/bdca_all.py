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
    """
    Bi-directional Cross-Attention module for fusing features from multiple streams.
    Each stream attends to all other streams bidirectionally.
    """
    def __init__(self, in_channels=144, num_streams=4, reduction_ratio=8):
        super(BiDirectionalCrossAttention, self).__init__()
        self.in_channels = in_channels
        self.num_streams = num_streams
        self.head_dim = in_channels // reduction_ratio
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V for each stream
        self.q_projections = nn.ModuleList([
            nn.Conv2d(in_channels, self.head_dim, kernel_size=1, bias=False) 
            for _ in range(num_streams)
        ])
        
        self.k_projections = nn.ModuleList([
            nn.Conv2d(in_channels, self.head_dim, kernel_size=1, bias=False) 
            for _ in range(num_streams)
        ])
        
        self.v_projections = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False) 
            for _ in range(num_streams)
        ])
        
        # Output projection for each stream
        self.out_projections = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False) 
            for _ in range(num_streams)
        ])
        
        # Layer normalization for each stream
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm([in_channels, 8, 8]) 
            for _ in range(num_streams)
        ])
        
        # Learnable mixing parameters for residual connections
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) 
            for _ in range(num_streams)
        ])
    
    
    def forward(self, stream_features):
        """
        Args:
            stream_features: List of 4 feature tensors, each with shape (B, 144, H, W)
        
        Returns:
            List of 4 enhanced feature tensors with the same shape
        """
        batch_size, channels, height, width = stream_features[0].shape
        
        # Generate Q, K, V for all streams
        queries = []
        keys = []
        values = []
        
        for i in range(self.num_streams):
            q = self.q_projections[i](stream_features[i])  # (B, head_dim, H, W)
            k = self.k_projections[i](stream_features[i])  # (B, head_dim, H, W)
            v = self.v_projections[i](stream_features[i])  # (B, C, H, W)
            
            # Reshape for attention computation
            q = q.flatten(2).transpose(1, 2)  # (B, H*W, head_dim)
            k = k.flatten(2).transpose(1, 2)  # (B, H*W, head_dim)
            v = v.flatten(2).transpose(1, 2)  # (B, H*W, C)
            
            queries.append(q)
            keys.append(k)
            values.append(v)
        
        # Bi-directional cross-attention
        enhanced_features = []
        
        for i in range(self.num_streams):
            attended_features = []
            
            # Stream i attends to all other streams (including itself)
            for j in range(self.num_streams):
                if i == j:
                    # Self-attention for the current stream
                    attn_scores = torch.bmm(queries[i], keys[j].transpose(1, 2)) * self.scale
                else:
                    # Cross-attention with other streams
                    attn_scores = torch.bmm(queries[i], keys[j].transpose(1, 2)) * self.scale
                
                # Apply softmax
                attn_weights = F.softmax(attn_scores, dim=-1)
                
                # Apply attention to values
                attended = torch.bmm(attn_weights, values[j])  # (B, H*W, C)
                attended_features.append(attended)
            
            # Aggregate attended features from all streams
            # Use learnable weights for different streams
            aggregated = sum(attended_features) / self.num_streams
            
            # Reshape back to spatial dimensions
            aggregated = aggregated.transpose(1, 2).reshape(batch_size, channels, height, width)
            
            # Apply output projection
            projected = self.out_projections[i](aggregated)
            
            # Residual connection with learnable parameter
            enhanced = self.alphas[i] * projected + stream_features[i]
            
            # Layer normalization
            enhanced = self.layer_norms[i](enhanced)
            
            enhanced_features.append(enhanced)
        
        return enhanced_features


class StreamFusionModule(nn.Module):
    """
    Complete fusion module that combines bi-directional cross-attention with 
    final feature aggregation for classification
    """
    def __init__(self, in_channels=144, num_streams=4):
        super(StreamFusionModule, self).__init__()
        
        # Bi-directional cross-attention
        self.cross_attention = BiDirectionalCrossAttention(
            in_channels=in_channels, 
            num_streams=num_streams
        )
        
        # Final fusion layer
        self.final_fusion = nn.Sequential(
            nn.Conv2d(in_channels * num_streams, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Self-attention on fused features
        self.self_attention = ConvolutionalSelfAttention(in_channels=in_channels)
        
        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, stream_features):
        """
        Args:
            stream_features: List of 4 feature tensors from different streams
        
        Returns:
            Fused feature tensor ready for classification
        """
        # Apply bi-directional cross-attention
        enhanced_features = self.cross_attention(stream_features)
        
        # Concatenate enhanced features
        concatenated = torch.cat(enhanced_features, dim=1)  # (B, 144*4, H, W)
        
        # Final fusion
        fused = self.final_fusion(concatenated)  # (B, 144, H, W)
        
        # Apply self-attention
        fused = self.self_attention(fused)
        
        # Global average pooling
        output = self.pool(fused).flatten(1)  # (B, 144)
        
        return output


# Modified Model class with the new fusion module
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Define the backbone (same as before)
        self.base_model = BaseModel()
        
        # Define the position embedding (same as before)
        self.patch_pos_embed = nn.ParameterList()
        for i in range(4):
            pos = torch.zeros(1, 144)
            for d in range(144):
                if d % 2 == 0:
                    pos[0, d] = np.sin(i / (10000 ** (2 * (d // 2) / 144)))
                else:
                    pos[0, d] = np.cos(i / (10000 ** (2 * (d // 2) / 144)))
            self.patch_pos_embed.append(nn.Parameter(pos, requires_grad=True))
        
        # Replace the original fusion with bi-directional cross-attention
        self.fusion_module = StreamFusionModule(in_channels=144, num_streams=4)
        
        # Classification heads (same as before)
        self.yaw_class = nn.Linear(144, 66)
        self.pitch_class = nn.Linear(144, 66)
        self.roll_class = nn.Linear(144, 66)

    def forward(self, x_parts):
        stream_features = []
        
        # 1. Process each stream through backbone
        for i in range(4):
            _, _, x3 = self.base_model(x_parts[i])  # (B, 144, 8, 8)
            
            # Add positional embedding
            pos_embed = self.patch_pos_embed[i]
            if torch.isnan(pos_embed).any() or torch.isinf(pos_embed).any():
                raise ValueError(f"Positional embedding for patch {i} contains NaN or Inf values.")
            
            pos_embed = pos_embed[:, :, None, None].expand(-1, -1, 8, 8)
            x3 = x3 + pos_embed
            
            stream_features.append(x3)
        
        # 2. Apply bi-directional cross-attention fusion
        fused_features = self.fusion_module(stream_features)  # (B, 144)
        
        # 3. Classification
        yaw_class = self.yaw_class(fused_features)
        pitch_class = self.pitch_class(fused_features)
        roll_class = self.roll_class(fused_features)
        
        return yaw_class, pitch_class, roll_class