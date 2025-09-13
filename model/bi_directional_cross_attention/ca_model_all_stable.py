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

class ConvolutionalMultiheadAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, kernel_size=3, padding=1):
        super(ConvolutionalMultiheadAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        # Use smaller dimension for Q and K to reduce computation
        self.qk_channels = max(in_channels // 8, num_heads)
        self.qk_channels = (self.qk_channels // num_heads) * \
            num_heads  # Ensure divisible by num_heads

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
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, query, key_value):
        """
        Args:
            query: tensor with shape [B, C, H, W]
            key_value: tensor with shape [B, C, H, W]
        Returns:
            attended_features: tensor with shape [B, C, H, W]
        """
        batch_size, C, H, W = query.shape
        
        # Reshape for attention computation
        q = self.q_proj(query.flatten(2).transpose(1, 2))  # [B, H*W, C]
        k = self.k_proj(key_value.flatten(2).transpose(1, 2))  # [B, H*W, C]
        v = self.v_proj(key_value.flatten(2).transpose(1, 2))  # [B, H*W, C]
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B, num_heads, H*W, C/num_heads]
        k = k.reshape(batch_size, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, H*W, H*W]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention weights
        x = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, H*W, C)  # [B, H*W, C]
        x = self.out_proj(x)
        
        # Reshape back to spatial dimensions
        x = x.transpose(1, 2).reshape(batch_size, C, H, W)
        
        return x

class BidirectionalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(BidirectionalCrossAttention, self).__init__()
        self.cross_attn = CrossAttention(dim, num_heads)
        self.proj = nn.Conv2d(dim*2, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling parameter
        
    def forward(self, x1, x2):
        """
        Bidirectional cross-attention between x1 and x2
        """
        # x1 as query, x2 as key/value
        x1_to_x2 = self.cross_attn(x1, x2)
        
        # x2 as query, x1 as key/value
        x2_to_x1 = self.cross_attn(x2, x1)
        
        # Concatenate and project
        x_cat = torch.cat([x1_to_x2, x2_to_x1], dim=1)
        x_fused = self.proj(x_cat)
        
        # Apply scaling and residual connection to x1
        x1_out = x1 + torch.tanh(self.gamma) * x_fused
        
        return x1_out

class CrossStreamFusion(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(CrossStreamFusion, self).__init__()
        self.dim = dim
        
        # Create bidirectional cross-attention modules for each master-slave pair
        self.cross_attentions = nn.ModuleList([
            nn.ModuleList([BidirectionalCrossAttention(dim, num_heads) for j in range(4) if j != i])
            for i in range(4)
        ])
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(dim*4, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        # Add a feed-forward network with scaling factor as mentioned in your paper
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim*4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim*4, dim, kernel_size=1)
        )
        self.gamma_ffn = nn.Parameter(torch.zeros(1))
        
    def forward(self, features):
        """
        Args:
            features: list of 4 feature maps, each with shape [B, C, H, W]
        Returns:
            fused_features: tensor with shape [B, C, H, W]
        """
        B, C, H, W = features[0].shape
        enhanced_features = []
        
        # For each stream as master
        for i in range(4):
            master = features[i]
            cross_attended = []
            
            # Cross-attend with all other streams as slaves
            slave_idx = 0
            for j in range(4):
                if j != i:
                    # Apply bidirectional cross-attention between master and slave
                    attended = self.cross_attentions[i][slave_idx](master, features[j])
                    cross_attended.append(attended)
                    slave_idx += 1
            
            # Average the cross-attended features
            master_enhanced = torch.stack(cross_attended).mean(dim=0)
            enhanced_features.append(master_enhanced)
        
        # Concatenate all enhanced features
        fused = torch.cat(enhanced_features, dim=1)
        fused = self.fusion_layer(fused)
        
        # Apply feed-forward network with scaling
        fused = fused + torch.tanh(self.gamma_ffn) * self.ffn(fused)
        
        return fused
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Define the backbone
        self.base_model = BaseModel()
        
        # Define the cross-stream fusion module (replacing the concatenation and reduction)
        self.cross_stream_fusion = CrossStreamFusion(dim=144, num_heads=8)
        
        # Define the self attention part
        self.convolutional_multihead_attention = ConvolutionalMultiheadAttention(in_channels=144)
        
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
            
        # Define the fully connected layers for angles
        self.yaw_class = nn.Linear(144, 66)
        self.pitch_class = nn.Linear(144, 66)
        self.roll_class = nn.Linear(144, 66)

    def forward(self, x_parts):
        x_features = []
        
        # 1. Go through 4 streams
        for i in range(4):
            _, _, x3 = self.base_model(x_parts[i])
            
            # Add positional embedding
            pos_embed = self.patch_pos_embed[i]
            if torch.isnan(pos_embed).any() or torch.isinf(pos_embed).any():
                raise ValueError(f"Positional embedding for patch {i} contains NaN or Inf values.")
            
            pos_embed = pos_embed[:, :, None, None].expand(-1, -1, 8, 8)  # [1, 144, 8, 8]
            
            # Add position embedding to this stream
            x3 = x3 + pos_embed
            
            x_features.append(x3)
            
        # 2. Apply cross-stream fusion with bidirectional cross-attention
        x4 = self.cross_stream_fusion(x_features)
        
        # 3. Apply self-attention for further refinement
        x4 = self.convolutional_multihead_attention(x4)
        
        # 4. Multibin classification and regression
        x5 = self.pool(x4).flatten(1)  # Pool down to (B, 144, 1, 1) and flatten to (B, 144)

        
        # 5. Predict angle bins
        yaw_class = self.yaw_class(x5)
        pitch_class = self.pitch_class(x5)
        roll_class = self.roll_class(x5)
        
        return yaw_class, pitch_class, roll_class