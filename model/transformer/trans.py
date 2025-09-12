import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=3, embed_dim=144):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Thay thế CNN bằng patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                   kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x: (B, 3, 64, 64) -> (B, 144, 8, 8) -> (B, 144, 64) -> (B, 64, 144)
        x = self.patch_embed(x)  # (B, 144, 8, 8)
        x = x.flatten(2).transpose(1, 2)  # (B, 64, 144)
        x = self.norm(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, 
                                             dropout=dropout, batch_first=True)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # Multi-head self-attention
        src2, _ = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout(src2))
        
        # Feed forward
        src2 = self.ffn(src)
        src = self.norm2(src + src2)
        
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=144, num_heads=8, num_layers=6, 
                 mlp_ratio=4, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class BiDirectionalCrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(BiDirectionalCrossAttentionLayer, self).__init__()
        self.num_streams = 4
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Query, Key, Value projections for each stream
        self.q_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, bias=False) 
            for _ in range(self.num_streams)
        ])
        self.k_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, bias=False) 
            for _ in range(self.num_streams)
        ])
        self.v_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, bias=False) 
            for _ in range(self.num_streams)
        ])
        
        # Output projections
        self.out_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) 
            for _ in range(self.num_streams)
        ])
        
        # Layer norms
        self.norms1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(self.num_streams)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(self.num_streams)])
        
        # FFN for each stream
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(self.num_streams)
        ])
        
        # Cross-stream interaction matrix (learnable)
        self.interaction_matrix = nn.Parameter(
            torch.eye(self.num_streams) + 0.1 * torch.randn(self.num_streams, self.num_streams)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, stream_features):
        batch_size = stream_features[0].size(0)
        seq_len = stream_features[0].size(1)
        
        # Generate Q, K, V for all streams
        queries, keys, values = [], [], []
        
        for i in range(self.num_streams):
            q = self.q_projections[i](stream_features[i])
            k = self.k_projections[i](stream_features[i])
            v = self.v_projections[i](stream_features[i])
            
            # Reshape for multi-head attention: (B, seq_len, embed_dim) -> (B, num_heads, seq_len, head_dim)
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            queries.append(q)
            keys.append(k)
            values.append(v)
        
        enhanced_streams = []
        
        # Bi-directional cross-attention with interaction matrix
        for i in range(self.num_streams):
            attended_heads = []
            
            for head in range(self.num_heads):
                head_attended = []
                
                # Stream i attends to all streams with learned interaction weights
                for j in range(self.num_streams):
                    # Attention scores
                    scores = torch.matmul(queries[i][:, head], keys[j][:, head].transpose(-2, -1)) * self.scale
                    
                    # Apply interaction matrix weight
                    interaction_weight = self.interaction_matrix[i, j]
                    scores = scores * interaction_weight
                    
                    # Softmax
                    attn_weights = F.softmax(scores, dim=-1)
                    attn_weights = self.dropout(attn_weights)
                    
                    # Apply to values
                    attended = torch.matmul(attn_weights, values[j][:, head])
                    head_attended.append(attended)
                
                # Combine attended features from all streams for this head
                combined_head = sum(head_attended) / self.num_streams
                attended_heads.append(combined_head)
            
            # Concatenate all heads: (B, num_heads, seq_len, head_dim) -> (B, seq_len, embed_dim)
            multi_head_output = torch.cat(attended_heads, dim=-1).transpose(1, 2).contiguous()
            multi_head_output = multi_head_output.view(batch_size, seq_len, self.embed_dim)
            
            # Output projection
            projected = self.out_projections[i](multi_head_output)
            
            # First residual connection and layer norm
            residual1 = self.norms1[i](stream_features[i] + self.dropout(projected))
            
            # FFN
            ffn_output = self.ffns[i](residual1)
            
            # Second residual connection and layer norm
            enhanced = self.norms2[i](residual1 + self.dropout(ffn_output))
            
            enhanced_streams.append(enhanced)
        
        return enhanced_streams

class MultiStreamCrossAttentionTransformer(nn.Module):
    def __init__(self, embed_dim=144, num_heads=8, num_layers=3, dropout=0.1):
        super(MultiStreamCrossAttentionTransformer, self).__init__()
        self.num_streams = 4
        self.embed_dim = embed_dim
        
        # Bi-directional cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            BiDirectionalCrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Stream-specific tokens (learnable)
        self.stream_tokens = nn.Parameter(torch.randn(4, 1, embed_dim))
        
    def forward(self, stream_features):
        # stream_features: List of 4 tensors, each (B, num_patches, embed_dim)
        batch_size = stream_features[0].size(0)
        
        # Add stream-specific tokens
        enhanced_streams = []
        for i, features in enumerate(stream_features):
            stream_token = self.stream_tokens[i].expand(batch_size, -1, -1)
            enhanced = torch.cat([stream_token, features], dim=1)
            enhanced_streams.append(enhanced)
        
        # Apply bi-directional cross-attention layers
        for layer in self.cross_attention_layers:
            enhanced_streams = layer(enhanced_streams)
            
        return enhanced_streams

class TransformerFusionModule(nn.Module):
    def __init__(self, embed_dim=144, num_heads=8):
        super(TransformerFusionModule, self).__init__()
        
        # Global fusion attention
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        # Final transformer layers
        self.final_transformer = TransformerEncoder(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            num_layers=2
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Class token for final representation
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, enhanced_streams):
        batch_size = enhanced_streams[0].size(0)
        
        # Concatenate all streams
        all_features = torch.cat(enhanced_streams, dim=1)  # (B, total_patches, embed_dim)
        
        # Add class token
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        all_features = torch.cat([class_tokens, all_features], dim=1)
        
        # Final transformer processing
        final_features = self.final_transformer(all_features)
        
        # Extract class token as final representation
        class_representation = final_features[:, 0]  # (B, embed_dim)
        
        return class_representation

class Model(nn.Module):
    def __init__(self, img_size=64, patch_size=8, embed_dim=144, num_heads=8):
        super(Model, self).__init__()
        
        # Patch embedding thay vì CNN backbone
        self.patch_embedding = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        
        # Positional embedding
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        # Transformer encoder cho mỗi stream
        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, num_layers=6)
        
        # Cross-attention giữa streams
        self.cross_attention = MultiStreamCrossAttentionTransformer(embed_dim, num_heads)
        
        # Final fusion
        self.fusion_module = TransformerFusionModule(embed_dim, num_heads)
        
        # Classification heads
        self.yaw_head = nn.Linear(embed_dim, 66)
        self.pitch_head = nn.Linear(embed_dim, 66)
        self.roll_head = nn.Linear(embed_dim, 66)
        
    def forward(self, x_parts):
        stream_features = []
        
        # Process each stream
        for i in range(4):
            # Patch embedding
            patches = self.patch_embedding(x_parts[i])  # (B, num_patches, embed_dim)
            
            # Add positional embedding
            patches = patches + self.pos_embedding
            
            # Transformer encoding
            encoded = self.transformer_encoder(patches)
            stream_features.append(encoded)
        
        # Cross-attention between streams
        enhanced_streams = self.cross_attention(stream_features)
        
        # Final fusion
        fused_representation = self.fusion_module(enhanced_streams)
        
        # Classification
        yaw = self.yaw_head(fused_representation)
        pitch = self.pitch_head(fused_representation)
        roll = self.roll_head(fused_representation)
        
        return yaw, pitch, roll