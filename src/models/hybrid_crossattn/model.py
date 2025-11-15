import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import math


class CrossAttention(nn.Module):
    """
    Cross-Attention module for information exchange between two sequences.
    
    Allows one sequence (query) to attend to another sequence (key-value).
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        """
        Args:
            d_model: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(CrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        """
        Args:
            query: Query tensor (batch, seq_len_q, d_model)
            key_value: Key-Value tensor (batch, seq_len_kv, d_model)
            
        Returns:
            Output tensor (batch, seq_len_q, d_model)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # Q: (batch, num_heads, seq_len_q, head_dim)
        # K: (batch, num_heads, seq_len_kv, head_dim)
        # V: (batch, num_heads, seq_len_kv, head_dim)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, num_heads, seq_len_q, seq_len_kv)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len_q, head_dim)
        
        # Concatenate heads and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(attn_output)
        
        return output


class HybridCrossAttentionClassifier(nn.Module):
    """
    Cross-Attention Hybrid Architecture: Mamba ⟷ LSTM
    
    Dual encoder architecture where Mamba and LSTM process input in parallel,
    then exchange information through bidirectional cross-attention before fusion.
    
    Architecture:
        Input → Mamba encoder → H_M ↘
                                      → Cross-Attention Exchange → Fusion → FC → Output
        Input → LSTM encoder  → H_L ↗
        
        Cross-Attention Exchange:
            H_L' = H_L + CrossAttn(Q=H_L, K,V=H_M)
            H_M' = H_M + CrossAttn(Q=H_M, K,V=H_L)
    """
    def __init__(self,
                 input_channels,
                 num_classes,
                 d_model=128,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 lstm_hidden=128,
                 lstm_layers=1,
                 fc_hidden_size=128,
                 dropout=0.3,
                 num_attn_heads=4,
                 fusion_method='concat'):
        """
        Args:
            input_channels: Number of input channels (e.g., 12 for ECG)
            num_classes: Number of output classes
            d_model: Mamba embedding dimension
            d_state: Mamba state dimension
            d_conv: Mamba convolution kernel size
            expand: Mamba expansion factor
            lstm_hidden: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            fc_hidden_size: Hidden size for FC layers
            dropout: Dropout rate
            num_attn_heads: Number of attention heads for cross-attention
            fusion_method: Method to fuse encoders ('concat')
        """
        super(HybridCrossAttentionClassifier, self).__init__()
        
        self.fusion_method = fusion_method
        
        # Mamba encoder branch
        self.mamba_proj = nn.Linear(input_channels, d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # LSTM encoder branch
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Project LSTM to same dimension as Mamba for cross-attention
        self.lstm_proj = nn.Linear(lstm_hidden, d_model) if lstm_hidden != d_model else nn.Identity()
        
        # Cross-attention modules for bidirectional information exchange
        self.cross_attn_lstm_to_mamba = CrossAttention(d_model, num_heads=num_attn_heads, dropout=dropout)
        self.cross_attn_mamba_to_lstm = CrossAttention(d_model, num_heads=num_attn_heads, dropout=dropout)
        
        # Layer normalization after cross-attention
        self.norm_lstm = nn.LayerNorm(d_model)
        self.norm_mamba = nn.LayerNorm(d_model)
        
        # Determine fusion output dimension
        if fusion_method == 'concat':
            fusion_dim = d_model * 2  # Both are projected to d_model
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Classifier head
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through dual encoders with cross-attention exchange
        
        Args:
            x: Input tensor of shape (batch, time_steps, input_channels)
            
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Branch 1: Mamba encoder
        mamba_input = self.mamba_proj(x)              # (batch, time_steps, d_model)
        H_M = self.mamba(mamba_input)                 # (batch, time_steps, d_model)
        
        # Branch 2: LSTM encoder
        lstm_out, (h_n, c_n) = self.lstm(x)          # lstm_out: (batch, time_steps, lstm_hidden)
        H_L = self.lstm_proj(lstm_out)                # (batch, time_steps, d_model)
        
        # Cross-Attention Exchange
        # H_L' = H_L + CrossAttn(Q=H_L, K,V=H_M)  - LSTM queries Mamba
        lstm_attended = self.cross_attn_lstm_to_mamba(H_L, H_M)  # (batch, time_steps, d_model)
        H_L_updated = self.norm_lstm(H_L + lstm_attended)        # Residual + LayerNorm
        
        # H_M' = H_M + CrossAttn(Q=H_M, K,V=H_L)  - Mamba queries LSTM
        mamba_attended = self.cross_attn_mamba_to_lstm(H_M, H_L)  # (batch, time_steps, d_model)
        H_M_updated = self.norm_mamba(H_M + mamba_attended)       # Residual + LayerNorm
        
        # Global pooling for both branches
        lstm_pooled = H_L_updated.mean(dim=1)         # (batch, d_model)
        mamba_pooled = H_M_updated.mean(dim=1)        # (batch, d_model)
        
        # Fusion
        if self.fusion_method == 'concat':
            fused = torch.cat([mamba_pooled, lstm_pooled], dim=1)  # (batch, d_model * 2)
        
        # Classification
        return self.fc(fused)  # (batch, num_classes)
