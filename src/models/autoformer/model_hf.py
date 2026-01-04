from transformers import AutoformerConfig, AutoformerModel

import torch
import torch.nn as nn


class AutoformerClassifier(nn.Module):
    """
    Autoformer classifier using HuggingFace implementation
    """
    
    def __init__(self, input_channels, num_classes, seq_len, prediction_length=1,
                 d_model=128, encoder_layers=2, decoder_layers=1, encoder_attention_heads=8,
                 decoder_attention_heads=8, dropout=0.1, fc_hidden_size=128):
        super(AutoformerClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.seq_len = seq_len
        
        # Autoformer uses lags: context_length + max(lags) = total input length
        # With seq_len=1000 and lags=[1], we need context_length=999
        self.lags_sequence = [0]
        self.context_length = seq_len - max(self.lags_sequence)
        
        # Configure Autoformer with memory optimization
        config = AutoformerConfig(
            input_size=input_channels,
            prediction_length=prediction_length,
            context_length=self.context_length,
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            decoder_attention_heads=decoder_attention_heads,
            dropout=dropout,
            attention_dropout=dropout,
            decoder_ffn_dim=d_model * 4,
            encoder_ffn_dim=d_model * 4,
            use_cache=False,  # Disable caching to save memory
            num_time_features=0,  # No additional time features
            scaling=True,  # Disable internal scaling for simplicity
            lags_sequence=self.lags_sequence,  # Minimal lags for classification task
        )
        
        self.autoformer = AutoformerModel(config)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_channels)
        Returns:
            out: (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Autoformer expects inputs in the format:
        # past_values: (batch_size, seq_len, input_channels) for multivariate
        # past_time_features: (batch_size, seq_len, num_time_features) - can be empty
        # past_observed_mask: (batch_size, seq_len, input_channels) - mask for missing values
        
        # Create time features (empty tensor since num_time_features=0)
        past_time_features = torch.zeros(
            (batch_size, self.seq_len, 0), 
            dtype=x.dtype, 
            device=device
        )
        
        # Create observed mask (all ones, indicating all values are observed)
        past_observed_mask = torch.ones(
            (batch_size, self.seq_len, self.input_channels),
            dtype=torch.bool,
            device=device
        )
        
        # Get Autoformer embeddings
        outputs = self.autoformer(
            past_values=x,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
        )
        
        # Use encoder last hidden state for classification
        # encoder_last_hidden_state: (batch, seq_len, d_model)
        embeddings = outputs.encoder_last_hidden_state
        
        # Pool across time steps (average pooling)
        features = embeddings.mean(dim=1)  # (batch, d_model)
        
        # Classification
        out = self.fc(features)
        return out
