from transformers import PatchTSTConfig, PatchTSTModel

import torch
import torch.nn as nn


class PatchTSTClassifier(nn.Module):
    """
    PatchTST classifier using HuggingFace implementation
    """
    
    def __init__(self, input_channels, num_classes, seq_len, patch_len=16, stride=8,
                 d_model=128, n_heads=8, num_layers=3, dropout=0.1, fc_hidden_size=128):
        super(PatchTSTClassifier, self).__init__()
        
        # Configure PatchTST with memory optimization
        config = PatchTSTConfig(
            num_input_channels=input_channels,
            context_length=seq_len,
            patch_length=patch_len,
            stride=stride,
            d_model=d_model,
            num_attention_heads=n_heads,
            num_hidden_layers=num_layers,
            dropout=dropout,
            head_dropout=dropout,
            prediction_length=1,  # Not used for classification
            use_cache=False,  # Disable caching to save memory
        )
        
        self.patchtst = PatchTSTModel(config)
        
        
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
        # Get PatchTST embeddings
        outputs = self.patchtst(x)
        embeddings = outputs.last_hidden_state  # (batch, num_channels, num_patches, d_model)
        
        # Pool across patches and channels
        # Average over channels (dim=1) and patches (dim=2)
        features = embeddings.mean(dim=[1, 2])  # (batch, d_model)
        
        # Classification
        out = self.fc(features)
        return out
