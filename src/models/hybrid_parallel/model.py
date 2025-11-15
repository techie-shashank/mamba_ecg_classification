import torch
import torch.nn as nn
from mamba_ssm import Mamba


class HybridParallelClassifier(nn.Module):
    """
    Parallel Hybrid Architecture: Mamba + LSTM with Fusion
    
    Processes input through both Mamba and LSTM encoders simultaneously,
    then concatenates their outputs for classification.
    
    Architecture:
        input → Mamba encoder → 
                              → concat → FC head → output
        input → LSTM encoder → 
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
            fusion_method: Method to fuse encoders ('concat')
        """
        super(HybridParallelClassifier, self).__init__()
        
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
        
        # Determine fusion output dimension
        if fusion_method == 'concat':
            fusion_dim = d_model + lstm_hidden
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
        Forward pass through parallel encoders
        
        Args:
            x: Input tensor of shape (batch, time_steps, input_channels)
            
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Mamba encoder branch
        mamba_input = self.mamba_proj(x)              # (batch, time_steps, d_model)
        mamba_out = self.mamba(mamba_input)           # (batch, time_steps, d_model)
        mamba_pooled = mamba_out.mean(dim=1)          # (batch, d_model)
        
        # LSTM encoder branch
        lstm_out, (h_n, c_n) = self.lstm(x)          # lstm_out: (batch, time_steps, lstm_hidden)
        lstm_pooled = lstm_out.mean(dim=1)            # (batch, lstm_hidden)
        
        # Fusion
        if self.fusion_method == 'concat':
            fused = torch.cat([mamba_pooled, lstm_pooled], dim=1)  # (batch, d_model + lstm_hidden)

        # Classification
        return self.fc(fused)  # (batch, num_classes)
