import torch
import torch.nn as nn
from mamba_ssm import Mamba


class HybridSerialClassifier(nn.Module):
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
                 dropout=0.3):
        super(HybridSerialClassifier, self).__init__()

        # Project input to Mamba embedding dimension
        self.input_proj = nn.Linear(input_channels, d_model)

        # Mamba block
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # LSTM block
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Classifier head
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, time_steps, input_channels)

        # Step 1: project input
        x = self.input_proj(x)          # (batch, time_steps, d_model)

        # Step 2: pass through Mamba
        x = self.mamba(x)               # (batch, time_steps, d_model)

        # Step 3: pass through LSTM
        lstm_out, _ = self.lstm(x)      # (batch, time_steps, lstm_hidden)

        # Step 4: take last timestep and classify
        out = lstm_out[:, -1, :]        # (batch, lstm_hidden)
        return self.fc(out)             # (batch, num_classes)
