import torch
import torch.nn as nn
from mamba_ssm import Mamba


class HybridSerialReversedClassifier(nn.Module):
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
        super(HybridSerialReversedClassifier, self).__init__()

        # LSTM block
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Project LSTM output to Mamba embedding dimension
        self.input_proj = nn.Linear(lstm_hidden, d_model)

        # Mamba block
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # Classifier head
        self.fc = nn.Sequential(
            nn.Linear(d_model, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, time_steps, input_channels)

        # Step 1: pass through LSTM
        lstm_out, _ = self.lstm(x)          # (batch, time_steps, lstm_hidden)

        # Step 2: project input
        mamba_inp = self.input_proj(lstm_out)          # (batch, time_steps, d_model)

        # Step 3: pass through Mamba
        mamba_out = self.mamba(mamba_inp)               # (batch, time_steps, d_model)

        # Step 4: take last timestep and classify
        out = mamba_out.mean(dim=1)             # Global average pooling across all timesteps
        return self.fc(out)             # (batch, num_classes)
