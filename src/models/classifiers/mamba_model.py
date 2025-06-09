import torch.nn as nn
from mamba_ssm import Mamba


class MambaClassifier(nn.Module):
    def __init__(self, input_channels, time_steps, num_classes, d_model=128, d_state=16, d_conv=4, expand=2):
        super(MambaClassifier, self).__init__()
        self.input_proj = nn.Linear(input_channels, d_model)

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, time_steps, input_channels)
        x = self.input_proj(x)          # (batch_size, time_steps, d_model)
        x = self.mamba(x)               # (batch_size, time_steps, d_model)
        out = x[:, -1, :]               # Use the last time step output
        return self.fc(out)
