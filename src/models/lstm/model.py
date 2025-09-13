import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(self, input_channels, num_classes, hidden_size=64, num_layers=1, fc_hidden_size=128, dropout=0.3):
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(input_size=input_channels,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)  # Add bidirectional capability

        # Update fully connected layer to handle bidirectional output (hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, fc_hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # output shape: (batch_size, time_steps, hidden_size * 2) for bidirectional
        out = lstm_out.mean(dim=1)  # Global average pooling across all timesteps
        return self.fc(out)
