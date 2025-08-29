import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(self, input_channels, num_classes, hidden_size=64, num_layers=1):
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(input_size=input_channels,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # output shape: (batch_size, time_steps, hidden_size)
        out = lstm_out[:, -1, :]  # Take output from last time step
        return self.fc(out)
