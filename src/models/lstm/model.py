import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(self, input_channels, num_classes, hidden_size=64, num_layers=1, fc_hidden_size=128, dropout=0.3, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_channels,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        # Calculate the actual LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Correctly sized fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # output shape: (batch_size, time_steps, hidden_size)
        out = lstm_out.mean(dim=1)  # Global average pooling across all timesteps
        return self.fc(out)
