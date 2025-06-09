import torch.nn as nn


class FCNClassifier(nn.Module):

    def __init__(self, input_channels, time_steps, num_classes):
        super(FCNClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),  # from (B, T, C) to (B, T*C)
            nn.Linear(time_steps * input_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)
