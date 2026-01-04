import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic residual block for ResNet"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNet"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetClassifier(nn.Module):
    """
    Generic ResNet classifier adapted for time series data.
    Supports ResNet18, 34, 50, 101, 152, etc.
    """

    def __init__(self, input_channels, num_classes, seq_len=None, block=Bottleneck, layers=None, 
                 fc_hidden_size=128, dropout=0.3):
        """
        Args:
            input_channels (int): Number of input channels
            num_classes (int): Number of output classes
            seq_len (int): Sequence length (optional, for consistency with other models)
            block (class): Block type - Bottleneck or BasicBlock
            layers (list): Number of blocks in each layer, e.g., [3, 4, 6, 3] for ResNet50
            fc_hidden_size (int): Hidden size for classification head
            dropout (float): Dropout rate
        """
        super(ResNetClassifier, self).__init__()
        
        if layers is None:
            layers = [3, 4, 6, 3]  # Default ResNet50 configuration
        
        self.in_channels = 64
        self.block = block
        
        # Initial convolution layer
        self.conv1 = nn.Conv1d(input_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        final_channels = 512 * block.expansion
        self.fc = nn.Sequential(
            nn.Linear(final_channels, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, num_classes)
        )

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """Build a residual layer with multiple blocks"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_channels, seq_len) or (batch_size, seq_len, input_channels)
        Returns:
            out: (batch_size, num_classes)
        """
        # input is (batch_size, seq_len, input_channels), transpose to (batch_size, input_channels, seq_len)
        x = x.transpose(1, 2)
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        out = self.fc(x)
        return out


class ResNet50Classifier(ResNetClassifier):
    """ResNet50 classifier - wrapper for convenience"""
    def __init__(self, input_channels, num_classes, seq_len=None, fc_hidden_size=128, dropout=0.3):
        super(ResNet50Classifier, self).__init__(
            input_channels, num_classes,
            seq_len=seq_len,
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            fc_hidden_size=fc_hidden_size,
            dropout=dropout
        )


class ResNet34Classifier(ResNetClassifier):
    """ResNet34 classifier - wrapper for convenience"""
    def __init__(self, input_channels, num_classes, seq_len=None, fc_hidden_size=128, dropout=0.3):
        super(ResNet34Classifier, self).__init__(
            input_channels, num_classes,
            seq_len=seq_len,
            block=BasicBlock,
            layers=[3, 4, 6, 3],
            fc_hidden_size=fc_hidden_size,
            dropout=dropout
        )


class ResNet18Classifier(ResNetClassifier):
    """ResNet18 classifier - wrapper for convenience"""
    def __init__(self, input_channels, num_classes, seq_len=None, fc_hidden_size=128, dropout=0.3):
        super(ResNet18Classifier, self).__init__(
            input_channels, num_classes,
            seq_len=seq_len,
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            fc_hidden_size=fc_hidden_size,
            dropout=dropout
        )
