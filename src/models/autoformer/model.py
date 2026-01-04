import torch
import torch.nn as nn
import torch.nn.functional as F
from models.autoformer.layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from models.autoformer.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from models.autoformer.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class AutoformerClassifier(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, input_channels, num_classes, seq_len, prediction_length=1,
                 d_model=128, encoder_layers=2, decoder_layers=1, encoder_attention_heads=8,
                 decoder_attention_heads=8, dropout=0.1, fc_hidden_size=128):
        super(AutoformerClassifier, self).__init__()
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.d_model = d_model

        # Decomposition
        moving_avg_kernel_size = 25  # Standard kernel size for series decomposition
        self.decomp = series_decomp(moving_avg_kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            input_channels, d_model, embed_type='timeF', freq='h', dropout=dropout
        )
        
        # Encoder
        d_ff = d_model * 4  # Feed-forward dimension
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor=1, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, encoder_attention_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg_kernel_size,
                    dropout=dropout,
                    activation='gelu'
                ) for _ in range(encoder_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )

        self.act = F.gelu
        self.dropout_layer = nn.Dropout(dropout)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, num_classes)
        )

    def forward(self, x_enc):
        """
        Args:
            x_enc: (batch_size, seq_len, input_channels)
        Returns:
            output: (batch_size, num_classes)
        """
        # Encoder embedding
        enc_out = self.enc_embedding(x_enc, None)
        
        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Classification head
        output = self.act(enc_out)
        output = self.dropout_layer(output)

        # Pool across timesteps: (batch_size, seq_len, d_model) → (batch_size, d_model)
        output = output.mean(dim=1)
        
        # Project to num_classes
        output = self.fc(output)
        
        return output
