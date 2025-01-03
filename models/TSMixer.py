from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Mixing import MixerLayer, TimeBatchNorm2d, feature_to_time, time_to_feature


class Model(nn.Module):
    """TSMixer model for time series forecasting.

    This model uses a series of mixer layers to process time series data,
    followed by a linear transformation to project the output to the desired
    prediction length.

    Attributes:
        mixer_layers: Sequential container of mixer layers.
        temporal_projection: Linear layer for temporal projection.

    Args:
        sequence_length: Length of the input time series sequence.
        prediction_length: Desired length of the output prediction sequence.
        input_channels: Number of input channels.
        output_channels: Number of output channels. Defaults to None.
        activation_fn: Activation function to use. Defaults to "relu".
        num_blocks: Number of mixer blocks. Defaults to 2.
        dropout_rate: Dropout rate for regularization. Defaults to 0.1.
        ff_dim: Dimension of feedforward network inside mixer layer. Defaults to 64.
        normalize_before: Whether to apply layer normalization before or after mixer layer.
        norm_type: Type of normalization to use. "batch" or "layer". Defaults to "batch".
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # Transform activation_fn to callable
        activation_fn = getattr(F, "relu")
        self.sequence_length = configs.seq_len
        self.prediction_length = configs.pred_len
        self.input_channels = configs.enc_in
        self.output_channels = configs.enc_in
        self.norm_type = configs.norm_type
        self.norm = TimeBatchNorm2d if self.norm_type == "batch" else nn.LayerNorm
        self.num_blocks = configs.num_blocks
        self.dropout_rate= 0.1,
        self.ff_dim = configs.ff_dim,
        self.normalize_before = configs.normalize_before,

        # Build mixer layers
        self.mixer_layers = self._build_mixer(
            num_blocks=self.num_blocks,
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            ff_dim=self.ff_dim,
            activation_fn=activation_fn,
            dropout_rate=0.1,
            sequence_length=self.sequence_length,
            normalize_before=self.normalize_before,
            norm_type=self.norm,
        )

        # Temporal projection layer
        self.temporal_projection = nn.Linear(self.sequence_length, self.prediction_length)

    def _build_mixer(
        self, num_blocks: int, input_channels: int, output_channels: int, **kwargs
    ):
        """Build the mixer blocks for the model.

        Args:
            num_blocks (int): Number of mixer blocks to be built.
            input_channels (int): Number of input channels for the first block.
            output_channels (int): Number of output channels for the last block.
            **kwargs: Additional keyword arguments for mixer layer configuration.

        Returns:
            nn.Sequential: Sequential container of mixer layers.
        """
        output_channels = output_channels if output_channels is not None else input_channels
        channels = [input_channels] * (num_blocks - 1) + [output_channels]

        return nn.Sequential(
            *[
                MixerLayer(input_channels=in_ch, output_channels=out_ch, **kwargs)
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TSMixer model.

        Args:
            x_hist (torch.Tensor): Input time series tensor.

        Returns:
            torch.Tensor: The output tensor after processing by the model.
        """
        x = self.mixer_layers(x_hist)

        x_temp = feature_to_time(x)
        x_temp = self.temporal_projection(x_temp)
        x = time_to_feature(x_temp)

        return x