"""Decoding model, per Duan et al. Sec 4.3 "Model Settings".

Paper text (verbatim, verified against duanWLDAYT24nn.pdf p.14):
  "The base model is a three layer convolutional neural network. The
  first layer performs temporal convolution for frequency information
  extraction, with filter size being (1, C), C is the number of
  channels. The second layer performs depthwise convolutions with
  temporal specific spatial filters, with filter size being (2, 32)
  across all three datasets. Third layer performs pointwise convolution
  operations for improvement on computational cost. Zero padding is
  added between neighboring layers to keep the data dimensionality."

The paper gives filter *shapes* but not output channel counts, which is
a genuine gap (also flagged in IMPLEMENTATION_MAPPING.md before any code
existed). Reading the layer-1 shape (1, C=22) together with "zero
padding ... to keep the data dimensionality" means layer 1 preserves
both the 400 (time) and 22 (channel) axes -- i.e. it is a same-padded
conv whose kernel spans the full channel width at each single time
step. Layer 2's "depthwise convolutions with temporal specific spatial
filters" is Duan's own description of an EEGNet-style depthwise conv:
one (or more) spatial filter(s) per layer-1 temporal filter. Reading
"(2, 32)" as (kernel_size_in_time=2, num_output_filters=32) reproduces
the paper's own number: with F1=16 layer-1 filters and a depth
multiplier of 2, depthwise conv yields exactly 16*2=32 channels,
matching literally. F1=16 is the one number Duan et al. do not give
anywhere (IMPLEMENTATION DECISION).

This module implements exactly those 3 conv layers plus the minimal
addition needed to get a 4-class prediction (pooling + linear head),
which the paper does not describe in this section but is required for
any classifier -- flagged here, not silently assumed.
"""
import torch
import torch.nn as nn

NUM_CHANNELS = 22
NUM_CLASSES = 4
F1 = 16
DEPTH_MULT = 2  # -> F1 * DEPTH_MULT = 32, matching Duan et al.'s "(2, 32)"


class MudviCNN(nn.Module):
    def __init__(self, num_channels: int = NUM_CHANNELS, num_classes: int = NUM_CLASSES,
                 f1: int = F1, depth_mult: int = DEPTH_MULT, dropout: float = 0.5):
        super().__init__()
        f2 = f1 * depth_mult

        # Layer 1: temporal conv, filter (1, C) -- same-padded on both axes.
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, num_channels), padding="same", bias=False),
            nn.BatchNorm2d(f1),
            nn.ELU(),
        )

        # Layer 2: depthwise conv, filter (2, ...), temporal-specific spatial filters.
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(f1, f2, kernel_size=(2, 1), padding="same", groups=f1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(4, 1)),
            nn.Dropout(dropout),
        )

        # Layer 3: pointwise conv, for computational efficiency.
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(f2, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(f2, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 22, 400) -> feature vector (B, f2), used by the memory's
        gradient-norm importance scoring extension point (not used by the
        baseline itself, kept for the future informativeness-weighted
        memory update if it is added later)."""
        x = x.unsqueeze(1).transpose(2, 3)  # (B, 1, 400, 22)
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 22, 400) raw segment -> (B, num_classes) logits."""
        feat = self.features(x)
        return self.classifier(feat)
