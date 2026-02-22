"""EEGNet — compact CNN for EEG classification (Lawhern et al. 2018).

Adapted for Muse 2 (4 channels, 256 Hz). Supports:
  - P300 detection (target vs non-target)
  - Gesture classification (idle / blink / clench / noise)
  - Unified multi-class mode
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet architecture for raw EEG classification.

    Input shape:  (batch, 1, n_channels, n_samples)
    Output shape: (batch, n_classes)
    """

    def __init__(
        self,
        n_channels: int = 4,
        n_samples: int = 230,
        n_classes: int = 2,
        dropout_rate: float = 0.5,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kern_length: int = 64,
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

        # Block 1: temporal convolution + depthwise spatial convolution
        self.block1 = nn.Sequential(
            # Temporal filter: learns band-pass-like filters
            nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial filter: learns cross-channel patterns
            _DepthwiseConv2d(F1, D, (n_channels, 1), bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        # Block 2: separable convolution
        self.block2 = nn.Sequential(
            _SeparableConv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        # Compute flattened size by running a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            dummy = self.block1(dummy)
            dummy = self.block2(dummy)
            self._flat_size = dummy.numel()

        self.classifier = nn.Linear(self._flat_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (predicted_class, probabilities) for a batch."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        return preds, probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _DepthwiseConv2d(nn.Module):
    """Depthwise convolution: each input channel gets its own set of filters."""

    def __init__(self, in_channels: int, depth_multiplier: int, kernel_size: tuple, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels * depth_multiplier,
            kernel_size,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _SeparableConv2d(nn.Module):
    """Separable convolution = depthwise + pointwise (1×1)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 padding: tuple = (0, 0), bias: bool = True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


# ── Pre-configured model factories ────────────────────────────

def create_p300_model(n_samples: int = 230) -> EEGNet:
    """EEGNet for P300 detection (binary: target vs non-target)."""
    return EEGNet(n_channels=4, n_samples=n_samples, n_classes=2)


def create_gesture_model(n_samples: int = 256, n_channels: int = 24) -> EEGNet:
    """EEGNet for gesture classification (idle / blink / clench / noise).

    Default n_channels=24: 4 raw EEG + 20 spectral band-power channels.
    """
    return EEGNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=4,
        F1=16,
        D=2,
        F2=32,
        kern_length=64,
    )


def create_unified_model(n_samples: int = 256) -> EEGNet:
    """EEGNet for all classes: idle / p300_target / blink / clench / noise."""
    return EEGNet(
        n_channels=4,
        n_samples=n_samples,
        n_classes=5,
        F1=16,
        D=2,
        F2=32,
        kern_length=64,
    )
