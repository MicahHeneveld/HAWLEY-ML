"""Simple 1D convolutional autoencoder for personal baseline error."""

from __future__ import annotations

import torch
from torch import nn


class PersonalAutoencoder(nn.Module):
    def __init__(self, input_length: int = 500, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(64 * 32, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (64, 32)),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
        )
        self.output_length = input_length

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        if decoded.size(-1) != self.output_length:
            if decoded.size(-1) > self.output_length:
                decoded = decoded[..., : self.output_length]
            else:
                decoded = nn.functional.pad(
                    decoded, (0, self.output_length - decoded.size(-1))
                )
        return decoded, latent
