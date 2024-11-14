import torch
import torch.nn as nn


class STFT(nn.Module):
    def __init__(self, window: int = 2048, hop: int = 512, use_window=True):
        super().__init__()
        self.window = window
        self.hop = hop
        self.window_type = torch.hann_window(self.window)
        self.use_window = use_window

    def forward(self, x):
        """
        :param x has shape [B, T]
        :return audio with shape [B, F, T]
        """
        assert len(x.shape) == 2
        self.window_type = self.window_type.to(x.device)
        y = torch.stft(
            x,
            self.window,
            self.hop,
            window=self.window_type if self.use_window else None,
            return_complex=True,
        )
        w = torch.abs(y)
        w = w.pow(0.3)
        return w
