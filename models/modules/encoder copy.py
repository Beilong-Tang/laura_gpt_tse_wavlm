import torch.nn as nn
from .lm import LanguageModel
from .stft import STFT

from einops import rearrange


class SpeechEncoder(nn.Module):
    def __init__(
        self,
        d=512,
        input_dim=1025,
        lm_num=6,
        nheads=16,
        hidden_dim=4 * 512,
        emb_t=259,
        mlp_norm_first=True,
        norm_first=False,
        use_window=True,
    ):
        super().__init__()
        self.stft = STFT(use_window=use_window)
        self.mlp = MLP(input_dim, hidden_dim, d, emb_t=emb_t, pre_norm=mlp_norm_first)
        self.lm = LanguageModel(d, lm_num, nheads, norm_first=norm_first)

    def forward(self, x):
        """
        :params x [B, T]
        :return [B, T, E]
        """
        y = self.stft(x)  # [B, F, T]
        y = rearrange(y, "b e t -> b t e")
        y = self.mlp(y)  #  [B, T, E]
        y = self.lm(y)  # [B, T, E]
        return y


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        pre_norm=True,
        activation="nn.ReLU",
        norm="nn.LayerNorm",
        dropout: float = 0.0,
        bias=True,
        emb_t=259,
    ):
        super().__init__()
        linear1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        linear2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        norm_layer = eval(norm)
        activation_layer = eval(activation)(input_dim)
        layers = []
        if pre_norm == True:
            layers.append(norm_layer(input_dim))
            layers.append(linear1)
            layers.append(activation_layer)
            layers.append(nn.Dropout(dropout))
            layers.append(linear2)
            layers.append(nn.Dropout(dropout))
        else:
            layers.append(linear1)
            layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer)
            layers.append(nn.Dropout(dropout))
            layers.append(linear2)
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        :params x [B, T, E']
        :return [B, T, E]
        """
        return self.layers(x)
