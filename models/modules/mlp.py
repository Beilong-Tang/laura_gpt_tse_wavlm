import torch.nn as nn


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
        activation_layer = eval(activation)()
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
