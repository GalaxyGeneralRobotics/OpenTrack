import torch
import torch.nn as nn

from track_mj.learning.models.dagger.policy_args import PolicyArgs


class MLP(nn.Module):
    def __init__(
        self,
        config: PolicyArgs,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()

        activate_final = config.mlp_activate_final
        hidden_dims = config.mlp_hidden_dim

        self.act = torch.nn.SiLU()
        layer_sizes = [input_dim, *hidden_dims, output_dim]
        layers = []
        for idx in range(len(layer_sizes) - 1):
            in_dim, out_dim = layer_sizes[idx], layer_sizes[idx + 1]
            layers.append(torch.nn.Linear(in_dim, out_dim))
            if idx < len(layer_sizes) - 2 or activate_final:
                layers.append(self.act)
        self.net = torch.nn.Sequential(*layers)
        self.out_head = torch.nn.Sequential(torch.nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        action = self.out_head(z)
        return action


class MLP_optimized(nn.Module):
    """
    Optimized MLP for DAgger/Behavioral Cloning.
    """

    def __init__(
        self,
        config: PolicyArgs,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()

        hidden_dims = config.mlp_hidden_dim
        use_layer_norm = getattr(config, "mlp_use_layer_norm", False)
        dropout_rate = getattr(config, "mlp_dropout_rate", 0.0)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.append(linear)

            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())

            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        output_linear = nn.Linear(prev_dim, output_dim)
        nn.init.xavier_uniform_(output_linear.weight, gain=0.01)
        nn.init.zeros_(output_linear.bias)
        layers.append(output_linear)

        self.net = nn.Sequential(*layers)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh(self.net(x))

    @torch.no_grad()
    def get_action_stats(self) -> dict:
        stats = {}
        for name, param in self.named_parameters():
            if "weight" in name:
                stats[f"{name}_norm"] = param.norm().item()
        return stats


class MLP_Absolute(nn.Module):
    """
    MLP for absolute motor target prediction (unbounded output).
    """

    def __init__(
        self,
        config: PolicyArgs,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()

        hidden_dims = config.mlp_hidden_dim
        use_layer_norm = getattr(config, "mlp_use_layer_norm", False)
        dropout_rate = getattr(config, "mlp_dropout_rate", 0.0)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.append(linear)

            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())

            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        output_linear = nn.Linear(prev_dim, output_dim)
        nn.init.xavier_uniform_(output_linear.weight, gain=1.0)
        nn.init.zeros_(output_linear.bias)
        layers.append(output_linear)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def get_action_stats(self) -> dict:
        stats = {}
        for name, param in self.named_parameters():
            if "weight" in name:
                stats[f"{name}_norm"] = param.norm().item()
        return stats
