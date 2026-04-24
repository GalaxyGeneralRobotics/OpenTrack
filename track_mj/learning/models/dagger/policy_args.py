from dataclasses import dataclass, asdict, fields, field
from ml_collections import config_dict

@dataclass
class PolicyArgs:
    policy_type: str = "mlp"

    # ===== model general config =====
    obs_dim: int = 0
    aux_obs_dim: int = 0
    act_dim: int = 0
    bf16: bool = False
    load_path: str = ""
    policy_obs_key: str = "state"
    policy_auxiliary_obs_key: str = "auxiliary_state"
    output_residual_action: bool = True

    # ===== MLP config =====
    mlp_hidden_dim: list[int] = field(default_factory=lambda: [512, 512, 256, 256, 128])
    mlp_activate_final: bool = False
    
    def to_config_dict(self) -> config_dict.ConfigDict:
        return config_dict.create(**asdict(self))
    
    @classmethod
    def from_config_dict(cls, cfg: config_dict.ConfigDict) -> "PolicyArgs":
        field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in cfg.items() if k in field_names}
        return cls(**filtered)


@dataclass
class ONNXPolicyArgs(PolicyArgs):
    onnx_dir: str = ""