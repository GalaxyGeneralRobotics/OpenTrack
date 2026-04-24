import os
import numpy as np
import onnxruntime as rt
import torch

from track_mj.learning.models.dagger.action_expert import MLP_optimized, MLP_Absolute
from track_mj.learning.models.dagger.policy_args import PolicyArgs, ONNXPolicyArgs


class MLP_Policy:
    def __init__(
        self,
        config: PolicyArgs,
    ):
        self.policy_obs_key = config.policy_obs_key
        self.policy_auxiliary_obs_key = config.policy_auxiliary_obs_key

        if config.output_residual_action:
            self.model = MLP_optimized(
                config=config,
                input_dim=config.obs_dim,
                output_dim=config.act_dim,
            )
        else:
            self.model = MLP_Absolute(
                config=config,
                input_dim=config.obs_dim,
                output_dim=config.act_dim,
            )

    @torch.no_grad()
    def infer(self, obs_dict: dict) -> torch.Tensor:
        obs = obs_dict.get(self.policy_obs_key, None)
        return self.model(obs)

    def save_pretrained(self, save_path, file_name_sufix: str = None, use_ddp: bool = False):
        if use_ddp:
            torch.save(self.model.module.state_dict(), os.path.join(save_path, f"mlp_action_expert{file_name_sufix}.pth"))
        else:
            torch.save(self.model.state_dict(), os.path.join(save_path, f"mlp_action_expert{file_name_sufix}.pth"))

    def compute_loss(self, obs_dict: dict, target_action: torch.Tensor, aux_loss_info: dict = None):
        obs: torch.Tensor = obs_dict.get(self.policy_obs_key, None)

        pred_action = self.model(obs)
        bc_loss = torch.nn.functional.mse_loss(pred_action, target_action)
        info = {"bc_loss": bc_loss.item(), "action": pred_action}
        return bc_loss, info


def get_policy(args: PolicyArgs):
    if args.policy_type != "mlp":
        raise ValueError(f"Only mlp policy is supported, got: {args.policy_type}")
    return MLP_Policy(config=args)


class MLP_Policy_ONNX:
    def __init__(self, config: ONNXPolicyArgs):
        self.policy_obs_key = config.policy_obs_key
        self.onnx_path = os.path.join(config.onnx_dir, "model.onnx")
        self.onnx_model = rt.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        print(f"Loaded ONNX model from {self.onnx_path}")

    def infer(self, obs_dict: dict, deterministic: bool = False, inference_method: str = "posterior") -> np.ndarray:
        obs = obs_dict.get(self.policy_obs_key, None)
        nn_action = self.onnx_model.run(["continuous_actions"], {"obs": obs})[0]
        return nn_action


def get_policy_onnx(args: ONNXPolicyArgs):
    if args.policy_type != "mlp":
        raise ValueError(f"Only mlp policy is supported, got: {args.policy_type}")
    return MLP_Policy_ONNX(args)
