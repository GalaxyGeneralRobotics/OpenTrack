import os
import json

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

from dataclasses import dataclass

import numpy as np
import onnxruntime as rt
import tyro
from tqdm import tqdm

import track_mj as tmj
from track_mj.envs.g1_tracking_dagger.play.play_g1_env_tracking_general import PlayG1TrackingGeneralEnv
from track_mj.learning.models.dagger.policy_args import PolicyArgs, ONNXPolicyArgs
from track_mj.learning.models.dagger.policy import get_policy_onnx

@dataclass
class Args:
    exp_name: str
    play_ref_motion: bool = False
    use_viewer: bool = False    # passive viewer (with display)
    use_renderer: bool = False  # r
    task: str = "G1Tracking"

    first_n_traj: int = -1
    
@dataclass
class State:
    info: dict
    obs: dict


def play(args: Args):
    env_class = tmj.registry.get(args.task, "tracking_dagger_play_env_class")
    task_cfg = tmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config

    exp_dir = tmj.constant.WANDB_PATH_LOG / "dagger" / args.exp_name
    assert exp_dir.exists(), f"Experiment directory does not exist: {exp_dir}"

    config_path = exp_dir / "checkpoints" / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # overwrite reference motions
    
    if args.first_n_traj > 0:
        for key in config["env_config"]["reference_traj_config"]["name"]:
            config["env_config"]["reference_traj_config"]["name"][key] = config["env_config"]["reference_traj_config"]["name"][key][: args.first_n_traj]

    env_cfg.reference_traj_config.name = config["env_config"]["reference_traj_config"]["name"]
    env_cfg.update(config["env_config"])

    policy_args = PolicyArgs.from_config_dict(config["policy_config"]["policy_args"])

    onnx_path = exp_dir / "checkpoints" / "model.onnx"
    assert onnx_path.exists(), f"ONNX model does not exist: {onnx_path}"
    
    env: PlayG1TrackingGeneralEnv = env_class(
        terrain_type=env_cfg.terrain_type,
        config=env_cfg,
        play_ref_motion=args.play_ref_motion,
        use_viewer=args.use_viewer,
        use_renderer=args.use_renderer,
        exp_name=args.exp_name,
    )
    
    policy = get_policy_onnx(ONNXPolicyArgs(**{
        "onnx_dir": str(onnx_path),
        **vars(policy_args)
    }))
    
    state = env.reset()

    first_dataset_name = next(iter(env_cfg.reference_traj_config.name.keys()))
    len_traj = env.th.traj.data.qpos.shape[0] - len(env_cfg.reference_traj_config.name[first_dataset_name]) - 1

    for i in tqdm(range(len_traj)):
        obs = state.obs[policy_args.policy_obs_key].reshape(1, -1).astype(np.float32)
        nn_state_dict = {
            policy_args.policy_obs_key: obs,
        }
        action = policy.infer(nn_state_dict)[0] # (1, action_dim) --> (action_dim,)
        
        state = env.step(state, action)
    
    env.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    play(args)