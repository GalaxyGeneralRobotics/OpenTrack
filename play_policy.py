import os
import json

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

import logging as python_logging
LOGGER = python_logging.getLogger()
LOGGER.setLevel(python_logging.INFO)

from absl import logging
logging.set_verbosity(logging.INFO)

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from tqdm import tqdm
import torch

from src.envs.g1.g1_tracking_env import default_config
from src.envs.g1.play_g1_tracking_env import PlayG1TrackingEnv


@dataclass
class Args:
    exp_name: str
    play_ref_motion: bool = False
    use_viewer: bool = False  # passive viewer (with display)
    use_renderer: bool = False  # renderer with video (headless mode)


@dataclass
class State:
    info: dict
    obs: dict


def get_latest_ckpt(path: Path) -> Path | None:
    ckpts = [ckpt for ckpt in path.glob("*") if not ckpt.name.endswith(".json")]
    ckpts.sort(key=lambda x: int(x.name))
    return ckpts[-1] if ckpts else None


def play(args: Args):
    task_cfg = default_config()
    env_cfg = task_cfg.env_config

    config_path = Path(__file__).parent / "experiments" / args.exp_name / "checkpoints" / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    env_cfg.update(config["env_config"])

    # env_cfg.reference_traj_config.name = {"lafan1": ["dance1_subject2"]}

    env = PlayG1TrackingEnv(
        terrain_type=env_cfg.terrain_type,
        config=env_cfg,
        play_ref_motion=args.play_ref_motion,
        use_viewer=args.use_viewer,
        use_renderer=args.use_renderer,
        exp_name=args.exp_name,
    )
    ckpt_path = Path(__file__).parent / "experiments" / args.exp_name / "checkpoints"
    latest_ckpt = get_latest_ckpt(ckpt_path)
    if latest_ckpt is None:
        raise FileNotFoundError("No checkpoint found.")

    policy_path = latest_ckpt / "policy.pt"
    policy_jit = torch.jit.load(policy_path, map_location="cpu")
    state = env.reset()

    len_traj = env.th.traj.data.qpos.shape[0] - env.th.n_trajectories - 1
    for i in tqdm(range(len_traj)):
        with torch.no_grad():
            action = policy_jit(torch.from_numpy(state.obs["state"].reshape(1, -1).astype(np.float32))).cpu().numpy()
        state = env.step(state, action)

    env.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    play(args)
