import os
import json

from httpx import get

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
from pathlib import Path

import track_mj as tmj
from track_mj.envs.g1_tracking.play.play_g1_env_tracking_general import PlayG1TrackingGeneralEnv


@dataclass
class Args:
    exp_name: str
    play_ref_motion: bool = False
    use_viewer: bool = False    # passive viewer (with display)
    use_renderer: bool = False  # renderer with video (headless mode)
    task: str = "G1Tracking"


@dataclass
class State:
    info: dict
    obs: dict


def get_latest_ckpt(tag):
    ckpt_dir = tmj.constant.WANDB_PATH_LOG / "track" / tag / "checkpoints"
    ckpts = [ckpt for ckpt in Path(ckpt_dir).glob("*") if not ckpt.name.endswith(".json")]
    ckpts.sort(key=lambda x: int(x.name))
    return ckpts[-1] if ckpts else None


def play(args: Args):
    env_class = tmj.registry.get(args.task, "tracking_play_env_class")
    task_cfg = tmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config
    config_path = tmj.constant.WANDB_PATH_LOG / "track" / args.exp_name / "checkpoints" / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    env_cfg.update(config["env_config"])
    env_cfg.reference_traj_config.name = config["env_config"]["reference_traj_config"]["name"]
    
    assert len(env_cfg.reference_traj_config.name) == 1, "Only one dataset is supported for now."

    env: PlayG1TrackingGeneralEnv = env_class(
        config=env_cfg,
        play_ref_motion=args.play_ref_motion,
        use_viewer=args.use_viewer,
        use_renderer=args.use_renderer,
        exp_name=args.exp_name,
    )

    ckpt_path = get_latest_ckpt(args.exp_name)
    onnx_path = ckpt_path / "policy.onnx"

    output_names = ["continuous_actions"]
    policy = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    state = env.reset()

    len_traj = env.th.traj.data.qpos.shape[0] - len(env_cfg.reference_traj_config.name[env_cfg.reference_traj_config.name.keys()[0]]) - 1
    for i in tqdm(range(len_traj)):
        onnx_input = {"obs": state.obs["state"].reshape(1, -1).astype(np.float32)}
        action = policy.run(output_names, onnx_input)[0][0]
        state = env.step(state, action)

    env.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    play(args)
