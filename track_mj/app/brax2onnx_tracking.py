from typing import Dict, Union, Tuple, Mapping
import functools
from absl import logging
from dataclasses import dataclass
import tyro

# --- Set environment variables ---
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from track_mj.eval.tracking.brax2onnx import get_latest_ckpt, convert_jax2onnx
from track_mj.envs.g1_tracking.utils.wrapper import wrap_fn


# --- CLI args ---
@dataclass
class Args:
    task: str
    exp_name: str


# --- Main entry point ---
def main(args: Args):
    import track_mj as tmj

    # import brax.training.agents.ppo.train as ppo
    from track_mj.learning.policy.ppo import train_tracking as ppo
    from brax.training.agents.ppo.networks import make_ppo_networks

    ckpt_path = tmj.constant.WANDB_PATH_LOG / "track" / args.exp_name / "checkpoints"
    latest_ckpt = get_latest_ckpt(ckpt_path)

    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found under: {ckpt_path}")

    logging.info(f"Using checkpoint: {latest_ckpt}")
    output_path = f"{latest_ckpt}/policy.onnx"

    env_class = tmj.registry.get(args.task, "tracking_train_env_class")
    task_cfg = tmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config
    policy_config = task_cfg.policy_config

    import json
    config_path = tmj.constant.WANDB_PATH_LOG / "track" / args.exp_name / "checkpoints" / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
        del config["policy_config"]["progress_fn"]
    env_cfg.update(config["env_config"])
    policy_config.update(config["policy_config"])

    env = env_class(terrain_type=env_cfg.terrain_type, config=env_cfg)
    env.prepare_trajectory(env._config.reference_traj_config.name)
    
    policy_obs_key = policy_config.network_factory.policy_obs_key

    network_factory = functools.partial(make_ppo_networks, **policy_config.network_factory)
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=0,
        episode_length=policy_config.episode_length,
        normalize_observations=False,
        restore_checkpoint_path=latest_ckpt,
        network_factory=network_factory,
        wrap_env_fn=wrap_fn,
        num_envs=1,
    )

    make_inference_fn, params, _ = train_fn(environment=env)
    inference_fn = make_inference_fn(params, deterministic=True)

    obs_size = env.observation_size
    act_size = env.action_size

    convert_jax2onnx(
        ckpt_dir=latest_ckpt,
        output_path=output_path,
        inference_fn=inference_fn,
        hidden_layer_sizes=policy_config.network_factory.policy_hidden_layer_sizes,
        obs_size=obs_size,
        action_size=act_size,
        policy_obs_key=policy_obs_key,
        jax_params=params,
        activation="swish",
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
