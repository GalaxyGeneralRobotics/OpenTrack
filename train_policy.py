import os

import logging as python_logging
LOGGER = python_logging.getLogger()
LOGGER.setLevel(python_logging.INFO)

from absl import logging
logging.set_verbosity(logging.INFO)

import functools
import time
import datetime
from dataclasses import dataclass
from pathlib import Path
import tyro
import wandb
import numpy as np
import jax.numpy as jp
from brax.training.agents.ppo.networks import make_ppo_networks

from src.learning.ppo import train_ppo as ppo
from src.envs.g1.g1_tracking_env import G1TrackingEnv, default_config
from src.envs.g1.wrapper import wrap_fn
from src.envs.g1.randomize import domain_randomize_model, domain_randomize_terrain


@dataclass
class Args:
    exp_name: str = "debug"
    num_timesteps: int = 3_000_000_000
    enable_randomize: bool = True
    terrain_type: str = "flat_terrain" # choose from flat_terrain, rough_terrain


def _setup_paths(exp_name: str) -> tuple[Path, Path]:
    logdir = Path(__file__).parent / "experiments" / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    return logdir, ckpt_path


def _apply_policy_args_to_config(args: Args, cfg, debug: bool):
    cfg.num_timesteps = args.num_timesteps
    if debug:
        cfg.training_metrics_steps = 1000
        cfg.num_evals = 0
        cfg.batch_size = 8
        cfg.num_minibatches = 2
        cfg.num_envs = cfg.batch_size * cfg.num_minibatches
        cfg.episode_length = 200
        cfg.unroll_length = 10
        cfg.num_updates_per_batch = 1
        cfg.action_repeat = 1
        cfg.num_timesteps = 100_000
        cfg.num_resets_per_eval = 1


def _prepare_training_params(cfg, ckpt_path: Path):
    params = cfg.to_dict()
    params.pop("network_factory", None)
    params["wrap_env_fn"] = wrap_fn
    network_fn = make_ppo_networks
    params["network_factory"] = (
        functools.partial(network_fn, **cfg.network_factory) if hasattr(cfg, "network_factory") else network_fn
    )
    params["save_checkpoint_path"] = ckpt_path
    return params


def _progress(num_steps, metrics, times, total_steps, debug_mode):
    now = time.monotonic()
    times.append(now)
    if metrics and not debug_mode:
        try:
            wandb.log(metrics, step=num_steps)
        except Exception as e:
            logging.warning(f"wandb.log failed: {e}")

    if len(times) < 2 or num_steps == 0:
        return
    step_times = np.diff(times)
    median_step_time = np.median(step_times)
    if median_step_time <= 0:
        return
    steps_logged = num_steps / len(step_times)
    est_seconds_left = (total_steps - num_steps) / steps_logged * median_step_time
    logging.info(f"NumSteps {num_steps} - EstTimeLeft {est_seconds_left:.1f}[s]")


def get_trajectory_handler(env):
    # load reference trajectory
    trajectory_data = env.prepare_trajectory(env._config.reference_traj_config.name)
    env.th.traj = None

    # output the dataset and observation info of general tracker
    print("=" * 50)
    print(
        f"Tracking {len(trajectory_data.split_points) - 1} trajectories with {trajectory_data.qpos.shape[0]} timesteps, fps={1 / env.dt:.1f}"
    )
    print(f"Observation: {env._config.obs_keys}")
    print(f"Privileged state: {env._config.privileged_obs_keys}")
    print("=" * 50)

    return trajectory_data


def train(args: Args):
    task_cfg = default_config()
    env_cfg = task_cfg.env_config
    policy_cfg = task_cfg.policy_config

    timestamp = datetime.datetime.now().strftime("%m%d%H%M")
    args.exp_name = f"{timestamp}_{args.exp_name}"

    debug_mode = "debug" in args.exp_name
    logdir, ckpt_path = _setup_paths(args.exp_name)

    _apply_policy_args_to_config(args, policy_cfg, debug_mode)
    env_cfg.history_len = 0

    env_cfg.enable_randomize = args.enable_randomize
    env_cfg.push_config.enable = args.enable_randomize
    env_cfg.terrain_type = args.terrain_type

    policy_params = _prepare_training_params(policy_cfg, ckpt_path)

    wandb.init(
        project="any2track",
        name=args.exp_name,
        mode="online" if not debug_mode else "disabled"
    )
    wandb.config.update(task_cfg.to_dict())
    config_path = ckpt_path / "config.json"
    config_path.write_text(task_cfg.to_json_best_effort(indent=4))

    train_fn = functools.partial(ppo.train, **policy_params)
    times = [time.monotonic()]

    env = G1TrackingEnv(terrain_type=env_cfg.terrain_type, config=env_cfg)
    trajectory_data = get_trajectory_handler(env)

    if env_cfg.terrain_type == "rough_terrain":
        hfield_data = jp.asarray(np.load("data/hfield/terrain.npz")["hfield_data"])
        dr_func = functools.partial(domain_randomize_terrain, all_hfield_data=hfield_data)
    else:
        dr_func = domain_randomize_model

    make_inference_fn, params, _ = train_fn(
        environment=env,
        trajectory_data=trajectory_data,
        progress_fn=lambda s, m: _progress(s, m, times, policy_cfg.num_timesteps, debug_mode),
        policy_params_fn=lambda *args: None,
        randomization_fn=dr_func if env_cfg.enable_randomize else None,
    )
    logging.info(f"Run {args.exp_name} Train done.")


if __name__ == "__main__":
    train(tyro.cli(Args))
