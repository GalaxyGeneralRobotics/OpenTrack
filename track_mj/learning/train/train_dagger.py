import os
import sys

from track_mj.learning.policy.dagger import dagger_horizon

if "LOCAL_RANK" in os.environ:
    device_offset = int(os.environ.get("CUDA_DEVICE_OFFSET", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(os.environ["LOCAL_RANK"]) + device_offset)
    print(f"[Pre-init] Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} for rank {os.environ.get('RANK', '?')}")

import inspect
import functools
import time
import pytz
import json
from ml_collections import config_dict
from typing import List, Tuple, Dict, Any, Union
import torch
import torch.distributed as dist

from track_mj.learning.models.dagger.policy_args import PolicyArgs

from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from absl import logging
from typing import Any, Callable, Optional, Tuple
import tqdm
import tyro
import wandb
import numpy as np
import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src import mjx_env

WANDB_PROJECT = os.environ.get("WANDB_PROJECT")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")

from brax.training.agents.ppo.networks import make_ppo_networks

import track_mj as tmj
from track_mj import update_file_handler
from track_mj.constant import WANDB_PATH_LOG
from track_mj.envs.g1_tracking_dagger.utils.wrapper import wrap_fn
from track_mj.dr.domain_randomize_tracking_dagger import (
    domain_randomize,
    domain_randomize_terrain,
)


# ========================== Args ==========================


@dataclass
class TrainingConfig:
    # ===== training config =====
    # In Tracking: 
    #   - n_envs=32768 (bsz=1024, n_mini_b=32, unroll_l=20).
    #   - Collect transition seq of shape (1024*32)*20) per ppo update from all devices.
    #   - Optimize model with transition seq with shape (1024*20) per minibatch.
    #   - n_timesteps=3_000_000_000, with (ceil(n_timesteps / (bsz * n_mini_b * unroll_l * act_repeat))) = 4578 steps

    num_envs: int = 2048 * 8
    num_training_steps: int = 400_000
    verbose: bool = False
    save_ckpts: int = 40
    log_freq: int = 1_00
    lr: float = 1e-4 * 8
    lr_scheduler: str = "cosine_annealing"      # "constant", "cosine_annealing"
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    use_test_set: bool = False
    load_pretrained_path: str = ""

@dataclass
class Args:
    task: str
    exp_name: str = "debug"
    exp_tags: str = None
    exp_notes: str = None
    seed: int = 42
    convert_onnx: bool = True

    use_ddp: bool = False

    policy_type: str = "mlp"
    student_obs_frame: str = "local"   # "actor_root" / "local"
    dagger_config_path: str = ""
    student_use_residual_action: bool = True
    dagger_horizon: int = 1
    dagger_learning_epochs: int = 10

    training: TrainingConfig = field(default_factory=TrainingConfig)
    policy: PolicyArgs = field(default_factory=PolicyArgs)

    
# ========================== Helper Functions ==========================


def get_ddp_params():
    dist.init_process_group("nccl", device_id=0)
    world_size = int(os.environ.get("WORLD_SIZE"))
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(0)
    return rank, world_size

def _prepare_exp_name(task: str, exp_name: str) -> str:
    r"""
    timestamp_task_expname
    """
    cst_time = datetime.now(pytz.timezone('Asia/Shanghai'))
    timestamp = cst_time.strftime("%m%d%H%M")
    return f"{timestamp}_{task}_{exp_name}"

def _parse_exp_tags(tags):
    r"""
    Parse tags like `"'[tag1, tag2]'" into a list.
    """
    if isinstance(tags, list):
        return tags
    if isinstance(tags, str):
        cleaned = tags.strip()
        if (cleaned.startswith('[') and cleaned.endswith(']')) or \
           (cleaned.startswith('(') and cleaned.endswith(')')) or \
           (cleaned.startswith('"') and cleaned.endswith('"')) or \
           (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1]
        # Handle quoted tags
        result = []
        for tag in cleaned.split(','):
            tag = tag.strip()
            if tag.startswith('"') and tag.endswith('"') or \
               tag.startswith("'") and tag.endswith("'"):
                tag = tag[1:-1]
            if tag:  # Ensure no empty tags are added
                result.append(tag)
        return result
    return [str(tags)]


def parse_dagger_config(dagger_config_path: str) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray], Dict[str, int]]:
    r"""
    Returns:
        privileged_obs_keys: List[str], the privileged observation keys used by teacher policies.
        traj_cluster_sample_probs: np.ndarray, the sampling probabilities for each motion cluster.
        traj_sample_probs: Dict[str, np.ndarray], the sampling probabilities for each trajectory.
        traj_sample_cluster_ids: Dict[str, int], the cluster id for each trajectory.
    """
    assert dagger_config_path != "", "Dagger config path must be specified."
    dagger_config_path = os.path.join(os.getcwd(), dagger_config_path)
    assert os.path.exists(dagger_config_path), f"Dagger config path {dagger_config_path} does not exist."

    with open(dagger_config_path, 'r') as f:
        dagger_config = json.load(f)
    assert "motion_clusters" in dagger_config, "dagger config must contain motion_clusters field."
    motion_clusters = dagger_config.get("motion_clusters", None)

    last_cluster_teacher_obs = None
    all_trajs = []
    traj_cluster_sample_probs = []
    traj_sample_probs = []
    traj_sample_cluster_ids = []

    ref_motion_root = os.path.join(os.getcwd(), "storage", "data", "mocap", "lafan1", "UnitreeG1")

    for cluster_id, cluster in enumerate(motion_clusters):
        _cluster_id = cluster["cluster_id"]
        _teacher_ckpt_dir = os.path.join(os.getcwd(), cluster["teacher_ckpt_dir"])
        _motions = cluster["motions"]
        _cluster_sample_prob = cluster["prob"]

        try:
            with open(os.path.join(_teacher_ckpt_dir, "checkpoints", "config.json"), 'r') as f:
                _teacher_env_cfg = json.load(f)["env_config"]
            _cluster_teacher_obs = _teacher_env_cfg["obs_keys"]

            if last_cluster_teacher_obs is not None:
                if set(_cluster_teacher_obs) != set(last_cluster_teacher_obs):
                    raise ValueError("Privileged obs keys set not consistent")
                if _cluster_teacher_obs != last_cluster_teacher_obs:
                    raise ValueError("Privileged obs keys order are not consistent")
            last_cluster_teacher_obs = _cluster_teacher_obs

            for _m in _motions:
                if not os.path.exists(os.path.join(ref_motion_root, _m + ".npz")):
                    raise FileNotFoundError(f"Reference motion {_m} does not exist.")
                traj_sample_cluster_ids.append(cluster_id)
            if set(_motions).intersection(set(all_trajs)):
                raise ValueError(f"Motion names in cluster id {_cluster_id} overlap with previous clusters.")
            all_trajs.extend(_motions)
            traj_cluster_sample_probs.append(np.float32(_cluster_sample_prob))
            traj_sample_probs.extend([np.float32(_cluster_sample_prob) / np.float32(len(_motions))] * len(_motions))
        
        except Exception as e:
            print(f"Error in cluster_id: {_cluster_id}, teacher_ckpt_dir: {_teacher_ckpt_dir}")
            print(f"Error details: {str(e)}")
            raise e

        print(f"Finished checking cluster id {_cluster_id}, with teacher ckpt dir {_teacher_ckpt_dir}")

    assert last_cluster_teacher_obs is not None, "No motion clusters found in dagger config."
    traj_cluster_sample_probs = np.array(traj_cluster_sample_probs, dtype=np.float32)
    traj_cluster_sample_probs = traj_cluster_sample_probs / np.sum(traj_cluster_sample_probs, dtype=np.float32)

    traj_sample_probs = np.array(traj_sample_probs, dtype=np.float32)
    traj_sample_probs = traj_sample_probs / np.sum(traj_sample_probs, dtype=np.float32)

    print(f"\nPrivileged observation keys from DAgger config: {last_cluster_teacher_obs}\n")

    traj_sample_probs = {k: v for k, v in zip(all_trajs, traj_sample_probs)}
    traj_sample_cluster_ids = {k: v for k, v in zip(all_trajs, traj_sample_cluster_ids)}

    return last_cluster_teacher_obs, traj_cluster_sample_probs, traj_sample_probs, traj_sample_cluster_ids

def _apply_policy_args_to_config(args: Args, policy_cfg: config_dict.ConfigDict, debug: bool):

    policy_cfg.num_envs = args.training.num_envs
    policy_cfg.num_training_steps = args.training.num_training_steps
    policy_cfg.debug = debug
    policy_cfg.verbose = args.training.verbose
    policy_cfg.save_freq = args.training.num_training_steps // args.training.save_ckpts
    policy_cfg.log_freq = args.training.log_freq
    policy_cfg.lr = args.training.lr
    policy_cfg.lr_scheduler = args.training.lr_scheduler
    policy_cfg.weight_decay = args.training.weight_decay
    policy_cfg.max_grad_norm = args.training.max_grad_norm
    policy_cfg.use_test_set = args.training.use_test_set

    policy_cfg.policy_args = args.policy.to_config_dict()

    if debug:
        policy_cfg.num_envs = 16
        policy_cfg.num_training_steps = 2000
        policy_cfg.save_freq = 20
        policy_cfg.log_freq = 2
        policy_cfg.episode_length = 1000
        policy_cfg.action_repeat = 1

def _apply_env_args_to_config(
        args: Args, env_cfg: config_dict.ConfigDict,
        privileged_obs_keys: List[str], traj_cluster_sample_probs: np.ndarray, traj_sample_probs: Dict[str, np.ndarray], traj_sample_cluster_ids: Dict[str, int]
    ):

    # read priv obs from teacher config
    env_cfg.privileged_obs_keys = None          # clear first
    env_cfg.dagger_config_path = args.dagger_config_path
    env_cfg.privileged_obs_keys, env_cfg.traj_cluster_sample_probs = privileged_obs_keys, traj_cluster_sample_probs.tolist()
    env_cfg.traj_sample_cluster_ids = list(traj_sample_cluster_ids.items())
    env_cfg.traj_sample_probs = list({k: float(v) for k, v in traj_sample_probs.items()}.items())
    env_cfg.student_use_residual_action = args.student_use_residual_action
    env_cfg.dagger_horizon = args.dagger_horizon
    env_cfg.dagger_learning_epochs = args.dagger_learning_epochs

    # Ensure the actual loaded reference trajectories follow the DAgger config
    all_dagger_trajs = list(traj_sample_probs.keys())
    assert len(all_dagger_trajs) > 0, "No trajectories found after parsing DAgger config."
    if env_cfg.reference_traj_config.name is None or len(env_cfg.reference_traj_config.name) == 0:
        env_cfg.reference_traj_config.name = {"lafan1": all_dagger_trajs}
    else:
        dataset_name = next(iter(env_cfg.reference_traj_config.name.keys()))
        env_cfg.reference_traj_config.name = {dataset_name: all_dagger_trajs}

    cluster_counts: Dict[int, int] = {}
    for _, cluster_id in traj_sample_cluster_ids.items():
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if (not args.use_ddp) or local_rank == 0:
        print(
            f"DAgger trajectory override: dataset={list(env_cfg.reference_traj_config.name.keys())}, "
            f"num_trajectories={len(all_dagger_trajs)}, cluster_counts={cluster_counts}"
        )

    # TODO: modify student obs
    env_cfg.obs_frame = args.student_obs_frame

    # Modify obs keys according to policy type
    if args.policy.policy_type in ["mlp"]:
        env_cfg.obs_keys = [
                "gyro_pelvis",
                "gvec_pelvis",
                "joint_pos",
                "joint_vel",
                "last_motor_targets",
                "dif_joint_pos",
                "dif_joint_vel",
                "ref_root_height",
                "ref_feet_height",
            ]
        env_cfg.auxiliary_obs_keys = []
    
    # env_cfg.privileged_obs_keys = env_cfg.privileged_obs_keys
    env_cfg.obs_keys = sorted(list(set(env_cfg.obs_keys)))
    env_cfg.auxiliary_obs_keys = sorted(list(set(env_cfg.auxiliary_obs_keys)))


def _enable_debug_mode():
    jax.config.update("jax_traceback_filtering", "off")
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)
    # jax.config.update("jax_disable_jit", True)


def _setup_paths(exp_name: str) -> tuple[Path, Path]:
    logdir = Path(WANDB_PATH_LOG) / "dagger" / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    update_file_handler(filename=f"{logdir}/info.log")
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    return logdir, ckpt_path

def _log_checkpoint_path(ckpt_path: Path):
    logging.info(f"Checkpoint path: {ckpt_path}")

def _prepare_training_params(policy_cfg: config_dict.ConfigDict, student_obs_size: Dict, student_act_size: int, ckpt_path: Path, student_use_residual_action: bool, load_pretrained_path: str = "") -> Dict[str, Any]:

    assert policy_cfg.policy_args.policy_obs_key in student_obs_size, \
        f"Policy obs key '{policy_cfg.policy_args.policy_obs_key}' not in student observation keys {list(student_obs_size.keys())}."
    assert len(student_obs_size[policy_cfg.policy_args.policy_obs_key]) == 1, \
        f"Observation dimension for policy obs key '{policy_cfg.policy_args.policy_obs_key}' should be 1D, but got {student_obs_size[policy_cfg.policy_args.policy_obs_key]}."
    assert policy_cfg.policy_args.policy_type is not None and policy_cfg.policy_args.policy_type != "", "Policy type must be specified in policy args."
    
    policy_cfg.policy_args.obs_dim = student_obs_size[policy_cfg.policy_args.policy_obs_key][0]
    # Auxiliary observation is currently unused by the student policy.
    policy_cfg.policy_args.aux_obs_dim = 0
    policy_cfg.policy_args.act_dim = student_act_size
    policy_cfg.save_dir = str(ckpt_path)
    policy_cfg.policy_args.load_path = load_pretrained_path
    policy_cfg.policy_args.output_residual_action = student_use_residual_action

    params = policy_cfg.to_dict()
    params["wrap_env_fn"] = wrap_fn
    policy_args = PolicyArgs.from_config_dict(params["policy_args"])
    params["policy_args"] = policy_args
    return params

def _init_wandb(args: Args, exp_name, env_class, task_cfg, ckpt_path, config_fname="config.json"):
    wandb.init(
        name=exp_name,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group="DAgger",
        config={
            "num_training_steps": args.training.num_training_steps,
            "task": args.task,
            "group": "DAgger",
        },
        dir=os.path.join(WANDB_PATH_LOG, "dagger"),
        tags=_parse_exp_tags(args.exp_tags),
        notes=args.exp_notes,
    )
    wandb.config.update(task_cfg.to_dict())
    wandb.save(inspect.getfile(env_class))
    config_path = ckpt_path / config_fname
    config_path.write_text(task_cfg.to_json_best_effort(indent=4))


def _progress(num_steps, metrics, times, total_steps, debug_mode):
    r"""
    Log metrcis to wandb. Estimate remaining time.

    Args:
        num_steps (int):current number of steps
        metrics (dict): metrics to log
        times (list): list of time stamps
        total_steps: int, total number of steps
        debug_mode: bool, whether in debug mode
    """
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

def _report_training_time(times):
    if len(times) > 1:
        logging.info("Done training.")
        logging.info(f"Time to JIT compile: {times[1] - times[0]:.2f}s")
        logging.info(f"Time to train: {times[-1] - times[1]:.2f}s")


def get_trajectory_handler(env, args: Args, traj_sample_cluster_ids: Dict[str, int], trajectory_sample_probs: Dict[str, np.ndarray] = None, use_ddp: bool = False):
    # load reference trajectory
    assert traj_sample_cluster_ids is not None, "traj_sample_cluster_ids must be provided."

    trajectory_data = env.prepare_trajectory(
        env._config.reference_traj_config.name, traj_sample_cluster_ids=traj_sample_cluster_ids, trajectory_sample_probs=trajectory_sample_probs)
    obs_size = env.observation_size
    act_size = env.action_size
    env.th.traj = None

    # output the dataset and observation info of general tracker
    print("=" * 50)
    print(
        f"Tracking {len(trajectory_data.split_points) - 1} trajectories with {trajectory_data.qpos.shape[0]} timesteps, fps={1 / env.dt:.1f}"
    )
    print(f"Observation: {env._config.obs_keys}")
    print(f"Privileged state: {env._config.privileged_obs_keys}")
    print(f"Auxiliary state: {env._config.auxiliary_obs_keys}")
    print("=" * 50)

    return trajectory_data, obs_size, act_size


def train(args: Args):
    student_env_class = tmj.registry.get(args.task, "tracking_dagger_train_env_class")
    student_task_cfg = tmj.registry.get(args.task, "tracking_dagger_config")
    student_env_cfg = student_task_cfg.env_config
    student_policy_cfg = student_task_cfg.policy_config

    exp_name = _prepare_exp_name(args.task, args.exp_name)
    debug_mode = "debug" in exp_name

    if debug_mode:
        _enable_debug_mode()

    logdir, ckpt_path = _setup_paths(exp_name)
    _log_checkpoint_path(ckpt_path)

    # handle DDP settings
    if args.use_ddp:
        rank, world_size = get_ddp_params()
    else:
        rank, world_size = 0, 1
    
    device_th = f"cuda:0"           # we have set CUDA_VISIBLE_DEVICES already (if use_ddp)
    device_jax = jax.devices()[0]

    # for breakpoint debug
    if os.environ.get("WAIT_FOR_DEBUGGER"):
        import debugpy
        port = 5678 + int(os.environ.get("LOCAL_RANK", 0))
        debugpy.listen(("0.0.0.0", port))
        print(f"[Rank {rank}] Waiting for debugger on port {port}...")
        if rank == 1:
            debugpy.wait_for_client()
            print(f"[Rank 1] Debugger attached, continuing...")
        if args.use_ddp:
            dist.barrier()
            print(f"[Rank {rank}] All ranks synchronized, starting training...")

    with jax.default_device(device_jax):
        with jax.disable_jit():

            privileged_obs_keys, traj_cluster_sample_probs, traj_sample_probs, traj_sample_cluster_ids = parse_dagger_config(args.dagger_config_path)
            _apply_policy_args_to_config(args, student_policy_cfg, debug_mode)
            _apply_env_args_to_config(args, student_env_cfg, privileged_obs_keys, traj_cluster_sample_probs, traj_sample_probs, traj_sample_cluster_ids)

            if args.task == "G1TrackingGeneralTerrainDR":
                hfield_data = jp.asarray(np.load("storage/data/hfield/terrain.npz")["hfield_data"])
                student_policy_cfg.randomization_fn = functools.partial(domain_randomize_terrain, all_hfield_data=hfield_data)
                del hfield_data
                assert student_env_cfg.terrain_type == "rough_terrain"
            elif args.task == "G1TrackingGeneralDR":
                assert student_policy_cfg.randomization_fn == domain_randomize
            elif args.task == "G1TrackingGeneral":
                assert student_policy_cfg.randomization_fn == None
                
            env = student_env_class(terrain_type=student_env_cfg.terrain_type, config=student_env_cfg)
            # _eval_env = env_class(terrain_type=env_cfg.terrain_type, config=env_cfg)

            trajectory_data, obs_size, act_size = get_trajectory_handler(env, args, traj_sample_cluster_ids, traj_sample_probs)

        policy_params = _prepare_training_params(student_policy_cfg, obs_size, act_size, ckpt_path, args.student_use_residual_action, args.training.load_pretrained_path)

        if rank == 0:
            print(policy_params['policy_args'])

        if not debug_mode and rank == 0:
            _init_wandb(args, exp_name, student_env_class, student_task_cfg, ckpt_path)
            
        train_fn = functools.partial(dagger_horizon.dagger, rank=rank, world_size=world_size, **policy_params)

    if args.use_ddp and dist.is_initialized():
        dist.barrier()

    times = [time.monotonic()]      # global 

    train_fn(
        student_environment=env,
        trajectory_data=trajectory_data,
        seed=args.seed,
        dagger_cfg_path = os.path.join(os.getcwd(), args.dagger_config_path),
        student_use_residual_action = args.student_use_residual_action,
        dagger_horizon = args.dagger_horizon,
        dagger_learning_epochs = args.dagger_learning_epochs,
        progress_fn=lambda s, m: _progress(
            num_steps=s,
            metrics=m,
            times=times,
            total_steps=student_policy_cfg.num_training_steps,
            debug_mode=debug_mode
        ),
    )

    _report_training_time(times)

    logging.info(f"Run {exp_name} DAgger done.")

    if args.convert_onnx and rank == 0:

        try:
            from track_mj.eval.dagger.torch2onnx import convert_torch2onnx, get_latest_ckpt
            ckpt_path = str(get_latest_ckpt(ckpt_path))
            output_path = os.path.join(os.path.dirname(ckpt_path), "model.onnx")
            
            convert_torch2onnx(
                ckpt_path=ckpt_path,
                output_path=output_path,
                policy_args=policy_params["policy_args"],
            )

        except ImportError:
            logging.warning("onnx conversion failed due to package missing.")

    if args.use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    train(tyro.cli(Args))
