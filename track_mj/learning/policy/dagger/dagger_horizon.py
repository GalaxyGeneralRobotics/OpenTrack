import os
import json
import time
import logging
from collections import deque, defaultdict
from contextlib import contextmanager
from typing import Dict, List, Union, Callable, Optional

from click import Option
import jax
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from torch.utils.dlpack import from_dlpack as torch_from_dlpack
from jax.dlpack import from_dlpack as jax_from_dlpack
import numpy as np
import onnx
import onnxruntime as rt
from onnx import helper, numpy_helper, TensorProto, ModelProto, GraphProto

from mujoco_playground._src import mjx_env


import track_mj as gmj
from track_mj.envs.g1_tracking_dagger import g1_tracking_constants as consts
from track_mj.learning.models.dagger.policy_args import PolicyArgs
from track_mj.learning.models.dagger.policy import get_policy
from track_mj.learning.policy.dagger import metrics_aggregator as metric_logger
from track_mj.utils.dataset.traj_class import TrajectoryData
from track_mj.learning.policy.dagger.utils import TorchJaxEnv, TorchJaxState, set_seed, AuxTrainingInfoHandler
from track_mj.envs.g1_tracking_dagger.train.g1_env_tracking_general import G1TrackingGeneralEnv
from track_mj.envs.g1_tracking_dagger.train.g1_env_tracking_general_dr import G1TrackingGeneralDREnv
from track_mj.envs.g1_tracking_dagger.train.g1_env_tracking_general_terrain_dr import G1TrackingGeneralTerrainDREnv

import einops


RT_INP_OBS = "obs"
RT_OUP_ACT = "continuous_actions"


# ==============================================================================================
# Performance Timer
# ==============================================================================================


class PerformanceTimer:
    """Elegant performance monitoring tool for tracking execution time of various parts of the training loop"""

    def __init__(self, window_size: int = 100):
        self.timers = defaultdict(lambda: deque(maxlen=window_size))
        self.current_section = None
        self.section_start = None

    @contextmanager
    def timer(self, section_name: str):
        """Context manager for automatic code block timing"""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.timers[section_name].append(elapsed)

    def get_stats(self, section_name: str = None):
        """Get statistics for a section (mean, min, max)"""
        if section_name:
            if section_name not in self.timers or len(self.timers[section_name]) == 0:
                return None
            times = list(self.timers[section_name])
            return {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'count': len(times)
            }
        else:
            # Return statistics for all sections
            return {name: self.get_stats(name) for name in self.timers.keys()}

    def print_summary(self, total_time: float = None):
        """Print performance summary sorted by average time in descending order"""
        stats_dict = self.get_stats()
        if not stats_dict:
            return

        # Sort by average time in descending order
        sorted_sections = sorted(
            stats_dict.items(),
            key=lambda x: x[1]['mean'] if x[1] else 0,
            reverse=True
        )

        print("\n" + "=" * 80)
        print("Performance Breakdown (per step)")
        print("=" * 80)
        print(f"{'Section':<30} {'Mean (ms)':<12} {'Std (ms)':<12} {'% of Total':<12}")
        print("-" * 80)

        total_measured = sum(s[1]['mean'] for s in sorted_sections if s[1])

        for section_name, stats in sorted_sections:
            if stats is None:
                continue
            mean_ms = stats['mean'] * 1000
            std_ms = stats['std'] * 1000
            percentage = (stats['mean'] / total_measured * 100) if total_measured > 0 else 0

            print(f"{section_name:<30} {mean_ms:<12.2f} {std_ms:<12.2f} {percentage:<12.1f}")

        print("-" * 80)
        print(f"{'Total measured time':<30} {total_measured * 1000:<12.2f} ms/step")
        if total_time:
            print(f"{'Actual step time':<30} {total_time * 1000:<12.2f} ms/step")
            overhead = total_time - total_measured
            print(f"{'Overhead (logging, etc.)':<30} {overhead * 1000:<12.2f} ms/step")
        print("=" * 80 + "\n")


# ==============================================================================================
# DDP Utils
# ==============================================================================================


def is_ddp():
    """Checks if the script is running in a DDP environment."""
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

def validate_ddp_params(rank: int, world_size: int):
    expected_world_size = int(os.environ.get("WORLD_SIZE"))
    expected_rank = int(os.environ["LOCAL_RANK"])
    assert world_size == expected_world_size, f"Expected world_size {expected_world_size}, got {world_size}"
    assert rank == expected_rank, f"Expected rank {expected_rank}, got {rank}"

def auto_set_barrier(use_ddp):
    if use_ddp:
        dist.barrier()


# ==============================================================================================
# Teacher Utils
# ==============================================================================================


def make_param_model(example_teacher_path: str, param_model_path: str) -> None:
    if os.path.exists(param_model_path):
        return

    model: ModelProto = onnx.load(example_teacher_path)
    graph: GraphProto = model.graph

    existing_inputs = {inp.name for inp in graph.input}

    new_inputs = list(graph.input)
    keep_initializers = []

    for init in list(graph.initializer):
        name = init.name
        if name in existing_inputs:
            continue

        elem_type = init.data_type
        shape = init.dims
        vi = helper.make_tensor_value_info(
            name,
            elem_type if elem_type != 0 else TensorProto.FLOAT,
            shape if len(shape) > 0 else []
        )
        new_inputs.append(vi)

    del graph.input[:]
    graph.input.extend(new_inputs)

    del graph.initializer[:]
    graph.initializer.extend(keep_initializers)  # Usually empty

    onnx.checker.check_model(model)
    onnx.save(model, param_model_path)
    print(f"[ParamModel] Wrote parameterized model to: {param_model_path}")

def extract_initializer_dict(teacher_onnx_path: str) -> Dict[str, np.ndarray]:
    model = onnx.load(teacher_onnx_path)
    d = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        d[init.name] = arr
    return d

def load_teacher_weights_bulk(motion_clusters) -> List[Dict[str, np.ndarray]]:
    all_w = []
    for cls_cfg in motion_clusters:
        p = cls_cfg["teacher_onnx_path"]
        print(f"[Weights] Loading weights from {p} ...")
        w = extract_initializer_dict(p)
        all_w.append(w)
    return all_w


# ==============================================================================================
# Action and State Utils
# ==============================================================================================


def get_student_state(
        state_tj: TorchJaxState,
        policy_args: PolicyArgs,
    ) -> torch.Tensor:
    obs = torch_from_dlpack(state_tj.state_mjx.obs[policy_args.policy_obs_key])
    # auxiliary_obs = torch_from_dlpack(state_tj.state_mjx.obs[policy_args.policy_auxiliary_obs_key])

    return {
        policy_args.policy_obs_key: obs.to(torch.float32),
        # policy_args.policy_auxiliary_obs_key: auxiliary_obs.to(torch.float32)
    }

def get_teacher_actions_param(
        state_tj: TorchJaxState,
        teacher_weights_list: List[Dict[str, np.ndarray]],
        num_clusters: int,
        device: torch.device,
        num_envs: int,
        act_dim: int,
        param_sess: rt.InferenceSession,
        weight_input_names: List[str],
    ):

    teacher_state = torch_from_dlpack(state_tj.state_mjx.obs["privileged_state"])
    env_cluster_ids = torch_from_dlpack(state_tj.state_mjx.info["ref_motion_cluster_id"]).long()
    teacher_actions = torch.zeros((num_envs, act_dim), device=device)

    for cluster_id in range(num_clusters):
        mask = (env_cluster_ids == cluster_id)
        if not mask.any():
            continue

        teacher_obs_np = teacher_state[mask].cpu().numpy()
        feed = {RT_INP_OBS: teacher_obs_np}

        teacher_weight = teacher_weights_list[cluster_id]

        for w_name in weight_input_names:
            if w_name == RT_INP_OBS:
                continue
            arr = teacher_weight.get(w_name, None)
            if arr is None:
                raise KeyError(f"[Teacher {cluster_id}] missing weight '{w_name}'")
            feed[w_name] = arr

        teacher_action_np = param_sess.run([RT_OUP_ACT], feed)[0]       # [M, act_dim]
        teacher_actions[mask] = torch.from_numpy(teacher_action_np).to(device, torch.float32)

    return teacher_actions


# ==============================================================================================
# DAgger
# ==============================================================================================


def dagger(
    student_environment: mjx_env.MjxEnv,
    trajectory_data: TrajectoryData,
    dagger_cfg_path: str,
    seed: int = 0,
    # ====== Set via Env's policy_config ======
    num_envs: int = 4096,
    num_training_steps: int = 400_000,
    debug: bool = False,
    verbose: bool = False,
    device: str = "cuda:0",
    save_freq: int = 500,
    log_freq: int = 100,
    lr: float = 1e-4,
    lr_scheduler: str = "constant",     # "constant", "cosine_annealing"
    weight_decay: float = 1e-2,
    max_grad_norm: float = 1.0,
    use_test_set: bool = False,
    save_dir = "",

    do_exclude_joints: bool = False,

    wrap_env: bool = True,
    episode_length: int = 1000,
    action_repeat: int = 1,
    wrap_env_fn: Optional[Callable] = None,
    randomization_fn: Optional[Callable] = None,

    # ====== Passed at function calling ======
    student_use_residual_action: bool = True,
    dagger_horizon: int = 1,
    dagger_learning_epochs: int = 10,
    progress_fn: Optional[Callable] = None,

    policy_args: PolicyArgs = PolicyArgs(),

    rank: int = 0,
    world_size: int = 1,
):
    """
    DAgger training
    """
    
    assert save_freq % log_freq == 0 and log_freq > 0, "save_freq must be multiple of log_freq"

    # handle DDP
    use_ddp = is_ddp()
    if use_ddp:
        validate_ddp_params(rank, world_size)
        seed = seed + rank
        device = f"cuda:0"           # we have set CUDA_VISIBLE_DEVICES already (if use_ddp)
    else:
        rank, world_size = 0, 1
    
    num_envs_per_rank = num_envs // world_size
    set_seed(seed)

    if use_ddp:
        logging.info(
            f"[DDP Init] rank={rank}/{world_size - 1}, "
            f"world_size={world_size}, "
            f"device={device}, "
            f"seed={seed}, "
            f"num_envs_per_rank={num_envs_per_rank} "
            f"(global={num_envs_per_rank * world_size}), "
        )
    else:
        logging.info(
            f"[Single GPU/CPU Init] "
            f"device={device}, "
            f"seed={seed}, "
            f"num_envs={num_envs_per_rank}, "
        )        

    # ====== TorchJax Environment ======

    if rank == 0:
        print("Initializing environment...")
    profile_step = deque(maxlen=100)

    with open(dagger_cfg_path, 'r') as f:
        dagger_cfg = json.load(f)
    assert "motion_clusters" in dagger_cfg, "dagger config must contain motion_clusters field."

    tj_env = TorchJaxEnv(
        environment=student_environment,
        wrap_env=wrap_env,
        num_envs=num_envs_per_rank,
        episode_length=episode_length,
        action_repeat=action_repeat,
        wrap_env_fn=wrap_env_fn,
        local_device_count=1,
        randomization_fn=randomization_fn,
        seed=seed,
        verbose=verbose,
        th_device=device,
        debug=debug,
    )

    state_tj = tj_env.init_state(trajectory_data)

    # ====== Student & DDP wrapping ======
    if rank == 0:
        print(f"Initializing {policy_args.policy_type} policy...")
    student_nn = get_policy(policy_args)
    student_nn.model.to(device)
    if use_ddp:
        student_nn.model = DDP(student_nn.model, device_ids=[0])    # each process only sees one GPU， we have set CUDA_VISIBLE_DEVICES already (if use_ddp)
        if rank == 0:
            print(f"Model wrapped with DDP on rank {rank}")
    auto_set_barrier(use_ddp)

    optimizer = optim.AdamW(student_nn.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    if lr_scheduler == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-6)
    elif lr_scheduler == "constant":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    else:
        raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}. Supported: 'constant', 'cosine_annealing'")

    # ====== Teacher (parameterized ONNX) ======
    if rank == 0:
        print("LR Scheduler:", lr_scheduler)
        print("Preparing parameterized teacher model ...")

    motion_clusters = dagger_cfg["motion_clusters"]
    num_clusters = len(motion_clusters)

    if rank == 0:
        print(f"Number of motion clusters: {num_clusters}")

    param_model_path = os.path.join(save_dir if save_dir else ".", "param_model.onnx")
    os.makedirs(os.path.dirname(param_model_path), exist_ok=True)

    # only rank 0 creates the parameterized model
    if rank == 0:
        make_param_model(os.path.join(os.getcwd(), motion_clusters[0]["teacher_onnx_path"]), param_model_path)

    # synchronize to ensure file is created
    auto_set_barrier(use_ddp)

    teacher_weights_list = load_teacher_weights_bulk(motion_clusters)

    so = rt.SessionOptions()
    providers = [(
        "CUDAExecutionProvider",
        {
            "device_id": 0,         # always 0 because each process only sees one GPU
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": "1",
            "arena_extend_strategy": "kSameAsRequested",
        }
    ), "CPUExecutionProvider"]
    
    if rank == 0:
        print(f"[ORT] Creating single CUDA session from {param_model_path}, num={len(teacher_weights_list)}")

    param_sess = rt.InferenceSession(param_model_path, sess_options=so, providers=providers)
    input_names = [i.name for i in param_sess.get_inputs()]
    assert RT_INP_OBS in input_names, f"Param model must have input '{RT_INP_OBS}'"
    weight_input_names = [n for n in input_names if n != RT_INP_OBS]

    # Training vars
    best_test_length = 0.0
    best_total_loss = float('inf')

    print("Starting training...")
    print("=" * 60)

    # Initialize performance timer
    metrics_aggregator = metric_logger.DDPMetricsLogger(
        base_logger= metric_logger.EpisodeMetricsLogger(
            steps_between_logging=float('inf'),  # Disable auto-logging, we control it manually
            progress_fn=progress_fn,
        ),
        rank=rank,
        world_size=world_size,
        use_ddp=use_ddp,
    )

    # Initialize aux training info handler
    aux_loss_info_handler = AuxTrainingInfoHandler(
        policy_args=policy_args,
        num_envs=num_envs_per_rank, 
        horizon=dagger_horizon,
        device=torch.device(device)
    )

    perf_timer = PerformanceTimer(window_size=100)

    # Main training loop
    for step_id_train in range(num_training_steps):
        _t_step_start = time.perf_counter()

        nn_state_dict_th_list = []  # for student
        student_action_th_list = []
        teacher_action_th_list = []
        dones_th_list = []  # collect dones after each step for mu_reg_loss
        
        # Get student action
        with perf_timer.timer("1.env_rollout"):
            for _ in range(dagger_horizon):
                # jax to torch (prepare for student forward)
                nn_state_dict = get_student_state(state_tj, policy_args)
                nn_state_dict_th = {k: v.to(device) for k, v in nn_state_dict.items()}
                # 1. student forward
                student_action_th = student_nn.infer(nn_state_dict_th).detach()
                # 2. teacher forward
                teacher_action_th = get_teacher_actions_param(
                    state_tj,
                    teacher_weights_list,
                    num_clusters,
                    device,
                    num_envs_per_rank,
                    policy_args.act_dim,
                    param_sess,
                    weight_input_names
                )
                if not student_use_residual_action:
                    # motor targets
                    teacher_action_th = torch_from_dlpack(
                        tj_env._get_motor_targets_fn(
                            state_tj.state_mjx,
                            jax_from_dlpack(teacher_action_th.clone()),
                            use_residual_action=True,           # teacher is always residual
                            trajectory_data=trajectory_data,
                        )
                    )
                    if do_exclude_joints:
                        teacher_action_th = teacher_action_th[..., tj_env._active_actuator_to_full]
                        
                # 3. step environment
                state_tj = tj_env.step(state_tj, student_action_th, trajectory_data)
                
                # save for loss computation
                nn_state_dict_th_list.append(nn_state_dict_th)
                student_action_th_list.append(student_action_th)
                teacher_action_th_list.append(teacher_action_th)
                
                # dones_th_list[h] indicates dones AFTER step h, used to determine if step h and h+1 are in same episode
                dones_th_list.append(torch_from_dlpack(state_tj.state_mjx.info["episode_done"]).bool())

        # Compute loss
        with perf_timer.timer("2.optimizer_step"):
            # process student state_dict: [B, H, ...] -> [B*H, ...]
            nn_state_dict_th_batch = {}
            for key in nn_state_dict_th_list[0].keys():
                nn_state_dict_th_batch[key] = torch.stack([step[key] for step in nn_state_dict_th_list], dim=1)  # [B, H, ...]
                nn_state_dict_th_batch[key] = einops.rearrange(nn_state_dict_th_batch[key], "B H ... -> (B H) ...")
            
            # process teacher action: [B, H, ...] -> [B*H, ...]
            teacher_action_th_batch = torch.stack([step for step in teacher_action_th_list], dim=1)  # [B, H, ...]
            teacher_action_th_batch = einops.rearrange(teacher_action_th_batch, "B H ... -> (B H) ...")
            # process student action collected during rollout: [B, H, ...] -> [B*H, ...]
            student_action_th_batch = torch.stack([step for step in student_action_th_list], dim=1)  # [B, H, ...]
            student_action_th_batch = einops.rearrange(student_action_th_batch, "B H ... -> (B H) ...")

            aux_loss_info = aux_loss_info_handler.get_aux_loss_info(
                dones_list=dones_th_list,
            )

            auto_set_barrier(use_ddp)
            
            for _ in range(dagger_learning_epochs):
                # compute loss
                loss, info_student = student_nn.compute_loss(nn_state_dict_th_batch, teacher_action_th_batch, aux_loss_info)
                # update model
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    student_nn.model.parameters() if not use_ddp else student_nn.model.module.parameters(),
                    max_grad_norm
                )
                optimizer.step()
            scheduler.step()

        # Handle episode completion and logging
        with perf_timer.timer("3.step_overhead"):
            # jax, [#envs,]
            dones = state_tj.state_mjx.info["episode_done"]
            metrics = state_tj.state_mjx.info["episode_metrics"]
            # torch, scaler
            total_loss = loss.clone().detach().cpu()
            mae_loss = torch.mean(torch.abs(teacher_action_th_batch - student_action_th_batch)).cpu()
            # bc_loss, kl_loss, mu_reg_loss
            bc_loss = info_student.get("bc_loss", 0.0)
            kl_loss = info_student.get("kl_loss", 0.0)
            mu_reg_loss = info_student.get("mu_reg_loss", 0.0)

        _t_step_elapsed = time.perf_counter() - _t_step_start
        profile_step.append(_t_step_elapsed)

        # Logging and saving
        training_metrics = {
            "total_loss": total_loss.item(),
            "mae_loss": mae_loss.item(),
            "bc_loss": bc_loss,
            "kl_loss": kl_loss,
            "mu_reg_loss": mu_reg_loss,
        }
        timing_stats = perf_timer.get_stats()
        for section_name, stats in timing_stats.items():
            if stats:
                training_metrics[f"{section_name}_ms"] = stats['mean'] * 1000

        metrics_aggregator.update_episode_metrics(
            aggregated_metrics=metrics,
            dones=dones,
            training_metrics=training_metrics,
        )

        if (step_id_train % log_freq == 0 and step_id_train > 0) or debug:

            # Aggregate and log metrics
            metrics_aggregator.aggregate_and_log()

            if rank == 0:
                dones_th = torch_from_dlpack(dones)
                done_lengths = torch.mean(torch_from_dlpack(state_tj.state_mjx.info["episode_metrics"]["length"])[dones_th.bool()]).item()
                
                # Print performance breakdown
                avg_step_time = np.mean(profile_step)
                perf_timer.print_summary(total_time=avg_step_time)

                # Get aggregated losses from training_metrics_buffer
                if len(metrics_aggregator.base_logger._training_metrics_buffer["mae_loss"]) > 0:
                    global_total_loss = np.mean(metrics_aggregator.base_logger._training_metrics_buffer["total_loss"])
                    global_mae_loss = np.mean(metrics_aggregator.base_logger._training_metrics_buffer["mae_loss"])
                    global_bc_loss = np.mean(metrics_aggregator.base_logger._training_metrics_buffer["bc_loss"])
                    global_kl_loss = np.mean(metrics_aggregator.base_logger._training_metrics_buffer["kl_loss"])
                else:
                    global_total_loss = 0.0
                    global_mae_loss = 0.0
                    global_bc_loss = 0.0
                    global_kl_loss = 0.0

                if use_test_set:
                    # TODO: we don't care about generalization ability yet
                    raise NotImplementedError("Test set evaluation is not implemented yet.")
                else:
                    if step_id_train % save_freq == 0:
                        os.makedirs(save_dir, exist_ok=True)
                        student_nn.save_pretrained(save_dir, file_name_sufix="_step_" + str(step_id_train).zfill(10), use_ddp=use_ddp)
                        logging.info(f"Model saved at step {step_id_train} to {save_dir}")

                    if global_total_loss < best_total_loss:
                        best_total_loss = global_total_loss
                        os.makedirs(save_dir, exist_ok=True)
                        # remove old best model
                        for f in os.listdir(save_dir):
                            if f.endswith("_best.pth"):
                                os.remove(os.path.join(save_dir, f))
                        student_nn.save_pretrained(save_dir, file_name_sufix="_step_" + str(step_id_train).zfill(10) + "_best", use_ddp=use_ddp)
                        logging.info(f"New best model saved at step {step_id_train} with Total loss {best_total_loss:.4f}")
                
                    print(
                        f"step={step_id_train:<6} "
                        f"avg-done-length(rank0)={done_lengths} "
                        f"BC-Total-Loss={global_total_loss:.2e} "
                        f"BC-MAE-Loss={global_mae_loss:.2e} "
                        f"Best-Total-Loss={best_total_loss:.2e} "
                        f"KL-Loss={global_kl_loss:.2e} "
                        f"BC-Loss={global_bc_loss:.2e} "
                    )

            # Synchronize all ranks after logging and saving
            auto_set_barrier(use_ddp)

        if verbose:
            tj_env.view(state_tj)

    if rank == 0:
        print("=" * 60)
        print("Training completed!")
        print(f"Best Total Loss: {best_total_loss:.4f}")
        print(f"Model saved to: {save_dir}")
    auto_set_barrier(use_ddp)
