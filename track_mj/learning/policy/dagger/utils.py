import functools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
from brax import base
from brax import envs
from brax.training.types import PRNGKey
from mujoco_playground._src import mjx_env
import jax
import jax.numpy as jnp
import numpy as np
import torch
import random
from jax.dlpack import from_dlpack as jax_from_dlpack
from torch.utils.dlpack import from_dlpack as torch_from_dlpack

from track_mj.utils.dataset.traj_class import TrajectoryData
from track_mj.envs.g1_tracking_dagger.train.g1_env_tracking_general import G1TrackingGeneralEnv
from track_mj.learning.models.dagger.policy_args import PolicyArgs


# ==============================================================================================
# General Utils
# ==============================================================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==============================================================================================
# Torch-Jax hybrid Env Utils
# ==============================================================================================


@dataclass
class TorchJaxState:
    state_th: dict
    state_mjx: mjx_env.State

def _maybe_wrap_env(
    env: envs.Env,
    wrap_env: bool,
    num_envs: int,
    episode_length: Optional[int],
    action_repeat: int,
    local_device_count: int,
    key_env: PRNGKey,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
    """Wraps the environment for training/eval if wrap_env is True."""
    if not wrap_env:
        return env
    if episode_length is None:
        raise ValueError('episode_length must be specified in ppo.train')
    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // local_device_count
        # all devices gets the same randomization rng
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(
            randomization_fn, rng=randomization_rng
        )
    if wrap_env_fn is not None:
        wrap_for_training = wrap_env_fn
    else:
        # wrap_for_training = envs.training.wrap
        raise NotImplementedError("Default wrap function is not implemented.")
    env = wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )  # pytype: disable=wrong-keyword-args
    return env


class TorchJaxEnv:
    def __init__(
            self,
            environment: mjx_env.MjxEnv,
            wrap_env: bool = False,
            num_envs: int = 1,
            episode_length: int = 1,
            action_repeat: int = 1,
            local_device_count: int = 1,
            wrap_env_fn: Optional[Callable] = None,
            randomization_fn: Optional[Callable] = None,
            seed: int = 0,
            verbose: bool = False,
            th_device: str = "cuda:0",
            debug: bool = True,
    ):
        r"""
        A Torch-Jax hybrid environment class.
        """

        self.num_envs = num_envs
        self.mj_model = environment.mj_model
        self.verbose = verbose
        self.th_device = th_device
        self.debug = debug
        # self._active_actuator_to_full = torch_from_dlpack(environment._active_actuator_to_full)

        key = jax.random.PRNGKey(seed)
        global_key, local_key = jax.random.split(key)
        local_key, key_env, eval_key = jax.random.split(local_key, 3)

        assert local_device_count >= 1, "local_device_count must be at least 1."

        env = _maybe_wrap_env(
            environment,
            wrap_env=wrap_env,
            num_envs=num_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
            local_device_count=1,
            key_env=key_env,
            wrap_env_fn=wrap_env_fn,
            randomization_fn=randomization_fn,
        )
        self._reset_fn = jax.jit(env.reset)
        self._step_fn = jax.jit(env.step)
        self._get_motor_targets_fn = jax.jit(env._get_motor_targets)

        self.key_env = key_env

        if self.verbose:
            import mujoco
            import mujoco.viewer
            self._mj_data = mujoco.MjData(self.mj_model)
            self._viewer = mujoco.viewer.launch_passive(self.mj_model, self._mj_data)

        self._action_th = torch.zeros(
            (self.num_envs, env.action_size), device=self.th_device
        )

    def init_state(self, trajectory_data: TrajectoryData) -> TorchJaxState:
        key_env = self.key_env
        key_envs = jax.random.split(key_env, self.num_envs)
        state_mjx = self._reset_fn(key_envs, trajectory_data)
        return TorchJaxState(state_mjx=state_mjx, state_th=dict())
    
    def step(self, state: TorchJaxState, action: torch.Tensor, trajectory_data: TrajectoryData) -> TorchJaxState:
        self._action_th[:] = action
        state_mjx = state.state_mjx
        action_jx = jax_from_dlpack(self._action_th)
        state_mjx = self._step_fn(state_mjx, action_jx, trajectory_data)
        return TorchJaxState(state_mjx=state_mjx, state_th=dict())
    
    def view(self, state: TorchJaxState, sim_id: int = 0) -> None:
        mjx_data = state.state_mjx.data
        if self.verbose:
            import mujoco
            mujoco.mjx.get_data_into(
                self._mj_data,
                self.mj_model,
                jax.tree_map(lambda x: x[sim_id], mjx_data),
            )
            mujoco.mj_forward(self.mj_model, self._mj_data)
            self._viewer.sync()
        else:
            raise RuntimeError("Viewer is not available in headless mode.")
        

# ==============================================================================================
# Auxiliary Training Info Handler
# ==============================================================================================


class AuxTrainingInfoHandler:

    def __init__(self, policy_args: PolicyArgs, num_envs: int, horizon: int, device: torch.device):
        self.policy_args = policy_args
        self.num_envs = num_envs
        self.horizon = horizon
        self.device = device


    def get_aux_loss_info(self, dones_list: List[torch.Tensor]):
        return {}
