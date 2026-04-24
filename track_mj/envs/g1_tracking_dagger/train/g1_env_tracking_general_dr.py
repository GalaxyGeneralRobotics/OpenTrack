from typing import Any, Dict, Optional, Union, Tuple, List, Callable
from ml_collections import config_dict
from dataclasses import replace
import os
import jax
import jax.numpy as jp
from functools import partial
import numpy as np
from tqdm import tqdm

import mujoco
from mujoco import MjData, mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

import track_mj as tmj  
from track_mj.envs.g1_tracking_dagger.train import base_env as g1_base
from track_mj.envs.g1_tracking_dagger.train import g1_env_tracking_general
from track_mj.envs.g1_tracking_dagger import g1_tracking_constants as consts
from track_mj.utils.dataset.traj_class import (
    Trajectory,
    TrajectoryData,
    interpolate_trajectories,
    recalculate_traj_angular_velocity,
    recalculate_traj_linear_velocity,
    recalculate_traj_joint_velocity,
)
from track_mj.utils.dataset.traj_handler import TrajectoryHandler, TrajCarry
from track_mj.utils.mujoco import mj_jntname2qposid, mj_jntid2qposid
from track_mj.utils.dataset.traj_process import ExtendTrajData
from track_mj.utils import math as gmth
from track_mj.dr.domain_randomize_tracking_dagger import (
    domain_randomize,
    domain_randomize_terrain,
    domain_randomize_motor_ctrl,
)

ENABLE_PUSH = True
EPISODE_LENGTH = 1000


def g1_tracking_general_dr_task_config() -> config_dict.ConfigDict:

    env_config = config_dict.create(
        terrain_type="flat_terrain",
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=EPISODE_LENGTH,
        action_repeat=1,
        action_scale=1.0,
        recalculate_vel_in_reward=True,
        recalculate_vel_in_reference_motion=True,
        history_len=0,
        soft_joint_pos_limit_factor=0.95,
        student_use_residual_action=True,
        dagger_horizon=1,
        dagger_learning_epochs=10,
        reference_traj_config=config_dict.create(
            name={"lafan1": consts.LAFAN1_SPECIALIST_DATASETS_1},
            random_start=True,
            fixed_start_frame=0,        # only works if random_start is False
        ),
        termination_config=config_dict.create(
            root_height_threshold=0.3,
            rigid_body_dif_threshold=0.5,
        ),
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gravity=0.05,
                gyro=0.2,
                root_pos=0.0,
                root_rot=0.0,
                root_linvel=0.0,
                root_angvel=0.0,
                torso_pos=0.00,
                torso_rot=0.0,
                root_xy_reset=0.1,
                root_yaw_reset=0.27,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Tracking related rewards.
                rigid_body_pos_tracking_upper=1.0,
                rigid_body_pos_tracking_lower=0.5,
                rigid_body_rot_tracking=0.5,
                rigid_body_linvel_tracking=0.5,
                rigid_body_angvel_tracking=0.5,
                joint_pos_tracking=0.75,
                joint_vel_tracking=0.5,
                roll_pitch_tracking=1.0,
                root_linvel_tracking=1.0,
                root_angvel_tracking=1.0,
                root_height_tracking=1.0,
                feet_height_tracking=1.0,
                feet_pos_tracking=2.1,
                penalty_action_rate=-0.5,
                penalty_torque=-0.00002,
                smoothness_joint=-1e-6,
                dof_pos_limit=-10,
                dof_vel_limit=-5,
                collision=-10,
                termination=-200,
            ),
            auxiliary=config_dict.create(
                upper_body_sigma=1.0,
                lower_body_sigma=1.0,
                feet_pos_sigma=1.0,
                body_rot_sigma=1.0,
                feet_rot_sigma=1.0,
                body_linvel_sigma=5.0,
                feet_linvel_sigma=1.0,
                body_angvel_sigma=50.0,
                feet_angvel_sigma=1.0,
                joint_pos_sigma=10.0,
                joint_vel_sigma=1.0,
                root_pos_sigma=0.5,
                root_rot_sigma=1.0,
                root_linvel_sigma=1.0,
                root_angvel_sigma=10.0,
                roll_pitch_sigma=0.2,
                # aux height and contact
                root_height_sigma=0.1,
                feet_height_sigma=0.1,
                global_feet_vel_threshold=0.5,
                global_feet_height_threshold=0.04,
                feet_linvel_threshold=0.1,
                feet_angvel_threshold=0.1,
                feet_slipping_sigma=2.0,
            ),
            penalize_collision_on=[
                ["left_hand_collision", "left_thigh"],
                ["right_hand_collision", "right_thigh"],
                ["left_hand_collision", "right_hand_collision"],
                ["left_hand_collision", "right_wrist_pitch_collision"],
                ["right_hand_collision", "left_wrist_pitch_collision"],
            ],
        ),
        push_config=config_dict.create(
            enable=ENABLE_PUSH,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        obs_scales_config=config_dict.create(joint_vel=0.05, dif_joint_vel=0.05),
        obs_keys = None,
        auxiliary_obs_keys = None,
        privileged_obs_keys = None,
        history_keys=[
            "gyro_pelvis",
            "gvec_pelvis",
            "joint_pos",
            "joint_vel",
        ],
    )

    policy_config = config_dict.create(
        # ====== Used in dagger training function ======
        num_envs = 4096,                        # 4096
        num_training_steps = 400_000,         # 400_000
        debug = False,
        verbose = False,
        save_freq = 500,
        log_freq = 100,
        lr = 1e-4,
        lr_scheduler = "constant",
        weight_decay = 1e-2,
        max_grad_norm = 1.0,
        use_test_set = False,
        save_dir = "",
        wrap_env=True,
        # environment wrapper
        episode_length=EPISODE_LENGTH,
        action_repeat=1,
        wrap_env_fn=None,
        randomization_fn=domain_randomize,
        
        policy_args=None,
    )

    config = config_dict.create(
        env_config=env_config,
        policy_config=policy_config,
    )
    return config


tmj.registry.register("G1TrackingGeneralDR", "tracking_dagger_config")(g1_tracking_general_dr_task_config())


def torque_step_dr(
        rng: jax.Array,
        model: mjx.Model,
        data: mjx.Data,
        qpos_des: jax.Array,
        kps: jax.Array,
        kds: jax.Array,
        kp_scale: jax.Array,
        kd_scale: jax.Array,
        rfi_lim_scale: jax.Array,
        torque_limit: jax.Array,
        n_substeps: int = 1,
) -> tuple[jax.Array, mjx.Data, jax.Array]:
    def single_step(carry, _):
        rng, data, _ = carry
        rng, rng_rfi = jax.random.split(rng, 2)

        # pd control
        pos_err = qpos_des - data.qpos[7:]
        vel_err = -data.qvel[6:]
        torque = (kp_scale * kps) * pos_err + (kd_scale * kds) * vel_err

        # rfi noise
        rfi_noise = rfi_lim_scale * jax.random.uniform(rng_rfi, shape=torque.shape, minval=-1.0, maxval=1.0) * torque_limit
        torque += rfi_noise

        # clip
        torque = jp.clip(torque, -torque_limit, torque_limit)

        # apply torque
        data = data.replace(ctrl=torque)
        data = mjx.step(model, data)

        return (rng, data, torque), None

    initial_torque = jp.zeros_like(torque_limit) #
    (final_rng, final_data, final_torque), _ = jax.lax.scan(single_step, (rng, data, initial_torque), (), n_substeps) #

    return final_rng, final_data, final_torque


@tmj.registry.register("G1TrackingGeneralDR", "tracking_dagger_train_env_class")
class G1TrackingGeneralDREnv(g1_env_tracking_general.G1TrackingGeneralEnv):

    def reset(self, rng: jax.Array, trajectory_data: TrajectoryData = None) -> mjx_env.State:
        # only use key to choose a new start
        if trajectory_data is None:
            trajectory_data = self.th.traj.data
        rng, _noise_rng = jax.random.split(rng, 2)

        carry = self.th.reset_state_with_trajectory(trajectory_data, TrajCarry(rng, self.th.init_state()))
        init_traj_data = self.th.get_current_traj_data_with_trajectory(trajectory_data, carry)  # get traj state

        # add noise to RSI-ed actor root pos & rot
        _noise_rng, xy_noise_rng = jax.random.split(_noise_rng, 2)
        xy_noise = (
            (2 * jax.random.uniform(xy_noise_rng, (2,)) - 1.0) \
            * self._config.noise_config.level \
            * self._config.noise_config.scales.root_xy_reset
        )
        noisy_init_qpos = init_traj_data.qpos.at[:2].add(xy_noise)

        _noise_rng, yaw_noise_rng = jax.random.split(_noise_rng, 2)
        yaw_noise = (
            (2 * jax.random.uniform(yaw_noise_rng, ()) - 1.0) \
            * self._config.noise_config.level \
            * self._config.noise_config.scales.root_yaw_reset
        )
        yaw_noise_quat = gmth.angle2quat(jp.array([0.0, 0.0, yaw_noise]), backend=jp, scalar_first=True)
        noisy_init_root_quat = math.quat_mul(yaw_noise_quat, init_traj_data.qpos[3:7])
        noisy_init_qpos = noisy_init_qpos.at[3:7].set(noisy_init_root_quat)

        data = mjx_env.init(
            self.mjx_model, qpos=noisy_init_qpos, qvel=init_traj_data.qvel, ctrl=noisy_init_qpos[7:]
        )

        traj_no = carry.traj_state.traj_no

        # update to get the reference trajectory step
        carry = self.th.update_state_with_trajectory(trajectory_data, carry)
        traj_data = self.th.get_current_traj_data_with_trajectory(trajectory_data, carry)

        rng = carry.key

        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        rng, dr_ctrl_dict = domain_randomize_motor_ctrl(rng)

        info = {
            "rng": rng,
            "step": 0,
            # history
            "last_motor_targets": data.qpos[7:][self.action_joint_ids],
            "last_action": jp.zeros(self.action_size),
            "last_root_pos": data.qpos[:3],
            'last_root_ori': data.qpos[3:7],
            'last_dof_pos': data.qpos[7:],
            'last_rigid_body_pos': data.xpos,
            'last_rigid_body_ori': data.xquat,
            "last_joint_vel": jp.zeros(self.num_joints),
            # reference trajectory info
            "traj_no": traj_no,
            "traj_info": carry,
            # domain rand
            # push
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            # ctrl
            "kp_scale": dr_ctrl_dict["kp_scale"],
            "kd_scale": dr_ctrl_dict["kd_scale"],
            "rfi_lim_scale": dr_ctrl_dict["rfi_lim_scale"],
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        obs, history = self._get_obs(data, traj_data, info)
        if self._config.history_len > 0:
            _, init_history = self._get_obs(data, init_traj_data, info)
            info["previous_obs"] = jp.stack([init_history] * self._config.history_len, axis=0)

            obs["state"] = jp.concatenate([obs["state"], info["previous_obs"].flatten()], axis=0)
            obs["privileged_state"] = jp.concatenate([obs["privileged_state"], info["previous_obs"].flatten()], axis=0)
            info["previous_obs"] = jp.concatenate([info["previous_obs"][1:], history[None, :]], axis=0)

        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array, trajectory_data: TrajectoryData = None) -> mjx_env.State:
        if trajectory_data is None:
            trajectory_data = self.th.traj.data

        state.info["rng"], push1_rng, push2_rng = jax.random.split(state.info["rng"], 3)

        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push_signal = jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= push_signal
        push *= self._config.push_config.enable
        qvel = state.data.qvel
        qvel = qvel.at[:2].set(qvel[:2] + push * push_magnitude)
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        traj_data = self.th.get_current_traj_data_with_trajectory(trajectory_data, state.info["traj_info"])
        motor_targets = self._get_motor_targets(state, action, use_residual_action=self._config.student_use_residual_action, trajectory_data=trajectory_data)

        state.info["rng"], data, torque = torque_step_dr(
            state.info["rng"],
            self.mjx_model,
            state.data,
            motor_targets,
            kps=self._kps,
            kds=self._kds,
            kp_scale=state.info["kp_scale"],
            kd_scale=state.info["kd_scale"],
            rfi_lim_scale=state.info["rfi_lim_scale"],
            torque_limit=self.torque_limit,
            n_substeps=self.n_substeps,
        )
        rewards = self._get_reward(data, traj_data, action, motor_targets, torque, state.info)
        rewards = {k: v * rewards[k] for k, v in self._config.reward_config.scales.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, a_max=10000.0)

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
        state.info["push"] = push
        state.info["push_step"] += 1
        state.info["step"] += 1

        # update history
        state.info["last_motor_targets"] = motor_targets.copy()
        state.info["last_action"] = action.copy()
        state.info["last_root_pos"] = data.qpos[:3].copy()
        state.info["last_root_ori"] = data.qpos[3:7].copy()
        state.info["last_dof_pos"] = data.qpos[7:].copy()
        state.info["last_rigid_body_pos"] = data.xpos.copy()
        state.info["last_rigid_body_ori"] = data.xquat.copy()
        state.info["last_joint_vel"] = data.qvel[6:].copy()

        # get termination
        termination = self._get_termination(data, traj_data, state.info)

        # reference trajectory step
        state.info["traj_info"] = self.th.update_state_with_trajectory(trajectory_data, state.info["traj_info"])
        traj_data = self.th.get_current_traj_data_with_trajectory(trajectory_data, state.info["traj_info"])

        # get truncated conditions
        truncated = (state.info["step"] >= self._config.episode_length) | (
            state.info["traj_info"].traj_state.traj_no != state.info["traj_no"]
        )
        state.info["truncation"] = truncated.astype(jp.float32)

        done = termination | truncated
        state.info["step"] = jp.where(done, 0, state.info["step"])
        done = done.astype(reward.dtype)

        obs, history = self._get_obs(data, traj_data, state.info)
        if self._config.history_len > 0:
            obs["state"] = jp.concatenate([obs["state"], state.info["previous_obs"].flatten()], axis=0)
            obs["privileged_state"] = jp.concatenate(
                [obs["privileged_state"], state.info["previous_obs"].flatten()], axis=0
            )
            state.info["previous_obs"] = jp.concatenate([state.info["previous_obs"][1:], history[None, :]], axis=0)

        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        # manual reset
        state = jax.lax.cond(
            done, partial(self._reset_and_update_state, trajectory_data=trajectory_data), lambda x: x, state
        )
        return state
