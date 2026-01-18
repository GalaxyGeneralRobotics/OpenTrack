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

from src.envs.g1 import base_env as g1_base
from src.envs.g1 import g1_tracking_constants as consts
from src.utils.dataset.traj_class import (
    Trajectory,
    TrajectoryData,
    interpolate_trajectories,
    recalculate_traj_angular_velocity,
    recalculate_traj_linear_velocity,
    recalculate_traj_joint_velocity,
)
from src.utils.dataset.traj_handler import TrajectoryHandler, TrajCarry
from src.utils.mujoco import mj_jntname2qposid, mj_jntid2qposid
from src.utils.dataset.traj_process import ExtendTrajData
from src.utils import math as gmth

ENABLE_RANDOMIZE = True
EPISODE_LENGTH = 1000


def default_config() -> config_dict.ConfigDict:

    env_config = config_dict.create(
        terrain_type="flat_terrain",
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=EPISODE_LENGTH,
        action_repeat=1,
        action_scale=1.0,
        recalculate_velocity=True,
        history_len=79,
        enable_randomize=ENABLE_RANDOMIZE,
        soft_joint_pos_limit_factor=0.95,
        reference_traj_config=config_dict.create(
            name={"lafan1": consts.LAFAN1_DATASETS},
            random_start=True,
            fixed_start_frame=0,  # only works if random_start is False
            add_pertubation=False,  # only for test
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
                penalty_action_rate=-0.5,
                penalty_torque=-0.00002,
                dof_pos_limit=-10,
                dof_vel_limit=-5,
                collision=-10,
                termination=-200,
                # root vel
                root_linvel_tracking=1.0,
                root_angvel_tracking=1.0,
                root_height_tracking=1.0,
                feet_height_tracking=1.0,
                feet_pos_tracking=2.1,
                smoothness_joint=-1e-6,
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
            enable=ENABLE_RANDOMIZE,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        obs_scales_config=config_dict.create(joint_vel=0.05),
        obs_keys=[
            "gyro_pelvis",
            "gvec_pelvis",
            "joint_pos",
            "joint_vel",
            "last_motor_targets",
            "dif_joint_pos",
            "dif_joint_vel",
            "ref_root_linvel",
            "ref_root_angvel",
            "ref_root_height",
            "ref_feet_height",
        ],
        privileged_obs_keys=[
            "gyro_pelvis",
            "gvec_pelvis",
            "linvel_pelvis",
            "dif_torso_rp",
            "joint_pos",
            "joint_vel",
            "last_motor_targets",
            "dif_joint_pos",
            "dif_joint_vel",
            "feet_contact",
            "dif_feet_height",
            "dif_root_height",
            "dif_root_linvel",
            "dif_root_angvel",
            "dif_rigid_body_pos_local",
            "dif_rigid_body_rot_local",
            "dif_rigid_body_linvel_local",
            "dif_rigid_body_angvel_local",
        ],
        history_keys=[
            "gyro_pelvis",
            "gvec_pelvis",
            "joint_pos",
            "joint_vel",
        ],
    )

    policy_config = config_dict.create(
        num_timesteps=3_000_000_000,
        max_devices_per_host=8,
        # high-level control flow
        wrap_env=True,
        # environment wrapper
        num_envs=32768,  # 8192(256*32), 16384(512*32), 32768(1024*32)
        episode_length=EPISODE_LENGTH,
        action_repeat=1,
        # ppo params
        learning_rate=3e-4,
        entropy_cost=0.01,
        discounting=0.97,
        unroll_length=20,
        batch_size=1024,  # 256, 512, 1024
        num_minibatches=32,  # 8, 16, 32
        num_updates_per_batch=4,
        num_resets_per_eval=0,
        normalize_observations=False,
        reward_scaling=1.0,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        normalize_advantage=True,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 512, 256, 256, 128),
            value_hidden_layer_sizes=(512, 512, 256, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
        seed=0,
        # eval
        num_evals=5,
        # training metrics
        log_training_metrics=True,
        training_metrics_steps=int(1e6),  # 1M
        # callbacks
        save_checkpoint_path=None,
        restore_checkpoint_path=None,
        restore_params=None,
        restore_value_fn=True,
    )

    config = config_dict.create(
        env_config=env_config,
        policy_config=policy_config,
    )
    return config


def get_collision_contact(contact: Any, geom1: int, geom2: int) -> Tuple[jax.Array, jax.Array]:
    """Get the contact point between two geoms"""
    mask = (jp.array([geom1, geom2]) == contact.geom).all(axis=1)
    mask |= (jp.array([geom2, geom1]) == contact.geom).all(axis=1)
    idx = jp.where(mask, contact.dist, 1e4).argmin()
    dist = contact.dist[idx] * mask[idx]
    pos = contact.pos[idx]
    return dist < 0, pos


def get_feet_contact(self, data: mjx.Data) -> jax.Array:
    """
    Returns the contact state of the left and right feet.

    Contact state encoding:
        -1: fully in air (no contact)
         1: partial contact (some or all foot geoms in contact)
    """
    left_contacts = jp.array(
        [geoms_colliding(data, geom_id, self.geom_id_floor) for geom_id in self.geom_ids_left_feet]
    )
    right_contacts = jp.array(
        [geoms_colliding(data, geom_id, self.geom_id_floor) for geom_id in self.geom_ids_right_feet]
    )

    left_state = jp.where(left_contacts.any(), 1, -1)
    right_state = jp.where(right_contacts.any(), 1, -1)

    return jp.array([left_state, right_state])


class G1TrackingEnv(g1_base.G1Env):
    @property
    def action_size(self) -> int:
        return len(self.action_joint_names)

    @property
    def pred_size(self) -> int:
        # delta_root_ori + delta_root_angvel + delta_dof_vel
        return len(self.action_joint_names) + 3 + 3

    def __init__(
        self,
        terrain_type: str = "flat_terrain",
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.task_to_xml(terrain_type).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        self.num_joints = self.mjx_model.nq - 7
        self.episode_length = self._config.episode_length

        self.action_joint_names = consts.ACTION_JOINT_NAMES.copy()
        self.action_joint_ids = []
        for j_name in self.action_joint_names:
            self.action_joint_ids.append(self.mj_model.actuator(j_name).id)
        self.action_joint_ids = jp.array(self.action_joint_ids)

        self.obs_joint_names = consts.OBS_JOINT_NAMES.copy()
        self.obs_joint_ids = []
        for j_name in self.obs_joint_names:
            self.obs_joint_ids.append(self.mj_model.actuator(j_name).id)
        self.obs_joint_ids = jp.array(self.obs_joint_ids)

        self._up_vec = jp.array([0.0, 0.0, 1.0])
        self._left_vec = jp.array([0.0, 1.0, 0.0])
        self._default_qpos = jp.array(consts.DEFAULT_QPOS[7:])

        # Note: First joint is freejoint.
        self._kps = jp.array(consts.KPs)
        self._kds = jp.array(consts.KDs)
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        waist_indices = []
        waist_joint_names = ["waist_yaw", "waist_roll", "waist_pitch"]
        for joint_name in waist_joint_names:
            waist_indices.append(self._mj_model.joint(f"{joint_name}_joint").qposadr - 7)
        self._waist_indices = jp.array(waist_indices)

        arm_indices = []
        arm_joint_names = ["shoulder_roll", "shoulder_yaw", "wrist_roll", "wrist_pitch", "wrist_yaw"]
        for side in ["left", "right"]:
            for joint_name in arm_joint_names:
                arm_indices.append(self._mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7)
        self._arm_indices = jp.array(arm_indices)

        hip_indices = []
        hip_joint_names = ["hip_roll", "hip_yaw"]
        for side in ["left", "right"]:
            for joint_name in hip_joint_names:
                hip_indices.append(self._mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7)
        self._hip_indices = jp.array(hip_indices)

        knee_indices = []
        knee_joint_names = ["knee"]
        for side in ["left", "right"]:
            for joint_name in knee_joint_names:
                knee_indices.append(self._mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7)
        self._knee_indices = jp.array(knee_indices)

        self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._torso_imu_site_id = self._mj_model.site("imu_in_torso").id
        self._pelvis_imu_site_id = self._mj_model.site("imu_in_pelvis").id

        self._feet_site_id = jp.array([self._mj_model.site(name).id for name in consts.FEET_SITES])
        self._feet_all_site_id = jp.array([self._mj_model.site(name).id for name in consts.FEET_ALL_SITES])
        self._hands_site_id = jp.array([self._mj_model.site(name).id for name in consts.HAND_SITES])
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = jp.array([self._mj_model.geom(name).id for name in consts.FEET_GEOMS])

        foot_linvel_sensor_adr = []
        for site in consts.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        self.penalize_collision_pair = jp.array(
            [
                [self.mj_model.geom(pair[0]).id, self.mj_model.geom(pair[1]).id]
                for pair in self._config.reward_config.penalize_collision_on
            ]
        )

        # bodies
        self.body_id_pelvis = self.mj_model.body("pelvis").id
        self.body_id_torso = self.mj_model.body("torso_link").id
        self.body_names_left_leg = ["left_knee_link", "left_ankle_roll_link"]
        self.body_ids_left_leg = jp.array([self.mj_model.body(n).id for n in self.body_names_left_leg])
        self.body_names_right_leg = ["right_knee_link", "right_ankle_roll_link"]
        self.body_ids_right_leg = jp.array([self.mj_model.body(n).id for n in self.body_names_right_leg])
        self.upper_body_ids = jp.array([self.mj_model.body(n).id for n in consts.UPPER_BODY_LINKs])
        self.lower_body_ids = jp.array([self.mj_model.body(n).id for n in consts.LOWER_BODY_LINKs])
        self.upper_body_joints = jp.array([self.mj_model.joint(n).id for n in consts.UPPER_BODY_JOINTs])
        self.feet_ids = jp.array([self.mj_model.body(n).id for n in consts.FEET_LINKs])
        self.shoulder_ids = jp.array([self.mj_model.body(n).id for n in consts.SHOULDER_LINKs])
        self.valid_body_ids = jp.concatenate((self.lower_body_ids, self.upper_body_ids))  # link of id 0 is world!

        self.dof_pos_lower_limit = jp.array([item[0] for item in consts.RESTRICTED_JOINT_RANGE])
        self.dof_pos_upper_limit = jp.array([item[1] for item in consts.RESTRICTED_JOINT_RANGE])
        self.dof_vel_limit = jp.array(consts.DOF_VEL_LIMITS)
        self.torque_limit = jp.array(consts.TORQUE_LIMIT)

        # reference trajectory
        self.th: TrajectoryHandler = None
        if self._config.reference_traj_config.random_start:
            self._th_params = {"random_start": True}
        else:
            self._th_params = {
                "random_start": False,
                "fixed_start_conf": [0, self._config.reference_traj_config.fixed_start_frame],
            }
        self._data = mujoco.MjData(self._mj_model)

    def reset(self, rng: jax.Array, trajectory_data: TrajectoryData = None) -> mjx_env.State:
        # only use key to choose a new start
        if trajectory_data is None:
            trajectory_data = self.th.traj.data
        carry = self.th.reset_state_with_trajectory(trajectory_data, TrajCarry(rng, self.th.init_state()))
        init_traj_data = self.th.get_current_traj_data_with_trajectory(trajectory_data, carry)
        data = mjx_env.init(
            self.mjx_model, qpos=init_traj_data.qpos, qvel=init_traj_data.qvel, ctrl=init_traj_data.qpos[7:]
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

        info = {
            "rng": rng,
            "step": 0,
            # history
            "last_motor_targets": data.qpos[7:][self.action_joint_ids],
            "last_action": jp.zeros(self.action_size),
            "last_root_pos": data.qpos[:3],
            "last_root_ori": data.qpos[3:7],
            "last_dof_pos": data.qpos[7:],
            "last_rigid_body_pos": data.xpos,
            "last_rigid_body_ori": data.xquat,
            "last_joint_vel": jp.zeros(self.num_joints),
            # reference trajectory info
            "traj_no": traj_no,
            "traj_info": carry,
            # domain rand
            # push
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        obs, history = self._get_obs(data, traj_data, info)
        if self._config.history_len > 0:
            _, init_history = self._get_obs(data, init_traj_data, info)
            init_history_action = jp.concatenate([init_history, info["last_motor_targets"]], axis=0)
            obs["history_state"] = jp.stack([init_history_action] * self._config.history_len, axis=0).flatten()
            info["current_history"] = history

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

        # set motor target
        # action is defined as the deviation from reference motion
        traj_data = self.th.get_current_traj_data_with_trajectory(trajectory_data, state.info["traj_info"])
        lower_motor_targets = traj_data.qpos[7:][self.action_joint_ids] + action * self._config.action_scale

        motor_targets = self._default_qpos.copy()
        motor_targets = motor_targets.at[self.action_joint_ids].set(lower_motor_targets)
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        torque = data.actuator_force[self.action_joint_ids]

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
            obs["history_state"] = jp.concatenate(
                [
                    state.obs["history_state"].reshape(self._config.history_len, -1)[1:],
                    jp.concatenate([state.info["current_history"], state.info["last_motor_targets"]], axis=0)[None, :],
                ],
                axis=0,
            ).flatten()
            state.info["current_history"] = history

        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        # manual reset
        state = jax.lax.cond(
            done, partial(self._reset_and_update_state, trajectory_data=trajectory_data), lambda x: x, state
        )
        return state

    def _reset_and_update_state(self, current_state: mjx_env.State, trajectory_data: TrajectoryData) -> mjx_env.State:
        """Helper function to perform reset and update state."""
        reset_rng, current_state.info["rng"] = jax.random.split(current_state.info["rng"])
        reset_state = self.reset(reset_rng, trajectory_data)
        current_state = current_state.replace(data=reset_state.data, obs=reset_state.obs)
        for key in reset_state.info.keys():
            current_state.info[key] = reset_state.info[key]

        return current_state

    def load_trajectory(self, traj: Trajectory = None, warn: bool = True) -> None:
        th_params = self._th_params if self._th_params is not None else {}
        return TrajectoryHandler(model=self._mj_model, warn=warn, traj=traj, control_dt=self.dt, **th_params)

    def prepare_trajectory(self, dataset_dict: Dict[str, List[str]]) -> Trajectory:
        all_trajectories = []
        for dataset_name, traj_names in dataset_dict.items():
            path_to_datasets = os.path.join(os.getcwd(), "data", "mocap", dataset_name)
            for idx, t_name in enumerate(tqdm(traj_names)):
                # load the npz file
                traj_path = os.path.join(path_to_datasets, "UnitreeG1", f"{t_name}.npz")
                traj = Trajectory.load(traj_path, backend=np)

                if not traj.data.is_complete:
                    traj = self.extend_motion(traj)
                    traj.info.model = traj.info.model.to_numpy()
                    traj.data = traj.data.to_numpy()
                    traj.save(traj_path)  # save trajectory before recalculating velocity

                # recalculate velocity
                if self._config.recalculate_velocity:
                    traj = recalculate_traj_angular_velocity(traj, frequency=1.0 / self.dt, backend=np)
                    traj = recalculate_traj_linear_velocity(traj, frequency=1.0 / self.dt, backend=np)
                    traj = recalculate_traj_joint_velocity(traj, frequency=1.0 / self.dt, backend=np)
                all_trajectories.append(traj)

        # concatenate trajectories
        if len(all_trajectories) == 1:
            trajectory = all_trajectories[0]
        else:
            traj_datas = [t.data for t in all_trajectories]
            traj_infos = [t.info for t in all_trajectories]
            traj_data, traj_info = TrajectoryData.concatenate(traj_datas, traj_infos, backend=np)
            trajectory = Trajectory(traj_info, traj_data)

        # load trajectory again to ensure the latest transformed trajectories is loaded
        trajectory.info.model = trajectory.info.model.to_jax()
        trajectory.data = trajectory.data.to_jax_fp32()
        self.th = self.load_trajectory(trajectory, warn=False)

        return trajectory.data

    def extend_motion(self, traj: Trajectory) -> Trajectory:
        assert traj.data.n_trajectories == 1

        traj_data, traj_info = interpolate_trajectories(traj.data, traj.info, 1.0 / self.dt)
        traj = Trajectory(info=traj_info, data=traj_data)

        self.th = TrajectoryHandler(model=self._mj_model, warn=True, traj=traj, control_dt=self.dt, random_start=False, fixed_start_conf=(0, 0))
        traj_data, traj_info = self.th.traj.data, self.th.traj.info

        callback = ExtendTrajData(self, model=self._mj_model, n_samples=traj_data.n_samples)
        self.play_trajectory(n_episodes=self.th.n_trajectories, callback_class=callback)
        traj_data, traj_info = callback.extend_trajectory_data(traj_data, traj_info)
        traj = replace(traj, data=traj_data, info=traj_info)

        return traj

    def play_trajectory(
        self,
        n_episodes: int = None,
        n_steps_per_episode: int = None,
        callback_class: Callable = None,
        quiet: bool = False,
    ) -> None:
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones in the trajectories at every step.

        Args:
            n_episodes (int): Number of episode to replay.
            n_steps_per_episode (int): Number of steps to replay per episode.
            callback_class: Object to be called at each step of the simulation.
            quiet (bool): If True, disable tqdm.
        """

        assert self.th is not None

        if not self.th.is_numpy:
            was_jax = True
            self.th.to_numpy()
        else:
            was_jax = False

        traj_info = TrajCarry(key=jax.random.PRNGKey(123), traj_state=self.th.init_state())
        traj_data_sample = self.th.get_current_traj_data(traj_info, np)

        highest_int = np.iinfo(np.int32).max
        if n_episodes is None:
            n_episodes = highest_int
        for i in range(n_episodes):
            if n_steps_per_episode is None:
                nspe = self.th.len_trajectory(traj_info.traj_state.traj_no) - traj_info.traj_state.subtraj_step_no
            else:
                nspe = n_steps_per_episode

            for j in tqdm(range(nspe), disable=quiet):
                self._mj_model, self._data, traj_info = callback_class(
                    self, self._mj_model, self._data, traj_data_sample, traj_info
                )

                traj_data_sample = self.th.get_current_traj_data(traj_info, np)

        if was_jax:
            self.th.to_jax()

    def set_sim_state_from_traj_data(self, data: mujoco.MjData, traj_data: TrajectoryData, carry: TrajCarry) -> MjData:
        """
        Sets the Mujoco datastructure to the state specified in the trajectory data.

        Args:
            data (MjData): The Mujoco data structure.
            traj_data: The trajectory data containing state information.
            carry (Carry): Additional carry information.

        Returns:
            MjData: The updated Mujoco data structure.
        """
        robot_free_jnt_qpos_id_xy = np.array(mj_jntname2qposid("root", self._mj_model))[:2]
        free_jnt_qpos_id = np.concatenate(
            [
                mj_jntid2qposid(i, self._mj_model)
                for i in range(self._mj_model.njnt)
                if self._mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
            ]
        ).reshape(-1, 7)
        all_free_jnt_qpos_id_xy = free_jnt_qpos_id[:, :2].reshape(-1)
        traj_state = carry.traj_state
        # get the initial state of the current trajectory
        traj_data_init = self.th.traj.data.get(traj_state.traj_no, traj_state.subtraj_step_no_init, np)
        # subtract the initial state from the current state
        traj_data.qpos[all_free_jnt_qpos_id_xy] -= traj_data_init.qpos[robot_free_jnt_qpos_id_xy]

        if traj_data.xpos.size > 0:
            data.xpos = traj_data.xpos
        if traj_data.xquat.size > 0:
            data.xquat = traj_data.xquat
        if traj_data.cvel.size > 0:
            data.cvel = traj_data.cvel
        if traj_data.qpos.size > 0:
            data.qpos = traj_data.qpos
        if traj_data.qvel.size > 0:
            data.qvel = traj_data.qvel

        return data

    def _get_termination(self, data: mjx.Data, traj_data: TrajectoryData, info: dict[str, Any]) -> jax.Array:
        # fall_termination = jp.abs(data.qpos[2] - traj_data.qpos[2]) > self._config.termination_config.root_height_threshold
        fall_termination = (
            jp.abs(data.xpos[self.shoulder_ids[0], 2] - traj_data.xpos[self.shoulder_ids[0], 2])
            > self._config.termination_config.root_height_threshold
        )
        # fall_termination = self.get_gravity(data, "pelvis")[2] < 0.0

        # NOTICE: rigid body with id 0 is world!
        dif_rigid_body_pos_local = gmth.calculate_dif_rigid_body_pos_local(data, traj_data)
        norm_dif_rigid_body_pos_local = jp.linalg.norm(dif_rigid_body_pos_local[self.valid_body_ids, :], axis=-1)
        rigid_body_position_termination = jp.any(
            norm_dif_rigid_body_pos_local > self._config.termination_config.rigid_body_dif_threshold
        )

        return (
            fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any() | rigid_body_position_termination
        )

    def _get_obs(self, data: mjx.Data, traj_data: TrajectoryData, info: dict[str, Any]) -> mjx_env.Observation:
        # body pose
        gyro_pelvis = self.get_gyro(data, "pelvis")
        gvec_pelvis = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
        linvel_pelvis = self.get_local_linvel(data, "pelvis")
        dif_torso_rp = gmth.calculate_dif_torso_rp(data, traj_data)

        def quat_to_angvel(current_q, last_q):
            # calculate angular velocity in current_q coordination
            delta_q = math.quat_mul(math.quat_inv(last_q), current_q)
            w, x, y, z = delta_q
            angle = 2.0 * jp.arccos(jp.clip(w, -1.0, 1.0))
            axis = jp.array([x, y, z])
            axis = axis / (jp.linalg.norm(axis) + 1e-9)
            angvel = axis * angle / self.dt
            return angvel

        avg_gyro_pelvis = quat_to_angvel(data.qpos[3:7], info["last_root_ori"])

        ref_gyro_pelvis = traj_data.qvel[3:6]
        ref_gvec_pelvis = traj_data.site_xmat[self._pelvis_imu_site_id].reshape(3, 3).T @ jp.array([0, 0, -1])

        # joint
        joint_pos = data.qpos[7:]
        joint_vel = data.qvel[6:]
        avg_joint_vel = (traj_data.qpos[7:] - info["last_dof_pos"]) / self.dt

        # reference
        dif_joint_pos = traj_data.qpos[7:] - joint_pos
        dif_joint_vel = traj_data.qvel[6:] - joint_vel

        # hint state
        dif_rigid_body_pos_local = gmth.calculate_dif_rigid_body_pos_local(data, traj_data).flatten()
        dif_rigid_body_rot_local = gmth.calculate_dif_rigid_body_rot_local(data, traj_data).flatten()
        dif_rigid_body_linvel_local = gmth.calculate_dif_rigid_body_linvel_local(data, traj_data).flatten()
        dif_rigid_body_angvel_local = gmth.calculate_dif_rigid_body_angvel_local(data, traj_data).flatten()
        feet_contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])

        traj_root_rot_mat = math.quat_to_mat(traj_data.qpos[3:7])
        root_rot_mat = math.quat_to_mat(data.qpos[3:7])
        dif_root_linvel = traj_root_rot_mat.T @ traj_data.qvel[:3] - root_rot_mat.T @ data.qvel[:3]
        dif_root_angvel = traj_data.qvel[3:6] - data.qvel[3:6]

        ref_feet_height = traj_data.site_xpos[self._feet_all_site_id, 2]
        dif_feet_height = traj_data.site_xpos[self._feet_all_site_id, 2] - data.site_xpos[self._feet_all_site_id, 2]

        # add uniform noise to the observation
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro_pelvis = (
            gyro_pelvis
            + (2 * jax.random.uniform(noise_rng, shape=gyro_pelvis.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gvec_pelvis = (
            gvec_pelvis
            + (2 * jax.random.uniform(noise_rng, shape=gvec_pelvis.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_pos = (
            joint_pos
            + (2 * jax.random.uniform(noise_rng, shape=joint_pos.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_pos
        )

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        noisy_dif_joint_pos = traj_data.qpos[7:] - noisy_joint_pos
        noisy_dif_joint_vel = traj_data.qvel[6:] - noisy_joint_vel

        state_dict = {
            "gyro_pelvis": noisy_gyro_pelvis * self._config.obs_scales_config.joint_vel,
            "gvec_pelvis": noisy_gvec_pelvis,
            "joint_pos": (noisy_joint_pos - self._default_qpos)[self.obs_joint_ids],
            "joint_vel": noisy_joint_vel[self.obs_joint_ids] * self._config.obs_scales_config.joint_vel,
            "last_motor_targets": info["last_motor_targets"],
            "last_action": info["last_action"],
            "dif_joint_pos": noisy_dif_joint_pos,
            "dif_joint_vel": noisy_dif_joint_vel * self._config.obs_scales_config.joint_vel,
            "ref_feet_height": ref_feet_height,
            "ref_root_height": traj_data.qpos[2],
            "ref_root_linvel": (traj_root_rot_mat.T @ traj_data.qvel[:3]) * self._config.obs_scales_config.joint_vel,
            "ref_root_angvel": traj_data.qvel[3:6] * self._config.obs_scales_config.joint_vel,
        }

        privileged_state_dict = {
            "gyro_pelvis": gyro_pelvis * self._config.obs_scales_config.joint_vel,
            "avg_gyro_pelvis": avg_gyro_pelvis * self._config.obs_scales_config.joint_vel,
            "gvec_pelvis": gvec_pelvis,
            "linvel_pelvis": linvel_pelvis * self._config.obs_scales_config.joint_vel,
            "root_height": data.qpos[2],
            "dif_torso_rp": dif_torso_rp,
            "joint_pos": (joint_pos - self._default_qpos)[self.obs_joint_ids],
            "joint_vel": joint_vel[self.obs_joint_ids] * self._config.obs_scales_config.joint_vel,
            "avg_joint_vel": avg_joint_vel * self._config.obs_scales_config.joint_vel,
            "last_motor_targets": info["last_motor_targets"],
            "last_action": info["last_action"],
            "dif_joint_pos": dif_joint_pos,
            "dif_joint_vel": dif_joint_vel * self._config.obs_scales_config.joint_vel,
            "feet_contact": feet_contact,
            "dif_feet_height": dif_feet_height,
            "dif_root_height": traj_data.qpos[2] - data.qpos[2],
            "dif_rigid_body_pos_local": dif_rigid_body_pos_local,
            "dif_rigid_body_rot_local": dif_rigid_body_rot_local,
            "dif_rigid_body_linvel_local": dif_rigid_body_linvel_local * self._config.obs_scales_config.joint_vel,
            "dif_rigid_body_angvel_local": dif_rigid_body_angvel_local * self._config.obs_scales_config.joint_vel,
            "dif_root_linvel": dif_root_linvel * self._config.obs_scales_config.joint_vel,
            "dif_root_angvel": dif_root_angvel * self._config.obs_scales_config.joint_vel,
            # aux / world model ref
            "ref_gyro_pelvis": ref_gyro_pelvis * self._config.obs_scales_config.joint_vel,
            "ref_gvec_pelvis": ref_gvec_pelvis,
            "ref_root_height": traj_data.qpos[2],
            "ref_joint_pos": traj_data.qpos[7:],
            "ref_joint_vel": traj_data.qvel[6:],
        }

        state = jp.hstack([state_dict[k] for k in self._config.obs_keys])
        privileged_state = jp.hstack([privileged_state_dict[k] for k in self._config.privileged_obs_keys])
        current_history = jp.hstack([state_dict[k] for k in self._config.history_keys])

        # Nan to 0
        state = jp.nan_to_num(state)
        privileged_state = jp.nan_to_num(privileged_state)
        current_history = jp.nan_to_num(current_history)

        return {
            "state": state,
            "privileged_state": privileged_state,
        }, current_history

    def _get_reward(
        self,
        data: mjx.Data,
        traj_data: TrajectoryData,
        action: jax.Array,
        motor_targets: jax.Array,
        torque: jax.Array,
        info: dict[str, Any],
    ) -> dict[str, jax.Array]:
        dif_rigid_body_pos_local = gmth.calculate_dif_rigid_body_pos_local(data, traj_data)
        dif_rigid_body_rot_local = gmth.calculate_dif_rigid_body_rot_local(data, traj_data)
        dif_joint_pos = traj_data.qpos[7:] - data.qpos[7:]

        if self._config.recalculate_velocity:
            dif_rigid_body_linvel_local = gmth.calculate_dif_rigid_body_linvel_local_differential(
                data, traj_data, info["last_rigid_body_pos"]
            )
            dif_rigid_body_angvel_local = gmth.calculate_dif_rigid_body_angvel_local_differential(
                data, traj_data, info["last_rigid_body_ori"]
            )
            dif_joint_vel = traj_data.qvel[6:] - ((data.qpos[7:] - info["last_dof_pos"]) / self.dt)
            dif_root_linvel = gmth.calculate_dif_root_linvel_local_differential(data, traj_data, info["last_root_pos"])
            dif_root_angvel = gmth.calculate_dif_root_angvel_local_differential(data, traj_data, info["last_root_ori"])
        else:
            traj_root_rot_mat = math.quat_to_mat(traj_data.qpos[3:7])
            root_rot_mat = math.quat_to_mat(data.qpos[3:7])
            dif_rigid_body_linvel_local = gmth.calculate_dif_rigid_body_linvel_local(data, traj_data)
            dif_rigid_body_angvel_local = gmth.calculate_dif_rigid_body_angvel_local(data, traj_data)
            dif_joint_vel = traj_data.qvel[6:] - data.qvel[6:]
            dif_root_linvel = traj_root_rot_mat.T @ traj_data.qvel[:3] - root_rot_mat.T @ data.qvel[:3]
            dif_root_angvel = traj_data.qvel[3:6] - data.qvel[3:6]

        dif_root_height = traj_data.qpos[2] - data.qpos[2]
        dif_feet_height = traj_data.site_xpos[self._feet_all_site_id, 2] - data.site_xpos[self._feet_all_site_id, 2]
        dif_torso_rp = gmth.calculate_dif_torso_rp(data, traj_data)

        global_ref_feet_vel = jp.linalg.norm(traj_data.cvel[self.feet_ids, 3:], axis=-1)
        termination = self._get_termination(data, traj_data, info)

        reward_dict = {
            # tracking reward
            "rigid_body_pos_tracking_upper": self._reward_rigid_body_pos_tracking_upper(dif_rigid_body_pos_local),
            "rigid_body_pos_tracking_lower": self._reward_rigid_body_pos_tracking_lower(dif_rigid_body_pos_local),
            "feet_pos_tracking": self._reward_feet_pos_tracking(dif_rigid_body_pos_local),
            "rigid_body_rot_tracking": self._reward_rigid_body_rot_tracking(dif_rigid_body_rot_local),
            "rigid_body_linvel_tracking": self._reward_rigid_body_linvel_tracking(dif_rigid_body_linvel_local),
            "rigid_body_angvel_tracking": self._reward_rigid_body_angvel_tracking(dif_rigid_body_angvel_local),
            "joint_pos_tracking": self._reward_joint_pos_tracking(dif_joint_pos),
            "joint_vel_tracking": self._reward_joint_vel_tracking(dif_joint_vel),
            "root_linvel_tracking": self._reward_root_linvel_tracking(dif_root_linvel),
            "root_angvel_tracking": self._reward_root_angvel_tracking(dif_root_angvel),
            "roll_pitch_tracking": self._reward_roll_pitch_tracking(dif_torso_rp),
            # penalty reward
            "penalty_torque": self._reward_penalty_torque(torque),
            "penalty_action_rate": self._reward_penalty_action_rate(motor_targets, info["last_motor_targets"]),
            "action_deviation": self._reward_action_deviation(action),
            "dof_pos_limit": self._reward_dof_pos_limit(data.qpos[7:]),
            "dof_vel_limit": self._reward_dof_vel_limit(data.qvel[6:]),
            "torque_limit": self._reward_torque_limit(torque),
            "collision": self._reward_collision(data),
            "feet_linvel_deviation": self._reward_feet_linvel_deviation(data, global_ref_feet_vel),
            "feet_angvel_deviation": self._reward_feet_angvel_deviation(data, global_ref_feet_vel),
            "termination": self._reward_termination(termination),
            "feet_height_tracking": self._reward_feet_height_tracking(dif_feet_height),
            "root_height_tracking": self._reward_root_height_tracking(dif_root_height),
            "smoothness_joint": self._reward_smoothness_joint(data, info["last_joint_vel"]),
        }

        reward_dict = jax.tree_util.tree_map(lambda x: jp.where(jp.isnan(x), 0.0, x), reward_dict)

        return reward_dict

    def _reward_rigid_body_pos_tracking_upper(self, dif_rigid_body_pos: jax.Array) -> jax.Array:
        upper_body_diff = dif_rigid_body_pos[self.upper_body_ids, :]
        diff_body_pos_dist_upper = jp.sum(jp.abs(upper_body_diff), axis=(-2, -1))
        r_body_pos_upper = jp.exp(-diff_body_pos_dist_upper / self._config.reward_config.auxiliary.upper_body_sigma)

        return r_body_pos_upper

    def _reward_rigid_body_pos_tracking_lower(self, dif_rigid_body_pos: jax.Array) -> jax.Array:
        lower_body_diff = dif_rigid_body_pos[self.lower_body_ids, :]
        diff_body_pos_dist_lower = jp.sum(jp.abs(lower_body_diff), axis=(-2, -1))
        r_body_pos_lower = jp.exp(-diff_body_pos_dist_lower / self._config.reward_config.auxiliary.lower_body_sigma)

        return r_body_pos_lower

    def _reward_feet_pos_tracking(self, dif_rigid_body_pos: jax.Array) -> jax.Array:
        feet_pos_diff = dif_rigid_body_pos[self.feet_ids, :]
        feet_pos_dist = jp.sum(jp.abs(feet_pos_diff), axis=(-2, -1))

        rew = jp.exp(-feet_pos_dist / self._config.reward_config.auxiliary.feet_pos_sigma)
        return rew

    def _reward_rigid_body_rot_tracking(self, dif_rigid_body_rot: jax.Array) -> jax.Array:
        diff_body_rot_dist = 2 * jp.arccos(dif_rigid_body_rot[self.valid_body_ids, 0]).mean(axis=-1)
        rew = jp.exp(-diff_body_rot_dist / self._config.reward_config.auxiliary.body_rot_sigma)
        return rew

    def _reward_rigid_body_linvel_tracking(self, dif_rigid_body_linvel: jax.Array) -> jax.Array:
        diff_body_linvel_dist = (dif_rigid_body_linvel[self.valid_body_ids, :] ** 2).mean(axis=(-1, -2))

        rew = jp.exp(-diff_body_linvel_dist / self._config.reward_config.auxiliary.body_linvel_sigma)
        return rew

    def _reward_rigid_body_angvel_tracking(self, dif_rigid_body_angvel: jax.Array) -> jax.Array:
        diff_body_angvel_dist = (dif_rigid_body_angvel[self.valid_body_ids, :] ** 2).mean(axis=(-1, -2))

        rew = jp.exp(-diff_body_angvel_dist / self._config.reward_config.auxiliary.body_angvel_sigma)
        return rew

    def _reward_joint_pos_tracking(self, dif_joint_pos: jax.Array) -> jax.Array:
        diff_joint_pos_dist = jp.sum(jp.abs(dif_joint_pos), axis=-1)

        rew = jp.exp(-diff_joint_pos_dist / self._config.reward_config.auxiliary.joint_pos_sigma)
        return rew

    def _reward_joint_vel_tracking(self, dif_joint_vel: jax.Array) -> jax.Array:
        diff_joint_vel_dist = jp.sum(jp.abs(dif_joint_vel), axis=-1) * self.dt

        rew = jp.exp(-diff_joint_vel_dist / self._config.reward_config.auxiliary.joint_vel_sigma)
        return rew

    def _reward_root_linvel_tracking(self, dif_root_linvel: jax.Array) -> jax.Array:
        diff_root_linvel_dist = jp.sum(jp.abs(dif_root_linvel), axis=-1)

        rew = jp.exp(-diff_root_linvel_dist / self._config.reward_config.auxiliary.root_linvel_sigma)
        return rew

    def _reward_root_angvel_tracking(self, dif_root_angvel: jax.Array) -> jax.Array:
        diff_root_angvel_dist = jp.sum(jp.abs(dif_root_angvel), axis=-1)

        rew = jp.exp(-diff_root_angvel_dist / self._config.reward_config.auxiliary.root_angvel_sigma)
        return rew

    def _reward_roll_pitch_tracking(self, dif_rp: jax.Array) -> jax.Array:
        diff_rp_dist = jp.sum(jp.abs(dif_rp), axis=-1)

        rew = jp.exp(-diff_rp_dist / self._config.reward_config.auxiliary.roll_pitch_sigma)
        return rew

    def _reward_penalty_torque(self, torque: jax.Array) -> jax.Array:
        return jp.sum(jp.square(torque), axis=-1)

    def _reward_penalty_action_rate(self, action: jax.Array, last_action: jax.Array) -> jax.Array:
        return jp.sum(jp.square(last_action - action), axis=-1)

    def _reward_termination(self, termination: jax.Array) -> jax.Array:
        return termination

    def _reward_dof_pos_limit(self, dof_pos: jp.ndarray) -> jp.ndarray:
        # Penalize joints if they cross soft limits.
        out_of_limits = -jp.clip(dof_pos - self._soft_lowers, None, 0.0)
        out_of_limits += jp.clip(dof_pos - self._soft_uppers, 0.0, None)
        return jp.clip(jp.sum(out_of_limits), 0.0, 100.0)

    def _reward_dof_vel_limit(self, dof_vel: jp.ndarray) -> jp.ndarray:
        out_of_limits = jp.clip(jp.abs(dof_vel) - self.dof_vel_limit, 0.0, 1.0)
        penalty = jp.sum(out_of_limits, axis=-1)

        return penalty

    def _reward_torque_limit(self, torque: jax.Array) -> jp.ndarray:
        out_of_limits = jp.clip(jp.abs(torque) - self.torque_limit, min=0.0)
        penalty = jp.sum(out_of_limits, axis=-1)

        return penalty

    def _reward_action_deviation(self, action: jax.Array) -> jax.Array:
        penalty = jp.sum(jp.square(action[self.upper_body_joints]), axis=-1)

        return penalty

    def _reward_collision(self, data: mjx.Data) -> jax.Array:
        pair_geom1 = self.penalize_collision_pair[:, 0]
        pair_geom2 = self.penalize_collision_pair[:, 1]

        collided_values = jax.vmap(partial(geoms_colliding, data))(pair_geom1, pair_geom2)

        return jp.sum(collided_values, axis=-1)

    def _reward_feet_linvel_deviation(self, data: mjx.Data, global_ref_feet_vel: jax.Array) -> jax.Array:
        current_global_feet_vel = jp.linalg.norm(data.cvel[self.feet_ids, 3:], axis=-1)
        slipping_conditions = (
            jp.abs(global_ref_feet_vel) < self._config.reward_config.auxiliary.global_feet_vel_threshold
        )
        left_foot_penalty = jax.lax.cond(
            slipping_conditions[0], lambda _: jp.square(current_global_feet_vel[0]), lambda _: 0.0, None
        )
        right_foot_penalty = jax.lax.cond(
            slipping_conditions[1], lambda _: jp.square(current_global_feet_vel[1]), lambda _: 0.0, None
        )
        rew = jp.clip(left_foot_penalty + right_foot_penalty, 0, 25)

        return rew

    def _reward_feet_angvel_deviation(self, data: mjx.Data, global_ref_feet_vel: jax.Array) -> jax.Array:
        current_global_feet_vel = jp.linalg.norm(data.cvel[self.feet_ids, :3], axis=-1)
        slipping_conditions = (
            jp.abs(global_ref_feet_vel) < self._config.reward_config.auxiliary.global_feet_vel_threshold
        )
        left_foot_penalty = jax.lax.cond(
            slipping_conditions[0], lambda _: jp.square(current_global_feet_vel[0]), lambda _: 0.0, None
        )
        right_foot_penalty = jax.lax.cond(
            slipping_conditions[1], lambda _: jp.square(current_global_feet_vel[1]), lambda _: 0.0, None
        )
        rew = left_foot_penalty + right_foot_penalty

        return rew

    def _reward_root_height_tracking(self, dif_root_height: jax.Array) -> jax.Array:
        diff_root_height_dist = jp.abs(dif_root_height)

        rew = jp.exp(-diff_root_height_dist / self._config.reward_config.auxiliary.root_height_sigma)
        return rew

    def _reward_feet_height_tracking(self, dif_feet_height: jax.Array) -> jax.Array:
        diff_feet_height_dist = jp.sum(jp.abs(dif_feet_height), axis=-1)

        rew = jp.exp(-diff_feet_height_dist / self._config.reward_config.auxiliary.feet_height_sigma)
        return rew

    def _reward_smoothness_joint(self, data: mjx.Data, last_joint_vel):
        qvel = data.qvel[6:]
        qacc = (qvel - last_joint_vel) / self.dt
        cost = jp.sum(0.02 * jp.square(qvel) + jp.square(qacc))
        return cost
