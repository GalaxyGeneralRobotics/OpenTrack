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
from track_mj.utils.dataset.traj_process import ExtendTrajData, SmoothStartEndTransition
from track_mj.utils import math as gmth

ENABLE_PUSH = True
EPISODE_LENGTH = 1000


def g1_tracking_general_task_config() -> config_dict.ConfigDict:

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
        randomization_fn=None,
        
        policy_args=None,
    )

    config = config_dict.create(
        env_config=env_config,
        policy_config=policy_config,
    )
    return config


tmj.registry.register("G1TrackingGeneral", "tracking_dagger_config")(g1_tracking_general_task_config())


def torque_step(
    rng: jax.Array,
    model: mjx.Model,
    data: mjx.Data,
    qpos_des: jax.Array,
    kps: jax.Array,
    kds: jax.Array,
    torque_limit: jax.Array,
    n_substeps: int = 1,
) -> tuple[jax.Array, mjx.Data, jax.Array]:
    def single_step(carry, _):
        rng, data, _ = carry
        rng, rng_rfi = jax.random.split(rng, 2)

        # pd control
        pos_err = qpos_des - data.qpos[7:]
        vel_err = -data.qvel[6:]
        torque = (kps) * pos_err + (kds) * vel_err

        # clip
        torque = jp.clip(torque, -torque_limit, torque_limit)

        # apply torque
        data = data.replace(ctrl=torque)
        data = mjx.step(model, data)

        return (rng, data, torque), None

    initial_torque = jp.zeros_like(torque_limit)
    (final_rng, final_data, final_torque), _ = jax.lax.scan(single_step, (rng, data, initial_torque), (), n_substeps)

    return final_rng, final_data, final_torque


def get_collision_contact(contact: Any, geom1: int, geom2: int) -> Tuple[jax.Array, jax.Array]:
    """Get the contact point between two geoms"""
    mask = (jp.array([geom1, geom2]) == contact.geom).all(axis=1)
    mask |= (jp.array([geom2, geom1]) == contact.geom).all(axis=1)
    idx = jp.where(mask, contact.dist, 1e4).argmin()
    dist = contact.dist[idx] * mask[idx]
    pos = contact.pos[idx]
    return dist < 0, pos


@tmj.registry.register("G1TrackingGeneral", "tracking_dagger_train_env_class")
class G1TrackingGeneralEnv(g1_base.G1Env):
    @property
    def action_size(self) -> int:
        return len(self.action_joint_names)


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
        self.key_body_ids = jp.array([self.mj_model.body(n).id for n in consts.KEY_BODY_LINKs])
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

        self.recalculate_vel_in_reward = self._config.recalculate_vel_in_reward
        self.recalculate_vel_in_reference_motion = self._config.recalculate_vel_in_reference_motion

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


    def _get_motor_targets(self, state: mjx_env.State, action: jax.Array, use_residual_action: bool, trajectory_data: TrajectoryData = None) -> jax.Array:

        if trajectory_data is None:
            trajectory_data = self.th.traj.data
        motor_targets = self._default_qpos.copy()
        
        def residual_fn(_):
            traj_data = self.th.get_current_traj_data_with_trajectory(trajectory_data, state.info["traj_info"])
            lower_motor_targets = traj_data.qpos[7:][self.action_joint_ids] + action * self._config.action_scale
            return motor_targets.at[self.action_joint_ids].set(lower_motor_targets)
        
        def absolute_fn(_):
            return motor_targets.at[self.action_joint_ids].set(action * self._config.action_scale)
        
        motor_targets = jax.lax.cond(
            use_residual_action,
            residual_fn,
            absolute_fn,
            operand=None
        )
        
        return motor_targets

    def step(self, state: mjx_env.State, action: jax.Array, trajectory_data: TrajectoryData = None) -> mjx_env.State:
        if trajectory_data is None:
            trajectory_data = self.th.traj.data
        
        state.info["rng"], push1_rng, push2_rng = jax.random.split(state.info["rng"], 3)

        traj_data = self.th.get_current_traj_data_with_trajectory(trajectory_data, state.info["traj_info"])
        motor_targets = self._get_motor_targets(state, action, use_residual_action=self._config.student_use_residual_action, trajectory_data=trajectory_data)

        state.info["rng"], data, torque = torque_step(
            state.info["rng"],
            self.mjx_model,
            state.data,
            motor_targets,
            kps=self._kps,
            kds=self._kds,
            torque_limit=self.torque_limit,
            n_substeps=self.n_substeps,
        )
        rewards = self._get_reward(data, traj_data, action, motor_targets, torque, state.info)
        rewards = {k: v * rewards[k] for k, v in self._config.reward_config.scales.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, a_max=10000.0)

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
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

    def _reset_and_update_state(self, current_state: mjx_env.State, trajectory_data: TrajectoryData) -> mjx_env.State:
        """Helper function to perform reset and update state."""
        reset_rng, current_state.info["rng"] = jax.random.split(current_state.info["rng"])
        reset_state = self.reset(reset_rng, trajectory_data)
        current_state = current_state.replace(data=reset_state.data, obs=reset_state.obs)
        for key in reset_state.info.keys():
            current_state.info[key] = reset_state.info[key]

        return current_state

    def load_trajectory(self, traj: Trajectory = None, warn: bool = True, trajectory_sample_probs: Dict[str, np.ndarray] = None) -> None:
        th_params = self._th_params if self._th_params is not None else {}
        return TrajectoryHandler(model=self._mj_model, warn=warn, traj=traj, control_dt=self.dt, traj_sample_probs=trajectory_sample_probs, **th_params)

    def prepare_trajectory(
            self, dataset_dict: Dict[str, List[str]],
            traj_sample_cluster_ids: List[Tuple[str, int]]=None, trajectory_sample_probs: List[Tuple[str, np.ndarray]] = None
        ) -> Trajectory:
        all_trajectories = []
        all_traj_names = []
        
        for dataset_name, traj_names in dataset_dict.items():
            print(f"Loading dataset: {dataset_name} with {len(traj_names)} trajectories.")
            dataset_dir = os.path.join(os.getcwd(), "storage", "data", "mocap", dataset_name)

            for idx, t_name in enumerate(tqdm(traj_names)):
                # load the npz file
                traj_path = os.path.join(dataset_dir, "UnitreeG1", f"{t_name}.npz")
                # import ipdb; ipdb.set_trace()
                if os.path.exists(traj_path):
                    traj = Trajectory.load(traj_path, backend=jp)
                    if not traj.data.is_complete:
                        print(f"Trajectory {t_name} is not complete. Extending...")
                        traj = self.extend_motion(traj)
                        traj.save(traj_path)  # save trajectory before recalculating velocity
                    print(f"Loaded trajectory {t_name}")

                    # recalculate velocity
                    if self.recalculate_vel_in_reference_motion:
                        traj = recalculate_traj_angular_velocity(traj, frequency=1.0 / self.dt, backend=jp)
                        traj = recalculate_traj_linear_velocity(traj, frequency=1.0 / self.dt, backend=jp)
                        traj = recalculate_traj_joint_velocity(traj, frequency=1.0 / self.dt, backend=jp)
                    all_trajectories.append(traj)
                    all_traj_names.append(os.path.join(t_name))
                    continue
                raise FileNotFoundError(f"Motion file {traj_path} not found.")
            
        assert len(all_trajectories) > 0, "No valid trajectories found in the dataset."

        # concatenate trajectories
        if len(all_trajectories) == 1:
            trajectory = all_trajectories[0]
        else:
            traj_datas = [t.data for t in all_trajectories]
            traj_infos = [t.info for t in all_trajectories]
            traj_data, traj_info = TrajectoryData.concatenate(traj_datas, traj_infos, backend=jp)
            trajectory = Trajectory(traj_info, traj_data)
        
        # sort
        trajectory_sample_probs = dict(trajectory_sample_probs)
        traj_sample_cluster_ids = dict(traj_sample_cluster_ids)

        trajectory_sample_probs = [trajectory_sample_probs[name] for name in all_traj_names] if trajectory_sample_probs is not None else None
        self.traj_sample_cluster_ids = [traj_sample_cluster_ids[name] for name in all_traj_names] if traj_sample_cluster_ids is not None else None

        self.th = self.load_trajectory(trajectory, warn=False, trajectory_sample_probs=trajectory_sample_probs)          # includes a `filter_and_extend` & `interpolate_trajectories()` calling

        return trajectory.data

    def preprocess_trajectory(self, dataset_dict: Dict[str, List[str]], batch_idx: int, num_batches: int, smooth_start_end: bool = True) -> Trajectory:
        all_trajectories = []
        num_trajectory = sum(len(traj_names) for traj_names in dataset_dict.values())

        if num_batches > 1 and batch_idx is not None:
            num_trajectories_per_batch = num_trajectory // num_batches
            reminder = num_trajectory % num_batches

            print(num_trajectory, batch_idx, num_batches, reminder)

            if batch_idx < reminder:
                start_idx = batch_idx * (num_trajectories_per_batch + 1)
                end_idx = start_idx + num_trajectories_per_batch + 1
            else:
                start_idx = batch_idx * num_trajectories_per_batch + reminder
                end_idx = start_idx + num_trajectories_per_batch

            print("====================================================================================")
            print(f"num_batches: {num_batches}, batch_idx: {batch_idx}, start_idx: {start_idx}, end_idx: {end_idx}")
            print("====================================================================================")
        else:
            start_idx, end_idx = 0, num_trajectory

        current_idx = 0

        for dataset_name, traj_names in dataset_dict.items():
            print(f"Loading dataset: {dataset_name} with {len(traj_names)} trajectories.")
            dataset_dir = os.path.join(os.getcwd(), "storage", "data", "mocap", dataset_name)

            for idx, t_name in enumerate(tqdm(traj_names)):
                if current_idx >= end_idx:
                    break

                if current_idx >= start_idx:
                    # load the npz file
                    traj_path = os.path.join(dataset_dir, "UnitreeG1", f"{t_name}.npz")

                    if os.path.exists(traj_path):
                        traj = Trajectory.load(traj_path, backend=jp)

                        if not traj.data.is_complete:
                            print(f"Trajectory {t_name} is not complete. Extending...")
                            traj = self.extend_motion(traj, smooth_start_end=smooth_start_end)
                            traj.save(traj_path)  # save trajectory before recalculating velocity
                        print(f"Loaded trajectory {t_name}")

                        # recalculate velocity
                        if self.recalculate_vel_in_reference_motion:
                            traj = recalculate_traj_angular_velocity(traj, frequency=1.0 / self.dt, backend=jp)
                            traj = recalculate_traj_linear_velocity(traj, frequency=1.0 / self.dt, backend=jp)
                            traj = recalculate_traj_joint_velocity(traj, frequency=1.0 / self.dt, backend=jp)
                        all_trajectories.append(traj)
                    else:
                        raise FileNotFoundError(f"Motion file {traj_path} not found.")

                current_idx += 1

            if current_idx >= end_idx:
                break
        
        print(f"Batch {batch_idx} loaded {len(all_trajectories)} trajectories. Desired range: [{start_idx}, {end_idx}).")
        assert len(all_trajectories) > 0, "No valid trajectories found in the dataset."

        return None

    def extend_motion(self, traj: Trajectory, smooth_start_end: bool = True) -> Trajectory:
        assert traj.data.n_trajectories == 1
        
        traj_data, traj_info = interpolate_trajectories(traj.data, traj.info, 1.0 / self.dt)
        traj = Trajectory(info=traj_info, data=traj_data)

        self.th = TrajectoryHandler(
            model=self._mj_model, warn=True, traj=traj, control_dt=self.dt, random_start=False, fixed_start_conf=(0, 0)
        )

        traj = self.th.traj

        # Move start/end smoothing to the penultimate preprocessing step.
        if smooth_start_end:
            start_end_transition_smoother = SmoothStartEndTransition(model=self._mj_model, traj=traj)
            traj = start_end_transition_smoother.run_interp(return_backend=jp)  # use default params

            # Rebuild trajectory handler because smoothing changes trajectory content/length.
            self.th = TrajectoryHandler(
                model=self._mj_model, warn=True, traj=traj, control_dt=self.dt, random_start=False, fixed_start_conf=(0, 0)
            )

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

    def traj_render(self, traj_info: TrajCarry, record: bool = False) -> np.ndarray:
        if self._viewer is None:
            viewer_params = {"geom_group_visualization_on_startup": [0, 2], "default_camera_mode": "follow"}
            self._viewer = MujocoViewer(self._mj_model, self.dt, record=record, **viewer_params)

            headless = viewer_params.get("headless", False)

            if not headless:
                # register stop function to be called at exit
                atexit.register(self.stop)

        return self._viewer.render(self._data, traj_info, record)

    def stop(self) -> None:
        if self._viewer is not None:
            self._video_file_path = self._viewer.stop()
            del self._viewer
            self._viewer = None

    def _get_termination(self, data: mjx.Data, traj_data: TrajectoryData, info: dict[str, Any]) -> jax.Array:
        fall_termination = jp.abs(data.qpos[2] - traj_data.qpos[2]) > self._config.termination_config.root_height_threshold

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

        # joint
        joint_pos = data.qpos[7:]
        joint_vel = data.qvel[6:]

        # reference
        dif_joint_pos = traj_data.qpos[7:] - joint_pos
        dif_joint_vel = traj_data.qvel[6:] - joint_vel

        feet_contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])

        traj_root_rot_mat = math.quat_to_mat(traj_data.qpos[3:7])
        root_rot_mat = math.quat_to_mat(data.qpos[3:7])

        dif_root_linvel_local = traj_root_rot_mat.T @ traj_data.qvel[:3] - root_rot_mat.T @ data.qvel[:3]
        dif_root_angvel_local = traj_data.qvel[3:6] - data.qvel[3:6]
        ref_feet_height = traj_data.site_xpos[self._feet_all_site_id, 2]
        dif_feet_height = traj_data.site_xpos[self._feet_all_site_id, 2] - data.site_xpos[self._feet_all_site_id, 2]

        # hint state
        dif_rigid_body_pos_local = gmth.calculate_dif_rigid_body_pos_local(data, traj_data).flatten()
        dif_rigid_body_rot_local = gmth.calculate_dif_rigid_body_rot_local(data, traj_data).flatten()
        dif_rigid_body_linvel_local = gmth.calculate_dif_rigid_body_linvel_local(data, traj_data).flatten()
        dif_rigid_body_angvel_local = gmth.calculate_dif_rigid_body_angvel_local(data, traj_data).flatten()

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

        # if w < 0, use -q
        state_dict = {
            "gyro_pelvis": noisy_gyro_pelvis * self._config.obs_scales_config.joint_vel,
            "gvec_pelvis": noisy_gvec_pelvis,
            "joint_pos": (noisy_joint_pos - self._default_qpos)[self.obs_joint_ids],
            "joint_vel": noisy_joint_vel[self.obs_joint_ids] * self._config.obs_scales_config.joint_vel,
            "last_motor_targets": info["last_motor_targets"],
            "dif_joint_pos": noisy_dif_joint_pos,
            "dif_joint_vel": noisy_dif_joint_vel * self._config.obs_scales_config.dif_joint_vel,
            "ref_root_height": traj_data.qpos[2],
            "ref_feet_height": ref_feet_height,
        }

        privileged_state_dict = {
            "gyro_pelvis": gyro_pelvis * self._config.obs_scales_config.joint_vel,
            "gvec_pelvis": gvec_pelvis,
            "linvel_pelvis": linvel_pelvis * self._config.obs_scales_config.joint_vel,
            "dif_torso_rp": dif_torso_rp,
            "joint_pos": (joint_pos - self._default_qpos)[self.obs_joint_ids],
            "joint_vel": joint_vel[self.obs_joint_ids] * self._config.obs_scales_config.joint_vel,
            "last_motor_targets": info["last_motor_targets"],
            "dif_joint_pos": dif_joint_pos,
            "dif_joint_vel": dif_joint_vel * self._config.obs_scales_config.dif_joint_vel,
            "feet_contact": feet_contact,
            "dif_feet_height": dif_feet_height,
            "dif_rigid_body_pos_local": dif_rigid_body_pos_local,
            "dif_rigid_body_rot_local": jax.lax.cond(dif_rigid_body_rot_local[0] < 0, lambda x: -x, lambda x: x, dif_rigid_body_rot_local),
            "dif_rigid_body_linvel_local": dif_rigid_body_linvel_local * self._config.obs_scales_config.dif_joint_vel,
            "dif_rigid_body_angvel_local": dif_rigid_body_angvel_local * self._config.obs_scales_config.dif_joint_vel,
            "dif_root_height": traj_data.qpos[2] - data.qpos[2],
            "dif_root_linvel_local": dif_root_linvel_local * self._config.obs_scales_config.dif_joint_vel,
            "dif_root_angvel_local": dif_root_angvel_local * self._config.obs_scales_config.dif_joint_vel,
            "ref_root_height": traj_data.qpos[2],
            "ref_feet_height": ref_feet_height,
        }

        state = jp.hstack([state_dict[k] for k in self._config.obs_keys])
        privileged_state = jp.hstack([privileged_state_dict[k] for k in self._config.privileged_obs_keys])
        current_history = jp.hstack([state_dict[k] for k in self._config.history_keys])

        # Nan to 0
        state = jp.nan_to_num(state)
        privileged_state = jp.nan_to_num(privileged_state)
        current_history = jp.nan_to_num(current_history)

        return {"state": state, "privileged_state": privileged_state}, current_history

    def _get_reward(
        self,
        data: mjx.Data,
        traj_data: TrajectoryData,
        action: jax.Array,
        motor_targets: jax.Array,
        torque: jax.Array,
        info: dict[str, Any],
    ) -> dict[str, jax.Array]:

        traj_root_rot_mat = math.quat_to_mat(traj_data.qpos[3:7])
        root_rot_mat = math.quat_to_mat(data.qpos[3:7])

        dif_rigid_body_pos = gmth.calculate_dif_rigid_body_pos_local(data, traj_data)
        dif_rigid_body_rot = gmth.calculate_dif_rigid_body_rot_local(data, traj_data)
        dif_rigid_body_linvel = jax.lax.cond(self.recalculate_vel_in_reward, lambda _: gmth.calculate_dif_rigid_body_linvel_local_differential(data, traj_data, info['last_rigid_body_pos'], self.dt), lambda _: gmth.calculate_dif_rigid_body_linvel_local(data, traj_data), operand=None)
        dif_rigid_body_angvel = jax.lax.cond(self.recalculate_vel_in_reward, lambda _: gmth.calculate_dif_rigid_body_angvel_local_differential(data, traj_data, info['last_rigid_body_ori'], self.dt), lambda _: gmth.calculate_dif_rigid_body_angvel_local(data, traj_data), operand=None)
        dif_root_linvel = jax.lax.cond(self.recalculate_vel_in_reward, lambda _: gmth.calculate_dif_root_linvel_local_differential(data, traj_data, info['last_root_pos'], self.dt), lambda _: traj_root_rot_mat.T @ traj_data.qvel[:3] - root_rot_mat.T @ data.qvel[:3], operand=None)
        dif_root_angvel = jax.lax.cond(self.recalculate_vel_in_reward, lambda _: gmth.calculate_dif_root_angvel_local_differential(data, traj_data, info['last_root_ori'], self.dt), lambda _: traj_data.qvel[3:6] - data.qvel[3:6], operand=None)
        
        dif_joint_pos = traj_data.qpos[7:] - data.qpos[7:]
        dif_joint_vel = jax.lax.cond(self.recalculate_vel_in_reward, lambda _: traj_data.qvel[6:] - ((data.qpos[7:] - info['last_dof_pos']) / self.dt), lambda _: traj_data.qvel[6:] - data.qvel[6:], operand=None)

        dif_root_height = traj_data.qpos[2] - data.qpos[2]
        dif_feet_height = traj_data.site_xpos[self._feet_all_site_id, 2] - data.site_xpos[self._feet_all_site_id, 2]
        dif_torso_rp = gmth.calculate_dif_torso_rp(data, traj_data)

        termination = self._get_termination(data, traj_data, info)

        reward_dict = {
            # tracking reward
            "rigid_body_pos_tracking_upper": self._reward_rigid_body_pos_tracking_upper(dif_rigid_body_pos),
            "rigid_body_pos_tracking_lower": self._reward_rigid_body_pos_tracking_lower(dif_rigid_body_pos),
            "rigid_body_rot_tracking": self._reward_rigid_body_rot_tracking(dif_rigid_body_rot),
            "rigid_body_linvel_tracking": self._reward_rigid_body_linvel_tracking(dif_rigid_body_linvel),
            "rigid_body_angvel_tracking": self._reward_rigid_body_angvel_tracking(dif_rigid_body_angvel),
            "feet_pos_tracking": self._reward_feet_pos_tracking(dif_rigid_body_pos),
            "feet_rot_tracking": self._reward_feet_rot_tracking(dif_rigid_body_rot),
            "joint_pos_tracking": self._reward_joint_pos_tracking(dif_joint_pos),
            "joint_vel_tracking": self._reward_joint_vel_tracking(dif_joint_vel),
            "root_linvel_tracking": self._reward_root_linvel_tracking(dif_root_linvel),
            "root_angvel_tracking": self._reward_root_angvel_tracking(dif_root_angvel),
            "roll_pitch_tracking": self._reward_roll_pitch_tracking(dif_torso_rp),
            # penalty reward
            "penalty_torque": self._reward_penalty_torque(torque),
            "penalty_action_rate": self._reward_penalty_action_rate(motor_targets, info["last_motor_targets"]),
            "dof_pos_limit": self._reward_dof_pos_limit(data.qpos[7:]),
            "dof_vel_limit": self._reward_dof_vel_limit(data.qvel[6:]),
            "collision": self._reward_collision(data),
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

    def _reward_feet_rot_tracking(self, dif_rigid_body_rot: jax.Array) -> jax.Array:
        diff_feet_rot_dist = 2 * jp.arccos(dif_rigid_body_rot[self.feet_ids, 0]).mean(axis=-1)
        rew = jp.exp(-diff_feet_rot_dist / self._config.reward_config.auxiliary.feet_rot_sigma)
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

    def _reward_collision(self, data: mjx.Data) -> jax.Array:
        pair_geom1 = self.penalize_collision_pair[:, 0]
        pair_geom2 = self.penalize_collision_pair[:, 1]

        collided_values = jax.vmap(partial(geoms_colliding, data))(pair_geom1, pair_geom2)

        return jp.sum(collided_values, axis=-1)

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
