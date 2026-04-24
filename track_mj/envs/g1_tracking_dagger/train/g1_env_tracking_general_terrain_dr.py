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
from track_mj.envs.g1_tracking_dagger.train import g1_env_tracking_general, g1_env_tracking_general_dr
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


def g1_tracking_general_terrain_dr_task_config() -> config_dict.ConfigDict:

    env_config = config_dict.create(
        terrain_type="rough_terrain",
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
        randomization_fn=domain_randomize_terrain,
        
        policy_args=None,
    )

    config = config_dict.create(
        env_config=env_config,
        policy_config=policy_config,
    )
    return config

tmj.registry.register("G1TrackingGeneralTerrainDR", "tracking_dagger_config")(g1_tracking_general_terrain_dr_task_config())

@tmj.registry.register("G1TrackingGeneralTerrainDR", "tracking_dagger_train_env_class")
class G1TrackingGeneralTerrainDREnv(g1_env_tracking_general_dr.G1TrackingGeneralDREnv):
    pass