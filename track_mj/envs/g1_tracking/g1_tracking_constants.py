# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constants for G1."""

from pathlib import Path
import numpy as np
from track_mj.constant import PATH_STORAGE, PATH_ASSET, WANDB_PATH_LOG

ROOT_PATH = PATH_ASSET / "unitree_g1"

FLAT_TERRAIN_XML = ROOT_PATH / "scene_mjx_flat_terrain.xml"
ROUGH_TERRAIN_XML = ROOT_PATH / "scene_mjx_rough_terrain.xml"

NUM_JOINT = 29

def task_to_xml(task_name: str) -> Path:
    xml_map = {
        "flat_terrain": FLAT_TERRAIN_XML,
        "rough_terrain": ROUGH_TERRAIN_XML,
    }
    if task_name not in xml_map:
        raise ValueError(f"Unsupported task_name {task_name}.")
    return xml_map[task_name]


FEET_SITES = [
    "left_foot",
    "right_foot",
]

LEFT_FEET_SITE = "left_foot"

RIGHT_FEET_SITE = "right_foot"

FEET_ALL_SITES = [
    "left_foot",
    "right_foot",
    "left_foot_top",
    "right_foot_top",
]

LEFT_FEET_ALL_SITES = [
    "left_foot",
    "left_foot_top",
]

RIGHT_FEET_ALL_SITES = [
    "right_foot",
    "right_foot_top",
]

HAND_SITES = [
    "left_palm",
    "right_palm",
]

LEFT_FEET_GEOMS = ["left_foot"]
RIGHT_FEET_GEOMS = ["right_foot"]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

ROOT_BODY = "torso_link"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

RESTRICTED_JOINT_RANGE = (
    # Left leg.
    (-2.5307, 2.8798),
    (-0.5236, 2.9671),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    # Right leg.
    (-2.5307, 2.8798),
    (-0.5236, 2.9671),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    # Waist.
    (-2.618, 2.618),
    (-0.52, 0.52),
    (-0.52, 0.52),
    # Left shoulder.
    (-3.0892, 2.6704),
    (-1.5882, 2.2515),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
    (-1.61443, 1.61443),
    (-1.61443, 1.61443),
    # Right shoulder.
    (-3.0892, 2.6704),
    (-2.2515, 1.5882),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
    (-1.61443, 1.61443),
    (-1.61443, 1.61443),
)

DOF_VEL_LIMITS = [
    32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
    32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
    32.0, 37.0, 37.0,
    37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0,
    37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0
]

ACTION_JOINT_NAMES = [
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # -------------- tracking only --------------
    # waist
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
]

OBS_JOINT_NAMES = [
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
]

# fmt: off
TORQUE_LIMIT = np.array([
    88., 139., 88., 139., 50., 50.,
    88., 139., 88., 139., 50., 50.,
    88., 50., 50.,
    25., 25., 25., 25., 25., 5., 5.,
    25., 25., 25., 25., 25., 5., 5.,
])

DEFAULT_QPOS = np.float32([
    0, 0, 0.79,
    1, 0, 0, 0,
    -0.1, 0, 0, 0.3, -0.2, 0,
    -0.1, 0, 0, 0.3, -0.2, 0,
    0, 0, 0,
    0.2, 0.3, 0, 1.28, 0, 0, 0,
    0.2, -0.3, 0, 1.28, 0, 0, 0,
])

# v2
KPs = np.float32([
    100, 100, 100, 200, 80, 20,
    100, 100, 100, 200, 80, 20,
    300, 300, 300,
    90, 60, 20, 60, 20, 20, 20,
    90, 60, 20, 60, 20, 20, 20,
])

KDs = np.float32([
    2, 2, 2, 4, 2, 1,
    2, 2, 2, 4, 2, 1,
    10, 10, 10,
    2, 2, 1, 1, 1, 1, 1,
    2, 2, 1, 1, 1, 1, 1,
])
# fmt: on

UPPER_BODY_LINKs = [
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

LOWER_BODY_LINKs = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link"
]

UPPER_BODY_JOINTs = [
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
]

KEY_BODY_LINKs = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "torso_link"
]

FEET_LINKs = [
    "left_ankle_roll_link",
    "right_ankle_roll_link"
]

SHOULDER_LINKs = [
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link"
]

LAFAN1_DATASETS = [
    'dance1_subject1',
    'dance1_subject2',
    'dance1_subject3',
    'dance2_subject1',
    'dance2_subject2',
    'dance2_subject3',
    'dance2_subject4',
    'dance2_subject5',
    'fallAndGetUp1_subject1',
    'fallAndGetUp1_subject4',
    'fallAndGetUp1_subject5',
    'fallAndGetUp2_subject2',
    'fallAndGetUp2_subject3',
    'fallAndGetUp3_subject1',
    'fight1_subject2',
    'fight1_subject3',
    'fight1_subject5',
    'fightAndSports1_subject1',
    'fightAndSports1_subject4',
    'jumps1_subject1',
    'jumps1_subject2',
    'jumps1_subject5',
    'run1_subject2',
    'run1_subject5',
    'run2_subject1',
    'run2_subject4',
    'sprint1_subject2',
    'sprint1_subject4',
    'walk1_subject1',
    'walk1_subject2',
    'walk1_subject5',
    'walk2_subject1',
    'walk2_subject3',
    'walk2_subject4',
    'walk3_subject1',
    'walk3_subject2',
    'walk3_subject3',
    'walk3_subject4',
    'walk3_subject5',
    'walk4_subject1',
]

LAFAN1_SPECIALIST_DATASETS_1 = [
    'dance1_subject1',
    'dance1_subject2',
    'dance1_subject3',
    'dance2_subject1',
    'dance2_subject2',
    'dance2_subject3',
    'dance2_subject4',
    'dance2_subject5',
]

LAFAN1_SPECIALIST_DATASETS_2 = [
    'fallAndGetUp1_subject1',
    'fallAndGetUp1_subject4',
    'fallAndGetUp1_subject5',
    'fallAndGetUp2_subject2',
    'fallAndGetUp2_subject3',
    'fallAndGetUp3_subject1',
]


TEST = [
    # "Random_001_Tennis 001",
    "Random_001_Tennis 001 no"
]