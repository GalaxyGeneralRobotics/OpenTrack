import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


def calculate_dif_rigid_body_pos_local(current_data: mujoco.MjData, trajectory_data: mujoco.MjData):
    """
    Calculate the difference in rigid body positions between the current data and the reference data.
    """
    current_root_pos_g = current_data.qpos[:3]
    current_root_quat_g = current_data.qpos[3:7]
    current_root_rot_g = R.from_quat(current_root_quat_g[[1, 2, 3, 0]])

    ref_root_pos_g = trajectory_data.qpos[:3]
    ref_root_quat_g = trajectory_data.qpos[3:7]
    ref_root_rot_g = R.from_quat(ref_root_quat_g[[1, 2, 3, 0]])

    current_xpos_g = current_data.xpos
    ref_xpos_g = trajectory_data.xpos

    current_xpos_translated = current_xpos_g - current_root_pos_g
    current_xpos_l = current_root_rot_g.inv().apply(current_xpos_translated)

    ref_xpos_translated = ref_xpos_g - ref_root_pos_g
    ref_xpos_l = ref_root_rot_g.inv().apply(ref_xpos_translated)

    dif_rigid_body_pos_local = ref_xpos_l - current_xpos_l

    return dif_rigid_body_pos_local


def linvel_from_pos_diff(pos_curr: np.ndarray, pos_last: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute linear velocity from position difference.
    
    Args:
        pos_curr: Current position, shape (3,)
        pos_last: Last position, shape (3,)
        dt: Time step
        
    Returns:
        Linear velocity in world frame, shape (3,)
    """
    return (pos_curr - pos_last) / dt


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """
    Convert a unit quaternion (scalar-first: w, x, y, z) to a 3x3 rotation matrix.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions (scalar-first convention: w, x, y, z).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_inv(q: np.ndarray) -> np.ndarray:
    """
    Inverse of a quaternion (scalar-first convention: w, x, y, z).
    Assumes unit quaternion.
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotate_vector_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate vector v by quaternion q (scalar-first convention).
    """
    q_v = np.array([0.0, v[0], v[1], v[2]])
    q_conj = quat_inv(q)
    rotated = quat_mul(quat_mul(q, q_v), q_conj)
    return rotated[1:4]


def angvel_from_quat_diff(
    quat_curr: np.ndarray, 
    quat_last: np.ndarray, 
    dt: float,
    frame: str = "world"
) -> np.ndarray:
    """
    Compute angular velocity from quaternion difference using angle-axis extraction.
    
    Quaternion convention: scalar-first (w, x, y, z).
    
    Args:
        quat_curr: Current quaternion, shape (4,), scalar-first
        quat_last: Last quaternion, shape (4,), scalar-first
        dt: Time step
        frame: One of:
            - "world": angular velocity expressed in world frame
            - "local_last_frame": angular velocity expressed in last frame's body coordinate
            - "local_current_frame": angular velocity expressed in current frame's body coordinate
               
    Returns:
        Angular velocity, shape (3,)
    """
    if frame == "world":
        # delta_q represents rotation from last to current in world frame
        delta_q = quat_mul(quat_curr, quat_inv(quat_last))
    elif frame == "local_last_frame":
        # delta_q represents rotation from last to current in last frame's body coordinate
        delta_q = quat_mul(quat_inv(quat_last), quat_curr)
    elif frame == "local_current_frame":
        # First compute in world frame, then transform to current body frame
        delta_q = quat_mul(quat_curr, quat_inv(quat_last))
    else:
        raise ValueError(f"frame must be 'world', 'local_last_frame', or 'local_current_frame', got {frame}")
    
    w, x, y, z = delta_q[0], delta_q[1], delta_q[2], delta_q[3]
    
    # Extract angle from quaternion: angle = 2 * arccos(|w|)
    angle = 2.0 * np.arccos(np.clip(np.abs(w), 0.0, 1.0))
    
    # Extract axis (handle numerical stability)
    axis = np.array([x, y, z])
    axis_norm = np.linalg.norm(axis) + 1e-9
    axis = axis / axis_norm
    
    # Angular velocity = axis * angle / dt
    angvel = axis * angle / dt
    
    # Handle sign of quaternion (if w < 0, the rotation is > 180 deg, flip direction)
    if w < 0:
        angvel = -angvel
    
    # For local_current_frame, rotate the world-frame angular velocity to current body frame
    if frame == "local_current_frame":
        angvel = rotate_vector_by_quat(angvel, quat_inv(quat_curr))
    
    return angvel