import argparse
import os

import numpy as np
import jax
import jax.numpy as jnp
import time

import mujoco
import mujoco.viewer

from track_mj.utils.dataset.traj_class import Trajectory

import osqp
import scipy.sparse as sp
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from track_mj.envs.g1_tracking import g1_tracking_constants as consts


# ##################### Quaternion Utilities #####################


def quat_conjugate(q):
    """Return quaternion conjugate."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_to_yaw(q):
    """Extract yaw (rotation around z-axis) from quaternion."""
    w, x, y, z = q
    # Yaw from quaternion (Z-up convention)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return yaw

def quat_from_yaw(yaw):
    """Quaternion representing rotation around z-axis by yaw."""
    return np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])

def quat_mul(q1, q2):
    # (w,x,y,z) * (w,x,y,z)
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rotation_error_under_small_angle_A2B(R_target, R_current):
    """
    Calculate the rotation error between two rotation matrices (from target to current!!!) under small angle approximation.

    Mathematical Principle:
        The rotation vector (axis-angle representation) can be extracted from the skew-symmetric
        part of (R_err - I), which under small angles simplifies to:
        
        r_err ≈ 0.5 * vex(R_err - R_err^T)
        
        where vex() is the inverse operation of the skew-symmetric matrix operator, extracting
        the 3D vector from a skew-symmetric matrix.

    Args:
        R_target (np.ndarray): Target rotation matrix of shape (3, 3) in SO(3).
                              Represents the desired orientation.
        R_current (np.ndarray): Current rotation matrix of shape (3, 3) in SO(3).
                               Represents the actual orientation.

    Returns:
        np.ndarray: Rotation error vector of shape (3,) in axis-angle representation.
                   The direction indicates the rotation axis, and the magnitude (norm)
                   indicates the rotation angle in radians. This represents the minimal
                   rotation needed to align R_current with R_target.

    Note:
        This approximation is valid only for small rotation errors (typically < 15-30 degrees).
        For larger errors, use the full logarithmic map instead.
    """
    R_err = R_target.T @ R_current
    r_err = 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1]
    ])
    return r_err

def slerp_rotation_matrix(R1, R2, alpha):
    r"""
    Spherical linear interpolation between two rotation matrices.
    """
    rot1 = R.from_matrix(R1)
    rot2 = R.from_matrix(R2)

    key_times = [0, 1]
    key_rots = R.from_quat([rot1.as_quat(), rot2.as_quat()])
    slerp = Slerp(key_times, key_rots)
    
    return slerp(alpha).as_matrix()

def interp_qpos(q_start: np.ndarray, q_end: np.ndarray, alpha: float) -> np.ndarray:
    """
    Interpolate between two qpos configurations, handling yaw wrapping.

    Args:
        q_start (np.ndarray): Starting qpos (n,).
        q_end (np.ndarray): Ending qpos (n,).
        alpha (float): Interpolation factor [0, 1].

    Returns:
        np.ndarray: Interpolated qpos (n,).
    """
    q_interp = np.zeros_like(q_start)
    # interpolate base position linearly
    q_interp[0:3] = (1 - alpha) * q_start[0:3] + alpha * q_end[0:3]
    # interpolate base quat with slerp
    quat_start_xyzw = np.roll(q_start[3:7], -1)  # wxyz -> xyzw
    quat_end_xyzw = np.roll(q_end[3:7], -1)      # wxyz -> xyzw
    key_times = [0, 1]
    key_rots = R.from_quat([quat_start_xyzw, quat_end_xyzw])
    slerp = Slerp(key_times, key_rots)
    quat_interp_xyzw = slerp(alpha).as_quat()
    q_interp[3:7] = np.roll(quat_interp_xyzw, 1)  # xyzw -> wxyz
    # interpolate joint angles linearly
    q_interp[7:] = (1 - alpha) * q_start[7:] + alpha * q_end[7:]

    return q_interp
    

# ##################### IK Essentials #####################


class StartEndTransitionParams:

    # damping factor, to control the IK step size
    lam = 1e-3

    # q_pos weight, to control the importance of qpos in IK optimization
    # a larger value will make the joint angles change as little as possible while completing the IK action
    W_pos = np.array([
        1., 1., 1., 2.5, 1., 1.,    # base
        1., 0., 1., 0., 0., 0.,     # left leg
        1., 0., 1., 0., 0., 0.,     # right leg
        0., 0., 0.,
        1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1.,
    ])

    # qpos weight scaling factor, too large will cause IK not to converge
    W_pos_scale = 0.5          

    # end-effector weight for IK
    # right foot position / left foot position / right foot rotation / left foot rotation / base position
    W_ee = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    # integration time step for IK
    IK_dt = 1.0

def IK_swing_foot(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, 
                  fixed_foot_pos, fixed_foot_rot, 
                  swing_foot_pos, swing_foot_rot, 
                  swing_foot_pos_end, swing_foot_rot_end, 
                  q_ref, swing_foot_id=1, traj_len=20, foot_h=0.05, com_off=0.03):

    left_foot_site_id = mj_model.site(consts.LEFT_FEET_SITE).id
    right_foot_site_id = mj_model.site(consts.RIGHT_FEET_SITE).id

    Jp_r = np.zeros((3, mj_model.nv))
    Jp_l = np.zeros((3, mj_model.nv))
    Jr_r = np.zeros((3, mj_model.nv))
    Jr_l = np.zeros((3, mj_model.nv))
    Jcom = np.zeros((2, mj_model.nv))
    Jcom[:, :2] = np.eye(2)

    nv = mj_model.nv
    Id = sp.eye(nv, format='csc')

    q_cur = q_ref.copy()
    qpos_traj = []

    solver = osqp.OSQP()
    solver_initialized = False
    
    # QP params
    lam = StartEndTransitionParams.lam
    w_rf, w_lf, w_rrot, w_lrot, w_base = StartEndTransitionParams.W_ee
    dq_bound = 10.0
    W_qpos = StartEndTransitionParams.W_pos
    W_qpos_sp = sp.diags(W_qpos, format="csc")

    for t in range(0, traj_len + 1):
        if swing_foot_id == 1:
            alpha = t / traj_len
            alpha_smooth = 0.5 * (1 - np.cos(np.pi * alpha))
            xy_des = swing_foot_pos[0:2] + (swing_foot_pos_end[0:2] - swing_foot_pos[0:2]) * alpha_smooth
            
            s = t / traj_len
            if s <= 0.5:
                # swing up
                s_up = s / 0.5
                f = 10 * s_up ** 3 - 15 * s_up ** 4 + 6 * s_up ** 5
                z_des = swing_foot_pos[2] + foot_h * f
            else:
                # swing down
                s_down = (s - 0.5) / 0.5
                f = 1 - (10 * s_down ** 3 - 15 * s_down ** 4 + 6 * s_down ** 5)
                z_des = swing_foot_pos[2] + foot_h * f

            rf_pos_target = np.array([xy_des[0], xy_des[1], z_des])
            lf_pos_target = fixed_foot_pos
            rf_rot_target = slerp_rotation_matrix(swing_foot_rot, swing_foot_rot_end, alpha)
            lf_rot_target = fixed_foot_rot
        else:

            alpha = t / traj_len
            alpha_smooth = 0.5 * (1 - np.cos(np.pi * alpha))
            xy_des = swing_foot_pos[0:2] + (swing_foot_pos_end[0:2] - swing_foot_pos[0:2]) * alpha_smooth
            
            s = t / traj_len
            if s <= 0.5:
                # swing up
                s_up = s / 0.5
                f = 10 * s_up ** 3 - 15 * s_up ** 4 + 6 * s_up ** 5
                z_des = swing_foot_pos[2] + foot_h * f
            else:
                # swing down
                s_down = (s - 0.5) / 0.5
                f = 1 - (10 * s_down ** 3 - 15 * s_down ** 4 + 6 * s_down ** 5)
                z_des = swing_foot_pos[2] + foot_h * f

            lf_pos_target = np.array([xy_des[0], xy_des[1], z_des])
            rf_pos_target = fixed_foot_pos
            lf_rot_target = slerp_rotation_matrix(swing_foot_rot, swing_foot_rot_end, alpha)
            rf_rot_target = fixed_foot_rot

        
        com = .5 * (xy_des + fixed_foot_pos[0:2]) + np.array([com_off, 0.0])

        # IK iteration
        prev_err_norm = float('inf')
        for ik_iter in range(1500):
            # forward
            mj_data.qpos[:] = q_cur
            mujoco.mj_forward(mj_model, mj_data)

            rf_pos_curr = mj_data.site_xpos[right_foot_site_id].astype(np.float64)
            lf_pos_curr = mj_data.site_xpos[left_foot_site_id].astype(np.float64)
            rf_rot_curr = mj_data.site_xmat[right_foot_site_id].reshape(3, 3).astype(np.float64)
            lf_rot_curr = mj_data.site_xmat[left_foot_site_id].reshape(3, 3).astype(np.float64)

            # err
            pos_err_rf = rf_pos_target - rf_pos_curr
            pos_err_lf = lf_pos_target - lf_pos_curr
            rot_err_r = rotation_error_under_small_angle_A2B(rf_rot_curr, rf_rot_target)
            rot_err_l = rotation_error_under_small_angle_A2B(lf_rot_curr, lf_rot_target)
            com_err = com - q_cur[0:2]

            err = np.concatenate([pos_err_rf, pos_err_lf, rot_err_r, rot_err_l, com_err])
            err_norm = np.linalg.norm(err)

            # adaptive stopping criteria
            if err_norm < 1e-3:
                break
            elif ik_iter > 50 and err_norm < 5e-3:
                break
            elif ik_iter > 100 and err_norm < 1e-2:
                break
            elif abs(prev_err_norm - err_norm) < 1e-6:
                break
            
            prev_err_norm = err_norm

            # Jacobian
            mujoco.mj_jac(mj_model, mj_data, Jp_r, Jr_r, 
                         mj_data.site_xpos[right_foot_site_id],
                         mj_model.site_bodyid[right_foot_site_id])
            mujoco.mj_jac(mj_model, mj_data, Jp_l, Jr_l, 
                         mj_data.site_xpos[left_foot_site_id], 
                         mj_model.site_bodyid[left_foot_site_id])

            J = np.vstack([Jp_r, Jp_l, Jr_r, Jr_l, Jcom])
            
            # balance the weights
            W_diag = np.concatenate([
                w_rf * np.ones(3),
                w_lf * np.ones(3), 
                w_rrot * 0.1 * np.ones(3),  # lower rotation weight
                w_lrot * 0.1 * np.ones(3),
                w_base * 0.5 * np.ones(2),  # lower com weight
            ])
            W = sp.diags(W_diag, format='csc')
            J_sp = sp.csc_matrix(J)

            H = (J_sp.T @ W @ J_sp) + lam * Id + StartEndTransitionParams.W_pos_scale * W_qpos_sp
            f = -(J_sp.T @ (W @ err))

            l = -dq_bound * np.ones(nv)
            u = dq_bound * np.ones(nv)

            if not solver_initialized:
                solver.setup(P=H, q=f, A=Id, l=l, u=u, verbose=False, 
                           polish=True,       # improve solution accuracy
                           adaptive_rho=True) # adaptive rho update
                solver_initialized = True
            else:
                try:
                    solver.update(P=H, q=f)
                except:
                    # reset if matrix structure changes
                    solver.setup(P=H, q=f, A=Id, l=l, u=u, verbose=False)

            res = solver.solve()
            
            if res.info.status != 'solved':
                print(f"OSQP status: {res.info.status} at step {t}, iter {ik_iter}")
                break

            dq = res.x

            q_next = q_cur.astype(np.float64).copy()
            mujoco.mj_integratePos(mj_model, q_next, dq, StartEndTransitionParams.IK_dt)
            q_cur = q_next

        # log warning if not fully converged
        if ik_iter >= 1499:
            print(f"IK not fully converged at step {t}, error: {err_norm:.4e}")
        
        qpos_traj.append(q_cur.copy())

    return np.array(qpos_traj)

def IK_foot(
        mj_model: mujoco.MjModel, mj_data: mujoco.MjData,
        q_transition_start, q_transition_end, q_ref, step_ratio=1, swing_foot_id=1, traj_len=20, foot_h=0.05, com_off=0.03
    ):
    r"""
    Aligns the stance foot between start and end configurations, then generates
    an IK trajectory for the swing foot to transition from start to end position.
    
    Args:
        q_transition_start: Initial qpos at the start of transition.
        q_transition_end: Target qpos at the end of transition.
        q_ref: Reference qpos used for IK optimization.
        step_ratio (float, optional): The ratio of this transition step in the overall step (0 to 1). Defaults to 1.
        swing_foot_id (int, optional): ID of the swing foot (1 for right, other for left). Defaults to 1.
        traj_len (int, optional): Length of the generated trajectory. Defaults to 20.
        foot_h (float, optional): Maximum height of the swing foot during transition. Defaults to 0.05.
        com_off (float, optional): Center of mass offset parameter. Defaults to 0.03.
    
    Returns:
        numpy.ndarray: Trajectory of qpos for the foot transition.
    
    Notes:
        - The function first aligns the stance foot position between start and end configurations
            by adjusting the base position (first 2 DOFs) of q_transition_end.
        - Then it calculates desired swing foot position based on step_ratio.
        - Finally calls IK_swing_foot to generate the full trajectory.
    """
    left_foot_site_id = mj_model.site(consts.LEFT_FEET_SITE).id
    right_foot_site_id = mj_model.site(consts.RIGHT_FEET_SITE).id

    # Align stance foot position between start and end by adjusting base position
    mj_data.qpos[:] = q_transition_start
    mujoco.mj_forward(mj_model, mj_data)
    l_foot_pos_before = mj_data.site_xpos[left_foot_site_id].copy()
    r_foot_pos_before = mj_data.site_xpos[right_foot_site_id].copy()

    mj_data.qpos[:] = q_transition_end
    mujoco.mj_forward(mj_model, mj_data)
    l_foot_pos_after = mj_data.site_xpos[left_foot_site_id].copy()
    r_foot_pos_after = mj_data.site_xpos[right_foot_site_id].copy()

    if swing_foot_id == 1:
        q_transition_end[:2] += (l_foot_pos_before[0:2] - l_foot_pos_after[0:2])
    else:
        q_transition_end[:2] += (r_foot_pos_before[0:2] - r_foot_pos_after[0:2])

    # Initial foot position
    mj_data.qpos[:] = q_transition_start.copy()
    mujoco.mj_forward(mj_model, mj_data)
    lf_pos_init = mj_data.site_xpos[left_foot_site_id].copy()
    rf_pos_init = mj_data.site_xpos[right_foot_site_id].copy()

    lf_rot_init = mj_data.site_xmat[left_foot_site_id].copy().reshape(3, 3)
    rf_rot_init = mj_data.site_xmat[right_foot_site_id].copy().reshape(3, 3)

    # End foot position
    mj_data.qpos[:] = q_transition_end.copy()
    mujoco.mj_forward(mj_model, mj_data)
    lf_pos_end = mj_data.site_xpos[left_foot_site_id].copy()
    rf_pos_end = mj_data.site_xpos[right_foot_site_id].copy()

    lf_rot_end = mj_data.site_xmat[left_foot_site_id].copy().reshape(3, 3)
    rf_rot_end = mj_data.site_xmat[right_foot_site_id].copy().reshape(3, 3)

    # Inverse Kinematics to find the transition trajectory
    if swing_foot_id == 1:
        rf_pos_des = rf_pos_init + (rf_pos_end - rf_pos_init) * step_ratio
        # ik_traj = IK_swing_foot(mj_model, mj_data, lf_pos_init, lf_rot_init, rf_pos_init, rf_rot_end, rf_pos_des, rf_rot_end, q_ref, q_transition_end,
        #                         swing_foot_id=swing_foot_id, traj_len=traj_len, foot_h=foot_h, com_off=com_off)     # BUG?
        ik_traj = IK_swing_foot(
            mj_model=mj_model, mj_data=mj_data,
            fixed_foot_pos=lf_pos_init, fixed_foot_rot=lf_rot_init, 
            swing_foot_pos=rf_pos_init, swing_foot_rot=rf_rot_init, swing_foot_pos_end=rf_pos_des, swing_foot_rot_end=rf_rot_end,
            q_ref=q_ref, swing_foot_id=swing_foot_id, traj_len=traj_len, foot_h=foot_h, com_off=com_off
        )
    else:
        lf_pos_des = lf_pos_init + (lf_pos_end - lf_pos_init) * step_ratio
        # ik_traj = IK_swing_foot(mj_model, mj_data, rf_pos_init, rf_rot_init, lf_pos_init, lf_rot_end, lf_pos_des, lf_rot_end, q_ref, q_transition_end,
        #                         swing_foot_id=swing_foot_id, traj_len=traj_len, foot_h=foot_h, com_off=com_off)     # BUG?
        ik_traj = IK_swing_foot(
            mj_model=mj_model, mj_data=mj_data,
            fixed_foot_pos=rf_pos_init, fixed_foot_rot=rf_rot_init,
            swing_foot_pos=lf_pos_init, swing_foot_rot=lf_rot_init, swing_foot_pos_end=lf_pos_des, swing_foot_rot_end=lf_rot_end,
            q_ref=q_ref, swing_foot_id=swing_foot_id, traj_len=traj_len, foot_h=foot_h, com_off=com_off
        )

    return ik_traj


# ##################### Handler #####################


class StartEndTransitionHandler():

    def __init__(self, ori_traj: Trajectory, model: mujoco.MjModel, default_qpos: np.ndarray, transition_len_start: float):

        if isinstance(ori_traj.data.qpos, jax.Array):
            self.ori_qpos_traj = np.array(ori_traj.data.qpos)
        elif isinstance(ori_traj.data.qpos, np.ndarray):
            self.ori_qpos_traj = ori_traj.data.qpos
        else:
            raise TypeError("ori_traj.data.qpos must be jnp.ndarray or np.ndarray")
        if isinstance(default_qpos, jax.Array):
            self.default_qpos = np.array(default_qpos)
        elif isinstance(default_qpos, np.ndarray):
            self.default_qpos = default_qpos
        else:
            raise TypeError("default_qpos must be jnp.ndarray or np.ndarray")
        
        self.transition_len_start = transition_len_start

        self.mj_model = model
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.left_foot_site_id = self.mj_model.site(consts.LEFT_FEET_SITE).id
        self.right_foot_site_id = self.mj_model.site(consts.RIGHT_FEET_SITE).id

        self.upper_body_start_id = min([
            self.mj_model.joint(name).id for name in consts.UPPER_BODY_JOINTs
        ]) - 1 + 7  # 22

        self.q_transition_start: np.ndarray = None
        self.q_transition_end: np.ndarray = None
        self.qpos_traj: np.ndarray = None

        self.result = []

        self.__post_init__()

    def __post_init__(self):
        self.qpos_traj = self.ori_qpos_traj.copy()
        # align qpos trajectory yaw to default_qpos yaw
        q_init = self.ori_qpos_traj[0].copy()
        q_init_quat = q_init[3:7] / np.linalg.norm(q_init[3:7])
        yaw_init = quat_to_yaw(q_init_quat)
        q_yaw_inv = quat_from_yaw(-yaw_init)   # offset

        Rz = np.array([
            [np.cos(-yaw_init), -np.sin(-yaw_init), 0],
            [np.sin(-yaw_init),  np.cos(-yaw_init), 0],
            [0, 0, 1]
        ])
        
        # rotate all qpos in trajectory to face the same direction as default_qpos
        for i in range(len(self.qpos_traj)):
            q = self.qpos_traj[i, 3:7]
            q_new = quat_mul(q_yaw_inv, q)
            q_new /= np.linalg.norm(q_new)
            self.qpos_traj[i, 3:7] = q_new
            p = self.qpos_traj[i, :3]
            self.qpos_traj[i, :3] = Rz @ p
        
        self.q_transition_start = self.default_qpos.copy()
        self.q_transition_end = self.qpos_traj[0].copy()

    def set_transition_len(self, transition_len):
        self.transition_len_start = transition_len

    def compute_step_distance(self):
        self.mj_data.qpos[:] = self.q_transition_start
        mujoco.mj_forward(self.mj_model, self.mj_data)
        l_foot_pos_before_transition = self.mj_data.site_xpos[self.left_foot_site_id].copy()
        r_foot_pos_before_transition = self.mj_data.site_xpos[self.right_foot_site_id].copy()

        self.mj_data.qpos[:] = self.q_transition_end
        mujoco.mj_forward(self.mj_model, self.mj_data)
        l_foot_pos_after_transition = self.mj_data.site_xpos[self.left_foot_site_id].copy()
        r_foot_pos_after_transition = self.mj_data.site_xpos[self.right_foot_site_id].copy()

        step_len_before_transition = np.linalg.norm(l_foot_pos_before_transition[0:2] - r_foot_pos_before_transition[0:2])
        step_len_after_transition = np.linalg.norm(l_foot_pos_after_transition[0:2] - r_foot_pos_after_transition[0:2])
        step_len_diff_before_after_transition = np.abs(step_len_after_transition - step_len_before_transition)
        return step_len_diff_before_after_transition
    
    def add_start_transition(
            self, num_steps,
            start_foot_id = 0, foot_h = 0.05, com_off = 0.03,
            single_step_len = 20, double_step_len = 10.0,
            default_pose_len = 120,
        ):

        for i in range(num_steps):
            step_ratio = (i + 1) / num_steps * 1.0

            if i == 0:
                q_ref = self.q_transition_start
            else:
                q_ref = self.result[-1].copy()

            # [single_step_len + 1, 7 + 29]
            ik_traj = IK_foot(
                mj_model=self.mj_model, mj_data=self.mj_data,
                q_transition_start=self.q_transition_start, q_transition_end=self.q_transition_end, q_ref=q_ref,
                step_ratio=step_ratio, swing_foot_id=start_foot_id, traj_len=single_step_len, foot_h=foot_h, com_off=com_off
            )

            for t in range(ik_traj.shape[0]):
                self.result.append(ik_traj[t])

            # add double stance between steps
            if i < num_steps - 1:
                for t in range(0, double_step_len):
                    self.result.append(ik_traj[-1])

            self.q_transition_start = ik_traj[-1]

            # change swing foot
            if start_foot_id == 0:
                start_foot_id = 1
            else:
                start_foot_id = 0

        q_before = ik_traj[-1]
        self.mj_data.qpos[:] = q_before
        mujoco.mj_forward(self.mj_model, self.mj_data)
        l_foot_pos_before = self.mj_data.site_xpos[self.left_foot_site_id].copy()

        # overwrite upper body during transition via interpolation
        for t in range(0, len(self.result)):
            q = self.result[t].copy()
            q[self.upper_body_start_id:] = self.default_qpos[self.upper_body_start_id:] + (self.qpos_traj[0][self.upper_body_start_id:] - self.default_qpos[self.upper_body_start_id:]) * (t / len(self.result))
            self.result[t] = q
        
        # interpolate to mitigate gap between end of transition and start of main traj
        q_after = self.qpos_traj[0]
        self.mj_data.qpos[:] = q_after
        mujoco.mj_forward(self.mj_model, self.mj_data)
        l_foot_pos_after = self.mj_data.site_xpos[self.left_foot_site_id].copy()

        xy_offset = l_foot_pos_before[0:2] - l_foot_pos_after[0:2]

        q_start = self.result[-1]
        q_end = self.qpos_traj[0].copy()
        q_end[:2] += xy_offset

        # transition_ik_traj = self.generate_smooth_transition(q_start, q_end, int(self.transition_len_start))
        # for q in transition_ik_traj:
        #     self.result.append(q)

        for t in range(0, int(self.transition_len_start)):
            q = interp_qpos(q_start, q_end, t / self.transition_len_start)
            self.result.append(q)

        # apply xy offset to the rest of the main trajectory        
        for t in range(1, len(self.qpos_traj)):
            q = self.qpos_traj[t].copy()
            q[:2] += xy_offset
            self.result.append(q)

        # insert default_pose_len default pose at the beginning
        for t in range(default_pose_len):
            self.result.insert(0, self.default_qpos.copy())

    def generate_smooth_transition(self, q_start, q_end, transition_len):
        traj = []
        
        for t in range(transition_len):
            alpha = t / (transition_len - 1)
            q_interp = np.zeros_like(q_start)
            q_interp[:7] = interp_qpos(q_start[:7], q_end[:7], alpha)
            joint_alpha = self.smooth_alpha(alpha)
            q_interp[7:] = q_start[7:] + (q_end[7:] - q_start[7:]) * joint_alpha
            q_interp = self.kinematic_correction(q_interp, alpha)
            
            traj.append(q_interp)
        
        return traj

    def smooth_alpha(self, alpha):
        return 0.5 * (1 - np.cos(np.pi * alpha))

    def kinematic_correction(self, q, alpha):
        self.mj_data.qpos[:] = q
        mujoco.mj_forward(self.mj_model, self.mj_data)
        l_foot_z = self.mj_data.site_xpos[self.left_foot_site_id][2]
        r_foot_z = self.mj_data.site_xpos[self.right_foot_site_id][2]
        min_foot_z = min(l_foot_z, r_foot_z)
        
        if min_foot_z < 0.02:
            q[2] += (0.02 - min_foot_z) * 0.5
        
        return q
