from copy import deepcopy
from dataclasses import replace
import jax.numpy as jnp
import mujoco
import numpy as np
from types import ModuleType

from track_mj.utils.dataset.traj_class import (
    Trajectory,
    TrajectoryInfo,
    TrajectoryModel,
    TrajectoryData,
    SingleData)
from track_mj.utils.dataset.traj_handler import TrajCarry
from track_mj.utils.dataset.start_end_transition_handler import StartEndTransitionHandler
from track_mj.envs.g1_tracking import g1_tracking_constants as consts

class ReplayCallback:

    """Base class that can be used to do things while replaying a trajectory."""

    @staticmethod
    def __call__(env, model, data, traj_sample, carry):
        data = env.set_sim_state_from_traj_data(data, traj_sample, carry)
        mujoco.mj_forward(model, data)
        carry = env.th.update_state(carry)
        return model, data, carry


class ExtendTrajData(ReplayCallback):

    def __init__(self, env, n_samples, model, body_names=None, site_names=None):
        self.b_names, self.b_ids = self.get_body_names_and_ids(env._mj_model, body_names)
        self.s_names, self.s_ids = self.get_site_names_and_ids(env._mj_model, site_names)
        dim_qpos, dim_qvel = 0, 0
        for i in range(model.njnt):
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                dim_qpos += 7
                dim_qvel += 6
            else:
                dim_qpos += 1
                dim_qvel += 1
        self.recorder = dict(xpos=np.zeros((n_samples, model.nbody, 3)),
                             xquat=np.zeros((n_samples, model.nbody, 4)),
                             cvel=np.zeros((n_samples, model.nbody, 6)),
                             subtree_com=np.zeros((n_samples, model.nbody, 3)),
                             site_xpos=np.zeros((n_samples, model.nsite, 3)),
                             site_xmat=np.zeros((n_samples, model.nsite, 9)),
                             qpos=np.zeros((n_samples, dim_qpos)),
                             qvel=np.zeros((n_samples, dim_qvel)))
        self.traj_model = TrajectoryModel(njnt=model.njnt,
                                          jnt_type=jnp.array(model.jnt_type),
                                          nbody=model.nbody,
                                          body_rootid=jnp.array(model.body_rootid),
                                          body_weldid=jnp.array(model.body_weldid),
                                          body_mocapid=jnp.array(model.body_mocapid),
                                          body_pos=jnp.array(model.body_pos),
                                          body_quat=jnp.array(model.body_quat),
                                          body_ipos=jnp.array(model.body_ipos),
                                          body_iquat=jnp.array(model.body_iquat),
                                          nsite=model.nsite,
                                          site_bodyid=jnp.array(model.site_bodyid),
                                          site_pos=jnp.array(model.site_pos),
                                          site_quat=jnp.array(model.site_quat))
        self.current_length = 0

    def __call__(self, env, model: mujoco.MjModel, data: mujoco.MjData, traj_sample: SingleData, carry: TrajCarry):
        r"""
        Args:
            env: exp env
            traj_sample: get by `traj_handler.get_current_traj_data(traj_carry, np)`, will call Trajectory.data.get(...)
        """
        model, data, carry = super().__call__(env, model, data, traj_sample, carry)

        self.recorder["xpos"][self.current_length] = data.xpos[self.b_ids]
        self.recorder["xquat"][self.current_length] = data.xquat[self.b_ids]
        self.recorder["cvel"][self.current_length] = data.cvel[self.b_ids]
        self.recorder["subtree_com"][self.current_length] = data.subtree_com[self.b_ids]
        self.recorder["site_xpos"][self.current_length] = data.site_xpos[self.s_ids]
        self.recorder["site_xmat"][self.current_length] = data.site_xmat[self.s_ids]

        # add joint properties
        self.recorder["qpos"][self.current_length] = data.qpos
        self.recorder["qvel"][self.current_length] = data.qvel

        self.current_length += 1

        return model, data, carry

    def extend_trajectory_data(self, traj_data: TrajectoryData, traj_info: TrajectoryInfo):
        assert self.current_length == traj_data.qpos.shape[0]
        assert traj_info.model.njnt == self.traj_model.njnt
        converted_data = {}
        for key, value in self.recorder.items():
            converted_data[key] = jnp.array(value)
        return (traj_data.replace(**converted_data),
                replace(traj_info, body_names=self.b_names if len(self.b_names) > 0 else None,
                        site_names=self.s_names if len(self.s_names) > 0 else None,
                        model=self.traj_model))

    @staticmethod
    def get_body_names_and_ids(model, keys=None):
        """
        Get the names of the bodies in the model. If keys is not None, only return the names of the bodies
        that are in keys, otherwise return all body names.

        Args:
            model: mujoco model
            keys: list of body names

        Returns:
            List of body names and list of body ids.
        """
        keys = deepcopy(keys)
        body_names = []
        ids = range(model.nbody)
        for i in ids:
            b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if (keys is not None) and (b_name in keys):
                keys.remove(b_name)
            body_names.append(b_name)
        assert keys is None or len(keys) == 0, f"Could not find the following body names: {keys}"
        return body_names, list(ids)

    @staticmethod
    def get_site_names_and_ids(model, keys=None):
        """
        Get the names of the sites in the model. If keys is not None, only return the names of the sites
        that are in keys, otherwise return all site names.

        Args:
            model: mujoco model
            keys: list of site names

        Returns:
            List of site names and list of site ids.
        """
        keys = deepcopy(keys)
        site_names = []
        ids = range(model.nsite)
        for i in ids:
            s_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            if (keys is not None) and (s_name in keys):
                keys.remove(s_name)
            site_names.append(s_name)
        assert keys is None or len(keys) == 0, f"Could not find the following site names: {keys}"
        return site_names, list(ids)

class SmoothStartEndTransition:

    def __init__(self, model: mujoco.MjModel, traj: Trajectory):
        self.mj_model = model
        self.traj = traj
        self._default_qpos = np.array(consts.DEFAULT_QPOS)
        
    def run_interp(
            self,
            default_pose_sec = 1.0,         # sec to maintain default pose at start
            single_step_th = 0.1,           # threshold to step > 1 step
            additional_transition_sec_start = 0.3,     # interp sec between last frame of IK and first frame of mocap
            start_foot_id = 1,              # 0: left foot, 1: right foot
            start_foot_h = 0.05,            # feet height during transition 1 (start)
            start_com_off = 0.03,           # CoM offset during transition 1 (start)
            start_step_sec = 0.3,           # single step sec during transition 1 (start)
            return_backend: ModuleType = np,
    ):
        default_pose_len = int(default_pose_sec * self.traj.info.frequency)
        additional_transition_len_start = int(additional_transition_sec_start * self.traj.info.frequency)
        start_step_len = int(start_step_sec * self.traj.info.frequency)

        transition_handler = StartEndTransitionHandler(
            ori_traj=self.traj, model=self.mj_model, default_qpos=self._default_qpos, 
            transition_len_start=additional_transition_len_start
        )
        # transition 1 (start)
        if(transition_handler.compute_step_distance() > single_step_th):
            num_steps = 2
        else:
            num_steps = 1

        transition_handler.add_start_transition(
            num_steps=num_steps,
            start_foot_id=start_foot_id,
            foot_h=start_foot_h,
            com_off=start_com_off,
            single_step_len=start_step_len,
            double_step_len=3,
            default_pose_len=default_pose_len,
        )

        qpos_traj = transition_handler.result
        qpos_traj = return_backend.array(qpos_traj)

        ret_traj = Trajectory(
            info = self.traj.info,
            data = TrajectoryData(
                qpos = qpos_traj,
                qvel = return_backend.zeros((qpos_traj.shape[0], self.traj.data.qvel.shape[-1])),
                split_points = return_backend.array([0, qpos_traj.shape[0]]),
                ),
            transitions = self.traj.transitions if self.traj.transitions is not None else None,
        )

        return ret_traj