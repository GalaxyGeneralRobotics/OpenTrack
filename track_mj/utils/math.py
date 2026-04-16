import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R
from mujoco.mjx._src import math


def rotate_obs(state, angle, idx_rot, idx_xvel, idx_yvel):
    """
    Function to rotate a state (or set of states) around the rotation axis.

    Args:
        state (list or np.array): Single state or multiple states to be rotated.
        angle (float): Angle of rotation in radians.
        idx_rot (int): Index of rotation angle entry in the state.
        idx_xvel (int): Index of x-velocity entry in the state.
        idx_yvel (int): Index of y-velocity entry in the state.

    Returns:
        np.array of rotated states.

    """

    state = np.array(state)
    rotated_state = state.copy()

    # add rotation to trunk rotation and transform to range [-pi, pi]
    rotated_state[idx_rot] = (state[idx_rot] + angle + np.pi) % (2 * np.pi) - np.pi
    # rotate x,y velocity
    rotated_state[idx_xvel] = np.cos(angle) * state[idx_xvel] - np.sin(angle) * state[idx_yvel]
    rotated_state[idx_yvel] = np.sin(angle) * state[idx_xvel] + np.cos(angle) * state[idx_yvel]

    return rotated_state


def transform_angle_2pi(angle):
    """
    Transforms an angle to be in [-pi, pi].

    Args:
        angle (float): Angle in radians.

    Returns:
        Angle in radians in [-pi, pi].

    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def calc_rel_positions(xpos, x_pos_main_body, backend):
    """
    Calculate the relative positions of the bodies in b_ids to the main body.

    Args:
        xpos (array): Data array containing the positions of the bodies.
        x_pos_main_body (array): Position of the main body.
        backend: Backend to use (either np or jnp).

    Returns:
        Array of relative positions.

    """

    rpos = xpos - x_pos_main_body

    return rpos


def calculate_relative_velocities(xvel, xvel_main_body, backend):
    """
    Calculate the relative velocity of two 6D velocities in world frame.

    Args:
        xvel (array): Data array containing the velocities of the bodies.
        xvel_main_body (array): Velocity of the main body.
        backend: Backend to use (either np or jnp).

    Returns:
        array: Relative 6D velocity vector. Shape (6,)
    """
    # Calculate the relative velocity
    relative_velocity = xvel - xvel_main_body  # Shape: (6,)

    return relative_velocity


def calc_rel_quaternions(xquat, xquat_main_body, backend):
    """
    Calculate the relative quaternions of the bodies in b_ids to the main body.

    Args:
        xquat (array): Data array containing the quaternions of the bodies (quaternions is expected to be scalar last).
        xquat_main_body (array): Quaternion of the main body (quaternions is expected to be scalar last).
        backend: Backend to use (either np or jnp).
    Returns:
        Array of relative quaternions, where the quaternions are scalar last.

    """
    if backend == np:
        R = np_R
    else:
        R = jnp_R

    rquat = (R.from_quat(xquat_main_body).inv() * R.from_quat(xquat)).as_quat()

    return rquat


def calculate_relative_rotation_matrices(main_rot, other_rots, backend):
    """
    Calculate the relative rotation matrices of N bodies with respect to a main rotation matrix.

    Args:
        main_rot (array): Rotation matrix of the main body in world frame. Shape (3, 3).
        other_rots (array): Rotation matrices of other bodies in world frame. Shape (N, 3, 3) where N is the number of other bodies.
        backend: Backend to use (either np or jnp).

    Returns:
        array: Relative rotation matrices of other bodies with respect to the main body. Shape (N, 3, 3).
    """
    # Ensure main_rot is a 2D array and other_rots is at least 3D
    main_rot = backend.atleast_2d(main_rot)  # Shape: (3, 3)
    other_rots = atleast_3d(other_rots, backend)  # Shape: (N, 3, 3)

    # Use the transpose of the main rotation matrix as the inverse
    main_rot_inv = main_rot.T  # Shape: (3, 3)

    # Calculate relative rotation matrices
    relative_rots = backend.einsum('ik,nkj->nij', main_rot_inv, other_rots)  # Shape: (N, 3, 3)

    return relative_rots


def calculate_global_rotation_matrices(main_rot, relative_rots, backend):
    """
    Calculate the global rotation matrices of N bodies in the world frame given a main rotation matrix.

    Args:
        main_rot (array): Rotation matrix of the main body in world frame. Shape (3, 3).
        relative_rots (array): Relative rotation matrices of other bodies with respect to the main body. Shape (N, 3, 3).
        backend: Backend to use (either np or jnp).

    Returns:
        array: Global rotation matrices of other bodies in the world frame. Shape (N, 3, 3).
    """
    # Ensure main_rot is a 2D array and relative_rots is at least 3D
    main_rot = backend.atleast_2d(main_rot)  # Shape: (3, 3)
    relative_rots = atleast_3d(relative_rots, backend)  # Shape: (N, 3, 3)

    # Calculate global rotation matrices
    global_rots = backend.einsum('ij,njk->nik', main_rot, relative_rots)  # Shape: (N, 3, 3)

    return global_rots


def calculate_relative_velocity_in_local_frame(vel_a, vel_b, rot_mat_w_a, rot_mat_a_b, backend):
    """
    Calculate the relative velocity vel_a-vel_b expressed in the local frame of vel_a.

    Args:
        vel_a (array): 6D velocity vector of body A in world frame. Shape (6,) or (batch_size, 6)
        vel_b (array): 6D velocity vector of body B in world frame. Shape (6,) or (batch_size, 6)
        rot_mat_w_a (array): Rotation matrix to transform from world frame to the frame of vel_a.
            Shape (3, 3) or (batch_size, 3, 3)
        rot_mat_a_b (array): Rotation matrix to transform from the frame of vel_a to the frame of vel_b.
            Shape (3, 3) or (batch_size, 3, 3)
        backend: Backend to use (either np or jnp).

    Returns:
        array: Relative 6D velocity vector of vel_a in the local frame of vel_b. Shape (batch_size, 6)
    """
    # Ensure the inputs are at least 2D arrays for batch processing
    vel_a = backend.atleast_1d(vel_a)  # Shape: (6,)
    vel_b = backend.atleast_2d(vel_b)  # Shape: (1, 6) or (batch_size, 6)
    rot_mat_w_a = backend.atleast_2d(rot_mat_w_a)  # Shape: (3, 3)
    rot_mat_a_b = atleast_3d(rot_mat_a_b, backend)  # Shape: (1, 3, 3) or (batch_size, 3, 3)

    # Extract linear and angular components (corrected based on your input structure)
    ang_a = vel_a[:3]  # Angular components of A (Shape: (3,))
    lin_a = vel_a[3:]  # Linear components of A (Shape: (3,))

    ang_b = vel_b[:, :3]  # Angular components of B (Shape: (batch_size, 3))
    lin_b = vel_b[:, 3:]  # Linear components of B (Shape: (batch_size, 3))

    # Transform the relativ linear velocity of vel_a-vel_b to the local frame of vel_a
    relative_lin_vel = backend.einsum('jk,ik->ij', rot_mat_w_a, lin_a - lin_b)  # Shape: (batch_size, 3)

    # Transform the angular velocity of vel_b to the local frame of vel_a (einsum is taking the transpose of rot_mat_a_b)
    ang_b_transformed_to_a = backend.einsum('ikj,ik->ij', rot_mat_a_b, ang_b)  # Shape: (batch_size, 3)

    # Calculate the relative velocities in the local frame of vel_b
    relative_ang_vel = ang_b_transformed_to_a - ang_a  # Shape: (batch_size, 3)

    # Combine into a 6D vector
    relative_velocity = backend.hstack([relative_ang_vel, relative_lin_vel])  # Shape: (batch_size, 6)

    return relative_velocity


def calc_rel_body_velocities(cvel, xmat_main_body, backend):
    """
    Calculate the relative velocities of the bodies in b_ids to the main body.

    Args:
        cvel (array): Data array containing the velocities of the bodies.
        xmat_main_body (array): Rotation matrix of the main body.
        backend: Backend to use (either np or jnp).

    Returns:
        Array of relative velocities.

    """

    rot_mat_main_body = xmat_main_body.reshape(3, 3)
    lin_vel = backend.einsum('ij,nj->ni', rot_mat_main_body, cvel[:, :3])
    rot_vel = backend.einsum('ij,nj->ni', rot_mat_main_body, cvel[:, 3:])
    rvel = backend.concatenate([lin_vel, rot_vel], axis=1)

    return rvel


def calc_site_velocities(site_ids, data, parent_body_id, root_body_id, backend, flg_local=False):
    """
    Calculate the velocities of a batch of sites in world frame.
    This function is implemented similarly to Mujoco's mj_objectVelocity function
    https://github.com/google-deepmind/mujoco/blob/1f9dca8bc4cfbfc23f68c6bda7cdb6abebfdeb98/src/engine/engine_support.c#L1294

    Args:
        site_ids (array): Site ids for which to calculate the velocities. Shape (batch_size,)
        data: Mujoco/Mjx Data structure.
        parent_body_id (array): Parent body ids.
        root_body_id (array): Root body ids.
        backend: Backend to use (either np or jnp).
        flg_local (bool, optional): Whether to return velocities in local coordinates. Defaults to False.

    Returns:
        array: Global 6D velocity vectors of the sites. Shape (batch_size, 6)

    """

    site_xpos = data.site_xpos[site_ids]
    site_xmat = data.site_xmat[site_ids].reshape(*site_ids.shape, 3, 3)
    parent_body_cvel = data.cvel[parent_body_id]
    root_subtree_com = data.subtree_com[root_body_id]

    return transform_motion(parent_body_cvel, site_xpos, root_subtree_com, site_xmat, backend, flg_local)


def calc_body_velocities(body_ids, data, root_body_id, backend, flg_local=False):
    """
    Calculate the velocities of a batch of bodies in world frame.
    This function is implemented similarly to Mujoco's mj_objectVelocity function
    https://github.com/google-deepmind/mujoco/blob/1f9dca8bc4cfbfc23f68c6bda7cdb6abebfdeb98/src/engine/engine_support.c#L1294

    Args:
        body_ids (array): Body ids for which to calculate the velocities. Shape (batch_size,)
        data: Mujoco/Mjx Data structure.
        root_body_id (array): Root body ids.
        backend: Backend to use (either np or jnp).
        flg_local (bool, optional): Whether to return velocities in local coordinates. Defaults to False.

    Returns:
        array: Global 6D velocity vectors of the bodies. Shape (batch_size, 6)

    """

    body_xpos = data.xpos[body_ids]
    body_xmat = data.xmat[body_ids].reshape(*body_ids.shape, 3, 3)
    root_subtree_com = data.subtree_com[root_body_id]
    body_cvel = data.cvel[body_ids]

    return transform_motion(body_cvel, body_xpos, root_subtree_com, body_xmat, backend, flg_local)


def transform_motion(vel, new_pos, old_pos, rot_mat_new2old, backend, flg_local=True):
    """
    Transforms a motion vector from one frame to another. This function is implemented similarly to Mujoco's
    mju_transformSpatial function, but is limited to velocities.
    (https://github.com/google-deepmind/mujoco/blob/1f9dca8bc4cfbfc23f68c6bda7cdb6abebfdeb98/src/engine/engine_util_spatial.c#L454).


    Args:
        vel (array): Motion vector to be transformed. Shape (6,) or (batch_size, 6)
        new_pos (array): Position of the new frame. Shape (3,) or (batch_size, 3)
        old_pos (array): Position of the old frame. Shape (3,) or (batch_size, 3)
        rot_mat_new2old (array): Rotation matrix from the new to the old frame. Shape (3, 3) or (batch_size, 3, 3)
        backend: Backend to use (either np or jnp).

    Returns:
        array: Transformed motion vector. Shape (batch_size, 6)

    """

    # Use atleast_2d to handle both single and batched inputs
    vel = backend.atleast_2d(vel)  # Shape: (batch_size, 6)
    new_pos = backend.atleast_2d(new_pos)  # Shape: (batch_size, 3)
    old_pos = backend.atleast_2d(old_pos)  # Shape: (batch_size, 3)
    assert rot_mat_new2old.shape[-2:] == (3, 3)
    rot_mat_new2old = atleast_3d(rot_mat_new2old, backend)  # Shape: (batch_size, 3, 3)

    # Step 1: Compute the relative position of the new frame w.r.t. the old frame
    lin_vel = vel[:, 3:]  # Shape: (batch_size, 3)
    rot_vel = vel[:, :3]  # Shape: (batch_size, 3)
    rpos = calc_rel_positions(new_pos, old_pos, backend)  # Shape: (batch_size, 3)

    # Step 2: Compute the linear velocity at the site
    lin_vel = lin_vel - backend.cross(rpos, rot_vel, axis=-1)
    if flg_local:
        lin_vel = backend.einsum('bij,bj->bi', rot_mat_new2old.transpose(0, 2, 1), lin_vel)

    # Step 3: Compute the angular velocity at the site
    if flg_local:
        rot_vel = backend.einsum('bij,bj->bi', rot_mat_new2old.transpose(0, 2, 1), rot_vel)
    else:
        rot_vel = rot_vel

    # Step 4: Combine linear and angular velocities into a 6D velocity vector
    xvel = backend.hstack([rot_vel, lin_vel])  # Shape: (batch_size, 6)

    return xvel


def calculate_relative_site_quatities(data, rel_site_ids, rel_body_ids, body_rootid, backend):

    if backend == np:
        R = np_R
    else:
        R = jnp_R

    # get site positions and rotations
    site_xpos_traj = data.site_xpos
    site_xmat_traj = data.site_xmat
    site_xpos_traj = site_xpos_traj[rel_site_ids]
    site_xmat_traj = site_xmat_traj[rel_site_ids]

    # get relevant properties and calculate site velocities
    main_site_id = 0  # --> zeroth index in rel_site_ids
    site_root_body_id = body_rootid[rel_body_ids]
    site_xvel = calc_site_velocities(rel_site_ids, data, rel_body_ids, site_root_body_id, backend)
    main_site_xvel = site_xvel[main_site_id]
    site_xvel = backend.delete(site_xvel, main_site_id, axis=0)

    # calculate the rotation matrix from main site to the other sites
    main_site_xmat_traj = site_xmat_traj[main_site_id].reshape(3, 3)
    site_xmat_traj = backend.delete(site_xmat_traj, main_site_id, axis=0).reshape(-1, 3, 3)
    rel_rot_mat = calculate_relative_rotation_matrices(main_site_xmat_traj, site_xmat_traj, backend)

    # calculate relative quantities
    main_site_xpos_traj = site_xpos_traj[main_site_id]
    site_xpos_traj = backend.delete(site_xpos_traj, main_site_id, axis=0)
    site_rpos = calc_rel_positions(site_xpos_traj, main_site_xpos_traj, backend)
    site_rangles = R.from_matrix(rel_rot_mat).as_rotvec()
    site_rvel = calculate_relative_velocity_in_local_frame(main_site_xvel, site_xvel, main_site_xmat_traj,
                                                           rel_rot_mat, backend)

    return site_rpos, site_rangles, site_rvel


def quaternion_angular_distance(q1, q2, backend):
    """
    Calculate the angular distance between two rotations represented by quaternions.

    Args:
        q1 (array): First quaternion. Shape (4,) or (batch_size, 4). (quaternions is expected to be scalar last)
        q2 (array): Second quaternion. Shape (4,) or (batch_size, 4). (quaternions is expected to be scalar last)

    Returns:
        array: Angular distance between the two rotations. Shape (batch_size,)
    """

    if backend == np:
        R = np_R
    else:
        R = jnp_R

    # Create Rotation objects for both quaternions
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)

    # Compute the relative rotation
    relative_rotation = r1.inv() * r2

    # Extract the angle of the relative rotation
    angular_distance = relative_rotation.magnitude()  # Returns the angle in radians
    return angular_distance


def quat2angle(quat, backend):
    """
    Converts a quaternion to an angle. (quaternions is expected to be scalar last)
    """
    if backend == np:
        R = np_R
    else:
        R = jnp_R
    return R.from_quat(quat).as_rotvec()

def angle2quat(angle, backend, scalar_first=False):
    """
    Converts an angle to a quaternion. (quaternions is expected to be scalar last)
    """
    if backend == np:
        R = np_R
    else:
        R = jnp_R
    return R.from_rotvec(angle).as_quat(scalar_first=scalar_first)


def quat_scalarfirst2scalarlast(quat):
    """
    Converts a quaternion from scalar-first to scalar-last representation.
    """
    return quat[..., [1, 2, 3, 0]]


def quat_scalarlast2scalarfirst(quat):
    """
    Converts a quaternion from scalar-last to scalar-first representation.
    """
    return quat[..., [3, 0, 1, 2]]


def atleast_3d(tensor, backend):
    """
    Ensures the tensor has at least 3 dimensions by adding axes at the front if necessary.

    Args:
        tensor: Input tensor (numpy or JAX array).
        backend: The numerical backend, either numpy or jax.numpy.

    Returns:
        A tensor with at least 3 dimensions.
    """
    while tensor.ndim < 3:
        tensor = backend.expand_dims(tensor, axis=0)
    
    return tensor


def base2navi_transform(base2world, backend=jnp):
    x = base2world[:, 0]
    x_proj = x.at[2].set(0.0)
    x_proj /= backend.linalg.norm(x_proj)
    z_axis = backend.array([0.0, 0.0, 1.0])
    y_axis = backend.cross(z_axis, x_proj)
    y_axis /= backend.linalg.norm(y_axis)
    x_axis = backend.cross(y_axis, z_axis)
    return backend.column_stack((x_axis, y_axis, z_axis))


def calculate_dif_rigid_body_pos_local(current_data, reference_data):
    current_root_pos_g = current_data.qpos[:3]
    current_root_quat_g = current_data.qpos[3:7]
    current_root_quat_g_inv = math.quat_inv(current_root_quat_g)

    ref_root_pos_g = reference_data.qpos[:3]
    ref_root_quat_g = reference_data.qpos[3:7]
    ref_root_quat_g_inv = math.quat_inv(ref_root_quat_g)

    current_xpos_g = current_data.xpos
    ref_xpos_g = reference_data.xpos

    current_xpos_translated = current_xpos_g - current_root_pos_g
    current_xpos_l = jax.vmap(math.rotate, in_axes=(0, None))(
        current_xpos_translated, current_root_quat_g_inv
    )

    ref_xpos_translated = ref_xpos_g - ref_root_pos_g
    ref_xpos_l = jax.vmap(math.rotate, in_axes=(0, None))(
        ref_xpos_translated, ref_root_quat_g_inv
    )

    dif_rigid_body_pos_local = ref_xpos_l - current_xpos_l

    return dif_rigid_body_pos_local

def calculate_dif_rigid_body_pos_actor_root(current_data, reference_data):
    current_root_pos_g = current_data.qpos[:3]
    current_root_quat_g = current_data.qpos[3:7]
    current_root_quat_g_inv = math.quat_inv(current_root_quat_g)

    ref_root_pos_g = reference_data.qpos[:3]
    ref_root_quat_g = reference_data.qpos[3:7]
    ref_root_quat_g_inv = math.quat_inv(ref_root_quat_g)

    current_xpos_g = current_data.xpos
    ref_xpos_g = reference_data.xpos

    current_xpos_translated = current_xpos_g - current_root_pos_g
    current_xpos_l = jax.vmap(math.rotate, in_axes=(0, None))(
        current_xpos_translated, current_root_quat_g_inv
    )

    ref_xpos_translated = ref_xpos_g - current_root_pos_g
    ref_xpos_actor_root = jax.vmap(math.rotate, in_axes=(0, None))(
        ref_xpos_translated, current_root_quat_g_inv
    )

    dif_rigid_body_pos_actor_root = ref_xpos_actor_root - current_xpos_l

    return dif_rigid_body_pos_actor_root

def calculate_dif_rigid_body_rot_local(current_data, reference_data):
    current_root_quat_g = current_data.qpos[3:7]
    current_root_quat_g_inv = math.quat_inv(current_root_quat_g)
    current_xquat_g = current_data.xquat
    current_body_rot_local = jax.vmap(lambda q_body, q_root_inv: math.quat_mul(q_root_inv, q_body), in_axes=(0, None))(
        current_xquat_g, current_root_quat_g_inv
    )

    ref_root_quat_g = reference_data.qpos[3:7]
    ref_root_quat_g_inv = math.quat_inv(ref_root_quat_g)
    ref_xquat_g = reference_data.xquat
    ref_body_rot_local = jax.vmap(lambda q_body, q_root_inv: math.quat_mul(q_root_inv, q_body), in_axes=(0, None))(
        ref_xquat_g, ref_root_quat_g_inv
    )

    dif_rigid_body_rot_local = jax.vmap(lambda q_ref_local, q_curr_local: math.quat_mul(q_ref_local, math.quat_inv(q_curr_local)), in_axes=(0, 0))(
        ref_body_rot_local, current_body_rot_local
    )

    return dif_rigid_body_rot_local

def calculate_dif_rigid_body_rot_actor_root(current_data, reference_data):
    current_root_quat_g = current_data.qpos[3:7]
    current_root_quat_g_inv = math.quat_inv(current_root_quat_g)
    current_xquat_g = current_data.xquat
    current_body_rot_local = jax.vmap(lambda q_body, q_root_inv: math.quat_mul(q_root_inv, q_body), in_axes=(0, None))(
        current_xquat_g, current_root_quat_g_inv
    )

    ref_root_quat_g = reference_data.qpos[3:7]
    ref_root_quat_g_inv = math.quat_inv(ref_root_quat_g)
    ref_xquat_g = reference_data.xquat
    ref_body_rot_actor_root = jax.vmap(lambda q_body, q_root_inv: math.quat_mul(q_root_inv, q_body), in_axes=(0, None))(
        ref_xquat_g, current_root_quat_g_inv
    )

    dif_rigid_body_rot_actor_root = jax.vmap(lambda q_ref_local, q_curr_local: math.quat_mul(q_ref_local, math.quat_inv(q_curr_local)), in_axes=(0, 0))(
        ref_body_rot_actor_root, current_body_rot_local
    )

    return dif_rigid_body_rot_actor_root

def calculate_dif_rigid_body_linvel_local(current_data, reference_data):
    current_root_quat_g = current_data.qpos[3:7]
    current_root_quat_g_inv = math.quat_inv(current_root_quat_g)
    current_linear_vel_g = current_data.cvel[:, 3:]

    ref_root_quat_g = reference_data.qpos[3:7]
    ref_root_quat_g_inv = math.quat_inv(ref_root_quat_g)
    ref_linear_vel_g = reference_data.cvel[:, 3:]

    current_linear_vel_l = jax.vmap(math.rotate, in_axes=(0, None))(
        current_linear_vel_g, current_root_quat_g_inv
    )
    ref_linear_vel_l = jax.vmap(math.rotate, in_axes=(0, None))(
        ref_linear_vel_g, ref_root_quat_g_inv
    )

    dif_local_body_lin_vel = ref_linear_vel_l - current_linear_vel_l

    return dif_local_body_lin_vel

def calculate_dif_rigid_body_linvel_actor_root(current_data, reference_data):
    current_root_quat_g = current_data.qpos[3:7]
    current_root_quat_g_inv = math.quat_inv(current_root_quat_g)
    current_linear_vel_g = current_data.cvel[:, 3:]

    ref_root_quat_g = reference_data.qpos[3:7]
    ref_root_quat_g_inv = math.quat_inv(ref_root_quat_g)
    ref_linear_vel_g = reference_data.cvel[:, 3:]

    current_linear_vel_l = jax.vmap(math.rotate, in_axes=(0, None))(
        current_linear_vel_g, current_root_quat_g_inv
    )
    ref_linear_vel_actor_root = jax.vmap(math.rotate, in_axes=(0, None))(
        ref_linear_vel_g, current_root_quat_g_inv
    )

    dif_actor_root_body_lin_vel = ref_linear_vel_actor_root - current_linear_vel_l

    return dif_actor_root_body_lin_vel

def calculate_dif_rigid_body_linvel_local_differential(current_data, reference_data, last_pos, dt):
    current_root_quat_g = current_data.qpos[3:7]
    current_root_quat_g_inv = math.quat_inv(current_root_quat_g)
    current_linear_vel_g = (current_data.xpos - last_pos) / dt

    ref_root_quat_g = reference_data.qpos[3:7]
    ref_root_quat_g_inv = math.quat_inv(ref_root_quat_g)
    ref_linear_vel_g = reference_data.cvel[:, 3:]

    current_linear_vel_l = jax.vmap(math.rotate, in_axes=(0, None))(
        current_linear_vel_g, current_root_quat_g_inv
    )
    ref_linear_vel_l = jax.vmap(math.rotate, in_axes=(0, None))(
        ref_linear_vel_g, ref_root_quat_g_inv
    )

    dif_local_body_lin_vel = ref_linear_vel_l - current_linear_vel_l

    return dif_local_body_lin_vel

def calculate_dif_rigid_body_linvel_actor_root_differential(current_data, reference_data, last_pos, dt):
    current_root_quat_g = current_data.qpos[3:7]
    current_root_quat_g_inv = math.quat_inv(current_root_quat_g)
    current_linear_vel_g = (current_data.xpos - last_pos) / dt

    ref_root_quat_g = reference_data.qpos[3:7]
    ref_root_quat_g_inv = math.quat_inv(ref_root_quat_g)
    ref_linear_vel_g = reference_data.cvel[:, 3:]

    current_linear_vel_l = jax.vmap(math.rotate, in_axes=(0, None))(
        current_linear_vel_g, current_root_quat_g_inv
    )
    ref_linear_vel_actor_root = jax.vmap(math.rotate, in_axes=(0, None))(
        ref_linear_vel_g, current_root_quat_g_inv
    )

    dif_actor_root_body_lin_vel = ref_linear_vel_actor_root - current_linear_vel_l

    return dif_actor_root_body_lin_vel

def calculate_dif_root_linvel_local_differential(current_data, reference_data, last_root_pos, dt):
    current_root_quat_g = current_data.qpos[3:7]
    current_root_quat_g_inv = math.quat_inv(current_root_quat_g)
    current_root_linear_vel_g = (current_data.qpos[:3] - last_root_pos) / dt

    ref_root_quat_g = reference_data.qpos[3:7]
    ref_root_quat_g_inv = math.quat_inv(ref_root_quat_g)
    ref_root_linear_vel_g = reference_data.qvel[:3]

    current_root_linear_vel_l = math.rotate(
        current_root_linear_vel_g, current_root_quat_g_inv
    )
    ref_root_linear_vel_l = math.rotate(
        ref_root_linear_vel_g, ref_root_quat_g_inv
    )

    dif_local_root_lin_vel = ref_root_linear_vel_l - current_root_linear_vel_l

    return dif_local_root_lin_vel

def calculate_dif_root_linvel_actor_root_differential(current_data, reference_data, last_root_pos, dt):
    current_root_quat_g = current_data.qpos[3:7]
    current_root_quat_g_inv = math.quat_inv(current_root_quat_g)
    current_root_linear_vel_g = (current_data.qpos[:3] - last_root_pos) / dt

    ref_root_quat_g = reference_data.qpos[3:7]
    ref_root_quat_g_inv = math.quat_inv(ref_root_quat_g)
    ref_root_linear_vel_g = reference_data.qvel[:3]

    current_root_linear_vel_l = math.rotate(
        current_root_linear_vel_g, current_root_quat_g_inv
    )
    ref_root_linear_vel_actor_root = math.rotate(
        ref_root_linear_vel_g, current_root_quat_g_inv
    )

    dif_actor_root_frame_root_lin_vel = ref_root_linear_vel_actor_root - current_root_linear_vel_l

    return dif_actor_root_frame_root_lin_vel

def calculate_dif_rigid_body_angvel_local(current_data, reference_data):
    current_root_quat_g = current_data.qpos[3:7]
    current_root_quat_g_inv = math.quat_inv(current_root_quat_g)
    current_angular_vel_g = current_data.cvel[:, :3]

    ref_root_quat_g = reference_data.qpos[3:7]
    ref_root_quat_g_inv = math.quat_inv(ref_root_quat_g)
    ref_angular_vel_g = reference_data.cvel[:, :3]

    current_angular_vel_l = jax.vmap(math.rotate, in_axes=(0, None))(
        current_angular_vel_g, current_root_quat_g_inv
    )
    ref_angular_vel_l = jax.vmap(math.rotate, in_axes=(0, None))(
        ref_angular_vel_g, ref_root_quat_g_inv
    )

    dif_local_body_ang_vel = ref_angular_vel_l - current_angular_vel_l

    return dif_local_body_ang_vel

def calculate_dif_rigid_body_angvel_actor_root(current_data, reference_data):
    current_root_quat_g = current_data.qpos[3:7]
    current_root_quat_g_inv = math.quat_inv(current_root_quat_g)
    current_angular_vel_g = current_data.cvel[:, :3]

    ref_root_quat_g = reference_data.qpos[3:7]
    ref_root_quat_g_inv = math.quat_inv(ref_root_quat_g)
    ref_angular_vel_g = reference_data.cvel[:, :3]

    current_angular_vel_l = jax.vmap(math.rotate, in_axes=(0, None))(
        current_angular_vel_g, current_root_quat_g_inv
    )
    ref_angular_vel_actor_root = jax.vmap(math.rotate, in_axes=(0, None))(
        ref_angular_vel_g, current_root_quat_g_inv
    )

    dif_actor_root_frame_body_ang_vel = ref_angular_vel_actor_root - current_angular_vel_l

    return dif_actor_root_frame_body_ang_vel

def calculate_dif_rigid_body_angvel_local_differential(current_data, reference_data, last_ori, dt):
    current_root_quat_g = current_data.qpos[3:7]
    current_root_quat_g_inv = math.quat_inv(current_root_quat_g)

    ref_root_quat_g = reference_data.qpos[3:7]
    ref_root_quat_g_inv = math.quat_inv(ref_root_quat_g)

    def quat_to_angvel(current_q, last_q):
        # delta_q = last_q_inv * current_q
        delta_q = math.quat_mul(math.quat_inv(last_q), current_q)
        w, x, y, z = delta_q
        angle = 2.0 * jnp.arccos(jnp.clip(w, -1.0, 1.0))
        axis = jnp.array([x, y, z])
        axis = axis / (jnp.linalg.norm(axis) + 1e-9)
        angvel = axis * angle / dt
        return angvel

    current_angular_vel_g = jax.vmap(quat_to_angvel)(current_data.xquat, last_ori)
    ref_angular_vel_g = reference_data.cvel[:, :3]

    current_angular_vel_l = jax.vmap(math.rotate, in_axes=(0, None))(
        current_angular_vel_g, current_root_quat_g_inv
    )
    ref_angular_vel_l = jax.vmap(math.rotate, in_axes=(0, None))(
        ref_angular_vel_g, ref_root_quat_g_inv
    )

    dif_local_body_ang_vel = ref_angular_vel_l - current_angular_vel_l
    return dif_local_body_ang_vel

def calculate_dif_rigid_body_angvel_actor_root_differential(current_data, reference_data, last_ori, dt):
    current_root_quat_g = current_data.qpos[3:7]
    current_root_quat_g_inv = math.quat_inv(current_root_quat_g)

    ref_root_quat_g = reference_data.qpos[3:7]
    ref_root_quat_g_inv = math.quat_inv(ref_root_quat_g)

    def quat_to_angvel(current_q, last_q):
        # delta_q = last_q_inv * current_q
        delta_q = math.quat_mul(math.quat_inv(last_q), current_q)
        w, x, y, z = delta_q
        angle = 2.0 * jnp.arccos(jnp.clip(w, -1.0, 1.0))
        axis = jnp.array([x, y, z])
        axis = axis / (jnp.linalg.norm(axis) + 1e-9)
        angvel = axis * angle / dt
        return angvel

    current_angular_vel_g = jax.vmap(quat_to_angvel)(current_data.xquat, last_ori)
    ref_angular_vel_g = reference_data.cvel[:, :3]

    current_angular_vel_l = jax.vmap(math.rotate, in_axes=(0, None))(
        current_angular_vel_g, current_root_quat_g_inv
    )
    ref_angular_vel_actor_root = jax.vmap(math.rotate, in_axes=(0, None))(
        ref_angular_vel_g, current_root_quat_g_inv
    )

    dif_actor_root_body_ang_vel = ref_angular_vel_actor_root - current_angular_vel_l
    return dif_actor_root_body_ang_vel

def calculate_dif_root_angvel_local_differential(current_data, reference_data, last_root_ori, dt):
    current_root_quat_g = current_data.qpos[3:7]
    current_root_quat_g_inv = math.quat_inv(current_root_quat_g)

    def quat_to_angvel(current_q, last_q):
        # delta_q = last_q_inv * current_q
        delta_q = math.quat_mul(math.quat_inv(last_q), current_q)
        w, x, y, z = delta_q
        angle = 2.0 * jnp.arccos(jnp.clip(w, -1.0, 1.0))
        axis = jnp.array([x, y, z])
        axis = axis / (jnp.linalg.norm(axis) + 1e-9)
        angvel = axis * angle / dt
        return angvel

    current_root_angular_vel_g = quat_to_angvel(current_data.qpos[3:7], last_root_ori)
    current_root_angular_vel_l = math.rotate(
        current_root_angular_vel_g, current_root_quat_g_inv
    )

    ref_root_angular_vel_l = reference_data.qvel[3:6]

    dif_local_root_ang_vel = ref_root_angular_vel_l - current_root_angular_vel_l
    return dif_local_root_ang_vel

def calculate_dif_root_angvel_actor_root_differential(current_data, reference_data, last_root_ori, dt):
    current_root_quat_g = current_data.qpos[3:7]
    current_root_quat_g_inv = math.quat_inv(current_root_quat_g)

    ref_root_quat_g = reference_data.qpos[3:7]
    ref_root_quat_g_inv = math.quat_inv(ref_root_quat_g)

    def quat_to_angvel(current_q, last_q):
        # delta_q = last_q_inv * current_q
        delta_q = math.quat_mul(math.quat_inv(last_q), current_q)
        w, x, y, z = delta_q
        angle = 2.0 * jnp.arccos(jnp.clip(w, -1.0, 1.0))
        axis = jnp.array([x, y, z])
        axis = axis / (jnp.linalg.norm(axis) + 1e-9)
        angvel = axis * angle / dt
        return angvel

    current_root_angular_vel_g = quat_to_angvel(current_data.qpos[3:7], last_root_ori)
    current_root_angular_vel_l = math.rotate(
        current_root_angular_vel_g, current_root_quat_g_inv
    )

    ref_root_angular_vel_l = reference_data.qvel[3:6]       # NOTE: qvel of root is w.r.t. present root orientation!
    ref_root_angular_vel_g = math.rotate(
        ref_root_angular_vel_l, ref_root_quat_g
    )
    ref_root_angular_vel_actor_root = math.rotate(
        ref_root_angular_vel_g, current_root_quat_g_inv
    )


    dif_local_root_ang_vel = ref_root_angular_vel_actor_root - current_root_angular_vel_l
    return dif_local_root_ang_vel


def calculate_dif_joint_pos(current_data, reference_data):
    current_joint_pos = current_data.qpos[7:]
    ref_joint_pos = reference_data.qpos[7:]

    dif_joint_pos = ref_joint_pos - current_joint_pos

    return dif_joint_pos


def calculate_dif_joint_vel(current_data, reference_data):
    current_joint_velocities = current_data.qvel[6:]
    ref_joint_velocities = reference_data.qvel[6:]

    dif_joint_velocities = ref_joint_velocities - current_joint_velocities

    return dif_joint_velocities


def calculate_dif_root_height(current_data, reference_data):
    current_root_height = current_data.qpos[2]
    ref_root_height = reference_data.qpos[2]

    dif_root_height = ref_root_height - current_root_height

    return dif_root_height


def calculate_dif_navi_torso_rp(current_data, reference_data):
    return calculate_dif_torso_rp(current_data, reference_data)


def calculate_dif_torso_rp(current_data, reference_data):
    current_pelvis_rot = current_data.site_xmat[0].reshape(3, 3)  # _pelvis_imu_site_id
    current_torso_rpy = jnp_R.from_matrix(current_pelvis_rot).as_euler("XYZ")

    ref_pelvis_rot = reference_data.site_xmat[0].reshape(3, 3)  # _pelvis_imu_site_id
    ref_torso_rpy = jnp_R.from_matrix(ref_pelvis_rot).as_euler("XYZ")

    dif_torso_rpy = ref_torso_rpy - current_torso_rpy

    return dif_torso_rpy[:2]

def quat_flip(quat: jnp.ndarray) -> jnp.ndarray:
    """
    Flip quaternion to ensure w >= 0 (keep sign consistency).
    Works for shape (4,) or (B,4).
    """
    if quat.ndim == 1:
        sign = jnp.sign(quat[0])
        sign = jnp.where(sign == 0, 1.0, sign)  # avoid w=0 sign=0
        return quat * sign
    elif quat.ndim == 2:
        sign = jnp.sign(quat[:, 0])
        sign = jnp.where(sign == 0, 1.0, sign)
        return quat * sign[:, None]
    else:
        raise ValueError(f"quat must have shape (4,) or (B,4), got {quat.shape}")


# ==================== Velocity from Position Difference ====================

def linvel_from_pos_diff(pos_curr: jnp.ndarray, pos_last: jnp.ndarray, dt: float) -> jnp.ndarray:
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


def angvel_from_quat_diff(
    quat_curr: jnp.ndarray, 
    quat_last: jnp.ndarray, 
    dt: float,
    frame: str = "world"
) -> jnp.ndarray:
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
        
    Notes:
        - "world": delta_q = q_curr * q_last^{-1}, result is omega in world frame.
        - "local_last_frame": delta_q = q_last^{-1} * q_curr, result is omega in last frame's body frame.
        - "local_current_frame": compute omega_world first, then rotate to current body frame.
    """
    if frame == "world":
        # delta_q represents rotation from last to current in world frame
        delta_q = math.quat_mul(quat_curr, math.quat_inv(quat_last))
    elif frame == "local_last_frame":
        # delta_q represents rotation from last to current in last frame's body coordinate
        delta_q = math.quat_mul(math.quat_inv(quat_last), quat_curr)
    elif frame == "local_current_frame":
        # First compute in world frame, then transform to current body frame
        delta_q = math.quat_mul(quat_curr, math.quat_inv(quat_last))
    else:
        raise ValueError(f"frame must be 'world', 'local_last_frame', or 'local_current_frame', got {frame}")
    
    w, x, y, z = delta_q[0], delta_q[1], delta_q[2], delta_q[3]
    
    # Extract angle from quaternion: angle = 2 * arccos(|w|)
    angle = 2.0 * jnp.arccos(jnp.clip(jnp.abs(w), 0.0, 1.0))
    
    # Extract axis (handle numerical stability)
    axis = jnp.array([x, y, z])
    axis_norm = jnp.linalg.norm(axis) + 1e-9
    axis = axis / axis_norm
    
    # Angular velocity = axis * angle / dt
    angvel = axis * angle / dt
    
    # Handle sign of quaternion (if w < 0, the rotation is > 180 deg, flip direction)
    angvel = jnp.where(w < 0, -angvel, angvel)
    
    # For local_current_frame, rotate the world-frame angular velocity to current body frame
    if frame == "local_current_frame":
        angvel = math.rotate(angvel, math.quat_inv(quat_curr))
    
    return angvel