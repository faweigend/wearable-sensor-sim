import numpy as np
from scipy.spatial.transform import Rotation as R


def batch_skew_symmetric_mat_to_ground_ang_vel(s_mat: np.ndarray) -> np.ndarray:
    """
    Extract angular velocity components from a batch of skew-symmetric matrices.

    This function takes a batch of skew-symmetric matrices and vectorizes the process
    of extracting angular velocity components. Each skew-symmetric matrix corresponds
    to a 3D angular velocity represented by omega_G (ground angular velocity), which is
    computed as the following:
        omega_G = [W[2,1], W[0,2], W[1,0]].

    Parameters:
    s_mat : np.ndarray
        A batch of skew-symmetric matrices, of shape (N, 3, 3), where N is the
        number of matrices in the batch.

    Returns:
    np.ndarray
        An array of shape (N, 3), representing angular velocity components extracted
        from the input batch of skew-symmetric matrices. Each row corresponds to
        omega_G of the corresponding matrix.
    """
    # Vectorized extraction of omega_G from skew matrix
    # omega = [W[2,1], W[0,2], W[1,0]]
    omega_g = np.empty((s_mat.shape[0], 3), dtype=float)
    omega_g[:, 0] = s_mat[:, 2, 1]
    omega_g[:, 1] = s_mat[:, 0, 2]
    omega_g[:, 2] = s_mat[:, 1, 0]
    return omega_g


def compute_angle_from_body_euler_xyz(
        a_xyz: np.ndarray,
        input_in_degrees: bool = True
) -> np.ndarray:
    """
    a_xyz: Nx3 angles in radians
    input_in_degrees: set True if your angles are in degrees (common in exported IK tables)
    """
    a_xyz = np.asarray(a_xyz, dtype=float)

    if input_in_degrees:
        a_xyz = np.deg2rad(a_xyz)

    # the opensim order is frontal (x), transverse (y), sagittal (z)
    # [N,3,3] array of rot mats
    r = R.from_euler('XYZ', a_xyz, degrees=False)
    ypr = r.as_euler('YZX', degrees=False)
    ypr[:, 0] = np.unwrap(ypr[:, 0])
    return ypr


def compute_rotmat_from_body_euler_xyz(
        a_xyz: np.ndarray,
        input_in_degrees: bool = True
) -> np.ndarray:
    """
    Batch of ground->body rotation matrices R_GB from the OpenSim body orientation
    columns (*_Ox_rot, *_Oy_rot, *_Oz_rot).

    This is the same convention compute_angle_from_body_euler_xyz() consumes. The
    opensim order is frontal (x), transverse (y), sagittal (z), body-fixed, so scipy's
    capital 'XYZ' (intrinsic). Kept here as one function so the IMU path and the insole
    path cannot drift apart: compute_angle_from_body_euler_xyz() re-expresses this same
    matrix in 'YZX', and the gyro/accel helpers rebuild it from those angles.

    Parameters:
    a_xyz : np.ndarray
        (N, 3) body orientation angles.
    input_in_degrees : bool
        True if a_xyz is in degrees (what the OpenSim export ships).

    Returns:
    np.ndarray
        (N, 3, 3) rotation matrices R_GB, mapping body-frame vectors to ground.
    """
    a_xyz = np.asarray(a_xyz, dtype=float)
    if input_in_degrees:
        a_xyz = np.deg2rad(a_xyz)
    return R.from_euler('XYZ', a_xyz, degrees=False).as_matrix()


def rotate_ground_to_body(v_g: np.ndarray, r_gb: np.ndarray) -> np.ndarray:
    """
    Express a batch of ground-frame vectors in the body frame: v_B = R_GB^T v_G.

    Parameters:
    v_g : np.ndarray   (N, 3) vectors in the ground/lab frame.
    r_gb : np.ndarray  (N, 3, 3) rotation matrices from compute_rotmat_from_body_euler_xyz.

    Returns:
    np.ndarray  (N, 3) the same vectors in the body frame.
    """
    return np.einsum('nij,nj->ni', np.transpose(r_gb, (0, 2, 1)), v_g)


def compute_gyro_from_body_euler_xyz(
        a_gb: np.ndarray,
        dt: float,
        input_in_degrees: bool = True,
        sequence: str = 'XYZ'
) -> np.ndarray:
    """
    # Subject-Independent, Biological Hip Moment Estimation During Multimodal Overground Ambulation Using Deep Learning
    # https://ieeexplore.ieee.org/document/9687847/
    Paper Appendix A Eq (6)-(8):
      W = d(R_GB)/dt * R_BG
      omega_G from W, then omega_B = R_BG * omega_G

    Inputs:
      a_xyz: Nx3 array of body orientation angles (OpenSim columns *_Ox_pos, *_Oy_pos, *_Oz_pos)
      dt: timestep [s]
      degrees: set True if your angles are in degrees (common in exported IK tables)
    Returns:
      gyro_B: Nx3 angular velocity expressed in body frame (sensor-aligned with body) using radians
    """
    if input_in_degrees:
        a_gb = np.deg2rad(a_gb)

    r_gb = R.from_euler(sequence, angles=a_gb, degrees=False).as_matrix()

    # Compute derivative of rotation matrix
    r_gb_dot = np.gradient(r_gb, dt, axis=0)

    # R_BG for each sample, shape (N,3,3)
    r_bg = np.swapaxes(r_gb, 1, 2)

    # Batched W = r_gb_dot * r_bg, shape (N,3,3)
    w_mat_g = r_gb_dot @ r_bg
    omega_gb = batch_skew_symmetric_mat_to_ground_ang_vel(w_mat_g)
    omega_b = r_bg @ omega_gb[:, :, np.newaxis]

    return omega_b.squeeze()


def compute_accel_from_body_pos_and_euler_xyz(
        p_gb: np.ndarray,
        a_gb: np.ndarray,
        dt: float,
        p_bc: np.ndarray,
        input_in_degrees: bool = True,
        gravity_g: np.ndarray = np.array([0.0, -9.81, 0.0]),
        sequence: str = "XYZ"
) -> np.ndarray:
    """
    Paper Appendix A Eq (9)-(12):
      p_GC = p_GB + R_GB * p_BC
      a_GC = d2(p_GC)/dt2 - g
      a_C  = R_BG * a_GC   (if sensor frame aligned with body frame)

    Inputs:
      p_gb: Nx3 reference-point trajectory in ground. With body_segments.csv this is the
            segment's CENTRE OF MASS, not the body origin -- the OpenSim Analysis Tool's
            BodyKinematics records "center of mass position and orientation" per body.
      p_bc: 3-vector offset [m] of the sensor from that reference point, in body
            coordinates. COM-relative for the same reason.
    Returns:
      accel_C: Nx3 specific force expressed in sensor/body frame
    """

    if input_in_degrees:
        a_gb = np.deg2rad(a_gb)

    # Reshape inputs
    p_bc = np.asarray(p_bc, dtype=float).reshape(3, 1)  # (3, 1) for direct broadcasting
    gravity_g = np.asarray(gravity_g, dtype=float).reshape(3, )

    # Batch compute rotation matrices R_GB (g_R_b), shape (N, 3, 3)
    r_gb = R.from_euler(sequence, angles=a_gb, degrees=False).as_matrix()

    # Compute p_GC(t) = p_GB(t) + R_GB @ p_BC
    # Batched matrix-vector: (N,3,3) @ (3,1) -> (N,3,1) -> squeeze -> (N,3)
    p_gc = p_gb + (r_gb @ p_bc).squeeze(-1)  # Eq. (9)

    # Second derivative of p_GC, then subtract gravity
    d_p_gc = np.gradient(p_gc, dt, axis=0)
    dd_p_gc = np.gradient(d_p_gc, dt, axis=0)
    a_gc = dd_p_gc - gravity_g  # Eq. (10)

    # Rotate to sensor/body frame (Eq. 11-12)
    # R_BG = R_GB^T for each sample, shape (N,3,3)
    r_bg = np.transpose(r_gb, (0, 2, 1))

    r_cg = r_bg  # (Eq 12)

    # Batched a_C = R_BG @ a_spec_G, shape (N,3)
    accel_c = (r_cg @ a_gc[..., None]).squeeze(-1)

    return accel_c
