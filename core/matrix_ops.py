import numpy as np
from scipy.spatial.transform import Rotation as R


def rotmat_from_sensor_offset_bc(
        a_xyz: np.ndarray = None,
        input_in_degrees: bool = True
) -> np.ndarray:
    """
    The constant body->sensor mounting rotation R_BC from an offset triple.

    Same angle convention as the OpenSim orientation columns the rest of this pipeline is
    built on: body-fixed, intrinsic 'XYZ' (frontal, transverse, sagittal), so scipy's
    capital 'XYZ'. config.IMU_ROT_OFFSETS_BC stores these in DEGREES, matching the
    *_O{x,y,z}_rot columns.

    R_BC maps sensor-frame vectors into the body frame, so a sensor whose case is rotated
    by a_xyz relative to the segment it is strapped to has orientation R_GC = R_GB @ R_BC.

    Parameters:
    a_xyz : np.ndarray | None
        (3,) mounting offset. None or all-zero returns exactly the identity, i.e. a sensor
        aligned with the body frame -- the assumption this pipeline made before offsets
        existed. The identity is returned bit-exact so a zero offset cannot perturb the
        numbers.
    input_in_degrees : bool
        True if a_xyz is in degrees.

    Returns:
    np.ndarray  (3, 3) rotation matrix R_BC.
    """
    if a_xyz is None:
        return np.eye(3)
    a_xyz = np.asarray(a_xyz, dtype=float).reshape(3, )
    if not a_xyz.any():
        return np.eye(3)
    return R.from_euler('XYZ', a_xyz, degrees=input_in_degrees).as_matrix()


def _check_sensor_offset(r_bc: np.ndarray) -> bool:
    """
    Validate a body->sensor rotation and report whether it is a no-op.

    Returns True when r_bc is None or exactly the identity, which lets the callers below
    keep their original code path untouched. That is what makes a zero mounting offset
    bit-identical to the pre-offset pipeline rather than merely equal to within 1e-15.

    Raises ValueError if r_bc is not a (3, 3) rotation: a mistyped offset should fail here,
    not silently skew a whole dataset.
    """
    if r_bc is None:
        return True
    r_bc = np.asarray(r_bc, dtype=float)
    if r_bc.shape != (3, 3):
        raise ValueError(f"r_bc must be a (3, 3) rotation matrix, got shape {r_bc.shape}")
    if not np.allclose(r_bc.T @ r_bc, np.eye(3), atol=1e-8):
        raise ValueError("r_bc is not orthonormal, so it is not a rotation matrix")
    if not np.isclose(np.linalg.det(r_bc), 1.0, atol=1e-8):
        raise ValueError(f"r_bc has determinant {np.linalg.det(r_bc)}, expected +1")
    return np.array_equal(r_bc, np.eye(3))


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
        input_in_degrees: bool = True,
        r_bc: np.ndarray = None
) -> np.ndarray:
    """
    a_xyz: Nx3 angles in radians
    input_in_degrees: set True if your angles are in degrees (common in exported IK tables)
    r_bc: optional constant (3,3) body->sensor mounting rotation from
          rotmat_from_sensor_offset_bc(). None or identity returns the body's own
          orientation, as before offsets existed.
    """
    a_xyz = np.asarray(a_xyz, dtype=float)

    if input_in_degrees:
        a_xyz = np.deg2rad(a_xyz)

    # the opensim order is frontal (x), transverse (y), sagittal (z)
    # [N,3,3] array of rot mats
    r = R.from_euler('XYZ', a_xyz, degrees=False)
    if not _check_sensor_offset(r_bc):
        # R_GC = R_GB @ R_BC, composed while still a rotation
        r = r * R.from_matrix(r_bc)
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
        sequence: str = 'XYZ',
        r_bc: np.ndarray = None
) -> np.ndarray:
    """
    # Subject-Independent, Biological Hip Moment Estimation During Multimodal Overground Ambulation Using Deep Learning
    # https://ieeexplore.ieee.org/document/9687847/
    Paper Appendix A Eq (6)-(8):
      W = d(R_GB)/dt * R_BG
      omega_G from W, then omega_B = R_BG * omega_G
    and with a mounting offset, Eq (8) continues into the sensor case:
      omega_C = R_BC^T * omega_B

    Inputs:
      a_xyz: Nx3 array of body orientation angles (OpenSim columns *_Ox_pos, *_Oy_pos, *_Oz_pos)
      dt: timestep [s]
      degrees: set True if your angles are in degrees (common in exported IK tables)
      r_bc: optional constant (3,3) body->sensor mounting rotation.
    Returns:
      gyro_C: Nx3 angular velocity expressed in the sensor frame using radians. Identical to
              the body frame when r_bc is None or identity.
    """
    identity_offset = _check_sensor_offset(r_bc)

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
    omega_b = (r_bg @ omega_gb[:, :, np.newaxis]).squeeze(-1)

    if identity_offset:
        return omega_b

    # omega_C = R_BC^T omega_B: same vector, read on the sensor's own axes
    return np.einsum('ij,nj->ni', np.asarray(r_bc, dtype=float).T, omega_b)


def compute_accel_from_body_pos_and_euler_xyz(
        p_gb: np.ndarray,
        a_gb: np.ndarray,
        dt: float,
        p_bc: np.ndarray,
        input_in_degrees: bool = True,
        gravity_g: np.ndarray = np.array([0.0, -9.81, 0.0]),
        sequence: str = "XYZ",
        r_bc: np.ndarray = None
) -> np.ndarray:
    """
    Paper Appendix A Eq (9)-(12):
      p_GC = p_GB + R_GB * p_BC
      a_GC = d2(p_GC)/dt2 - g
      a_C  = R_CG * a_GC,  R_CG = (R_GB R_BC)^T = R_BC^T R_BG
    which reduces to Eq (12)'s R_BG when the sensor frame is aligned with the body frame.

    Inputs:
      p_gb: Nx3 reference-point trajectory in ground. With body_segments.csv this is the
            segment's CENTRE OF MASS, not the body origin -- the OpenSim Analysis Tool's
            BodyKinematics records "center of mass position and orientation" per body.
      p_bc: 3-vector offset [m] of the sensor from that reference point, in body
            coordinates. COM-relative for the same reason.
      r_bc: optional constant (3,3) body->sensor mounting rotation.
    Returns:
      accel_C: Nx3 specific force expressed in the sensor frame. Identical to the body frame
               when r_bc is None or identity.
    """
    identity_offset = _check_sensor_offset(r_bc)

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

    if identity_offset:
        r_cg = r_bg  # (Eq 12)
    else:
        # R_CG = (R_GB R_BC)^T = R_BC^T R_BG  (Eq 12 with the imu rotated on its mount)
        r_cg = np.einsum('ij,njk->nik', np.asarray(r_bc, dtype=float).T, r_bg)

    # Batched a_C = R_BG @ a_spec_G, shape (N,3)
    accel_c = (r_cg @ a_gc[..., None]).squeeze(-1)

    return accel_c
