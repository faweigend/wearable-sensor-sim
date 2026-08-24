import os
import time
from pathlib import Path
import multiprocessing as mp
import numpy as np
import pandas as pd
import core.matrix_ops as mops
import core.utility as utility
import config
import logging


def queue_filler(source_path: Path):
    """
    Find all body_segments.csv in the dataset and fill a multi processing queue
    """
    logging.info(f"filling queue")
    q = mp.Queue()
    # find all csvs prepared by the osim pipeline
    for csv_file in source_path.rglob("body_segments.csv"):
        q.put(csv_file)
        if q.qsize() % 100 == 0:
            logging.info(f"Queue size {q.qsize()}")
    return q


def queue_worker(q, thread_idx: int):
    # create a worker logger to manage the following logging calls and add a THREAD identifier
    worker_logger = logging.getLogger()
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(f"[THREAD {thread_idx}] %(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    worker_logger.addHandler(console_handler)
    worker_logger.setLevel(logging.INFO)
    while q.qsize() > 0:
        csv_file_path = q.get()
        logging.info(f"Queue size {q.qsize()}: starting to process {csv_file_path}")
        body_seg_df = pd.read_csv(csv_file_path, low_memory=False)
        create_simulated_imu_csv(csv_file_path, body_seg_df=body_seg_df)
        create_simulated_insole_csv(csv_file_path, body_seg_df=body_seg_df)
    logging.info(f"exiting")

    worker_logger.removeHandler(console_handler)


def create_simulated_imu_csv(csv_file_path, overwrite: bool = True, body_seg_df=None):
    # write simulated IMUs into same directory
    out_dir = csv_file_path.parent
    out_path = out_dir / "simulated_imus.csv"

    if out_path.exists():
        if overwrite:
            os.remove(out_path)
            logging.info(f"file {out_path} already exists, deleted")
        else:
            logging.info(f"file {out_path} already exists, skipped")
            return

    # read prepared imu_sim file by the osim pipeline (unless the caller already has it)
    if body_seg_df is None:
        body_seg_df = pd.read_csv(csv_file_path, low_memory=False)
    sim_df = body_seg_df["std_time"]

    # Get time vector
    ts = body_seg_df["std_time"].to_numpy(dtype=np.float32)
    dts = float(np.median(np.diff(ts)))

    # simulate imus for all segments
    all_segments = config.IMU_POS_OFFSETS_BC.keys()
    for body_seg in all_segments:
        p_offset_bc = config.IMU_POS_OFFSETS_BC[body_seg]

        # Body-origin trajectory in ground: the point the offsets above are measured from
        orig_pos_cols = utility.require_position_columns(body_seg_df.columns, body_seg)
        # Position of body origin in ground (p_GB)
        p_gb = body_seg_df[orig_pos_cols].to_numpy(dtype=np.float32)

        # Orientation angles
        ang = body_seg_df[
            [f"{body_seg}_Ox_rot", f"{body_seg}_Oy_rot", f"{body_seg}_Oz_rot"]
        ].to_numpy(dtype=np.float32)

        ypr_angles = mops.compute_angle_from_body_euler_xyz(
            a_xyz=ang,
            input_in_degrees=True
        )
        angle_print_order = np.hstack([ypr_angles[:, np.newaxis, -1], ypr_angles[:, :2]])

        gyro_b = mops.compute_gyro_from_body_euler_xyz(
            a_gb=ypr_angles,
            dt=dts,
            input_in_degrees=False,
            sequence='YZX'
        )

        accel_b = mops.compute_accel_from_body_pos_and_euler_xyz(
            p_gb=p_gb,
            a_gb=ypr_angles,
            dt=dts,
            p_bc=p_offset_bc,
            input_in_degrees=False,
            sequence='YZX'
        )

        # add simulated IMUs to csv
        imu_desc = f"{config.IMU_NAME_MAPPING[body_seg]}"
        cols = ["angle_x", "angle_y", "angle_z",
                "gyro_x", "gyro_y", "gyro_z",
                "accel_x", "accel_y", "accel_z"]
        append_sim_df = pd.DataFrame(
            np.hstack([angle_print_order, gyro_b, accel_b]),
            columns=[f"sim_{imu_desc}_{x}" for x in cols]
        )
        sim_df = pd.concat([sim_df, append_sim_df], axis=1)

    # write to determined output directory
    sim_df.to_csv(out_path, index=False)
    logging.info(f"saved file {out_path}")


def create_simulated_insole_csv(csv_file_path, overwrite: bool = True, body_seg_df=None):
    """
    Write simulated_insoles.csv next to body_segments.csv: the ground reaction force and
    the centre of pressure, both in each foot's own frame.

    Needs grfs.csv in the same trial_segment directory.
    Raises if the csv has no calcn_{l,r}_orig_* columns: the CoP is referenced to the calcn
    body origin.
    """
    out_dir = csv_file_path.parent
    out_file_path = out_dir / "simulated_insoles.csv"
    grf_file_path = out_dir / "grfs.csv"

    # calcn body axis -> insole channel. y is superior, so it is the vertical load.
    insole_force_axis_names = {0: "force_ap", 1: "force_vert", 2: "force_ml"}
    insole_cop_axis_names = {0: "cop_ap", 1: "cop_vert", 2: "cop_ml"}

    if out_file_path.exists():
        if overwrite:
            os.remove(out_file_path)
            logging.info(f"file {out_file_path} already exists, deleted")
        else:
            logging.info(f"file {out_file_path} already exists, skipped")
            return

    if not grf_file_path.exists():
        raise UserWarning(f"no grfs.csv in {out_dir}, cannot simulate insoles, skipped")

    if body_seg_df is None:
        body_seg_df = pd.read_csv(csv_file_path, low_memory=False)
    grf_df = pd.read_csv(grf_file_path, low_memory=False)

    # The two files are written by the same osim pipeline on one time base. Assert it
    if len(body_seg_df) != len(grf_df) or not np.allclose(
            body_seg_df["std_time"].to_numpy(dtype=float),
            grf_df["std_time"].to_numpy(dtype=float)):
        logging.error(f"std_time mismatch between body_segments.csv and grfs.csv in "
                      f"{out_dir} ({len(body_seg_df)} vs {len(grf_df)} rows), skipped")
        return

    sim_df = body_seg_df["std_time"]

    for foot_seg, insole_desc in config.INSOLE_FOOT_SEGMENTS.items():
        side = foot_seg.rsplit("_", 1)[-1]

        # Foot orientation in ground, same convention as the IMU path
        ang = body_seg_df[
            [f"{foot_seg}_Ox_rot", f"{foot_seg}_Oy_rot", f"{foot_seg}_Oz_rot"]
        ].to_numpy(dtype=float)
        r_gb = mops.compute_rotmat_from_body_euler_xyz(a_xyz=ang, input_in_degrees=True)

        # Lab-frame GRF for this foot
        f_g = grf_df[
            [f"ground_force_vx_{side}", f"ground_force_vy_{side}", f"ground_force_vz_{side}"]
        ].to_numpy(dtype=float)

        # Invert the transform: express the force in the foot frame
        f_b = mops.rotate_ground_to_body(v_g=f_g, r_gb=r_gb)

        cols = [f"sim_{insole_desc}_{insole_force_axis_names[k]}" for k in range(3)]
        sim_df = pd.concat([sim_df, pd.DataFrame(f_b, columns=cols)], axis=1)

        p_cop_g = grf_df[
            [f"ground_force_px_{side}", f"ground_force_py_{side}", f"ground_force_pz_{side}"]
        ].to_numpy(dtype=float)

        orig_cols = utility.require_position_columns(body_seg_df.columns, foot_seg)
        p_calcn_orig = body_seg_df[orig_cols].to_numpy(dtype=float)
        cop_b = mops.rotate_ground_to_body(p_cop_g - p_calcn_orig, r_gb)

        # An unloaded foot has no CoP: the stored global CoP is (0,0,0) there
        cop_b[f_g[:, 1] <= config.VGRF_ZERO_THRESH, :] = 0.0

        cols = [f"sim_{insole_desc}_{insole_cop_axis_names[k]}" for k in range(3)]
        sim_df = pd.concat([sim_df, pd.DataFrame(cop_b, columns=cols)], axis=1)

    sim_df.to_csv(out_file_path, index=False)
    logging.info(f"saved file {out_file_path}")


if __name__ == "__main__":
    work_path = config.WORK_PATH

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    q = queue_filler(work_path)

    num_threads = config.NUM_THREADS
    logger.info(f"Running {num_threads} threads")

    threads = []
    for t_idx in range(num_threads):
        t = mp.Process(target=queue_worker, args=(q, t_idx,))
        threads.append(t)

    for t in threads:
        t.start()
        time.sleep(1)

    # Wait for all threads to finish
    for t in threads:
        t.join()
