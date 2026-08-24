from pathlib import Path

import numpy as np

WORK_PATH = Path()  # path to project to run simulation on

NUM_THREADS = 4

# Set offsets in body coordinates (can set sensor point to be different location from body origin)
# Set offsets to 0 for now (getting acceleration of body origin)
IMU_POS_OFFSETS_BC = {
    "femur_r": np.array([0.0, -0.2, 0.0]),  # thigh IMU
    "femur_l": np.array([0.0, -0.2, 0.0]),
    "tibia_r": np.array([0.0, -0.1, 0.0]),  # shank IMU
    "tibia_l": np.array([0.0, -0.1, 0.0]),
    "talus_r": np.array([0.0, 0.0, 0.0]),  # foot IMU
    "talus_l": np.array([0.0, 0.0, 0.0]),
    "pelvis": np.array([0.0, 0.0, 0.0])  # pelvis IMU
}

# TODO: rotation offset ignored for now
IMU_ROT_OFFSETS_BC = {
    "femur_r": np.array([0.0, 0.0, 0.0]),
    "femur_l": np.array([0.0, 0.0, 0.0]),
    "tibia_r": np.array([0.0, 0.0, 0.0]),
    "tibia_l": np.array([0.0, 0.0, 0.0]),
    "talus_r": np.array([0.0, 0.0, 0.0]),
    "talus_l": np.array([0.0, 0.0, 0.0]),
    "pelvis": np.array([0.0, 0.0, 0.0])
}

IMU_NAME_MAPPING = {
    "femur_r": "thigh_r",
    "femur_l": "thigh_l",
    "tibia_r": "shank_r",
    "tibia_l": "shank_l",
    "talus_r": "foot_r",
    "talus_l": "foot_l",
    "pelvis": "pelvis"
}

# matching scherpereel_2023's own description of their virtual insoles:
# "transformation matrices were inverted and used to convert the global
# force plate measures into the foot reference frame."
# F_foot = R_GB^T F_ground, with R_GB the calcn (foot) segment orientation.
INSOLE_FOOT_SEGMENTS = {
    "calcn_r": "insole_r",
    "calcn_l": "insole_l"
}

VGRF_ZERO_THRESH = 25.0
