import numpy as np
import cv2

from pose_estimation import utils, demo_frames


def init_memory():
    retv = {}

    # WGS84 Ellipsoid constants tailored for global Transverse Mercator (UTM Zone 35N)
    retv["EARTH_A"] = 6378137.0         # Semi-major axis (meters)
    retv["EARTH_F"] = 1.0 / 298.257223563 # Flattening factor
    retv["EARTH_B"] = retv["EARTH_A"] * (1.0 - retv["EARTH_F"])
    retv["EARTH_E2"] = (retv["EARTH_A"]**2 - retv["EARTH_B"]**2) / (retv["EARTH_A"]**2)

    # Kalman filter state
    kf_state = None

    # For visualization
    retv["debug_trail"] = []

    retv["prev_t"] = None
    return retv



def imu_pose_estimation( img, data, memory, outfile ):

    if memory == {}:
        memory = init_memory()
    
    t_curr = float(data["epoch"])
    lat = float(data["lat"])
    lon = float(data["lon"])
    x_gps, y_gps = utils.geo_to_metric( lat, lon, memory )

    yaw_rad = np.radians(float(data["yaw"]))
    pitch_rad = np.radians(float(data["pitch"]))
    roll_rad = np.radians(float(data["roll"]))

    # Handle the initialization frame
    if memory["prev_t"] is None:

        # Baseline initial state vector matching first known GPS coordinate
        memory["kf_state"] = np.array([x_gps, y_gps, yaw_rad])

        # Initial KF state covariance and noise
        memory["P"] = np.diag([1.0, 1.0, 0.1])  # Initial estimate uncertainty
        memory["Q"] = np.diag([0.2, 0.2, 0.02])  # Process noise (IMU / model drift)
        memory["R_mat"] = np.diag([2.0, 2.0, 0.1])  # Measurement noise (GPS drift)


        memory["prev_t"] = t_curr
        memory["x_fused"] = x_gps
        memory["y_fused"] = y_gps
        memory["yaw_fused"] = yaw_rad
        memory["sog"] = float(data["sog"])
        memory["yaw_speed"] = float(data["yaw_speed"])
        retv = {"x": memory["x_fused"], "y": memory["y_fused"], "roll": roll_rad, "pitch": pitch_rad, "yaw": yaw_rad}
        return retv, memory

    dt = t_curr - memory["prev_t"]
    if dt <= 0: dt = 0.001

    # Update timestamps
    memory["prev_t"] = t_curr

    # Kinematic Motion Model propagation
    prev_sog = memory["sog"]
    kf_yaw_prev = memory["kf_state"][2]

    memory["kf_state"][0] += prev_sog * np.sin(kf_yaw_prev) * dt
    memory["kf_state"][1] += prev_sog * np.cos(kf_yaw_prev) * dt
    memory["kf_state"][2] += np.radians(memory["yaw_speed"]) * dt

    memory["P"] += memory["Q"] * dt

    z_measure = np.array([x_gps, y_gps, yaw_rad])
    innovation = z_measure - memory["kf_state"]
    innovation[2] = (innovation[2] + np.pi) % (2 * np.pi) - np.pi

    S = memory["P"] + memory["R_mat"]
    K = memory["P"] @ np.linalg.inv(S)
    memory["kf_state"] = memory["kf_state"] + (K @ innovation)
    memory["P"] = (np.eye(3) - K) @ memory["P"]

    pose = {
        "x": memory["kf_state"][0],
        "y": memory["kf_state"][1],
        "roll": roll_rad,
        "pitch": pitch_rad,
        "yaw": memory["kf_state"][2]
    }

    if outfile:
        memory = demo_frames.make_one_frame( outfile, img, pose, data, memory )

    return pose, memory

