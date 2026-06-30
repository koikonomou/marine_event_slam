import numpy as np
import cv2

from pose_estimation import utils, demo_frames


def get_visual_velocity(gray_frame, dt, yaw_rad, memory):
    """
    Tracks keypoints between frames to compute displacement, translates pixels 
    to meters based on camera geometry, and rotates the velocity vector into world space.
    """
    if memory["prev_gray"] is None:
        # Initialize tracking points if empty or lost
        memory["prev_points"] = cv2.goodFeaturesToTrack(gray_frame, maxCorners=100, qualityLevel=0.01, minDistance=10)
        memory["prev_gray"] = gray_frame.copy()
        return 0.0, 0.0

    # Calculate optical flow
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(memory["prev_gray"], gray_frame, memory["prev_points"], None, **memory["lk_params"])

    if next_points is None or len(next_points) == 0:
        memory["prev_points"] = cv2.goodFeaturesToTrack(gray_frame, maxCorners=100, qualityLevel=0.01, minDistance=10)
        memory["prev_gray"] = gray_frame.copy()
        return 0.0, 0.0

    # Filter out bad tracking matches
    valid_prev = memory["prev_points"][status == 1]
    valid_next = next_points[status == 1]

    if len(valid_prev) < 5:  # Too few points tracked
        memory["prev_points"] = cv2.goodFeaturesToTrack(gray_frame, maxCorners=100, qualityLevel=0.01, minDistance=10)
        memory["prev_gray"] = gray_frame.copy()
        return 0.0, 0.0

    # Compute average displacements in pixel coordinates (dx, dy)
    displacements = valid_next - valid_prev
    avg_dx_px = np.median(displacements[:, 0])
    avg_dy_px = np.median(displacements[:, 1])

    # Convert pixel displacement to metrics using pinhole camera scaling (dx_m = dx_px * height / focal_length)
    # Assumes a downward/nadir-facing camera mapping water textures or surrounding structures.
    dx_meters_local = -(avg_dx_px * memory["h"]) / memory["fx"]
    dy_meters_local = -(avg_dy_px * memory["h"]) / memory["fy"]

    # Rotate the boat-local visual translation vector into the global ENU map frame
    # Yaw maps tracking variables to Global East (X) and Global North (Y)
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)

    vx_world = (dx_meters_local * cos_y - dy_meters_local * sin_y) / dt
    vy_world = (dx_meters_local * sin_y + dy_meters_local * cos_y) / dt

    # Dynamically regenerate strong tracking points for the next frame
    memory["prev_points"] = cv2.goodFeaturesToTrack(gray_frame, maxCorners=100, qualityLevel=0.01, minDistance=10)
    memory["prev_gray"] = gray_frame.copy()

    return vx_world, vy_world



def init_memory():
    retv = {}
    # Static
    retv["fx"] = 1
    retv["fy"] = 1
    retv["h"] = 0.2
    retv["alpha"] = 0.85

    # WGS84 Ellipsoid constants tailored for global Transverse Mercator (UTM Zone 35N)
    retv["EARTH_A"] = 6378137.0         # Semi-major axis (meters)
    retv["EARTH_F"] = 1.0 / 298.257223563 # Flattening factor
    retv["EARTH_B"] = retv["EARTH_A"] * (1.0 - retv["EARTH_F"])
    retv["EARTH_E2"] = (retv["EARTH_A"]**2 - retv["EARTH_B"]**2) / (retv["EARTH_A"]**2)

    # Lucas-Kanade Optical Flow Parameters
    retv["lk_params"] = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    # Previous frame
    retv["prev_gray"] = None
    retv["prev_points"] = None
    retv["prev_t"] = None

    # For visualization
    retv["debug_trail"] = []

    return retv



def pose_estimation( img, data, memory, outfile=None ):

    if memory == {}:
        memory = init_memory()
    
    t_curr = float(data["epoch"])
    lat = float(data["lat"])
    lon = float(data["lon"])

    yaw_rad = np.radians(float(data["yaw"]))
    pitch_rad = np.radians(float(data["pitch"]))
    roll_rad = np.radians(float(data["roll"]))

    x_gps, y_gps = utils.geo_to_metric( lat, lon, memory )
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Handle the initialization frame
    if memory["prev_t"] is None:
        memory["prev_t"] = t_curr
        memory["x_fused"] = x_gps
        memory["y_fused"] = y_gps
        memory["prev_gray"] = gray_frame
        memory["prev_points"] = cv2.goodFeaturesToTrack(gray_frame, maxCorners=100, qualityLevel=0.01, minDistance=10)
        retv = {"x": memory["x_fused"], "y": memory["y_fused"], "roll": roll_rad, "pitch": pitch_rad, "yaw": yaw_rad}
        return retv, memory

    dt = t_curr - memory["prev_t"]
    if dt <= 0: dt = 0.001

    # Extract dead-reckoning metrics via Optical Flow
    vx_vis, vy_vis = get_visual_velocity(gray_frame, dt, yaw_rad, memory)

    # Propagate previous fused position using visual velocity
    x_pred = memory["x_fused"] + (vx_vis * dt)
    y_pred = memory["y_fused"] + (vy_vis * dt)

    # Correction Step: Blend prediction with absolute GPS measurements
    memory["x_fused"] = memory["alpha"] * x_pred + (1.0 - memory["alpha"]) * x_gps
    memory["y_fused"] = memory["alpha"] * y_pred + (1.0 - memory["alpha"]) * y_gps

    # Update timestamps
    memory["prev_t"] = t_curr

    pose = {
        "x": memory["x_fused"],
        "y": memory["y_fused"],
        "roll": roll_rad,      # To further smooth, apply similar alpha blending:
        "pitch": pitch_rad,    # angle_fused = alpha*(angle_prev + speed*dt) + (1-alpha)*angle_raw
        "yaw": yaw_rad
    }

    if outfile:
        memory = demo_frames.make_one_frame( outfile, img, pose, data, memory )

    return pose, memory

