import numpy as np

import cv2

import pose_estimation as pest


def geo_to_metric( lat, lon, memory ):
    """Converts geographic coordinates to local meters relative to origin"""

    if not (memory.get("ref_lat") and memory.get("ref_lon")):
        memory["ref_lat"] = lat
        memory["ref_lon"] = lon
        return 0.0, 0.0

    lat_rad = np.radians(lat)
    ref_lat_rad = np.radians(memory["ref_lat"])
    delta_lon_rad = np.radians(lon - memory["ref_lon"])
    delta_lat_rad = np.radians(lat - memory["ref_lat"])

    sin_ref = np.sin(ref_lat_rad)
    r_n = memory["EARTH_A"] / np.sqrt(1.0 - memory["EARTH_E2"] * sin_ref**2)
    r_m = r_n * (1.0 - memory["EARTH_E2"]) / (1.0 - memory["EARTH_E2"] * sin_ref**2)

    x = delta_lon_rad * r_n * np.cos(ref_lat_rad)
    y = delta_lat_rad * r_m
    return x, y



def stabilize( img, imu ):
    """Transforms image to bring roll and pitch to horizon"""

    # Transform so that roll and pitch are zero. 
    REF_ROLL = 0
    REF_PITCH = 0

    rot = imu["roll"]*(180.0/np.pi) - REF_ROLL
    trY = (imu["pitch"]*(180.0/np.pi) - REF_PITCH) * pest.PITCH_TO_PXL
    trX = 0.0 # ignore yaw

    # Translation
    T_mat = np.eye(3)
    T_mat[0, 2] = -trX
    T_mat[1, 2] = -trY

    # Rotation around img center
    h, w = img.shape[:2]
    M_rot_2d = cv2.getRotationMatrix2D((w / 2, h / 2), -rot_degs, 1.0)
    R_mat = np.eye(3)
    R_mat[0:2, 0:3] = M_rot_2d

    # Compose all transforms and apply to image
    M_combined_3x3 = np.dot(R_mat, T_mat)
    M_tel_final = M_combined_3x3[0:2, 0:3]

    trans_img = cv2.warpAffine( img, M_tel_final, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return trans_img

