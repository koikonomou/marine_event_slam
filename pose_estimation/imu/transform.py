import numpy as np

import cv2

import pose_estimation as pest


def pose_estimation( img, imu, _ ):

    # Stabilize wrt horizon: transform so that roll
    # and pitch are zero. 
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




