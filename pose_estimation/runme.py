import sys, os, glob

import pandas as pd
import numpy as np

import cv2


imu = pd.read_csv( sys.argv[1] )
imu.columns = imu.columns.str.replace('"', '').str.strip()

fnames = glob.glob( sys.argv[2] )


sync_pairs = []
for fname in fnames:

    ts_str = os.path.splitext(os.path.basename(fname))[0] 
    ts = int( ts_str )
    imu["nsec"] = imu["epoch"] * 1E9
    idx_after = imu["nsec"].searchsorted( ts )

    if idx_after == 0: idx = 0
    elif idx_after == len(imu): idx = len(imu)-1
    else:
        diff_after = abs( ts-imu.loc[idx_after,"nsec"] )
        diff_before= abs( ts-imu.loc[idx_after-1,"nsec"] )
        if diff_after > diff_before: idx = idx_after
        else: idx = idx_after - 1

    img = cv2.imread( fname )

    print( ts )
    print( imu.loc[idx,'epoch'] )
    print( "---" )
    #pose, mem = pose_estimation( img, imu, prev_mem )


