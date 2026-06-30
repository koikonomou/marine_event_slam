import math
import os
import time
import cv2
import numpy as np

from pose_estimation import demo_frames


def pose_estimation(img, imu, prev_mem, outfile=None):
    focal_px = 800.0
    event_thresh = 0.20
    ACC_WINDOW = 7     # temporal accumulation window
    ACC_THRESHOLD = 4     # spikes needed to survive
    TIME_TAU = 25.0  # time surface decay (seconds)

    if not prev_mem:
        prev_mem.update({
            'lp': None,
            'x': 0.0,
            'y': 0.0,
            'yaw': None,
            'prev_t': None,
            'lut': None,
            'C': None,
            'event_buffer': [],
            'time_surface': None,
            'debug_trail': [],
            'ref_lat': None,
            'ref_lon': None,
            'gps_proj': None,   # pyproj Transformer (lat/lon → UTM)
            'ref_e':   None,    # UTM easting of origin
            'ref_n':   None,    # UTM northing of origin
        })

    if prev_mem['lut'] is None:
        x_arr = np.arange(256, dtype=np.float32)
        raw  = np.log(x_arr + 1e-6)
        scale = 255.0 / float(raw.max() - raw.min())
        prev_mem['lut'] = np.clip(np.round((raw - raw.min()) * scale), 0, 255).astype(np.uint8)
        prev_mem['C'] = max(1, int(round(event_thresh * scale)))

    heading = float(imu['heading'])
    sog_ms  = float(imu['sog']) * 0.51444
    yaw_spd = float(imu['yaw_speed'])
    pitch   = float(imu['pitch'])
    roll    = float(imu['roll'])
    epoch   = imu['epoch']

    dt = (float(epoch) - prev_mem['prev_t']) if prev_mem['prev_t'] else 1.0
    dt = max(min(dt, 5.0), 0.05)

    # Event generation
    g  = cv2.GaussianBlur(img[:, :, 1], (0, 0), 1.5)
    lc = cv2.LUT(g, prev_mem['lut'])

    vis_yaw = 0.0
    _t0     = None

    if prev_mem['lp'] is not None:
        diff = lc.astype(np.int16) - prev_mem['lp'].astype(np.int16)
        ev = np.zeros_like(img)
        ev[diff >  prev_mem['C'], 2] = 255   # red  = ON
        ev[diff < -prev_mem['C'], 0] = 255   # blue = OFF
        on_ch  = ev[:, :, 2]   # ON  events
        off_ch = ev[:, :, 0]   # OFF events
        ev_gray = np.clip(on_ch.astype(np.uint16) + off_ch.astype(np.uint16), 0, 255).astype(np.uint8)

        _t0 = time.perf_counter()

        # 1. Temporal spike accumulation
        prev_mem['event_buffer'].append((ev_gray > 0).astype(np.uint8))
        if len(prev_mem['event_buffer']) > ACC_WINDOW:
            prev_mem['event_buffer'].pop(0)

        if len(prev_mem['event_buffer']) >= ACC_THRESHOLD:
            acc_map = (np.sum(prev_mem['event_buffer'], axis=0)
                       >= ACC_THRESHOLD).astype(np.uint8) * 255
        else:
            acc_map = ev_gray

        # 2. Time surface
        h_img, w_img = ev_gray.shape
        if prev_mem['time_surface'] is None:
            prev_mem['time_surface'] = np.zeros((h_img, w_img), dtype=np.float32)
        prev_mem['time_surface'][ev_gray > 0] = float(epoch)
        ts_norm = np.exp(-(float(epoch) - prev_mem['time_surface']) / TIME_TAU).astype(np.float32)
        ts_norm[prev_mem['time_surface'] == 0] = 0.0
        prev_mem['ts_norm'] = ts_norm

        # 3. Optical flow yaw — LK feature tracking on log-LUT frames (upper half)
        #    Upper half = sky + horizon = stable features for yaw, no water noise
        h_of    = lc.shape[0] // 2
        mask_of = np.zeros_like(lc); mask_of[:h_of, :] = 255
        prev_pts = cv2.goodFeaturesToTrack(
            prev_mem['lp'], maxCorners=80, qualityLevel=0.01,
            minDistance=10, mask=mask_of)
        if prev_pts is not None and len(prev_pts) >= 3:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_mem['lp'], lc, prev_pts, None,
                winSize=(21, 21), maxLevel=3)
            good = status.flatten() == 1
            if good.sum() >= 5:
                dx = curr_pts[good, 0, 0] - prev_pts[good, 0, 0]
                vis_yaw = math.degrees(math.atan2(-float(np.median(dx)), focal_px))
        # Fall back to event polarity if optical flow has too few features
        if vis_yaw == 0.0 and len(prev_mem['event_buffer']) >= ACC_WINDOW:
            on_cols  = np.where(on_ch  > 0)[1]
            off_cols = np.where(off_ch > 0)[1]
            if len(on_cols) > 20 and len(off_cols) > 20:
                polarity_dx = float(on_cols.mean() - off_cols.mean())
                vis_yaw = math.degrees(math.atan2(-polarity_dx, focal_px))

    prev_mem['lp']     = lc
    prev_mem['prev_t'] = float(epoch)

    # Yaw — complementary filter: optical flow (primary) + IMU rate + compass anchor
    if prev_mem['yaw'] is None:
        prev_mem['yaw'] = heading
    prev_mem['yaw'] += 0.60 * vis_yaw + 0.40 * yaw_spd * dt
    prev_mem['yaw'] += 0.05 * (((heading - prev_mem['yaw']) + 180) % 360 - 180)
    prev_mem['yaw'] %= 360.0

    # Position — dead-reckoning with ZUPT
    hr = math.radians(prev_mem['yaw'])
    if sog_ms > 0.15:
        prev_mem['x'] += sog_ms * dt * math.sin(hr)
        prev_mem['y'] += sog_ms * dt * math.cos(hr)

    # GPS correction — only when moving, so moored vessel is never pulled to berth
    if sog_ms > 0.3:
        try:
            from pyproj import Transformer
            lat = float(imu['lat'])
            lon = float(imu['lon'])
            if prev_mem['ref_lat'] is None:
                prev_mem['ref_lat'] = lat
                prev_mem['ref_lon'] = lon
                zone = int((lon + 180) / 6) + 1
                epsg = 32600 + zone if lat >= 0 else 32700 + zone
                prev_mem['gps_proj'] = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
                prev_mem['ref_e'], prev_mem['ref_n'] = prev_mem['gps_proj'].transform(lon, lat)
            gps_e, gps_n = prev_mem['gps_proj'].transform(lon, lat)
            gps_e -= prev_mem['ref_e']
            gps_n -= prev_mem['ref_n']
            prev_mem['x'] += 0.10 * (gps_e - prev_mem['x'])
            prev_mem['y'] += 0.10 * (gps_n - prev_mem['y'])
        except (KeyError, TypeError, ValueError, Exception):
            pass



    if _t0 is not None:
        prev_mem['proc_ms'] = (time.perf_counter() - _t0) * 1000

    pose = {
        'x':     round(prev_mem['x'],   3),
        'y':     round(prev_mem['y'],   3),
        'roll':  round(roll,             3),
        'pitch': round(pitch,            3),
        'yaw':   round(prev_mem['yaw'], 3),
    }

    if outfile:
        prev_mem = demo_frames.make_one_frame( outfile, img, pose, imu, prev_mem )

    return pose, prev_mem


