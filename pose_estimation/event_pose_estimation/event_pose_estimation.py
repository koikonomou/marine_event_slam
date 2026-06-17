import math
import cv2
import numpy as np


def pose_estimation(img, imu, prev_mem):
    focal_px = 800.0
    event_thresh = 0.20

    if not prev_mem:   # empty dict on first call → initialise in place
        prev_mem.update({
            'lp': None,   # previous log-LUT frame,  events are computed as the difference between this and the current frame
            'pkp': None,   # previous ORB keypoints , needed to match features from the last frame to this
            'pdes': None,   # previous ORB descriptors,  
            'x': 0.0, 
            'y': 0.0,
            'yaw': None,
            'prev_t': None, # the timestamp of the previous frame. Used to compute the time-difference between frames
            'lut': None, #precomputed log lookup table. Build once, resused every call
            'C': None, # the event threshold expressed in LUT integer units
            'orb': None, # the ORB feature detector object
            'bf': None, # the BFMatcher
        })
    
    #     Detection Keypoints and Finding Descriptors
    if prev_mem['lut'] is None:
        x_arr = np.arange(256, dtype=np.float32)
        raw = np.log(x_arr + 1e-6)
        scale = 255.0 / float(raw.max() - raw.min())
        prev_mem['lut'] = np.clip(np.round((raw - raw.min()) * scale), 0, 255).astype(np.uint8)
        prev_mem['C'] = max(1, int(round(event_thresh * scale)))
        prev_mem['orb'] = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=15)
        prev_mem['bf'] = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    heading = float(imu['heading'])
    sog_ms = float(imu['sog']) * 0.51444 # knots → m/s
    yaw_spd = float(imu['yaw_speed'])
    pitch = float(imu['pitch'])
    roll = float(imu['roll'])
    epoch = imu['epoch']

    dt = (float(epoch) - prev_mem['prev_t']) if prev_mem['prev_t'] else 1.0
    dt = max(min(dt, 5.0), 0.05)

    # Events (log-LUT diff on green channel)
    g = cv2.GaussianBlur(img[:, :, 1], (0, 0), 1.5)
    lc = cv2.LUT(g, prev_mem['lut'])

    n_in, dx_px = 0, 0.0

    if prev_mem['lp'] is not None:
        diff = lc.astype(np.int16) - prev_mem['lp'].astype(np.int16)
        ev = np.zeros_like(img)
        ev[diff >  prev_mem['C'], 2] = 255   # red  = ON
        ev[diff < -prev_mem['C'], 0] = 255   # blue = OFF

        # Merge ON + OFF into single grayscale for ORB
        ev_gray = np.clip(ev[:, :, 2].astype(np.uint16) + ev[:, :, 0].astype(np.uint16),0, 255).astype(np.uint8)

        # Feature matching — RANSAC
        kp, des = prev_mem['orb'].detectAndCompute(ev_gray, None)
        if (prev_mem['pkp'] and prev_mem['pdes'] is not None and des is not None and len(des) >= 4):
            matches = sorted(prev_mem['bf'].match(prev_mem['pdes'], des), key=lambda m: m.distance)[:200]
            if len(matches) >= 4:
                p0 = np.float32([prev_mem['pkp'][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                p1 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                _, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 25.0)
                if mask is not None:
                    inl  = mask.ravel().astype(bool)
                    n_in = int(inl.sum())
                    if n_in > 0:
                        dx_px = float((p1 - p0)[inl][:, 0, 0].mean())

        prev_mem['pkp'] = kp
        prev_mem['pdes'] = des

    prev_mem['lp'] = lc
    prev_mem['prev_t'] = float(epoch) if epoch else None

    #  Yaw: complementary filter (visual + IMU rate, anchored to heading)
    if prev_mem['yaw'] is None:
        prev_mem['yaw'] = heading

    vis_yaw = -math.degrees(math.atan2(dx_px, focal_px))
    alpha = min(n_in / 60.0, 1.0) * 0.40
    prev_mem['yaw'] += alpha * vis_yaw + (1.0 - alpha) * yaw_spd * dt
    prev_mem['yaw'] += 0.05 * (((heading - prev_mem['yaw']) + 180) % 360 - 180)
    prev_mem['yaw'] %= 360.0

    # Position (SOG + fused heading)
    hr = math.radians(prev_mem['yaw'])
    prev_mem['x'] += sog_ms * dt * math.sin(hr)
    prev_mem['y'] += sog_ms * dt * math.cos(hr)

    pose = {
        'x': round(prev_mem['x'], 3),
        'y': round(prev_mem['y'], 3),
        'roll': round(roll, 3),
        'pitch': round(pitch, 3),
        'yaw': round(prev_mem['yaw'], 3),
    }
    return pose, prev_mem


if __name__ == '__main__':
    import argparse, csv, glob, os, sys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description='Event-based, marine pose_estimation')
    ap.add_argument('--frames', required=True, help='Folder of PNG frames')
    ap.add_argument('--imu',    required=True, help='IMU/GPS CSV')
    ap.add_argument('--out',    default='trajectory.png')
    args = ap.parse_args()

    imu_rows = list(csv.DictReader(open(args.imu)))
    frames   = sorted(glob.glob(os.path.join(args.frames, '*.png')))

    # GPS reference point for ENU conversion (first frame)
    def to_en(lat, lon, lat0, lon0):
        north = (lat - lat0) * 111_320.0
        east  = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
        return east, north

    ref_lat = ref_lon = None
    prev_mem = {}
    pose_xs, pose_ys = [], []
    gps_xs,  gps_ys  = [], []

    for fpath in frames:
        ts_s    = int(os.path.splitext(os.path.basename(fpath))[0]) * 1e-9
        nearest = min(imu_rows, key=lambda r: abs(float(r['epoch']) - ts_s))

        if ref_lat is None:
            ref_lat = float(nearest['lat'])
            ref_lon = float(nearest['lon'])

        imu = {k: float(nearest[k]) for k in ('epoch', 'sog', 'heading', 'pitch', 'roll', 'yaw_speed')}
        pose, _ = pose_estimation(cv2.imread(fpath), imu, prev_mem)
        pose_xs.append(pose['x'])
        pose_ys.append(pose['y'])

        gx, gy = to_en(float(nearest['lat']), float(nearest['lon']), ref_lat, ref_lon)
        gps_xs.append(gx)
        gps_ys.append(gy)

        print(f"pose  x={pose['x']:+7.2f}m  y={pose['y']:+7.2f}m  yaw={pose['yaw']:6.1f}°"
              f"  |  gps  x={gx:+7.2f}m  y={gy:+7.2f}m")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gps_xs,  gps_ys,  color='green',     lw=1.8,         label='GPS ground truth')
    ax.plot(pose_xs, pose_ys, color='steelblue',  lw=1.5, ls='--', label='Pose estimation')
    ax.scatter([gps_xs[0]],   [gps_ys[0]],   color='green',     s=70, zorder=5)
    ax.scatter([gps_xs[-1]],  [gps_ys[-1]],  color='darkgreen', s=70, zorder=5, marker='s')
    ax.scatter([pose_xs[0]],  [pose_ys[0]],  color='steelblue', s=70, zorder=5)
    ax.scatter([pose_xs[-1]], [pose_ys[-1]], color='navy',      s=70, zorder=5, marker='s')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title(f'Pose vs GPS  —  {ref_lat:.4f}°N  {ref_lon:.4f}°E')
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f'\nSaved → {args.out}')
