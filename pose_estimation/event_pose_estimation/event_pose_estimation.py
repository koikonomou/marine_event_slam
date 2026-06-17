import math
import cv2
import numpy as np


def pose_estimation(img, imu, prev_mem):
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
            # 1. Temporal spike accumulation
            'event_buffer': [],
            # 2. Time surface
            'time_surface': None,
            # loop closure
            'keyframes': [],
            'last_kf_t': None,
        })

    if prev_mem['lut'] is None:
        x_arr = np.arange(256, dtype=np.float32)
        raw  = np.log(x_arr + 1e-6)
        scale = 255.0 / float(raw.max() - raw.min())
        prev_mem['lut'] = np.clip(np.round((raw - raw.min()) * scale),0, 255).astype(np.uint8)
        prev_mem['C'] = max(1, int(round(event_thresh * scale)))

    heading = float(imu['heading'])
    sog_ms  = float(imu['sog']) * 0.51444
    yaw_spd = float(imu['yaw_speed'])
    pitch = float(imu['pitch'])
    roll = float(imu['roll'])
    epoch = imu['epoch']

    dt = (float(epoch) - prev_mem['prev_t']) if prev_mem['prev_t'] else 1.0
    dt = max(min(dt, 5.0), 0.05)

    # Event generation 
    g  = cv2.GaussianBlur(img[:, :, 1], (0, 0), 1.5)
    lc = cv2.LUT(g, prev_mem['lut'])

    vis_yaw = 0.0   # visual yaw estimate (event-based)

    if prev_mem['lp'] is not None:
        diff = lc.astype(np.int16) - prev_mem['lp'].astype(np.int16)
        ev = np.zeros_like(img)
        ev[diff >  prev_mem['C'], 2] = 255   # red  = ON
        ev[diff < -prev_mem['C'], 0] = 255   # blue = OFF
        on_ch  = ev[:, :, 2]   # ON  events
        off_ch = ev[:, :, 0]   # OFF events
        ev_gray = np.clip(on_ch.astype(np.uint16) + off_ch.astype(np.uint16), 0, 255).astype(np.uint8)

        # 1. Temporal spike accumulation 
        # Pixels must fire in >= ACC_THRESHOLD of last ACC_WINDOW frames.
        # Mountains and masts produce consistent edges → survive.
        # Random sea glitter fires once or twice → suppressed.
        prev_mem['event_buffer'].append((ev_gray > 0).astype(np.uint8))
        if len(prev_mem['event_buffer']) > ACC_WINDOW:
            prev_mem['event_buffer'].pop(0)

        if len(prev_mem['event_buffer']) >= ACC_THRESHOLD:
            acc_map = (np.sum(prev_mem['event_buffer'], axis=0)
                       >= ACC_THRESHOLD).astype(np.uint8) * 255
        else:
            acc_map = ev_gray

        #  2. Time surface 
        # Each pixel stores when it last fired an event (membrane potential).
        # ts_norm[y,x] = exp(-(now - last_t[y,x]) / tau): recent = bright.
        h_img, w_img = ev_gray.shape
        if prev_mem['time_surface'] is None:
            prev_mem['time_surface'] = np.zeros((h_img, w_img), dtype=np.float32)
        prev_mem['time_surface'][ev_gray > 0] = float(epoch)
        ts_norm = np.exp(-(float(epoch) - prev_mem['time_surface']) / TIME_TAU).astype(np.float32)
        ts_norm[prev_mem['time_surface'] == 0] = 0.0
        prev_mem['ts_norm'] = ts_norm 

        #  3. Event polarity yaw 
        # Camera rotates RIGHT → scene moves LEFT →
        #   ON  events appear on the LEFT  edge of bright objects
        #   OFF events appear on the RIGHT edge of bright objects
        # Spatial separation between ON and OFF centres → rotation direction.
        # No feature matching needed — pure event polarity geometry.
        on_cols  = np.where(on_ch  > 0)[1]
        off_cols = np.where(off_ch > 0)[1]
        if len(on_cols) > 20 and len(off_cols) > 20:
            polarity_dx = float(on_cols.mean() - off_cols.mean())
            vis_yaw = math.degrees(math.atan2(-polarity_dx, focal_px))

    prev_mem['lp'] = lc
    prev_mem['prev_t'] = float(epoch)

    # Loop closure 
    h_lc = lc.shape[0]
    proj = lc[:h_lc // 2, :].astype(np.float32).mean(axis=0)
    sig = cv2.resize(proj.reshape(1, -1), (64, 1)).flatten()
    sig -= sig.mean()
    d_cur = math.sqrt(float((sig * sig).sum())) + 1e-10

    if prev_mem['last_kf_t'] is None or float(epoch) - prev_mem['last_kf_t'] > 30:
        prev_mem['keyframes'].append({'x': prev_mem['x'], 'y': prev_mem['y'],'sig': sig.copy(), 't': float(epoch)})
        prev_mem['last_kf_t'] = float(epoch)

    best_score, best_kf = 0.0, None
    for kf in prev_mem['keyframes']:
        if float(epoch) - kf['t'] < 60:
            continue
        d_kf  = math.sqrt(float((kf['sig'] * kf['sig']).sum())) + 1e-10
        score = float((sig * kf['sig']).sum()) / (d_cur * d_kf)
        if score > best_score:
            best_score, best_kf = score, kf

    if best_score > 0.92 and best_kf is not None:
        prev_mem['x'] += 0.10 * (best_kf['x'] - prev_mem['x'])
        prev_mem['y'] += 0.10 * (best_kf['y'] - prev_mem['y'])

    # Yaw fusion 
    if prev_mem['yaw'] is None:
        prev_mem['yaw'] = heading

    # Polarity yaw
    prev_mem['yaw'] += 0.20 * vis_yaw + 0.80 * yaw_spd * dt
    prev_mem['yaw'] += 0.05 * (((heading - prev_mem['yaw']) + 180) % 360 - 180)
    prev_mem['yaw'] %= 360.0

    #Position
    hr = math.radians(prev_mem['yaw'])
    prev_mem['x'] += sog_ms * dt * math.sin(hr)
    prev_mem['y'] += sog_ms * dt * math.cos(hr)

    pose = {
        'x':     round(prev_mem['x'],   3),
        'y':     round(prev_mem['y'],   3),
        'roll':  round(roll,             3),
        'pitch': round(pitch,            3),
        'yaw':   round(prev_mem['yaw'], 3),
    }
    return pose, prev_mem


if __name__ == '__main__':
    import argparse, csv, glob, os, sys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description='Event-based marine pose estimation')
    ap.add_argument('--frames', required=True)
    ap.add_argument('--imu',    required=True)
    ap.add_argument('--out',    default='trajectory.png')
    args = ap.parse_args()

    imu_rows = list(csv.DictReader(open(args.imu)))
    frames   = sorted(glob.glob(os.path.join(args.frames, '*.png')))

    def to_en(lat, lon, lat0, lon0):
        north = (lat - lat0) * 111_320.0
        east  = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
        return east, north

    ref_lat = ref_lon = None
    prev_mem = {}
    pose_xs, pose_ys, gps_xs, gps_ys = [], [], [], []

    for fpath in frames:
        ts_s    = int(os.path.splitext(os.path.basename(fpath))[0]) * 1e-9
        nearest = min(imu_rows, key=lambda r: abs(float(r['epoch']) - ts_s))
        if ref_lat is None:
            ref_lat, ref_lon = float(nearest['lat']), float(nearest['lon'])
        imu = {k: float(nearest[k])
               for k in ('epoch', 'sog', 'heading', 'pitch', 'roll', 'yaw_speed')}
        pose, _ = pose_estimation(cv2.imread(fpath), imu, prev_mem)
        pose_xs.append(pose['x']); pose_ys.append(pose['y'])
        gx, gy = to_en(float(nearest['lat']), float(nearest['lon']), ref_lat, ref_lon)
        gps_xs.append(gx); gps_ys.append(gy)
        print(f"pose ({pose['x']:+7.2f},{pose['y']:+7.2f})  "
              f"gps ({gx:+7.2f},{gy:+7.2f})")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gps_xs,  gps_ys,  color='green',    lw=1.8, label='GPS')
    ax.plot(pose_xs, pose_ys, color='steelblue', lw=1.5, ls='--', label='Pose')
    ax.set_aspect('equal'); ax.grid(True)
    ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)')
    ax.set_title(f'Pose vs GPS'); ax.legend()
    plt.tight_layout(); plt.savefig(args.out, dpi=150)
    print(f'Saved → {args.out}')
