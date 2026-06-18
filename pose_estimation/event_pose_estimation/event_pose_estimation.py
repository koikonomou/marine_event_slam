import math
import os
import cv2
import numpy as np


def pose_estimation(img, imu, prev_mem, debug=False):
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
            # debug
            'debug_trail': [],
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

        # 3. Event polarity yaw
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
        prev_mem['keyframes'].append({'x': prev_mem['x'], 'y': prev_mem['y'], 'sig': sig.copy(), 't': float(epoch)})
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

    prev_mem['yaw'] += 0.20 * vis_yaw + 0.80 * yaw_spd * dt
    prev_mem['yaw'] += 0.05 * (((heading - prev_mem['yaw']) + 180) % 360 - 180)
    prev_mem['yaw'] %= 360.0

    # Position
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

    if debug:
        try:
            lat = float(imu['lat'])
            lon = float(imu['lon'])
        except (KeyError, TypeError):
            lat = lon = 0.0
        prev_mem['debug_trail'].append({'x': pose['x'], 'y': pose['y'], 'lat': lat, 'lon': lon})

        try:
            import geopandas as gpd
            from shapely.geometry import Point
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            trail   = prev_mem['debug_trail']
            pose_xs = [p['x'] for p in trail]
            pose_ys = [p['y'] for p in trail]
            gdf     = gpd.GeoDataFrame(
                geometry=[Point(p['lon'], p['lat']) for p in trail],
                crs='EPSG:4326'
            )
            gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
            ox, oy  = gdf_utm.geometry.iloc[0].x, gdf_utm.geometry.iloc[0].y
            gps_xs  = [p.x - ox for p in gdf_utm.geometry]
            gps_ys  = [p.y - oy for p in gdf_utm.geometry]
            errors  = [math.hypot(px - gx, py - gy)
                       for px, py, gx, gy in zip(pose_xs, pose_ys, gps_xs, gps_ys)]

            import contextily as ctx

            utm_crs = gdf_utm.crs

            # absolute UTM coords for GPS and pose (needed for tile fetch)
            gps_abs_x  = [p.x for p in gdf_utm.geometry]
            gps_abs_y  = [p.y for p in gdf_utm.geometry]
            pose_abs_x = [ox + px for px in pose_xs]
            pose_abs_y = [oy + py for py in pose_ys]

            buf = 50
            all_x = gps_abs_x + pose_abs_x
            all_y = gps_abs_y + pose_abs_y

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlim(min(all_x) - buf, max(all_x) + buf)
            ax.set_ylim(min(all_y) - buf, max(all_y) + buf)
            ax.plot(gps_abs_x,  gps_abs_y,  color='green',    lw=1.8, label='GPS')
            ax.plot(pose_abs_x, pose_abs_y, color='steelblue', lw=1.5, ls='--', label='Pose')
            ctx.add_basemap(ax, crs=utm_crs.to_string(),source=ctx.providers.OpenStreetMap.Mapnik, zoom='auto')
            ax.set_aspect('equal')
            ax.set_xlabel('UTM Easting (m)'); ax.set_ylabel('UTM Northing (m)')
            ax.set_title(f'Trajectory vs GPS — Syros  (final {errors[-1]:.1f} m  mean {sum(errors)/len(errors):.1f} m)')
            ax.legend()
            plt.tight_layout()
            out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trajectory.png')
            plt.savefig(out, dpi=150)
            plt.close(fig)
            # print(f'[debug] final={errors[-1]:.1f}m  mean={sum(errors)/len(errors):.1f}m')
        except Exception as e:
            print(f'[debug] plot error: {e}')

    return pose, prev_mem


