import math
import cv2

import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx

import matplotlib
import matplotlib.pyplot as plt

from pose_estimation import utils


def make_one_frame( outfile, img, pose, imu, prev_mem ):

    try:
        lat = float(imu['lat'])
        lon = float(imu['lon'])
    except (KeyError, TypeError):
        lat = lon = 0.0

    prev_mem['debug_trail'].append({'x': pose['x'], 'y': pose['y'], 'lat': lat, 'lon': lon, 'yaw': pose['yaw']})

    try:
        matplotlib.use('Agg')

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

        if 1 == 0:
            # arrow at the current vessel position pointing in the current heading direction
            yaw_now    = trail[-1]['yaw']
            arrow_len  = buf * 0.1
            ax.annotate('',
                xy=(pose_abs_x[-1] + arrow_len * math.sin(yaw_now),
                    pose_abs_y[-1] + arrow_len * math.cos(yaw_now)),
                xytext=(pose_abs_x[-1], pose_abs_y[-1]),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=2.5, mutation_scale=18))
        else:
            # non-directional marks
            ax.scatter( gps_abs_x[-1], gps_abs_y[-1], marker="x", color="darkgreen", s=100, lw=1, zorder=5 )
            ax.scatter( pose_abs_x[-1],pose_abs_y[-1],marker="x", color="blue", s=100, lw=1, zorder=5 )

        ctx.add_basemap(ax, crs=utm_crs.to_string(),
                        source=ctx.providers.OpenStreetMap.Mapnik, zoom="auto")
        ax.set_aspect('equal')
        ax.set_xlabel('UTM Easting (m)'); ax.set_ylabel('UTM Northing (m)')
        #ax.set_title(f'Trajectory vs GPS — Syros  (final {errors[-1]:.1f} m  mean {sum(errors)/len(errors):.1f} m)')
        ax.legend()


        # bottom left inset: original frame
        ax_cam = ax.inset_axes( [0.01, 0.55, 0.45, 0.45] )
        ax_cam.imshow( cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )
        ax_cam.axis( "off" )
        ax_cam.set_title( "Original", fontsize=8, pad=2 )

        # bottom right inset: stabilized frame
        ax_stb = ax.inset_axes( [0.5, 0.55, 0.45, 0.45] )
        stb_img = utils.stabilize( img, pose )
        ax_stb.imshow( cv2.cvtColor(stb_img, cv2.COLOR_BGR2RGB) )
        ax_stb.axis( "off" )
        ax_stb.set_title( "Stabilized", fontsize=8, pad=2 )

        plt.tight_layout()
        plt.savefig(outfile, dpi=150)
        plt.close(fig)
        # print(f'[debug] final={errors[-1]:.1f}m  mean={sum(errors)/len(errors):.1f}m')
    except Exception as e:
        print(f'[debug] plot error: {e}')

    return prev_mem

