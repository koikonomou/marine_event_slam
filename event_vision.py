"""
event_vision.py  (v2 — ego-motion compensated)
───────────────────────────────────────────────────────────────────────────────
Bio-Inspired Event Vision with Ego-Motion Compensation.

The Problem
───────────
When the ASV moves, the camera sees global scene motion. Every background
pixel changes intensity → massive spurious events → detector fires on
everything.

The Fix
───────
Before computing the log-luminance difference we estimate the camera's own
motion as a homography H using:

    1. ORB feature detection on both frames
    2. Brute-force feature matching
    3. RANSAC homography estimation
       • Background features AGREE on H  (inliers)
       • Obstacle features DISAGREE       (outliers — moved independently)
    4. Warp the previous frame with H
    5. Compute log-difference on the aligned pair
       → background cancels out, only real obstacles remain

Pipeline
────────
RGB frame_t
    │
    ├─► ORB features ──┐
    │                  ├─► RANSAC homography H  (ego-motion estimate)
    └─► ORB features ◄─┘
            │
            ▼
    log(frame_t) - log(warp(frame_{t-1}, H))
            │
            ▼
    Threshold C  →  ON/OFF event mask
            │
            ▼
    Morph open + close + dilate
            │
            ▼
    BBoxMerger  →  one box per obstacle
            │
            ▼
    MonocularDistance  →  distance_mono_m
            │
            ▼
    same message dict as detect()

Usage
─────
    from event_vision import EventVisionDetector

    self.event_detector = EventVisionDetector(
        self,
        moving_camera = True,   # ← set True when ASV is underway
        event_thresh  = 0.15,   # can be LOWER now (ego-motion removed)
    )

    message = self.event_detector.detect_bioinspired(
        img, camera_lat, camera_lon, camera_alt_m, camera_bearing_deg)
"""

import cv2
import numpy as np
import math
import base64
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
#  1.  EGO-MOTION COMPENSATOR
# ══════════════════════════════════════════════════════════════════════════════

class EgoMotionCompensator:
    """
    Estimates and removes the ASV camera's own motion between consecutive frames.

    How it works
    ────────────
    ORB keypoints are detected in both frames and matched. A homography H is
    estimated with RANSAC:
        • Static background features  → agree on H  (inliers, green)
        • Moving obstacle features    → disagree     (outliers, red)

    The previous frame is warped with H so it is geometrically aligned to the
    current frame. The subsequent log-difference therefore shows only objects
    that moved INDEPENDENTLY of the camera.

    Fallback
    ────────
    If too few features are found (texture-poor water surface) or too few
    RANSAC inliers are accepted, the previous frame is used un-warped.
    This gracefully degrades to the static-camera behaviour.

    Parameters
    ──────────
    n_features  : ORB features to detect per frame  (default 1000)
    min_matches : minimum matches to attempt homography  (default 20)
    ransac_thr  : RANSAC reprojection threshold in pixels  (default 3.0)
    min_inliers : minimum RANSAC inliers to trust H  (default 15)
    """

    def __init__(self, n_features=1000, min_matches=20,
                 ransac_thr=3.0, min_inliers=15):
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.ransac_thr  = ransac_thr

        self._orb     = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self._prev_bgr = None
        self._prev_kp  = None
        self._prev_des = None

        # Public debug info
        self.last_n_inliers = 0
        self.last_status    = "init"
        self.last_H         = None

    def reset(self):
        self._prev_bgr = None
        self._prev_kp  = None
        self._prev_des = None
        self.last_H    = None
        self.last_status = "init"

    def compensate(self, frame_bgr):
        """
        Returns
        ───────
        prev_warped  uint8 H×W×3  previous frame warped to current viewpoint
        H            3×3 float or None
        ready        bool  False on first call
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self._orb.detectAndCompute(gray, None)

        if self._prev_bgr is None or des is None or self._prev_des is None:
            self._update(frame_bgr, kp, des)
            self.last_status = "init"
            return None, None, False

        # ── Feature matching ──────────────────────────────────────────────────
        matches = self._matcher.match(self._prev_des, des)
        matches = sorted(matches, key=lambda m: m.distance)

        if len(matches) < self.min_matches:
            warped = self._prev_bgr.copy()
            self._update(frame_bgr, kp, des)
            self.last_status = f"fallback (only {len(matches)} matches)"
            return warped, None, True

        pts_p = np.float32([self._prev_kp[m.queryIdx].pt
                            for m in matches]).reshape(-1, 1, 2)
        pts_c = np.float32([kp[m.trainIdx].pt
                            for m in matches]).reshape(-1, 1, 2)

        # ── RANSAC homography ─────────────────────────────────────────────────
        H, mask = cv2.findHomography(pts_p, pts_c, cv2.RANSAC, self.ransac_thr)
        n_in = int(mask.sum()) if mask is not None else 0
        self.last_n_inliers = n_in

        if H is None or n_in < self.min_inliers:
            warped = self._prev_bgr.copy()
            self._update(frame_bgr, kp, des)
            self.last_status = f"fallback (only {n_in} inliers)"
            return warped, None, True

        # ── Warp previous frame to current viewpoint ──────────────────────────
        warped = cv2.warpPerspective(
            self._prev_bgr, H, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE)

        self.last_H      = H
        self.last_status = f"ok  inliers={n_in}/{len(matches)}"
        self._update(frame_bgr, kp, des)
        return warped, H, True

    def debug_vis(self, frame_bgr):
        """
        Visualises RANSAC inliers (green = background) vs outliers (red = obstacle).
        Useful for summer-school slides explaining ego-motion compensation.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self._orb.detectAndCompute(gray, None)
        vis = frame_bgr.copy()

        if self._prev_kp and des is not None and self._prev_des is not None:
            matches = self._matcher.match(self._prev_des, des)
            if len(matches) >= self.min_matches:
                pts_p = np.float32([self._prev_kp[m.queryIdx].pt
                                    for m in matches]).reshape(-1, 1, 2)
                pts_c = np.float32([kp[m.trainIdx].pt
                                    for m in matches]).reshape(-1, 1, 2)
                _, mask = cv2.findHomography(pts_p, pts_c,
                                             cv2.RANSAC, self.ransac_thr)
                for i, m in enumerate(matches):
                    pt  = tuple(map(int, kp[m.trainIdx].pt))
                    col = (0, 255, 0) if (mask is not None and mask[i]) \
                          else (0, 0, 255)
                    cv2.circle(vis, pt, 5, col, -1)

        cv2.putText(vis, f"Ego: {self.last_status}",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(vis, "GREEN=background (inlier)  RED=obstacle (outlier)",
                    (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return vis

    def _update(self, bgr, kp, des):
        self._prev_bgr = bgr.copy()
        self._prev_kp  = kp
        self._prev_des = des


# ══════════════════════════════════════════════════════════════════════════════
#  2.  EVENT SIMULATOR  (ego-motion aware)
# ══════════════════════════════════════════════════════════════════════════════

class _EventSimulator:
    def __init__(self, threshold=0.15, blur_sigma=1.5,
                 moving_camera=True, compensator=None):
        self.C             = threshold
        self.sigma         = blur_sigma
        self.moving_camera = moving_camera
        self._comp         = compensator
        self._lp           = None
        self._prev_bgr     = None
        self._eps          = 1e-6

    def reset(self):
        self._lp       = None
        self._prev_bgr = None
        if self._comp:
            self._comp.reset()

    def process(self, frame_bgr):
        """
        Returns  (event_frame, ready, status_string)
        """
        h, w = frame_bgr.shape[:2]

        if self.moving_camera and self._comp is not None:
            prev_warped, _, ready = self._comp.compensate(frame_bgr)
            if not ready:
                return np.zeros((h, w, 3), np.uint8), False, "init"

            gc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gp = cv2.cvtColor(prev_warped, cv2.COLOR_BGR2GRAY)
            if self.sigma > 0:
                gc = cv2.GaussianBlur(gc, (0, 0), self.sigma)
                gp = cv2.GaussianBlur(gp, (0, 0), self.sigma)

            diff   = (np.log(gc.astype(np.float32) + self._eps) -
                      np.log(gp.astype(np.float32) + self._eps))
            status = self._comp.last_status

        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if self.sigma > 0:
                gray = cv2.GaussianBlur(gray, (0, 0), self.sigma)
            lc = np.log(gray.astype(np.float32) + self._eps)
            if self._lp is None:
                self._lp = lc
                return np.zeros((h, w, 3), np.uint8), False, "init"
            diff    = lc - self._lp
            self._lp = lc
            status  = "static"

        ef = np.zeros_like(frame_bgr)
        ef[diff >  self.C, 2] = 255   # R = ON
        ef[diff < -self.C, 0] = 255   # B = OFF
        return ef, True, status


# ══════════════════════════════════════════════════════════════════════════════
#  3.  SHADOW SUPPRESSOR
# ══════════════════════════════════════════════════════════════════════════════

class ShadowSuppressor:
    """
    Suppresses boat-shadow events from the binary event mask.

    Shadows have three exploitable properties that distinguish them from
    real obstacles:

    Property 1 – Brightness
        Shadow pixels are DARK in the current frame.
        A real moving obstacle (boat hull, buoy) is NOT dark — it has its
        own colour.  We reject events where the absolute luminance of the
        current frame is below `dark_thresh`.

    Property 2 – Edge sharpness (Laplacian variance)
        Real object edges are SHARP (high local contrast).
        Shadow edges (penumbra) are SOFT / gradual.
        We compute the local Laplacian magnitude at each event pixel.
        Events with low Laplacian response are likely shadow edges.

    Property 3 – Colour (HSV saturation)
        Shadows shift the colour toward blue/grey — low saturation.
        Real objects (painted hulls, life jackets, buoys) have colour.
        We optionally reject events in regions of very low saturation.
        Disable with sat_thresh=0 if your boats are also grey/white.

    Parameters
    ──────────
    dark_thresh  : int    0-255  pixels darker than this are candidates for
                                 shadow suppression  (default 60)
                                 raise if shadows are lighter (midday sun)
                                 lower if legitimate dark objects are suppressed
    sharp_thresh : float  local Laplacian variance below this = soft edge = shadow
                                 (default 20.0)
                                 lower = more aggressive shadow removal
                                 raise if real object edges are being removed
    sat_thresh   : int    0-255  HSV saturation below this = likely shadow
                                 set 0 to disable colour check  (default 30)
    """

    def __init__(self, dark_thresh=60, sharp_thresh=20.0, sat_thresh=30):
        self.dark_thresh  = dark_thresh
        self.sharp_thresh = sharp_thresh
        self.sat_thresh   = sat_thresh

    def apply(self, event_mask, frame_bgr):
        """
        Parameters
        ──────────
        event_mask : uint8 H×W  binary event mask (0/255)
        frame_bgr  : uint8 H×W×3  current RGB frame

        Returns
        ───────
        clean_mask  : uint8 H×W  shadow-suppressed mask
        shadow_mask : uint8 H×W  what was removed  (for debug)
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # ── Property 1: brightness mask ───────────────────────────────────────
        # Pixels that are event-active AND very dark → likely shadow
        dark_pixels = (gray < self.dark_thresh).astype(np.uint8) * 255

        # ── Property 2: edge sharpness (local Laplacian variance) ─────────────
        # Compute absolute Laplacian — high where edges are sharp
        laplacian  = cv2.Laplacian(gray, cv2.CV_32F)
        lap_abs    = np.abs(laplacian)
        # Smooth to get local sharpness estimate
        lap_smooth = cv2.GaussianBlur(lap_abs, (15, 15), 0)
        soft_edges = (lap_smooth < self.sharp_thresh).astype(np.uint8) * 255

        # ── Property 3: colour / saturation mask ──────────────────────────────
        if self.sat_thresh > 0:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            low_sat = (hsv[:, :, 1] < self.sat_thresh).astype(np.uint8) * 255
        else:
            low_sat = np.zeros_like(gray)

        # ── Combine: pixel flagged as shadow if dark AND (soft OR desaturated) ─
        shadow_candidate = cv2.bitwise_and(
            dark_pixels,
            cv2.bitwise_or(soft_edges, low_sat)
        )

        # ── Remove shadow candidates from event mask ──────────────────────────
        clean_mask  = cv2.bitwise_and(event_mask,
                                      cv2.bitwise_not(shadow_candidate))
        shadow_mask = cv2.bitwise_and(event_mask, shadow_candidate)

        return clean_mask, shadow_mask

    def debug_vis(self, frame_bgr, shadow_mask, clean_mask):
        """
        Colour overlay for summer-school explanation:
            YELLOW  = shadow events (suppressed)
            CYAN    = real events   (kept)
        """
        vis = frame_bgr.copy()
        vis[shadow_mask > 0] = [0,   220, 220]   # yellow
        vis[clean_mask  > 0] = [220, 220,   0]   # cyan
        cv2.putText(vis, "YELLOW=shadow suppressed  CYAN=real obstacle",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return vis


# ══════════════════════════════════════════════════════════════════════════════
#  4.  MORPHOLOGICAL FILTER
# ══════════════════════════════════════════════════════════════════════════════

class _EventMorphFilter:
    def __init__(self, open_k=3, close_k=25, dilate_k=7):
        self._ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k,  open_k))
        self._kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        self._kd = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
                    if dilate_k > 0 else None)

    def apply(self, mask):
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._ko)
        m = cv2.morphologyEx(m,    cv2.MORPH_CLOSE, self._kc)
        if self._kd is not None:
            m = cv2.dilate(m, self._kd, iterations=1)
        return m


# ══════════════════════════════════════════════════════════════════════════════
#  4.  BBOX MERGER
# ══════════════════════════════════════════════════════════════════════════════

class _BBoxMerger:
    def __init__(self, dist_px=60):
        self.dist_px = dist_px

    def merge(self, boxes):
        if not boxes:
            return []
        boxes = [list(b) for b in boxes]
        changed = True
        while changed:
            changed, merged, used = False, [], [False]*len(boxes)
            for i in range(len(boxes)):
                if used[i]: continue
                cur = boxes[i][:]
                for j in range(i+1, len(boxes)):
                    if used[j]: continue
                    if self._close(cur, boxes[j]):
                        cur = [min(cur[0],boxes[j][0]), min(cur[1],boxes[j][1]),
                               max(cur[2],boxes[j][2]), max(cur[3],boxes[j][3])]
                        used[j] = True; changed = True
                merged.append(cur); used[i] = True
            boxes = merged
        return [tuple(b) for b in boxes]

    def _close(self, a, b):
        dx = max(0, max(a[0],b[0]) - min(a[2],b[2]))
        dy = max(0, max(a[1],b[1]) - min(a[3],b[3]))
        return math.sqrt(dx*dx + dy*dy) < self.dist_px


# ══════════════════════════════════════════════════════════════════════════════
#  5.  MONOCULAR DISTANCE
# ══════════════════════════════════════════════════════════════════════════════

class _MonocularDistance:
    def __init__(self, focal_px=700., cam_height_m=0.5,
                 tilt_deg=5., img_h=720, known_obj_h=1.8):
        self.f     = focal_px
        self.cam_h = cam_height_m
        self.tilt  = math.radians(tilt_deg)
        self.cy    = img_h / 2.
        self.Hr    = known_obj_h

    def estimate(self, bbox):
        x1, y1, x2, y2 = bbox
        h_px = max(y2-y1, 1); w_px = max(x2-x1, 1)
        d_h  = (self.Hr * self.f) / h_px
        d_w  = (self.Hr * 2. * self.f) / w_px
        ang  = self.tilt + math.atan2(y2 - self.cy, self.f)
        d_g  = self.cam_h / math.tan(ang) if ang > 0.02 else d_h
        vals = [v for v in [d_h, d_w, d_g] if 0.1 < v < 2000]
        return round(float(np.median(vals)) if vals else d_h, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  6.  PUBLIC CLASS
# ══════════════════════════════════════════════════════════════════════════════

class EventVisionDetector:
    """
    Drop-in replacement for detect() with ego-motion compensation.

    moving_camera=True   → ASV is underway  (uses RANSAC homography)
    moving_camera=False  → stationary test  (original log-diff behaviour)

    Integration (unchanged from v1):
        from event_vision import EventVisionDetector
        self.event_detector = EventVisionDetector(self, moving_camera=True)
        message = self.event_detector.detect_bioinspired(img, lat, lon, alt, bearing)
    """

    def __init__(self, base,
                 moving_camera  = True,
                 event_thresh   = 0.15,
                 morph_close_k  = 25,
                 merge_dist_px  = 60,
                 roi_frac       = 0.20,
                 focal_px       = 700.0,
                 cam_height_m   = 0.5,
                 tilt_deg       = 5.0,
                 known_obj_h    = 1.8,
                 min_blob_area  = 400,
                 orb_features   = 1000,
                 ransac_thr     = 3.0,
                 min_inliers    = 15,
                 # Shadow suppression
                 dark_thresh    = 60,    # pixels darker than this → shadow candidate
                 sharp_thresh   = 20.0,  # Laplacian below this → soft edge → shadow
                 sat_thresh     = 30):   # HSV saturation below this → shadow (0=off)

        self.base          = base
        self.roi_frac      = roi_frac
        self.min_blob_area = min_blob_area
        self._focal_px     = focal_px
        self._cam_height_m = cam_height_m
        self._tilt_deg     = tilt_deg
        self._known_obj_h  = known_obj_h

        self._comp = EgoMotionCompensator(
            n_features=orb_features,
            ransac_thr=ransac_thr,
            min_inliers=min_inliers,
        ) if moving_camera else None

        self._sim    = _EventSimulator(threshold=event_thresh,
                                       moving_camera=moving_camera,
                                       compensator=self._comp)
        self._morph  = _EventMorphFilter(close_k=morph_close_k)
        self._merger = _BBoxMerger(dist_px=merge_dist_px)
        self._shadow = ShadowSuppressor(dark_thresh=dark_thresh,
                                        sharp_thresh=sharp_thresh,
                                        sat_thresh=sat_thresh)

    def detect_bioinspired(self, img,
                           camera_lat=37.7749, camera_lon=-122.4194,
                           camera_alt_m=0.3,   camera_bearing_deg=0.0):
        h, w  = img.shape[:2]
        roi_y = int(h * self.roi_frac)

        ev_frame, ready, ego_status = self._sim.process(img)
        if not ready:
            return self._empty(img, camera_lat, camera_lon)

        ev_mask = (np.max(ev_frame, axis=2) > 0).astype(np.uint8) * 255
        if roi_y > 0:
            ev_mask[:roi_y, :] = 0

        clean = self._morph.apply(ev_mask)

        # Step 3b – shadow suppression
        clean, _ = self._shadow.apply(clean, img)

        n, _, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
        raw_boxes = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < self.min_blob_area:
                continue
            bx=stats[i,cv2.CC_STAT_LEFT]; bw=stats[i,cv2.CC_STAT_WIDTH]
            by=stats[i,cv2.CC_STAT_TOP];  bh=stats[i,cv2.CC_STAT_HEIGHT]
            raw_boxes.append((bx, by, bx+bw, by+bh))

        merged = self._merger.merge(raw_boxes)

        dist_est   = _MonocularDistance(focal_px=self._focal_px,
                                        cam_height_m=self._cam_height_m,
                                        tilt_deg=self._tilt_deg,
                                        img_h=h,
                                        known_obj_h=self._known_obj_h)
        b          = self.base
        detections = []
        annotated  = ev_frame.copy()

        for bbox in merged:
            x1, y1, x2, y2 = bbox
            if y2 <= roi_y:
                continue
            dist = dist_est.estimate(bbox)
            lat, lon, dist_gps, az = b.calculate_object_gps_from_bbox(
                bbox, camera_lat, camera_lon, camera_alt_m, camera_bearing_deg)

            cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,255),2)
            lbl = f"{dist:.1f}m"
            (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated,(x1,y1-th-6),(x1+tw+4,y1),(0,0,0),-1)
            cv2.putText(annotated,lbl,(x1+2,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),1)

            detections.append({
                "asv":             b.role,
                "mmsi":            b.status.get('mmsi'),
                "mission_uuid":    b.mission_uuid,
                "class":           "obstacle",
                "bbox":            bbox,
                "sensor_lon":      camera_lon,
                "sensor_lat":      camera_lat,
                "lon":             lon,
                "lat":             lat,
                "t":               datetime.now().isoformat(),
                "distance":        dist_gps,
                "distance_mono_m": dist,
                "azimuth":         az,
                "confidence":      1.0,
                "source":          "event_camera",
                "ego_status":      ego_status,
            })

        cv2.putText(annotated, f"ego: {ego_status}",
                    (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,100), 1)

        if b.video_write_on and b.writer is not None:
            b.writer.write(annotated)

        _, buf  = cv2.imencode('.jpg', annotated)
        payload = base64.b64encode(buf).decode('ascii')
        sdet    = sorted(detections,
                         key=lambda d: d.get("distance_mono_m", float('inf')))
        return {
            "asv": b.role, "lon": camera_lon, "lat": camera_lat,
            "t":   datetime.now().isoformat(),
            "detections": sdet, "source": "event_camera", "payload": payload,
        }

    def reset(self):
        self._sim.reset()

    def _empty(self, img, lat, lon):
        _, buf = cv2.imencode('.jpg', img)
        return {
            "asv": self.base.role, "lon": lon, "lat": lat,
            "t":   datetime.now().isoformat(),
            "detections": [], "source": "event_camera",
            "payload": base64.b64encode(buf).decode('ascii'),
        }
