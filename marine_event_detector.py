"""
marine_event_detector.py
────────────────────────────────────────────────────────────────────────────
Event-Only Marine Obstacle Detector — standalone viewer / batch processor.

Changes from previous version
──────────────────────────────
• EventSimulator: LUT-based log (no float32 per frame)
• EventSimulator: uses green channel instead of full BGR→Gray
• SeaSuppressor added (imported from event_vision.py)
• process_frame: pipeline order fixed →
      ev_mask → SeaSuppressor (raw sparse) → Morph → Shadow → Blobs
• 'T' key toggles the sea-vector debug window
• ShadowSuppressor no longer re-created every frame

Usage
─────
    python marine_event_detector.py --frames /path/to/images
    python marine_event_detector.py --video  /path/to/video.mp4
    python marine_event_detector.py --frames /path/to/images --auto --save out/
"""

import cv2
import numpy as np
import argparse
import os
import sys
import glob
import math

from event_vision import TemporalEventFilter, ShadowSuppressor as _SS


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT = dict(
    # ── EventSimulator ────────────────────────────────────────────────────────
    event_thresh  = 0.25,
    blur_sigma    = 1.5,

    # ── ROI ───────────────────────────────────────────────────────────────────
    roi_frac      = 0.20,

    # ── Sea suppression ───────────────────────────────────────────────────────
    sea_coherence = 0.50,   # structure-tensor coherence threshold (0–1)
    sea_magnitude = 2.0,    # minimum gradient magnitude to consider

    # ── Morphological cleanup ─────────────────────────────────────────────────
    morph_open_k  = 3,
    morph_close_k = 25,
    dilate_k      = 7,

    # ── Blob filter ───────────────────────────────────────────────────────────
    min_blob_area = 400,

    # ── BBox merger ───────────────────────────────────────────────────────────
    merge_iou     = 0.0,
    merge_dist_px = 60,

    # ── Shadow suppression ────────────────────────────────────────────────────
    dark_thresh   = 60,
    sharp_thresh  = 20.0,
    sat_thresh    = 30,

    # ── Distance estimation ───────────────────────────────────────────────────
    focal_px      = 700.0,
    cam_height_m  = 0.5,
    tilt_deg      = 5.0,
    known_obj_h   = 1.8,
)

PANEL_W = 640
PANEL_H = 360

COL_ON     = (0,   0,   255)
COL_OFF    = (255, 0,   0  )
COL_BOX    = (0,   255, 255)
COL_LABEL  = (0,   255, 255)
COL_DANGER = (0,   0,   255)
COL_OK     = (0,   200, 0  )
COL_WARN   = (0,   165, 255)


# ══════════════════════════════════════════════════════════════════════════════
#  1.  EVENT SIMULATOR  (LUT-based log, green channel)
# ══════════════════════════════════════════════════════════════════════════════

class EventSimulator:
    """
    Converts an RGB frame into a fake DVS event frame.

    Uses a pre-computed 256-entry LUT for log(x) — no float32 allocation
    or transcendental computation per frame.
    Operates on the green channel (index 1 in BGR) instead of full grayscale.

    Output
    ------
    event_frame  H×W×3 uint8   R=ON events, B=OFF events
    stats        dict   {on, off, total, sparsity_pct}
    ready        bool   False on the very first call
    """

    def __init__(self, threshold=0.25, blur_sigma=1.5):
        self.C     = threshold
        self.sigma = blur_sigma
        self._lp   = None
        self._eps  = 1e-6

        # ── Pre-compute log LUT ───────────────────────────────────────────────
        x   = np.arange(256, dtype=np.float32)
        raw = np.log(x + self._eps)
        self._log_min   = float(raw.min())
        self._log_scale = 255.0 / float(raw.max() - raw.min())
        self._lut       = np.clip(
            np.round((raw - self._log_min) * self._log_scale), 0, 255
        ).astype(np.uint8)
        self._C_lut = max(1, int(round(threshold * self._log_scale)))

    def reset(self):
        self._lp = None

    def process(self, frame_bgr):
        # Green channel — zero-copy view, no cvtColor
        gray = frame_bgr[:, :, 1]
        if self.sigma > 0:
            gray = cv2.GaussianBlur(gray, (0, 0), self.sigma)

        lc = cv2.LUT(gray, self._lut)   # uint8 LUT lookup, SIMD-friendly

        if self._lp is None:
            self._lp = lc.copy()
            h, w = frame_bgr.shape[:2]
            return (np.zeros((h, w, 3), np.uint8),
                    {'on': 0, 'off': 0, 'total': 0, 'sparsity_pct': 100.0},
                    False)

        diff  = lc.astype(np.int16) - self._lp.astype(np.int16)
        on_m  = diff >  self._C_lut
        off_m = diff < -self._C_lut

        ef = np.zeros_like(frame_bgr)
        ef[on_m,  2] = 255
        ef[off_m, 0] = 255

        n_on  = int(on_m.sum())
        n_off = int(off_m.sum())
        total = n_on + n_off
        px    = frame_bgr.shape[0] * frame_bgr.shape[1]
        self._lp = lc

        return (ef,
                {'on': n_on, 'off': n_off, 'total': total,
                 'sparsity_pct': round(100.0 * (1 - total / px), 1)},
                True)


# ══════════════════════════════════════════════════════════════════════════════
#  2.  MORPHOLOGICAL FILTER
# ══════════════════════════════════════════════════════════════════════════════

class EventMorphFilter:
    def __init__(self, open_k=3, close_k=25, dilate_k=7):
        self._ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k,  open_k))
        self._kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        self._kd = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
                    if dilate_k > 0 else None)

    def rebuild(self, open_k, close_k, dilate_k):
        self.__init__(open_k, close_k, dilate_k)

    def apply(self, raw_mask):
        filt = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  self._ko)
        filt = cv2.morphologyEx(filt,     cv2.MORPH_CLOSE, self._kc)
        if self._kd is not None:
            filt = cv2.dilate(filt, self._kd, iterations=1)
        return filt


# ══════════════════════════════════════════════════════════════════════════════
#  3.  BBOX MERGER
# ══════════════════════════════════════════════════════════════════════════════

class BBoxMerger:
    def __init__(self, iou_thresh=0.0, dist_px=60):
        self.iou_thresh = iou_thresh
        self.dist_px    = dist_px

    def merge(self, boxes):
        if not boxes:
            return []
        boxes = [list(b) for b in boxes]
        changed = True
        while changed:
            changed = False
            merged  = []
            used    = [False] * len(boxes)
            for i in range(len(boxes)):
                if used[i]:
                    continue
                cur = boxes[i][:]
                for j in range(i + 1, len(boxes)):
                    if used[j]:
                        continue
                    if self._should_merge(cur, boxes[j]):
                        cur = self._union(cur, boxes[j])
                        used[j] = True
                        changed = True
                merged.append(cur)
                used[i] = True
            boxes = merged
        return [tuple(b) for b in boxes]

    def _should_merge(self, a, b):
        xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
        xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        if inter > 0:
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            iou    = inter / (area_a + area_b - inter + 1e-6)
            if iou > self.iou_thresh:
                return True
        dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
        dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
        return (dx * dx + dy * dy) ** 0.5 < self.dist_px

    @staticmethod
    def _union(a, b):
        return [min(a[0], b[0]), min(a[1], b[1]),
                max(a[2], b[2]), max(a[3], b[3])]


# ══════════════════════════════════════════════════════════════════════════════
#  4.  MONOCULAR DISTANCE
# ══════════════════════════════════════════════════════════════════════════════

class MonocularDistance:
    def __init__(self, focal_px=700.0, cam_height_m=0.5,
                 tilt_deg=5.0, img_h=720, known_obj_h=1.8):
        self.f      = focal_px
        self.cam_h  = cam_height_m
        self.tilt   = np.deg2rad(tilt_deg)
        self.cy     = img_h / 2.0
        self.H_real = known_obj_h

    def estimate(self, bbox):
        x1, y1, x2, y2 = bbox
        h_px = max(y2 - y1, 1)
        w_px = max(x2 - x1, 1)
        d_h  = (self.H_real * self.f) / h_px
        d_w  = (self.H_real * 2.0 * self.f) / w_px
        delta = np.arctan2(y2 - self.cy, self.f)
        angle = self.tilt + delta
        d_g   = self.cam_h / np.tan(angle) if angle > 0.02 else d_h
        vals  = [v for v in [d_h, d_w, d_g] if 0.1 < v < 2000]
        dist  = round(float(np.median(vals)) if vals else d_h, 1)
        return dist, {'height': round(d_h, 1), 'width': round(d_w, 1),
                      'ground': round(d_g, 1) if angle > 0.02 else None}


# ══════════════════════════════════════════════════════════════════════════════
#  5.  FRAME SOURCE
# ══════════════════════════════════════════════════════════════════════════════

class FrameSource:
    def __init__(self, path):
        if os.path.isfile(path):
            cap = cv2.VideoCapture(path)
            self._frames = []
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                self._frames.append(f)
            cap.release()
            print(f"Loaded {len(self._frames)} frames from video.")
        elif os.path.isdir(path):
            exts  = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif',
                     '*.JPG', '*.JPEG', '*.PNG']
            files = sorted({f for e in exts
                            for f in glob.glob(os.path.join(path, e))})
            if not files:
                sys.exit(f"No image files in {path}")
            self._frames = [cv2.imread(f) for f in files]
            self._frames = [f for f in self._frames if f is not None]
            print(f"Loaded {len(self._frames)} images.")
        else:
            sys.exit(f"Not found: {path}")
        if not self._frames:
            sys.exit("No frames loaded.")
        self.idx = 0

    def __len__(self):  return len(self._frames)
    def current(self):  return self._frames[self.idx].copy()
    def next(self):     self.idx = min(self.idx + 1, len(self._frames) - 1)
    def prev(self):     self.idx = max(self.idx - 1, 0)


# ══════════════════════════════════════════════════════════════════════════════
#  6.  DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def sector_of(cx, w):
    return 'L' if cx < w // 3 else ('R' if cx > 2 * w // 3 else 'C')

def sector_colour(dist):
    if dist is None: return COL_OK
    if dist < 5:     return COL_DANGER
    if dist < 15:    return COL_WARN
    return COL_OK

def draw_detections(frame, detections, img_h, img_w):
    out     = frame.copy()
    nearest = {'L': None, 'C': None, 'R': None}
    for det in detections:
        s = det['sector']
        d = det['dist_m']
        if nearest[s] is None or d < nearest[s]:
            nearest[s] = d

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        dist  = det['dist_m']
        label = f"{dist:.1f}m  [{det['sector']}]"
        cv2.rectangle(out, (x1, y1), (x2, y2), COL_BOX, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_LABEL, 1)

    bar_h   = 30
    overlay = out.copy()
    for i, s in enumerate(['L', 'C', 'R']):
        x0, x1b = i * img_w // 3, (i + 1) * img_w // 3
        d        = nearest[s]
        col      = sector_colour(d)
        txt      = f"{s}: {d:.1f}m" if d is not None else f"{s}: --"
        cv2.rectangle(overlay, (x0, img_h - bar_h), (x1b, img_h), col, -1)
        cv2.putText(overlay, txt, (x0 + 6, img_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    return out, nearest


def make_display(frame, ev_frame, clean_mask, shadow_vis,
                 detections, ev_stats, p, show_debug=False):
    h, w      = frame.shape[:2]
    dist_est  = MonocularDistance(focal_px=p['focal_px'],
                                  cam_height_m=p['cam_height_m'],
                                  tilt_deg=p['tilt_deg'],
                                  img_h=h,
                                  known_obj_h=p['known_obj_h'])
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        det['sector'] = sector_of((x1 + x2) // 2, w)

    det_frame, nearest = draw_detections(frame, detections, h, w)

    def th(img, label, sub='', col=(220, 220, 220)):
        t = cv2.resize(img, (PANEL_W, PANEL_H))
        cv2.rectangle(t, (0, 0), (PANEL_W, 32), (0, 0, 0), -1)
        cv2.putText(t, label, (6, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
        if sub:
            cv2.putText(t, sub, (6, PANEL_H - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)
        return t

    p1 = th(ev_frame,
            f'Events  thresh={p["event_thresh"]:.3f}',
            f'ON={ev_stats["on"]}  OFF={ev_stats["off"]}  '
            f'sparsity={ev_stats["sparsity_pct"]}%',
            col=(100, 100, 255))

    if show_debug:
        p2 = th(shadow_vis,
                'Shadow suppression debug',
                'YELLOW=shadow removed  CYAN=real obstacle kept',
                col=(180, 180, 255))
    else:
        orient_vis = draw_orientation_field(clean_mask, block=8, scale=3)
        p2 = th(orient_vis,
                'Orientation field  (8-dir per 8×8 block)',
                f'GREEN=boat  RED=sea  '
                f'close={p["morph_close_k"]} merge_px={p["merge_dist_px"]}',
                col=(150, 255, 150))

    n   = len(detections)
    sec = ' | '.join(f'{s}:{nearest[s]:.1f}m' if nearest[s] else f'{s}:--'
                     for s in ['L', 'C', 'R'])
    p3  = th(det_frame,
             f'Detections  [{n} objects]',
             sec, col=(0, 255, 255))

    div = np.full((PANEL_H, 4, 3), 50, np.uint8)
    row = np.hstack([p1, div, p2, div, p3])

    hdr = np.zeros((38, row.shape[1], 3), np.uint8)
    cv2.putText(hdr,
        'Event-Only Marine Detector  |  '
        '[+/-]=thresh  [V/B]=close_k  [M/N]=merge_dist  '
        '[T]=sea vectors  [D]=shadow debug  [S]=save  [Q]=quit',
        (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 50), 1)

    return np.vstack([hdr, row])


# ══════════════════════════════════════════════════════════════════════════════
#  7.  CORE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_frame(frame, sim, temp_filter, morph, merger, shadow, p, show_debug_window=False):
    """
    Pipeline
    --------
    ev_mask  →  SeaSuppressor  →  MorphFilter  →  ShadowSuppressor  →  Blobs
    """
    h, w  = frame.shape[:2]
    roi_y = int(h * p['roi_frac'])

    ev_frame, ev_stats, ready = sim.process(frame)
    if not ready:
        blank = np.zeros((h, w, 3), np.uint8)
        return blank, np.zeros((h, w), np.uint8), blank, blank, [], ev_stats

    ev_mask = (np.max(ev_frame, axis=2) > 0).astype(np.uint8) * 255
    if roi_y > 0:
        ev_mask[:roi_y, :] = 0
    
    # ── Step 1: Temporal Sea Suppression ──────────────────────────────────────
    clean_mask = temp_filter.apply(ev_mask)

    # ── Step 2: Morphological cleanup ─────────────────────────────────────────
    clean_mask = morph.apply(clean_mask)


    # ── Step 3: Shadow suppression ────────────────────────────────────────────
    clean_mask, shadow_removed = shadow.apply(clean_mask, frame)
    shadow_vis = shadow.debug_vis(frame, shadow_removed, clean_mask)

    # ── Step 4: Blob detection + merge ────────────────────────────────────────
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        clean_mask, connectivity=8)
    raw_boxes = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < p['min_blob_area']:
            continue
        bx = stats[i, cv2.CC_STAT_LEFT];  bw = stats[i, cv2.CC_STAT_WIDTH]
        by = stats[i, cv2.CC_STAT_TOP];   bh = stats[i, cv2.CC_STAT_HEIGHT]
        raw_boxes.append((bx, by, bx + bw, by + bh))

    merged_boxes = merger.merge(raw_boxes)

    dist_est   = MonocularDistance(focal_px=p['focal_px'],
                                   cam_height_m=p['cam_height_m'],
                                   tilt_deg=p['tilt_deg'],
                                   img_h=h,
                                   known_obj_h=p['known_obj_h'])
    detections = []
    for bbox in merged_boxes:
        x1, y1, x2, y2 = bbox
        if y2 <= roi_y:
            continue
        dist, _ = dist_est.estimate(bbox)
        detections.append({
            'bbox':   bbox,
            'dist_m': dist,
            'sector': sector_of((x1 + x2) // 2, w),
        })

    detections.sort(key=lambda d: d['dist_m'])
    return ev_frame, clean_mask, shadow_vis, shadow_removed, detections, ev_stats

def draw_orientation_field(event_mask, block=8, scale=3):
    """
    For every block×block region of the event mask:
      - Compute gradient magnitude + direction
      - Bin into 8 directions (0°,45°,90°,135°,180°,225°,270°,315°)
      - Draw 8 spokes from block centre, length ∝ energy in that direction

    Colour encodes isotropy:
        GREEN  → one dominant direction  (boat edge)
        RED    → energy spread all ways  (sea / noise)
    """
    vis = cv2.cvtColor(event_mask, cv2.COLOR_GRAY2BGR)

    Ix  = cv2.Sobel(event_mask, cv2.CV_32F, 1, 0, ksize=3)
    Iy  = cv2.Sobel(event_mask, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(Ix ** 2 + Iy ** 2)
    ang = np.arctan2(Iy, Ix)               # −π … +π

    # 8 unit vectors for the spoke tips
    spoke_dirs = np.array([
        (math.cos(i * math.pi / 4), math.sin(i * math.pi / 4))
        for i in range(8)
    ])

    h, w = event_mask.shape[:2]

    for by in range(0, h - block + 1, block):
        for bx in range(0, w - block + 1, block):
            m = mag[by:by+block, bx:bx+block]
            a = ang[by:by+block, bx:bx+block]

            total = float(m.sum())
            if total < 1.0:
                continue                   # empty block, skip

            cx = bx + block // 2
            cy = by + block // 2

            # ── 8-bin histogram weighted by gradient magnitude ────────────
            bin_idx = (((a + math.pi) / (2 * math.pi)) * 8
                       ).astype(np.int32) % 8   # 0..7
            bins = np.zeros(8, dtype=np.float32)
            for k in range(8):
                bins[k] = m[bin_idx == k].sum()

            bins /= (bins.max() + 1e-9)          # normalise to [0, 1]

            # ── Isotropy → colour (green = peaked, red = spread) ──────────
            entropy   = float(-np.sum(
                bins * np.log2(bins + 1e-9))) / 3.0  # 3 = log2(8), max entropy
            entropy   = max(0.0, min(1.0, entropy))
            col_green = int(255 * (1.0 - entropy))
            col_red   = int(255 * entropy)
            colour    = (0, col_green, col_red)   # BGR

            # ── Draw 8 spokes ─────────────────────────────────────────────
            for k in range(8):
                if bins[k] < 0.05:
                    continue
                dx, dy   = spoke_dirs[k]
                length   = bins[k] * scale * (block // 2)
                ex = int(cx + dx * length)
                ey = int(cy + dy * length)
                cv2.line(vis, (cx, cy), (ex, ey), colour, 1)

            # Small dot at block centre
            cv2.circle(vis, (cx, cy), 1, (180, 180, 180), -1)

    cv2.putText(vis,
        "GREEN=dominant direction (boat)  RED=isotropic (sea)",
        (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
    return vis
# ══════════════════════════════════════════════════════════════════════════════
#  8.  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    p = DEFAULT.copy()
    if args.thresh   is not None: p['event_thresh']  = args.thresh
    if args.roi_frac is not None: p['roi_frac']      = args.roi_frac
    if args.focal    is not None: p['focal_px']      = args.focal
    if args.cam_h    is not None: p['cam_height_m']  = args.cam_h
    if args.tilt     is not None: p['tilt_deg']      = args.tilt
    if args.min_area is not None: p['min_blob_area'] = args.min_area
    if args.close_k  is not None: p['morph_close_k'] = args.close_k
    if args.merge_px is not None: p['merge_dist_px'] = args.merge_px

    src    = FrameSource(args.frames or args.video)
    sim    = EventSimulator(threshold=p['event_thresh'],
                            blur_sigma=p['blur_sigma'])


    temp_filter = TemporalEventFilter(window_size=3, threshold=2)
    morph  = EventMorphFilter(open_k=p['morph_open_k'],
                              close_k=p['morph_close_k'],
                              dilate_k=p['dilate_k'])
    merger = BBoxMerger(iou_thresh=p['merge_iou'],
                        dist_px=p['merge_dist_px'])
    shadow = _SS(dark_thresh=p['dark_thresh'],
                 sharp_thresh=p['sharp_thresh'],
                 sat_thresh=p['sat_thresh'])

    save_dir        = args.save
    show_debug      = False
    show_sea_vectors = False
    cache            = {}
    prev_idx         = -1

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    cv2.namedWindow('Event Detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Event Detector', PANEL_W * 3 + 12, PANEL_H + 38 + 20)

    print("\n── Event-Only Marine Detector ────────────────────────────────────")
    print(f"   Frames        : {len(src)}")
    print(f"   event_thresh  : {p['event_thresh']}   (tune: +/-)")
    print(f"   sea_coherence : {p['sea_coherence']}   (structure-tensor)")
    print(f"   morph_close_k : {p['morph_close_k']}     (tune: V/B)")
    print(f"   merge_dist_px : {p['merge_dist_px']}    (tune: M/N key)")
    print(f"   min_blob_area : {p['min_blob_area']} px²")
    print(f"   [T] key       : toggle sea-vector debug window")
    print("──────────────────────────────────────────────────────────────────\n")

    last_panel = None

    while True:
        idx   = src.idx
        frame = src.current()

        if idx not in cache or idx != prev_idx:
            if idx < prev_idx or idx > prev_idx + 1:
                sim.reset()
                temp_filter.reset()
                for wi in range(max(0, idx - 1), idx + 1):
                    process_frame(src._frames[wi], sim, temp_filter, morph, merger, shadow, p)

            res = process_frame(frame, sim, temp_filter, morph, merger, shadow, p, show_sea_vectors)
            cache    = {idx: res}
            prev_idx = idx

            ev_frame, clean_mask, shadow_vis, shadow_removed, detections, ev_stats = res
            print(f"  Frame {idx+1:03d}/{len(src)}  "
                  f"events={ev_stats['total']:7d}  "
                  f"sparsity={ev_stats['sparsity_pct']:5.1f}%  "
                  f"objects={len(detections)}  "
                  + ('  '.join(f"{d['sector']}:{d['dist_m']}m"
                               for d in detections[:5])))
        else:
            ev_frame, clean_mask, shadow_vis, shadow_removed, detections, ev_stats = cache[idx]

        panel = make_display(frame, ev_frame, clean_mask, shadow_vis,
                             detections, ev_stats, p, show_debug)
        last_panel = panel
        cv2.imshow('Event Detector', panel)

        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key in (ord(' '), ord('n')):
            src.next()
        elif key == ord('p'):
            src.prev()
        elif key in (ord('+'), ord('=')):
            p['event_thresh'] = min(p['event_thresh'] + 0.01, 0.50)
            sim.reset(); cache.clear(); prev_idx = -1
            print(f"  event_thresh → {p['event_thresh']:.3f}")
        elif key == ord('-'):
            p['event_thresh'] = max(p['event_thresh'] - 0.01, 0.01)
            sim.reset(); cache.clear(); prev_idx = -1
            print(f"  event_thresh → {p['event_thresh']:.3f}")
        elif key == ord('v'):
            p['morph_close_k'] = min(p['morph_close_k'] + 2, 71)
            morph.rebuild(p['morph_open_k'], p['morph_close_k'], p['dilate_k'])
            cache.clear(); prev_idx = -1
            print(f"  morph_close_k → {p['morph_close_k']}")
        elif key == ord('b'):
            p['morph_close_k'] = max(p['morph_close_k'] - 2, 3)
            morph.rebuild(p['morph_open_k'], p['morph_close_k'], p['dilate_k'])
            cache.clear(); prev_idx = -1
            print(f"  morph_close_k → {p['morph_close_k']}")
        elif key == ord('m'):
            p['merge_dist_px'] = min(p['merge_dist_px'] + 10, 300)
            merger = BBoxMerger(iou_thresh=p['merge_iou'],
                                dist_px=p['merge_dist_px'])
            cache.clear(); prev_idx = -1
            print(f"  merge_dist_px → {p['merge_dist_px']}")
        elif key == ord('n'):
            p['merge_dist_px'] = max(p['merge_dist_px'] - 10, 0)
            merger = BBoxMerger(iou_thresh=p['merge_iou'],
                                dist_px=p['merge_dist_px'])
            cache.clear(); prev_idx = -1
            print(f"  merge_dist_px → {p['merge_dist_px']}")
        elif key == ord('1'):
            p['dark_thresh'] = min(p['dark_thresh'] + 5, 200)
            shadow = _SS(dark_thresh=p['dark_thresh'],
                         sharp_thresh=p['sharp_thresh'],
                         sat_thresh=p['sat_thresh'])
            cache.clear(); prev_idx = -1
            print(f"  dark_thresh → {p['dark_thresh']}")
        elif key == ord('2'):
            p['dark_thresh'] = max(p['dark_thresh'] - 5, 10)
            shadow = _SS(dark_thresh=p['dark_thresh'],
                         sharp_thresh=p['sharp_thresh'],
                         sat_thresh=p['sat_thresh'])
            cache.clear(); prev_idx = -1
            print(f"  dark_thresh → {p['dark_thresh']}")
        elif key == ord('3'):
            p['sharp_thresh'] = min(p['sharp_thresh'] + 2.0, 100.0)
            shadow = _SS(dark_thresh=p['dark_thresh'],
                         sharp_thresh=p['sharp_thresh'],
                         sat_thresh=p['sat_thresh'])
            cache.clear(); prev_idx = -1
            print(f"  sharp_thresh → {p['sharp_thresh']}")
        elif key == ord('4'):
            p['sharp_thresh'] = max(p['sharp_thresh'] - 2.0, 2.0)
            shadow = _SS(dark_thresh=p['dark_thresh'],
                         sharp_thresh=p['sharp_thresh'],
                         sat_thresh=p['sat_thresh'])
            cache.clear(); prev_idx = -1
            print(f"  sharp_thresh → {p['sharp_thresh']}")
        elif key == ord('t'):
            # ── Toggle sea-vector debug window ────────────────────────────────
            show_sea_vectors = not show_sea_vectors
            if not show_sea_vectors:
                cv2.destroyWindow("Sea suppressor vectors")
            cache.clear(); prev_idx = -1
            print(f"  sea vectors → {'ON' if show_sea_vectors else 'OFF'}")
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('r'):
            sim.reset(); cache.clear(); prev_idx = -1
            print("  Reset.")
        elif key == ord('s'):
            if last_panel is not None:
                fname = (os.path.join(save_dir, f'frame_{idx+1:04d}.png')
                         if save_dir else f'frame_{idx+1:04d}.png')
                cv2.imwrite(fname, last_panel)
                print(f"  Saved → {fname}")

        if args.auto:
            src.next()
            if src.idx == len(src) - 1:
                break

    cv2.destroyAllWindows()

    if save_dir and not args.auto:
        print(f"\nBatch saving all frames → {save_dir}")
        sim.reset()
        for i in range(len(src)):
            src.idx = i
            if i == 0:
                sim.reset()
            res = process_frame(src.current(), sim, sea, morph,
                                merger, shadow, p)
            ev_frame, clean_mask, shadow_vis, _, detections, ev_stats = res
            panel = make_display(src.current(), ev_frame, clean_mask,
                                 shadow_vis, detections, ev_stats, p)
            cv2.imwrite(os.path.join(save_dir, f'frame_{i+1:04d}.png'), panel)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(src)}")
        print("Done.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Event-Only Marine Obstacle Detector',
        formatter_class=argparse.RawTextHelpFormatter)

    src_g = ap.add_mutually_exclusive_group(required=True)
    src_g.add_argument('--frames', metavar='DIR',  help='Folder of images')
    src_g.add_argument('--video',  metavar='FILE', help='Video file')

    ap.add_argument('--save',     metavar='DIR',   default=None)
    ap.add_argument('--thresh',   type=float,      default=None,
                    help=f'Event threshold (default {DEFAULT["event_thresh"]})')
    ap.add_argument('--close-k',  type=int,        default=None, dest='close_k',
                    help=f'Morph close kernel (default {DEFAULT["morph_close_k"]})')
    ap.add_argument('--merge-px', type=int,        default=None, dest='merge_px',
                    help=f'Merge distance px (default {DEFAULT["merge_dist_px"]})')
    ap.add_argument('--roi-frac', type=float,      default=None, dest='roi_frac',
                    help=f'Top fraction to ignore (default {DEFAULT["roi_frac"]})')
    ap.add_argument('--focal',    type=float,      default=None,
                    help=f'Focal length px (default {DEFAULT["focal_px"]})')
    ap.add_argument('--cam-h',    type=float,      default=None, dest='cam_h',
                    help=f'Camera height m (default {DEFAULT["cam_height_m"]})')
    ap.add_argument('--tilt',     type=float,      default=None,
                    help=f'Downward tilt degrees (default {DEFAULT["tilt_deg"]})')
    ap.add_argument('--min-area', type=int,        default=None, dest='min_area',
                    help=f'Min blob area px² (default {DEFAULT["min_blob_area"]})')
    ap.add_argument('--auto',     action='store_true',
                    help='Auto-advance (batch mode)')

    run(ap.parse_args())
