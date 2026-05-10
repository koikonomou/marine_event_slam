# Bio-Inspired Event Vision — Integration Guide
## What you received

| File | What it is |
|---|---|
| `event_vision.py` | The bio-inspired module. Drop next to `detect.py`. |
| `INTEGRATION.md` | This file. |

---

## What to install

```bash
pip install opencv-python numpy
```
Nothing else. No YOLO, no NCNN, no extra models.

---

## How to integrate (3 changes to detect.py)

### Change 1 — top of detect.py, add one import

```python
from event_vision import EventVisionDetector
```

---

### Change 2 — inside your class `__init__`, add one line

```python
def __init__(self, ...):
    # ... your existing code ...

    # ← ADD THIS (one line)
    self.event_detector = EventVisionDetector(
        self,
        event_thresh  = 0.25,   # tune: raise to suppress water noise
        morph_close_k = 25,     # tune: raise if boat appears fragmented
        merge_dist_px = 60,     # tune: raise if hull/mast appear separate
        roi_frac      = 0.20,   # ignore top 20% of frame (sky/wall)
        focal_px      = 700.0,  # your camera focal length in pixels
        cam_height_m  = 0.5,    # camera height above waterline (metres)
        tilt_deg      = 5.0,    # camera downward tilt (degrees)
    )
```

---

### Change 3 — call it wherever you call detect()

```python
# BEFORE (classic RGB):
message = self.detect(img, camera_lat, camera_lon, camera_alt_m, camera_bearing_deg)

# AFTER (bio-inspired events) — same arguments, same returned dict:
message = self.event_detector.detect_bioinspired(img, camera_lat, camera_lon,
                                                  camera_alt_m, camera_bearing_deg)

# Or run BOTH and compare:
classic_msg = self.detect(img, ...)
bio_msg     = self.event_detector.detect_bioinspired(img, ...)
```

---

## What comes back

The returned `message` dict is **identical** to `detect()`:

```python
{
    "asv":        ...,
    "lon":        camera_lon,
    "lat":        camera_lat,
    "t":          "2025-05-10T12:34:56.789",
    "source":     "event_camera",      # ← only difference from "camera"
    "payload":    "<base64 jpg>",      # annotated event image
    "detections": [
        {
            "asv":             ...,
            "mmsi":            ...,
            "mission_uuid":    ...,
            "class":           "obstacle",
            "bbox":            (x1, y1, x2, y2),
            "sensor_lon":      ...,
            "sensor_lat":      ...,
            "lon":             ...,
            "lat":             ...,
            "t":               ...,
            "distance":        ...,      # GPS-derived (same as classic)
            "distance_mono_m": 12.3,     # monocular camera estimate (NEW)
            "azimuth":         ...,
            "confidence":      1.0,
            "source":          "event_camera",
        },
        ...   # sorted by distance_mono_m, closest first
    ]
}
```

---

## Tuning parameters

Run `marine_event_detector.py` (the standalone test tool) on your frames
to find the right values interactively, then copy them into `__init__`:

| Parameter | Key in test tool | Effect |
|---|---|---|
| `event_thresh` | `+` / `-` | **Main dial.** Raise to suppress water noise (try 0.20–0.35). |
| `morph_close_k` | `V` / `B` | Raise if the boat splits into many fragments. |
| `merge_dist_px` | `M` / `N` | Raise if hull + mast appear as separate detections. |
| `roi_frac` | — | Set to the fraction of your frame that is sky/wall/horizon. |
| `focal_px` | — | Compute: `f = (img_width / 2) / tan(horizontal_FOV_radians / 2)` |
| `cam_height_m` | — | Measure from camera lens to waterline. |

---

## If the stream has gaps or restarts

```python
self.event_detector.reset()   # clears internal frame memory
```

---

## How it works (one paragraph for the report)

Instead of detecting objects in raw RGB frames, the bio-inspired approach
simulates a Dynamic Vision Sensor (DVS). For each consecutive pair of frames
the per-pixel log-luminance change is computed. Pixels where the change
exceeds threshold C fire an "event" (red = brightness increase, blue =
decrease). Because water ripples produce slow, small intensity changes
they do not cross the threshold, moving boats and people produce large,
fast changes and do. The resulting sparse event map is cleaned with
morphological open+close operations and nearby blobs are merged into one
bounding box per obstacle. Metric distance is estimated from the bounding
box geometry using a pinhole camera model fused with a ground-plane
projection.
