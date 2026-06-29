import argparse
import math
import os
import cv2
import numpy as np


def extract_affine_params(matrix):
    """Extracts translation dx, dy (pixels) and rotation (degrees) from an affine matrix."""
    if matrix is None:
        return 0.0, 0.0, 0.0
    dx = float(matrix[0, 2])
    dy = float(matrix[1, 2])

    # Isolate rotation safely using atan2 on the matrix scale parameters
    rotation_rad = math.atan2(matrix[1, 0], matrix[0, 0])
    rotation_deg = math.degrees(rotation_rad)
    return dx, dy, rotation_deg


def calculate_crop_margins(w, h, max_dx, max_dy, max_rot_deg, deep_margin_factor):
    """Calculates safe padding borders to crop out black edges entirely with a deeper margin buffer."""
    rad = math.radians(abs(max_rot_deg))
    sin_a = math.sin(rad)
    cos_a = math.cos(rad)

    # Calculate bounding envelope corners due to rotational shift
    rot_pad_x = int((w / 2) * (1 - cos_a) + (h / 2) * sin_a)
    rot_pad_y = int((h / 2) * (1 - cos_a) + (w / 2) * sin_a)

    # Total buffer requirement combines translation absolute peak + rotational envelope
    # MULTIPLIED by the custom user deep_margin_factor to trim even further inward
    pad_x = int(math.ceil((max_dx + rot_pad_x) * deep_margin_factor))
    pad_y = int(math.ceil((max_dy + rot_pad_y) * deep_margin_factor))

    # Clamp padding values to avoid collapsing the canvas image to zero pixels
    pad_x = min(pad_x, w // 3)
    pad_y = min(pad_y, h // 3)

    return max(0, pad_x), max(0, pad_y)


def stabilize_frame_at_index(
    idx,
    buffer,
    trajectory,
    window_size,
    output_folder,
    debug_folder,
    crop_tracking_list,
    scale_y_factor,
    pure_black_borders,
):
    """Smooths camera path tracks, applies pivot-centered rotations, and handles edge borders."""
    frame_name, curr_frame, feature_data = buffer[idx]
    src_pts, dst_pts, inliers = feature_data

    # --- DECOUPLED SMOOTHING WINDOWS ---
    # Keep rotation tight (default window_size, e.g., 3) to catch aggressive snapping
    rot_radius = window_size // 2
    rot_start = max(0, idx - rot_radius)
    rot_end = min(len(trajectory), idx + rot_radius + 1)
    smoothed_rot = np.mean(trajectory[rot_start:rot_end], axis=0)[2]

    # FORCE EXTRA SMOOTHING FOR Y-AXIS JITTER (Multiply window scope for translation)
    trans_window_size = window_size * 2 + 1  # Expands a window of 3 to 7 for translations
    trans_radius = trans_window_size // 2
    trans_start = max(0, idx - trans_radius)
    trans_end = min(len(trajectory), idx + trans_radius + 1)
    trans_window_pts = trajectory[trans_start:trans_end]

    smoothed_dx = np.mean(trans_window_pts, axis=0)[0]
    smoothed_dy = np.mean(trans_window_pts, axis=0)[1]

    # Compute decoupled correction targets (Smoothed Trajectory - Shaky Position)
    diff_dx = smoothed_dx - trajectory[idx][0]
    diff_dy = (smoothed_dy - trajectory[idx][1]) * scale_y_factor
    diff_angle_deg = smoothed_rot - trajectory[idx][2]

    # Track absolute correction numbers for final global crop pass
    crop_tracking_list.append((abs(diff_dx), abs(diff_dy), abs(diff_angle_deg)))

    h, w, c = curr_frame.shape
    center = (w // 2, h // 2)

    # Generate rotation matrix around the center pixel pivot
    rot_matrix = cv2.getRotationMatrix2D(center, diff_angle_deg, scale=1.0)

    # SAFE COLUMN INJECTION: Modify ONLY the third column [tx, ty] translation parameters
    rot_matrix[0, 2] += diff_dx
    rot_matrix[1, 2] += diff_dy

    # SELECT BORDER MODE: Force pure black if argument parameter flag is passed, otherwise mirror
    border_mode = cv2.BORDER_CONSTANT if pure_black_borders else cv2.BORDER_REFLECT

    # Apply warp using selected canvas edge fill rule (Pure Black = 0,0,0 value fallback)
    stabilized_frame = cv2.warpAffine(
        curr_frame,
        rot_matrix,
        (w, h),
        borderMode=border_mode,
        borderValue=(0, 0, 0),
    )

    # Temporarily write out full frame asset before global final crop pass
    temp_out_path = os.path.join(output_folder, f"temp_{frame_name}")
    cv2.imwrite(temp_out_path, stabilized_frame)

    print(
        f"Processed: {frame_name} -> dx: {diff_dx:+.1f}px, dy: {diff_dy:+.1f}px (Smoothed), rot: {diff_angle_deg:+.2f}°"
    )

    if debug_folder:
        img_left = curr_frame.copy()
        img_right = stabilized_frame.copy()

        marker_count = 0
        if inliers is not None and len(src_pts) > 0:
            inlier_mask = inliers.ravel()
            for i, is_inlier in enumerate(inlier_mask):
                if is_inlier:
                    p_left = (int(dst_pts[i][0][0]), int(dst_pts[i][0][1]))
                    p_right = (int(src_pts[i][0][0]), int(src_pts[i][0][1]))

                    if (
                        0 <= p_left[0] < w
                        and 0 <= p_left[1] < h
                        and 0 <= p_right[0] < w
                        and 0 <= p_right[1] < h
                    ):
                        cv2.circle(img_left, p_left, 5, (0, 255, 0), -1)
                        cv2.circle(img_right, p_right, 5, (0, 255, 0), -1)

                        label = str(marker_count)
                        cv2.putText(
                            img_left,
                            label,
                            (p_left[0] + 6, p_left[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 255),
                            1,
                        )
                        cv2.putText(
                            img_right,
                            label,
                            (p_right[0] + 6, p_right[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 255),
                            1,
                        )
                        marker_count += 1
                        if marker_count >= 40:
                            break

        combined = np.hstack((img_left, img_right))
        canvas = np.zeros((h + 60, w * 2, 3), dtype=np.uint8)
        canvas[0:h, 0 : w * 2] = combined

        caption_text = (
            f"Correction Applied -> Shift X: {diff_dx:+.2f}px | "
            f"Shift Y: {diff_dy:+.2f}px | Rotation: {diff_angle_deg:+.2f} deg"
        )
        cv2.putText(
            canvas,
            caption_text,
            (20, h + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(os.path.join(debug_folder, frame_name), canvas)


def stabilize_png_sequence(
    input_folder,
    output_folder,
    debug_folder=None,
    window_size=3,
    scale_y=0.1,
    pure_black=True,
    deep_margin=1.0,
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if debug_folder and not os.path.exists(debug_folder):
        os.makedirs(debug_folder)

    frame_files = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith(".png")]
    )
    num_frames = len(frame_files)

    if num_frames < 2:
        print("Error: Minimum of 2 PNG frames required.")
        return

    if window_size % 2 == 0:
        window_size += 1
    # Lookahead radius tracks the larger window to guarantee smooth data arrays
    lookahead_radius = (window_size * 2 + 1) // 2

    # High-yield configuration for extreme distance structural matching at low framerates
    orb = cv2.ORB_create(
        nfeatures=6000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, patchSize=31
    )
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    buffer = []
    trajectory = []
    crop_tracking_list = []

    print("Running Decoupled Centered 1 FPS Large Displacement Tracker...")

    # Initialize frame 0
    prev_name = frame_files[0]
    prev_frame = cv2.imread(os.path.join(input_folder, prev_name))
    h, w, c = prev_frame.shape
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

    # Tracks absolute position [X, Y, Angle in Degrees]
    current_absolute_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    trajectory.append(current_absolute_pos.copy())
    buffer.append((prev_name, prev_frame, ([], [], None)))

    for i in range(1, num_frames):
        curr_name = frame_files[i]
        curr_frame = cv2.imread(os.path.join(input_folder, curr_name))
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_kp, curr_des = orb.detectAndCompute(curr_gray, None)

        matrix = None
        src_pts, dst_pts, inliers = [], [], None

        if prev_des is not None and curr_des is not None:
            matches = bf.match(prev_des, curr_des)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) >= 4:
                src_pts = np.float32(
                    [prev_kp[m.queryIdx].pt for m in matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [curr_kp[m.trainIdx].pt for m in matches]
                ).reshape(-1, 1, 2)

                # High RANSAC outlier tolerance window to capture wide displacements
                matrix, inliers = cv2.estimateAffinePartial2D(
                    dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=30.0
                )
                if matrix is not None:
                    # Update previous buffer frame node retroactively with match data
                    buffer[-1] = (
                        buffer[-1][0],
                        buffer[-1][1],
                        (src_pts, dst_pts, inliers),
                    )

        if matrix is not None:
            dx, dy, rot_deg = extract_affine_params(matrix)
            # Accept extreme rotations up to 90 degrees since this is a 1 FPS dataset
            if abs(rot_deg) < 90.0:
                current_absolute_pos += [dx, dy, rot_deg]
            else:
                matrix = None
        
        if matrix is None:
            current_absolute_pos += [0.0, 0.0, 0.0]

        trajectory.append(current_absolute_pos.copy())
        buffer.append((curr_name, curr_frame, ([], [], None)))
        
        prev_gray, prev_kp, prev_des = curr_gray, curr_kp, curr_des

        # Stream process target frame if lookahead requirements are fulfilled
        if len(buffer) > lookahead_radius:
            target_idx = len(buffer) - 1 - lookahead_radius
            stabilize_frame_at_index(target_idx,buffer,trajectory,window_size,output_folder,debug_folder,crop_tracking_list,scale_y,pure_black,)
            # Flush remaining lagging frames trapped inside sliding window
    start_flush_idx = len(buffer) - lookahead_radius
    for idx in range(start_flush_idx, len(buffer)):
        stabilize_frame_at_index(idx,buffer,trajectory,window_size,output_folder,debug_folder,crop_tracking_list,scale_y,pure_black,)

    # FINAL PASS: Calculate unified crop margins using deep_margin factor expansion
    max_dx = max([item[0] for item in crop_tracking_list]) if crop_tracking_list else 0
    max_dy = max([item[1] for item in crop_tracking_list]) if crop_tracking_list else 0
    max_rot = max([item[2] for item in crop_tracking_list]) if crop_tracking_list else 0
    pad_x, pad_y = calculate_crop_margins(w, h, max_dx, max_dy, max_rot, deep_margin)
    print(f"\nAuto-Cropping Final Sequence -> Uniform border shaving X: {pad_x}px, Y: {pad_y}px")

    for file_name in frame_files:
        temp_path = os.path.join(output_folder, f"temp_{file_name}")
        final_path = os.path.join(output_folder, file_name)
        if os.path.exists(temp_path):
            img = cv2.imread(temp_path)
            cropped_img = img[pad_y : h - pad_y, pad_x : w - pad_x]
            cv2.imwrite(final_path, cropped_img)
            os.remove(temp_path)

    
    print("Pipeline Execution Complete!")


parser = argparse.ArgumentParser(description="Low Framerate Unconstrained 3-DOF Stabilizer Engine.")
parser.add_argument("-i", "--input", required=True, help="Folder containing shaky PNG frames")
parser.add_argument("-o", "--output", required=True, help="Folder where clean frames save")
parser.add_argument("-d","--debug",default=None,help="Optional folder for diagnostics visualizations",)
parser.add_argument("-w","--window",type=int,default=3,help="Rotation path smoothing frame size window context. Default 3.",)
parser.add_argument("-sy","--scaley",type=float,default=0.1,help="Vertical jitter dampening weight. Set to 0.0 to lock Y, 0.1 to smooth. Default 0.1.",)
parser.add_argument("-b","--black",action="store_true",help="Force pure matte black fills on rotated gaps. Blocks mirrored edge artifacts completely.",)
parser.add_argument("-dm","--deepmargin",type=float,default=1.0,help="Cropping boundary scale multiplier. Raise to 1.2 or 1.4 to slice deep black gaps. Default 1.0.",)

args = parser.parse_args()

stabilize_png_sequence(args.input,args.output,args.debug,args.window,args.scaley,args.black,args.deepmargin,)
