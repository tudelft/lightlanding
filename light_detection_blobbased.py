import time
import cv2
import numpy as np
from itertools import combinations
import math
from scipy.spatial.transform import Rotation as R
import os

os.environ["MAVLINK20"] = "1"
os.environ["MAVLINK_DIALECT"] = "common"
from pymavlink import mavutil

# =========================
# Input source configuration
# =========================
USE_VIDEO_FILE = False          # True = read from video, False = use RPi camera
VIDEO_PATH = "lightrecordingLshape.mp4" # Path to video file when USE_VIDEO_FILE=True

# intrinsics and distortion parameters
camera_matrix = np.array([
    [966.94734754,   0.0,         717.76863491],
    [0.0,            965.39535141, 509.88724998],
    [0.0,            0.0,         1.0]
], dtype=np.float32)

dist_coeffs = np.array([
    -0.12461181,
     0.00088134,
    -0.01019451,
     0.00861141
], dtype=np.float32)


def detect_leds(
    min_area: int = 1,
    max_area: int = 40000,
    brightness_threshold: int = 200,
) -> None:


    m = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)
    m.wait_heartbeat()

    # Example: Delft, NL
    LAT_DEG = 51.99042
    LON_DEG = 4.37549
    ALT_M = 5.0   # MSL altitude in meters

    def wait_cmd_ack(master, command_id, timeout=3.0):
        start = time.time()
        while time.time() - start < timeout:
            msg = master.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.5)
            if msg and msg.command == command_id:
                return msg
        return None

    def request_message(master, msg_id):
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
            0,
            float(msg_id), 0, 0, 0, 0, 0, 0
        )

    def recv_one(master, msg_type, timeout=2.0):
        start = time.time()
        while time.time() - start < timeout:
            msg = master.recv_match(type=msg_type, blocking=True, timeout=0.5)
            if msg:
                return msg
        return None

    def set_global_origin(master, LAT_DEG, LON_DEG, ALT_M):
        target_system = master.target_system
        target_component = master.target_component
        
        lat_int = int(LAT_DEG * 1e7)
        lon_int = int(LON_DEG * 1e7)
        alt_mm = int(ALT_M * 1000)

        print("Sending SET_GPS_GLOBAL_ORIGIN...")
        master.mav.set_gps_global_origin_send(
            target_system,
            lat_int,
            lon_int,
            alt_mm,
            int(time.time() * 1e6)  # time_usec
        )

#        time.sleep(5)

        #print("Sending MAV_CMD_DO_SET_HOME...")
        master.mav.command_long_send(
            target_system,
            target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_HOME,
            0,          # confirmation
            0,          # param1: 0 = use specified location, 1 = use current
            math.nan,   # param2: roll
            math.nan,   # param3: pitch
            math.nan,   # param4: yaw
            LAT_DEG,    # param5: latitude in degrees
            LON_DEG,    # param6: longitude in degrees
            ALT_M       # param7: altitude in meters (MSL)
        )

        ack = wait_cmd_ack(master, mavutil.mavlink.MAV_CMD_DO_SET_HOME, timeout=5.0)
        
        if ack:
            print(f"SET_HOME ACK result: {ack.result}")
        else:
            print("No COMMAND_ACK received for MAV_CMD_DO_SET_HOME")

        # Ask PX4 to send back the values it currently believes
        print("Requesting GPS_GLOBAL_ORIGIN and HOME_POSITION...")
        request_message(master, mavutil.mavlink.MAVLINK_MSG_ID_GPS_GLOBAL_ORIGIN)
        request_message(master, mavutil.mavlink.MAVLINK_MSG_ID_HOME_POSITION)

        gps_origin = recv_one(master, "GPS_GLOBAL_ORIGIN", timeout=3.0)
        home_pos = recv_one(master, "HOME_POSITION", timeout=3.0)

        if gps_origin:
            print("GPS_GLOBAL_ORIGIN received:")
            print(f"  lat={gps_origin.latitude / 1e7}")
            print(f"  lon={gps_origin.longitude / 1e7}")
            print(f"  alt_msl_m={gps_origin.altitude / 1000.0}")
        else:
            print("No GPS_GLOBAL_ORIGIN received")

        if home_pos:
            print("HOME_POSITION received:")
            print(f"  lat={home_pos.latitude / 1e7}")
            print(f"  lon={home_pos.longitude / 1e7}")
            print(f"  alt_msl_m={home_pos.altitude / 1000.0}")
            print(f"  local_x={home_pos.x} local_y={home_pos.y} local_z={home_pos.z}")
        else:
            print("No HOME_POSITION received")
            
        return None
            
    picam2 = None
    cap = None

    # -------------------------
    # Setup input source
    # -------------------------
    if USE_VIDEO_FILE:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {VIDEO_PATH}")
    else:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (1456, 1088), "format": "RGB888"}
        )
        picam2.configure(config)
        controls = {
        "ExposureTime": 4000,   # microseconds
        "AnalogueGain": 1.0}
        picam2.set_controls(controls)
        picam2.start()


    # try:
    start_time = time.time()
    global_tf_set = False 
    
    while True:
        # -------------------------
        # Read frame
        # -------------------------
        if USE_VIDEO_FILE:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame.")
                break
            else:
                timestamp = int(time.time()*1e6)
        else:
            frame = picam2.capture_array()
            timestamp = int(time.time()*1e6)

        frame = cv2.rotate(frame, cv2.ROTATE_180)
        green = frame  # or frame[:, :, 1] if you want the green channel only

        # -------------------------
        # Undistort frame
        # -------------------------
        h, w = frame.shape[:2]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            np.eye(3),
            camera_matrix,
            (w, h),
            cv2.CV_16SC2,
        )

        image_undistorted = cv2.remap(
            frame, map1, map2, interpolation=cv2.INTER_LINEAR
        )
        green_undistorted = cv2.remap(
            green, map1, map2, interpolation=cv2.INTER_LINEAR
        )

        # image_undistorted = cv2.rotate(image_undistorted, cv2.ROTATE_90_CLOCKWISE)  # Rotate if needed based on camera orientation
        # green_undistorted = cv2.rotate(green_undistorted, cv2.ROTATE_90_CLOCKWISE)  # Rotate if needed based on camera orientation

        annotated = image_undistorted.copy()
        green_undistorted = cv2.cvtColor(green_undistorted, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(green_undistorted, (7, 7), 1.5)

        # Threshold bright regions (likely LEDs)
        _, thresh = cv2.threshold(blurred, brightness_threshold, 255, cv2.THRESH_BINARY)

        # Clean up small noise
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

        # Find contours of bright blobs
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        circles = np.empty((0, 3), dtype=np.float32) 
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Filter by size
            if area < min_area or area > max_area:
                continue

            # Find enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(cnt)

            # # Skip tiny detections
            # if radius < 1:
            #     continue

            center = (int(x), int(y))
            # radius = int(radius)

            circles = np.append(circles, [[x, y, radius]], axis=0)

        print('total circles', len(circles))
        # deterministic order
        order = np.lexsort((circles[:, 1], circles[:, 0]))
        circles = circles[order]
        
        # fitered_circles = circles
        fitered_circles = filter_circles_same_line_similar_radius(circles, radius_tol=0.5, line_tol=5.0, min_group_size=4, cross_ratio_tol=0.02) #cr: 0.015
        print('filtered circles', len(fitered_circles))

        led_count = 0
        for circle in fitered_circles:    
            # Draw annotation
            center = (int(circle[0]), int(circle[1]))
            radius = int(circle[2])
            cv2.circle(annotated, center, radius + 1, (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"LED {led_count + 1}",
                (center[0] + 5, center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            led_count += 1

        # print(f"Detected LEDs: {led_count}")

        # Show images
        cv2.namedWindow("Thresholded", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Thresholded', 700, 700) 
        cv2.imshow("Thresholded", thresh)
        cv2.waitKey(1)
        
        cv2.namedWindow("Annotated", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Annotated', 700, 700) 
        cv2.imshow("Annotated", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        # print(len(fitered_circles), "circles after line/radius filtering")
        if (len(fitered_circles) == 8):
            # print("Attempting pose estimation with", len(fitered_circles), "circles...")
            image_points, object_points, info = detections_to_points(fitered_circles)
            # print("2D-3D correspondences:", len(image_points), len(object_points))

            if (len(image_points)%4 == 0) and (len(image_points) == len(object_points)):
                pose_dict = estimate_planar_pose(object_points, image_points, camera_matrix, dist_coeffs=np.zeros((1, 4)))
                # print('Reprojection error:', pose_dict["reprojection_error"])
                print('Reprojection error:', pose_dict["reprojection_error"])
                print("Estimated pose:", pose_dict["camera_position"]) if pose_dict["reprojection_error"] < 60 else print("Pose estimation failed")
                cam_to_w_xyz = p = pose_dict["camera_position"]
                body_to_w_quat = q = pose_dict["R_body_to_w"]

#                if ((time.time() - start_time >= 5) and not global_tf_set):
#                    set_global_origin(m, LAT_DEG, LON_DEG, ALT_M)
#                    global_tf_set = True 
                    
                if (time.time() - start_time >= 1):
                    msg = mavutil.mavlink.MAVLink_odometry_message(
                        timestamp,
                        mavutil.mavlink.MAV_FRAME_LOCAL_FRD,
                        mavutil.mavlink.MAV_FRAME_BODY_FRD,
                        *p,
                        [q[3], q[0], q[1], q[2]],  # w, x, y, z

                        # velocity (
                        float('nan'), float('nan'), float('nan'),
                        float('nan'), float('nan'), float('nan'),

                        # pose covariance (6x6 upper triangle = 21 values)
                        [
                            0.01, 0, 0, 0, 0, 0,
                            0.01, 0, 0, 0, 0,
                            0.02, 0, 0, 0,
                            0.01, 0, 0,
                            0.01, 0,
                            0.03
                        ],

                        # velocity covariance
                        [
                            0.02, 0, 0, 0, 0, 0,
                            0.02, 0, 0, 0, 0,
                            0.04, 0, 0, 0,
                            0.02, 0, 0,
                            0.02, 0,
                            0.04
                        ],

                        100,  # quality
                        mavutil.mavlink.MAV_ESTIMATOR_TYPE_VISION
                    )
                    m.mav.send(msg)
                    time.sleep(1/30.0)


                    
    # finally:
    #     # -------------------------
    #     # Cleanup
    #     # -------------------------
    #     if cap is not None:
    #         cap.release()

    #     if picam2 is not None:
    #         picam2.stop()

    #     cv2.destroyAllWindows()

def cross_ratio_1d(a, b, c, d):
    den = (a - d) * (b - c)
    if abs(den) < 1e-12:
        return np.inf
    return ((a - c) * (b - d)) / den

def filter_circles_same_line_similar_radius(
    circles: np.ndarray,
    radius_tol: float = 0.1,
    line_tol: float = 5.0,
    min_group_size: int = 4,
    cross_ratio_tol: float = 0.01
) -> np.ndarray:
    """
    Keep circles that belong to a group of circles lying approximately on the
    same line and having similar radii.

    Parameters
    ----------
    circles : np.ndarray
        Array of shape (N, 3), where each row is [x, y, r].
    radius_tol : float
        Allowed relative radius difference.
        Example: 0.25 means radii may differ by up to 25% from the group mean.
    line_tol : float
        Maximum perpendicular distance (in pixels) from the fitted line for a
        circle to be considered on that line.
    min_group_size : int
        Minimum number of circles needed to form a valid line group.
    cross_ratio_tol : float
        Tolerance for the cross-ratio test to identify equally spaced points.

    Returns
    -------
    np.ndarray
        Filtered array of circles, shape (M, 3).
    """
    circles = np.asarray(circles)

    if circles.ndim != 2 or circles.shape[1] != 3:
        raise ValueError("circles must have shape (N, 3)")

    n = len(circles)
    if n < min_group_size:
        return np.empty((0, 3), dtype=circles.dtype)

    best_groups = []

    # Try every pair of circles as a candidate line
    for i in range(n):
        x1, y1, r1 = circles[i]
        for j in range(i + 1, n):
            x2, y2, r2 = circles[j]

            dx = x2 - x1
            dy = y2 - y1

            norm = np.hypot(dx, dy)
            if norm < 1e-6:
                continue

            ux = dx / norm
            uy = dy / norm
        
            group_indices = [i, j]
            current_radii = [r1, r2]

            # Line equation based on pair (i, j)
            for k in range(n):
                if k == i or k == j:
                    continue

                x, y, r = circles[k]

                # Perpendicular distance from point to line through (x1,y1)-(x2,y2)
                dist = abs(dy * x - dx * y + x2 * y1 - y2 * x1) / norm

                mean_r = np.mean([r1, r2])
                radius_ok = abs(r - mean_r) <= radius_tol * mean_r
                line_ok = dist <= line_tol
                # print('line_ok, radius_ok', line_ok, radius_ok)
                if line_ok and radius_ok:
                    group_indices.append(k)
                    current_radii.append(r)

            if len(group_indices) >= min_group_size:
                valid = False
                for quad in set(combinations(group_indices, 4)):
                    projections = []
                    for idx in quad:
                        x, y, _ = circles[idx]
                        t = (x - x1) * ux + (y - y1) * uy
                        projections.append(t)

                    projections = np.sort(np.asarray(projections))
                    a, b, c, d = projections

                    cr = cross_ratio_1d(a, b, c, d)
                    if np.isfinite(cr) and abs(cr - 4/3) <= cross_ratio_tol:
                        # print('cross_ratio_tol_ok:', 'True')
                        valid = True
                        best_groups.extend(quad)
                        break

                if not valid:
                    # print('cross_ratio_tol_ok:', 'False')
                    continue

    best_groups = sorted(set(best_groups))
    print('len(best_groups)', len(best_groups))

#    if (len(best_groups)%4) != 0:
#        return np.empty((0, 3), dtype=circles.dtype)

    return circles[best_groups]

def detections_to_points(fitered_circles):
    """
    Convert 8 detected LED circles into ordered 2D-3D correspondences for an L-shape.

    Assumptions
    - Input contains exactly 8 circles: (x, y, r)
    - The LEDs form an L-shape:
        * long arm: 5 points total including the corner  -> 4 points away from corner
        * short arm: 4 points total including the corner -> 3 points away from corner
    - Corner LED is one of the larger-radius LEDs
    - Adjacent LEDs are 12.5 cm apart in 3D
    - Corner 3D coordinate is (0, 0, 0)
    - Long arm is mapped to +X
    - Short arm is mapped to +Y

    Parameters
    ----------
    fitered_circles : array-like, shape (8, 3)
        Each row is (x, y, r)

    Returns
    -------
    image_points : np.ndarray, shape (8, 2), dtype=np.float32
        2D image points in the same order as object_points

    object_points : np.ndarray, shape (8, 3), dtype=np.float32
        Corresponding 3D points:
            [0,0,0]
            [12.5,0,0], [25,0,0], [37.5,0,0], [50,0,0]
            [0,12.5,0], [0,25,0], [0,37.5,0]

    info : dict
        Extra debug information:
        - "corner_index"
        - "large_radius_indices"
        - "small_radius_indices"
        - "long_arm_indices"
        - "short_arm_indices"
        - "radius_threshold"
    """
    circles = np.asarray(fitered_circles, dtype=float)
    if circles.shape != (8, 3):
        raise ValueError(f"Expected shape (8, 3), got {circles.shape}")

    pts = circles[:, :2]
    radii = circles[:, 2]

    # ------------------------------------------------------------------
    # 1) Split circles into two radius groups using the largest gap in r
    # ------------------------------------------------------------------
    sort_idx = np.argsort(radii)
    sorted_r = radii[sort_idx]
    gaps = np.diff(sorted_r)

    if len(gaps) == 0:
        raise ValueError("Need at least 2 circles to split by radius.")

    split_at = int(np.argmax(gaps))
    radius_threshold = 0.5 * (sorted_r[split_at] + sorted_r[split_at + 1])

    large_radius_indices = np.where(radii > radius_threshold)[0].tolist()
    small_radius_indices = np.where(radii <= radius_threshold)[0].tolist()

    if len(large_radius_indices) == 0:
        raise ValueError("Could not find any large-radius LEDs. Check detections/radii.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def fit_line_direction(points_2d):
        """
        Return unit direction of best-fit line through 2D points using SVD.
        """
        p = np.asarray(points_2d, dtype=float)
        c = p.mean(axis=0)
        _, _, vt = np.linalg.svd(p - c)
        d = vt[0]
        d = d / np.linalg.norm(d)
        return d

    def line_fit_error_with_corner(corner_pt, arm_pts):
        """
        Mean squared orthogonal distance of points to the best-fit line.
        Fits a line through [corner + arm points].
        """
        all_pts = np.vstack([corner_pt[None, :], arm_pts])
        center = all_pts.mean(axis=0)
        _, _, vt = np.linalg.svd(all_pts - center)
        direction = vt[0]
        direction = direction / np.linalg.norm(direction)

        diffs = all_pts - center
        # Orthogonal distance to line
        proj = np.outer(diffs @ direction, direction)
        ortho = diffs - proj
        mse = np.mean(np.sum(ortho ** 2, axis=1))
        return mse, direction

    def angle_between_dirs_deg(d1, d2):
        """
        Acute angle between two undirected line directions in degrees.
        """
        c = abs(float(np.dot(d1, d2)))
        c = np.clip(c, -1.0, 1.0)
        angle = math.degrees(math.acos(c))
        # Because directions are undirected, angle is in [0, 90]
        return angle

    def sort_arm_points_from_corner(corner_pt, arm_pts, arm_indices):
        """
        Sort arm points by increasing distance from corner.
        """
        d = np.linalg.norm(arm_pts - corner_pt[None, :], axis=1)
        order = np.argsort(d)
        return arm_pts[order], [arm_indices[i] for i in order]

    # ------------------------------------------------------------------
    # 2) Find the corner among the large-radius LEDs
    #
    #    We try each large-radius point as a corner candidate.
    #    For each candidate, we partition the remaining 7 points into:
    #      - 4 points on the long arm
    #      - 3 points on the short arm
    #    and score how well each group forms a line with the corner.
    # ------------------------------------------------------------------
    best = None

    for corner_idx in large_radius_indices:
        corner_pt = pts[corner_idx]
        other_indices = [i for i in range(8) if i != corner_idx]

        for long_combo in combinations(other_indices, 4):
            long_indices = list(long_combo)
            short_indices = [i for i in other_indices if i not in long_indices]

            long_pts = pts[long_indices]
            short_pts = pts[short_indices]

            long_err, long_dir = line_fit_error_with_corner(corner_pt, long_pts)
            short_err, short_dir = line_fit_error_with_corner(corner_pt, short_pts)
            angle = angle_between_dirs_deg(long_dir, short_dir)

            # Prefer near-perpendicular arms; penalize if too parallel
            angle_penalty = 0.0
            if angle < 60.0:
                angle_penalty += (60.0 - angle) ** 2
            elif angle > 100.0:
                angle_penalty += (angle - 95.0) ** 2

            # Small bonus if corner is among the larger-radius points
            radius_bonus = -0.1 * radii[corner_idx]

            score = long_err + short_err + 0.01 * angle_penalty + radius_bonus

            if (best is None) or (score < best["score"]):
                best = {
                    "score": score,
                    "corner_idx": corner_idx,
                    "long_indices": long_indices,
                    "short_indices": short_indices,
                    "long_err": long_err,
                    "short_err": short_err,
                    "angle_deg": angle,
                }

    if best is None:
        raise ValueError("Could not identify a valid L-shape configuration.")

    # ------------------------------------------------------------------
    # 3) Sort points along each arm from the corner outward
    # ------------------------------------------------------------------
    corner_idx = best["corner_idx"]
    corner_pt = pts[corner_idx]

    long_pts = pts[best["long_indices"]]
    short_pts = pts[best["short_indices"]]

    long_pts_sorted, long_indices_sorted = sort_arm_points_from_corner(
        corner_pt, long_pts, best["long_indices"]
    )
    short_pts_sorted, short_indices_sorted = sort_arm_points_from_corner(
        corner_pt, short_pts, best["short_indices"]
    )

    # ------------------------------------------------------------------
    # 4) Build ordered 2D image points
    #
    # Order convention:
    #   0: corner
    #   1..4: long arm (+X)
    #   5..7: short arm (+Y)
    # ------------------------------------------------------------------
    image_points = np.vstack([
        corner_pt[None, :],
        long_pts_sorted,
        short_pts_sorted
    ]).astype(np.float32)

    # 3D object points in cm
    object_points = np.array([
        [0.0,  0.0,  0.0],   # corner
        [0.125, 0.0,  0.0],
        [0.250, 0.0,  0.0],
        [0.375, 0.0,  0.0],
        [0.500, 0.0,  0.0],   # long arm

        [0.0,  -0.125, 0.0],
        [0.0,  -0.250, 0.0],
        [0.0,  -0.375, 0.0],   # short arm
    ], dtype=np.float32)

    info = {
        "corner_index": corner_idx,
        "large_radius_indices": large_radius_indices,
        "small_radius_indices": small_radius_indices,
        "long_arm_indices": long_indices_sorted,
        "short_arm_indices": short_indices_sorted,
        "radius_threshold": float(radius_threshold),
        "fit_score": float(best["score"]),
        "arm_angle_deg": float(best["angle_deg"]),
    }

    return image_points, object_points, info

def reprojection_error(object_points, image_points, rvec, tvec, K, dist_coeffs):
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, K, dist_coeffs
    )
    projected = projected.reshape(-1, 2)
    err = np.mean(np.linalg.norm(projected - image_points, axis=1))
    return err, projected


def camera_position_from_pose(Rot, tvec):
    """
    OpenCV pose convention:
        X_cam = R * X_obj + t
    Camera center in object coordinates:
        C_obj = -R^T * t
    """
    return -Rot.T @ tvec


def estimate_planar_pose(object_points, image_points, K, dist_coeffs):
    """
    Estimate pose for coplanar object points.

    Parameters
    ----------
    object_points : (N,3) ndarray
        Coplanar 3D points, usually all Z=0.
    image_points : (N,2) ndarray
        Corresponding pixel coordinates.
    K : (3,3) ndarray
        Camera intrinsic matrix.
    dist_coeffs : ndarray or None
        Distortion coefficients. Set to zeros if unknown.

    Returns
    -------
    result : dict
        Contains pose, reprojection error, and camera position.
    """
    object_points = np.ascontiguousarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.ascontiguousarray(image_points, dtype=np.float64).reshape(-1, 2)

    K = np.asarray(K, dtype=np.float64)

    if object_points.shape[0] < 4:
        raise ValueError("Need at least 4 points")
    if image_points.shape[0] != object_points.shape[0]:
        raise ValueError("image_points and object_points must match in count")

    # IPPE is designed for planar pose estimation.
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE
    )

    if not success:
        # Fallback to iterative
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

    if not success:
        return {"success": False}

    # Optional refinement
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist_coeffs,
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    R_mat, _ = cv2.Rodrigues(rvec)
    R_cam_to_w = R_mat.T  # Camera orientation in world coordinates
    cam_orient_quat = R.from_matrix(R_cam_to_w).as_quat()  # (x, y, z, w)
    err, projected = reprojection_error(
        object_points, image_points, rvec, tvec, K, dist_coeffs
    )
    cam_pos = camera_position_from_pose(R_mat, tvec)

    # Check that all points are in front of the camera
    pts_cam = (R_mat @ object_points.T + tvec).T
    positive_depth = np.all(pts_cam[:, 2] > 0)

    R_cam_to_body = np.array([
    [ 0, -1,  0],
    [ 1,  0,  0],
    [ 0,  0,  1],
])
    
    R_body_to_w = R_cam_to_w @ R_cam_to_body.T
    R_body_to_w_quat =  R.from_matrix(R_body_to_w).as_quat()  # (x, y, z, w)
    return {
        "success": True,
        "rvec": rvec,
        "tvec": tvec,
        "R": R_mat,
        "camera_position": cam_pos,
        "camera_orientation": cam_orient_quat,
        "R_body_to_w": R_body_to_w_quat,
        "reprojection_error": err,
        "projected_points": projected,
        "positive_depth": positive_depth,
        "points_camera_frame": pts_cam,
    }



if __name__ == "__main__":
    detect_leds()
