import cv2
import numpy as np
from itertools import combinations

# =========================
# Input source configuration
# =========================
USE_VIDEO_FILE = True          # True = read from video, False = use RPi camera
VIDEO_PATH = "lightrecording.mp4" # Path to video file when USE_VIDEO_FILE=True

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
    min_area: int = 5,
    max_area: int = 40000,
    brightness_threshold: int = 160,
) -> None:

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
        picam2.start()

    # try:
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
            frame = picam2.capture_array()

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

        # deterministic order
        # order = np.lexsort((circles[:, 1], circles[:, 0]))
        # circles = circles[order]
        
        # fitered_circles = circles
        fitered_circles = filter_circles_same_line_similar_radius(circles, radius_tol=0.1, line_tol=5.0, min_group_size=4, cross_ratio_tol=0.05)

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

        print(f"Detected LEDs: {led_count}")

        # Show images
        cv2.imshow("Threshold", thresh)
        cv2.imshow("Annotated LEDs", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
        
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

    best_group = []

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
                if line_ok and radius_ok:
                    group_indices.append(k)
                    current_radii.append(r)

            if len(group_indices) >= min_group_size:
                valid = False
                for quad in set(combinations(group_indices, 4)):
                    print('quad', quad)
                    projections = []
                    for idx in quad:
                        x, y, _ = circles[idx]
                        t = (x - x1) * ux + (y - y1) * uy
                        projections.append(t)

                    projections = np.sort(np.asarray(projections))
                    a, b, c, d = projections

                    cr = cross_ratio_1d(a, b, c, d)
                    if np.isfinite(cr) and abs(cr - 4/3) <= cross_ratio_tol:
                        valid = True
                        best_group = quad
                        break

                if not valid:
                    continue

    best_group = sorted(set(best_group))
    if len(best_group) != 4:
        return np.empty((0, 3), dtype=circles.dtype)

    return circles[best_group]

if __name__ == "__main__":
    detect_leds()
