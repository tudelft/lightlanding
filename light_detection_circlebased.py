import cv2
import numpy as np


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


def detect_bright_led_rings(
    brightness_threshold: int = 160,
    min_radius: int = 1,
    max_radius: int = 80,
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

    try:
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
            blur = cv2.GaussianBlur(green_undistorted, (7, 7), 1.5)

            # Keep bright parts only
            _, bright = cv2.threshold(
                blur, brightness_threshold, 255, cv2.THRESH_BINARY
            )

            # Clean noise
            kernel = np.ones((3, 3), np.uint8)
            bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)

            # Use masked image for circle detection
            masked = cv2.bitwise_and(blur, blur, mask=bright)

            circles = cv2.HoughCircles(
                masked,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=10, # 15
                param1=100, # 100
                param2=25, # 20
                minRadius=min_radius,
                maxRadius=max_radius,
            )

            if circles is not None:
                circles = np.round(circles[0]).astype(int)   # shape: (N, 3)
                circles = filter_nested_circles(circles)     # shape: (M, 3)

            count = 0
            if circles is not None:
                for i, (x, y, r) in enumerate(circles, start=1):
                    cv2.circle(annotated, (x, y), r, (0, 255, 0), 2)
                    cv2.circle(annotated, (x, y), 2, (0, 0, 255), 3)
                    cv2.putText(
                        annotated,
                        f"Ring {i}",
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    count += 1

            print(f"Detected bright rings: {count}")

            cv2.imshow("Bright mask", bright)
            cv2.imshow("Annotated", annotated)

            key = cv2.waitKey(1 if not USE_VIDEO_FILE else 30) & 0xFF
            if key == ord("q"):
                break

    finally:
        # -------------------------
        # Cleanup
        # -------------------------
        if cap is not None:
            cap.release()

        if picam2 is not None:
            picam2.stop()

        cv2.destroyAllWindows()

def filter_nested_circles(circles, center_thresh=5):
    """
    Remove nested/duplicate circles with nearly the same center.
    Keeps the largest circle among overlapping-center detections.

    Input:
        circles: numpy array of shape (N, 3)
    Output:
        numpy array of shape (M, 3), dtype=int
    """
    if circles is None or len(circles) == 0:
        return np.empty((0, 3), dtype=int)

    circles = np.asarray(circles, dtype=int)

    # Sort by radius descending so larger circles are kept first
    circles = circles[np.argsort(circles[:, 2])[::-1]]

    filtered = []

    for x, y, r in circles:
        keep = True

        for fx, fy, fr in filtered:
            dist = np.hypot(x - fx, y - fy)

            # Same center => treat as duplicate/nested circle
            if dist < center_thresh:
                keep = False
                break

        if keep:
            filtered.append([x, y, r])

    return np.asarray(filtered, dtype=int)

if __name__ == "__main__":
    detect_bright_led_rings()
