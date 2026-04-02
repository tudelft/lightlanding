import cv2
import numpy as np
from picamera2 import Picamera2

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
    max_area: int = 2000,
    brightness_threshold: int = 220,
) -> None:

    # Picamera2 setup
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (1456, 1088), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    while True:
        frame = picam2.capture_array()

        # ---- Undistort frame ----
        h, w = frame.shape[:2]

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            np.eye(3),
            camera_matrix,   # or a new matrix if you want cropping
            (w, h),
            cv2.CV_16SC2
        )

        image_undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)


        if image_undistorted is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        # Keep a copy for drawing
        annotated = image_undistorted.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(image_undistorted, cv2.COLOR_BGR2GRAY)

        # Slight blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold bright regions (likely LEDs)
        _, thresh = cv2.threshold(blurred, brightness_threshold, 255, cv2.THRESH_BINARY)

        # Clean up small noise
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

        # Find contours of bright blobs
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        led_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Filter by size
            if area < min_area or area > max_area:
                continue

            # Find enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(cnt)

            # Skip tiny detections
            if radius < 2:
                continue

            center = (int(x), int(y))
            radius = int(radius)

            # Draw annotation
            cv2.circle(annotated, center, radius + 5, (0, 255, 0), 2)
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


if __name__ == "__main__":
    detect_leds()
