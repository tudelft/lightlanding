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


def detect_bright_led_rings(
    brightness_threshold: int = 230,
    min_radius: int = 3,
    max_radius: int = 80,
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

        annotated = image_undistorted.copy()
        image_undistorted = cv2.cvtColor(image_undistorted, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image_undistorted, (7, 7), 1.5)

        # Keep bright parts only
        _, bright = cv2.threshold(blur, brightness_threshold, 255, cv2.THRESH_BINARY)

        # Clean noise
        kernel = np.ones((3, 3), np.uint8)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)

        # Use masked image for circle detection
        masked = cv2.bitwise_and(blur, blur, mask=bright)

        circles = cv2.HoughCircles(
            masked,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=15,
            param1=100,
            param2=10,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        count = 0
        if circles is not None:
            circles = np.round(circles[0]).astype(int)
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
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(output_path, annotated)


if __name__ == "__main__":
    detect_bright_led_rings()
