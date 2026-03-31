import cv2
import numpy as np
from picamera2 import Picamera2

# =========================
# REPLACE WITH YOUR VALUES
# =========================
camera_matrix = np.array([
    [1000.0,    0.0, 728.0],
    [   0.0, 1000.0, 544.0],
    [   0.0,    0.0,   1.0]
], dtype=np.float32)

dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)

marker_size = 0.10   # marker size in meters
target_id = 0

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# Picamera2 setup for IMX296 resolution
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1456, 1088), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

while True:
    frame = picam2.capture_array()

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        ids = ids.flatten()
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, marker_id in enumerate(ids):
            if marker_id == target_id:
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[i]],
                    marker_size,
                    camera_matrix,
                    dist_coeffs
                )

                rvec = rvec[0][0]
                tvec = tvec[0][0]

                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

                x, y, z = tvec
                text = f"ID {marker_id} X:{x:.2f} Y:{y:.2f} Z:{z:.2f} m"
                print(text)

                cv2.putText(
                    frame,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("IMX296 ArUco Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
