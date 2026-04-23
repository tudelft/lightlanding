import cv2
import numpy as np
from picamera2 import Picamera2


# Camera matrix:
 # [[966.94734754   0.         717.76863491]
 # [  0.         965.39535141 509.88724998]
 # [  0.           0.           1.        ]]
# Distortion coeffs:
 # [-0.12461181  0.00088134 -0.01019451  0.00861141]


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

marker_size = 0.075   # meters
target_id = 13

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# Picamera2 setup
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1456, 1088), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

# Precompute optimal new camera matrix without distortion
h, w = 1088, 1456
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w, h), 1, (w, h)
)

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

    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    # ---- ArUco detection on original frame ----
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

    # ---- Show both windows ----
    cv2.imshow("Original (Pose Estimation)", frame)
    cv2.imshow("Undistorted Image", undistorted)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
