import cv2
import numpy as np
from picamera2 import Picamera2

camera_matrix_rgb = np.array( [[1.13783006e+03, 0.00000000e+00, 9.99899908e+02],
 [0.00000000e+00, 1.14071831e+03, 5.99492820e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeffs_rgb = np.array(
 [-0.08491671, -0.09462636,  0.1612735,  -0.09637632])
 
# camera_matrix_mono = np.array(
 # [[972.41752602,   0.,         719.86748972],
 # [  0.,         970.82689346, 520.66180438],
 # [  0.,           0.,           1.        ]])
# dist_coeffs_mono = np.array([-0.13573729,  0.03353202, -0.0345132,   0.01030255])

marker_size = 0.198   # meters
target_id = 0

camera_matrix = camera_matrix_rgb
dist_coeffs = dist_coeffs_rgb

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# Picamera2 setup
picam2 = Picamera2(0)
full_size = picam2.camera_properties["PixelArraySize"]
print('full_size', full_size)
config = picam2.create_video_configuration(
    main={"size": full_size, "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

# Precompute optimal new camera matrix without distortion
w, h = full_size
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w, h), 1, (w, h)
)

while True:
    frame = picam2.capture_array()

    # ---- Undistort frame ----
    h, w = frame.shape[:2]

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    camera_matrix,
    dist_coeffs,
    (w, h),
    np.eye(3),
    balance=1.0)
            
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        np.eye(3),
        new_K,   # or a new matrix if you want cropping
        (w, h),
        cv2.CV_16SC2
    )

    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    # ---- ArUco detection on original frame ----
    corners, ids, _ = detector.detectMarkers(undistorted)

    if ids is not None:
        ids = ids.flatten()
        cv2.aruco.drawDetectedMarkers(undistorted, corners, ids)

        for i, marker_id in enumerate(ids):
            if marker_id == target_id:
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[i]],
                    marker_size,
                    new_K,
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
                
                R_wld_to_cam, _ = cv2.Rodrigues(rvec)
                T_wld_to_cam = np.eye(4)
                T_wld_to_cam[:3, :3] = R_wld_to_cam
                T_wld_to_cam[:3, 3] = tvec.flatten()

                # Invert transform
                T_cam_to_wld = np.linalg.inv(T_wld_to_cam)

                x, y, z = T_cam_to_wld[:3,3]
                text2 = f"ID {marker_id} X:{x:.2f} Y:{y:.2f} Z:{z:.2f} m"
                cv2.putText(
                    undistorted,
                    text2,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


    # ---- Show both windows ----
    cv2.imshow("Original", frame)
    cv2.imshow("Undistorted Image  (Pose Estimation)", undistorted)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
