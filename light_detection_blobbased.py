import threading
import time
import cv2
import numpy as np
from itertools import combinations
import math
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist

import os
os.environ["MAVLINK20"] = "1"
os.environ["MAVLINK_DIALECT"] = "common"
from pymavlink import mavutil

from helpers.utils import scale_camera_matrix, rotate_intrinsics_180
from helpers.mavlink_utils import get_fc_time_us, wait_cmd_ack, request_message, recv_one, set_global_origin
from helpers.poseestimation import pose_from_colored_leds, estimate_planar_pose, order_l_shape_markers
from helpers.led_detection import filter_circles_same_line_similar_radius, cross_ratio_1d   

# =========================
# Input source configuration
# =========================
USE_VIDEO_FILE = False          # True = read from video, False = use RPi camera
VIDEO_PATH = "lightrecordingLshape.mp4" # Path to video file when USE_VIDEO_FILE=True
CONNECT_MAVLINK = True             # Whether to connect to MAVLink and send odometry messages
MAVLINK_MULTIPLE_CONNECTIONS = False  # If we are also sending Mocap data to drone on serial then set this to True to avoid conflicts. Requires mavlink_routerd running on the pi.

if (not MAVLINK_MULTIPLE_CONNECTIONS):
    serial_ip = "/dev/ttyACM0"  # Serial port for MAVLink connection
else:
    serial_ip = "udpout:127.0.0.1:14600"  # UDP port for MAVLink connection

rgb_cameratype = 'fisheye' # 'fisheye' or 'pinhole'
mono_cameratype = 'fisheye' # 'fisheye' or 'pinhole'

markertype = 'aruco'  # 'Lshape' or 'aruco'
show_visualization = True

# L-shape marker setup
radius_tol=0.5 
line_tol=5.0
min_group_size=4 
cross_ratio_tol=0.025

# Camera setup
blur_window = (9, 9) # (9, 9) for monochrome global shutter
exposure_time_rgb = 5000 # microseconds
exposure_time_mono = 20000 # microseconds

# Pose estimation acceptance criteria
brightness_threshold = 35 # 60 for monochrome global shutter
reproj_threshold = 5 # default 5

# ArUco setup
marker_size = 0.198   # meters
target_id = 0

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

  
# intrinsics and distortion parameters (first flight test)
# computed at full_resolution of mono camera (1456 x 1088)
camera_matrix_mono = np.array(
 [[972.41752602,   0.,         719.86748972],
 [  0.,         970.82689346, 520.66180438],
 [  0.,           0.,           1.        ]])
dist_coeffs_mono = np.array([-0.13573729,  0.03353202, -0.0345132,   0.01030255])

# computed at full_resolution of rgb camera (1945 x 1097)
camera_matrix_rgb = np.array( [[1.13783006e+03, 0.00000000e+00, 9.99899908e+02],
 [0.00000000e+00, 1.14071831e+03, 5.99492820e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeffs_rgb = np.array(
 [-0.08491671, -0.09462636,  0.1612735,  -0.09637632])


# camera_matrix_rgb_perspective = np.array(
#  [[2.36184664e+03, 0.00000000e+00, 7.68344401e+02],
#  [0.00000000e+00, 2.37450698e+03, 5.08886362e+02],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
 
# dist_coeffs_rgb_perspective = np.array(
# [-2.20055223e-01, -1.12110606e+00, -4.76585431e-03, -8.49651511e-04,
#  -1.26268143e+02,  2.03709304e-01, -1.09829966e-01, -1.41235708e+02,
#   0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#   0.00000000e+00,  0.00000000e+00])

s=0.6 # scaling down the camera image and intrinsics for faster processing since we only care about large bright blobs (LEDs)

def detect_lights_sendodometry(
    brightness_threshold,
    min_area: int = 1,
    max_area: int = 40000,
) -> None:

    global camera_matrix_rgb, dist_coeffs_rgb, camera_matrix_mono, dist_coeffs_mono

    if (markertype == 'Lshape'):
        camera_matrix = camera_matrix_rgb
        dist_coeffs = dist_coeffs_rgb 

    elif (markertype == 'aruco'):
        camera_matrix = camera_matrix_mono #camera_matrix_mono
        dist_coeffs = dist_coeffs_mono #dist_coeffs_mono

    if CONNECT_MAVLINK:
        m = mavutil.mavlink_connection(serial_ip, baud=115200)
        
        print("Waiting heartbeat...")
        m.wait_heartbeat()
        print("Heartbeat received")
    
        print("Waiting for FC boot time...")
        fc_time_us = get_fc_time_us(m)
        print(f"FC time received: {fc_time_us}")

        # Estimate offset between companion monotonic clock
        # and FC boot clock
        companion_monotonic_us = int(time.monotonic() * 1e6)
        offset_us = fc_time_us - companion_monotonic_us
        print(f"Offset: {offset_us}")

        # Example: Delft, NL
        LAT_DEG = 51.99042
        LON_DEG = 4.37549
        ALT_M = 5.0   # MSL altitude in meters
        
            
    global_tf_set = False 
    if (not global_tf_set and CONNECT_MAVLINK):
        time.sleep(3)
        set_global_origin(m, LAT_DEG, LON_DEG, ALT_M)
        global_tf_set = True 
        time.sleep(1)

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

        cam_port = 1 if markertype == 'aruco' else 0  # Use camera port 1 (monochrome global shutter camera) for ArUco, port 0 (RGB camera) for L-shape
        picam2 = Picamera2(cam_port)

        full_size = picam2.camera_properties["PixelArraySize"]
        config = picam2.create_video_configuration(
             main={"size": full_size, "format": "RGB888"},
             buffer_count=1,
             queue=False)
             
        picam2.configure(config)
        controls = {
        "ScalerCrop": (0, 0, *full_size),
        "ExposureTime": exposure_time_rgb if markertype == 'Lshape' else exposure_time_mono,   # microseconds
        "AnalogueGain": 1.0}
        picam2.set_controls(controls)
        picam2.start()

        camera_matrix = scale_camera_matrix(camera_matrix, s)

    # try:
    start_time = time.time()

#    cv2.waitKey(1)
    if (show_visualization):
       cv2.namedWindow("Thresholded", cv2.WINDOW_NORMAL)
       cv2.resizeWindow('Thresholded', 700, 700) 

       cv2.namedWindow("Annotated", cv2.WINDOW_NORMAL)
       cv2.resizeWindow('Annotated', 700, 700) 

       cv2.namedWindow("Annotated_colors", cv2.WINDOW_NORMAL)
       cv2.resizeWindow('Annotated_colors', 700, 700) 

       cv2.namedWindow("Undistorted Image", cv2.WINDOW_NORMAL)
       cv2.resizeWindow('Undistorted Image', 700, 700) 

    camera_matrix = rotate_intrinsics_180(camera_matrix, s*full_size[0], s*full_size[1]) # because frame is rotated below
        
    while True:
#        time.sleep(1)
        # -------------------------
        # Read frame
        # -------------------------
        if USE_VIDEO_FILE:
            image_capture_time_usec = int(time.monotonic() * 1e6)
            mavlink_timestamp = image_capture_time_usec + offset_us
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame.")
                break

        else:
            # Use monotonic clock, NOT time.time()
            image_capture_time_usec = int(time.monotonic() * 1e6)
            mavlink_timestamp = image_capture_time_usec + offset_us
            frame = picam2.capture_array()
            
        height, width  = frame.shape[:2]
        frame = cv2.resize(frame, (int(s*width), int(s*height)))
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        red = frame[:, :, 0]
        green = frame[:, :, 1]

        green = diff_image_RG = red.astype(np.int16) - green.astype(np.int16)
        image_undistorted = None
        # -------------------------
        # Undistort frame
        # -------------------------
        h, w = frame.shape[:2]

        if ((markertype == 'Lshape' and rgb_cameratype == 'fisheye') or (markertype == 'aruco' and mono_cameratype == 'fisheye')):
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
                new_K,
                (w, h),
                cv2.CV_16SC2,
            )
            
            image_undistorted = cv2.remap(
                frame, map1, map2, interpolation=cv2.INTER_LINEAR)

            green_undistorted = cv2.remap(
               green, map1, map2, interpolation=cv2.INTER_LINEAR)

        elif ((markertype == 'Lshape' and rgb_cameratype == 'pinhole') or (markertype == 'aruco' and mono_cameratype == 'pinhole')):
            # Compute new camera matrix
            new_K, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix,
                dist_coeffs,
                (w, h),
                alpha=0.5,      # similar to fisheye balance=0.0
                newImgSize=(w, h)
            )

            # Precompute remapping
            map1, map2 = cv2.initUndistortRectifyMap(
                camera_matrix,
                dist_coeffs,
                R=np.eye(3),
                newCameraMatrix=new_K,
                size=(w, h),
                m1type=cv2.CV_16SC2,
            )

            # Undistort images
            image_undistorted = cv2.remap(
                frame,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR
            )

            green_undistorted = cv2.remap(
                green,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR
            )

        if (markertype == 'Lshape'):
            print('Searching for LEDs...')
            blurred = cv2.GaussianBlur(image_undistorted, blur_window, 0)

            annotated = blurred.copy()
            annotated_colors = blurred.copy()
#            blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            blurred = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
            
            # threshold for top 10%
            brightness_mask = np.percentile(blurred, 90)

            # select pixels above threshold
            top_pixels = blurred[blurred >= brightness_mask]

            avg_top_10_intensities = np.mean(top_pixels)
            # brightness_threshold = avg_top_10_intensities - 10

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


#            print('total circles', len(circles))
            # deterministic order
            order = np.lexsort((circles[:, 1], circles[:, 0]))
            circles = circles[order]
            
            # fitered_circles = circles
            fitered_circles = filter_circles_same_line_similar_radius(circles, radius_tol, line_tol, min_group_size, cross_ratio_tol) #cr: 0.015
            filteredcircles_avgcolor = []
#            print('filtered circles', len(fitered_circles))

            for circle in fitered_circles:    
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])
                h, w = annotated_colors.shape[:2]
                y, x = np.ogrid[:h, :w]
                inside_circle = (x-center[0])**2 + (y-center[1])**2 <= radius**2
                led_mean = int(green_undistorted[inside_circle].mean())
                filteredcircles_avgcolor.append(led_mean)

            filteredcircles_avgcolor_sorted = np.argsort(filteredcircles_avgcolor)
        
            if (show_visualization):
               led_count = 0
               for circle in fitered_circles:    
                   # Draw annotation
                   center = (int(circle[0]), int(circle[1]))
                   radius = int(circle[2])
                   h, w = annotated_colors.shape[:2]
                   y, x = np.ogrid[:h, :w]
                   inside_circle = (x-center[0])**2 + (y-center[1])**2 <= radius**2
                   led_mean = int(green_undistorted[inside_circle].mean())
                   led_type = np.where(filteredcircles_avgcolor_sorted==led_count)[0] <= 3  #first 4 LEDs based on min avg intensity
                   ann_circle_color = (0,255,0) if (led_type==1) else (0,0,255)
                   
                   filteredcircles_avgcolor.append(led_mean)
                   cv2.circle(annotated_colors, center, radius + 1, ann_circle_color, 2)
                   cv2.putText(
                      annotated_colors,
                      f"Int {led_mean}",
                      (center[0] + 5, center[1] - 5),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.5,
                      ann_circle_color,
                      1,
                      cv2.LINE_AA,
                      )

                   led_count += 1

            # print(f"Detected LEDs: {led_count}")

               ## Show images
               cv2.imshow("Thresholded", thresh)
               cv2.imshow("Annotated_colors", annotated_colors)
               cv2.imshow("Undistorted Image", image_undistorted)
               cv2.waitKey(1)

    #        if cv2.waitKey(1) & 0xFF == ord("q"):
    #            cv2.destroyAllWindows()
    #            break

            # print(len(fitered_circles), "circles after line/radius filtering")
            if (len(fitered_circles) == 8):
                # print("Attempting pose estimation with", len(fitered_circles), "circles...")
                image_points, object_points, pose_dict = pose_from_colored_leds(fitered_circles, filteredcircles_avgcolor_sorted, new_K, np.zeros((1, 4)))

                #image_points, object_points, info = order_l_shape_markers(fitered_circles)
                # print("2D-3D correspondences:", len(image_points), len(object_points))
                
                if (show_visualization):
                   led_count = 0
                   for image_point in image_points:    
                       # Draw annotation
                       center = (int(image_point[0]), int(image_point[1]))
                       #print('center', center)
                       cv2.circle(annotated, center, 10, (0, 255, 0), 2)
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

#                cv2.imshow("Annotated", annotated)              
#                print(f"Detected LEDs: {led_count}")

                if (len(image_points)%4 == 0) and (len(image_points) == len(object_points)):
                    #pose_dict = estimate_planar_pose(object_points, image_points, new_K, np.zeros((1, 4)))
                    # print('Reprojection error:', pose_dict["reprojection_error"])
                    print('Reprojection error:', pose_dict["reprojection_error"])
                    print('Positive depth', pose_dict["positive_depth"])
                    x = pose_dict["camera_position"][0]
                    y = pose_dict["camera_position"][1]
                    z = pose_dict["camera_position"][2]
                    text = f"Drone location: X:{x:.2f} Y:{y:.2f} Z:{z:.2f} m, {pose_dict["positive_depth"]}, {pose_dict["reprojection_error"]:.2f}"

                    print("Estimated pose:", x, y, z) if (pose_dict["reprojection_error"] < reproj_threshold and pose_dict["positive_depth"]) else print("Pose estimation failed")

                    if (show_visualization and pose_dict["reprojection_error"] < reproj_threshold):
                       cv2.putText(
                       annotated,
                       text,
                       (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.8,
                       (0, 255, 0),
                       2
                       )

                       projected = pose_dict["projected_points"]
                       for p_img, p_proj in zip(image_points, projected):
                           cv2.circle(annotated, tuple(p_img.astype(int)), 10, (0,255,0), -1)
                           cv2.circle(annotated, tuple(p_proj.astype(int)), 10, (0,0,255), -1)
                           cv2.line(annotated,
                               tuple(p_img.astype(int)),
                               tuple(p_proj.astype(int)),
                               (255,0,0), 5)
                       cv2.imshow("Annotated", annotated)              
                       cv2.waitKey(1)


                    cam_to_w_xyz = p = pose_dict["camera_position"]
                    body_to_w_quat = q = pose_dict["camera_orientation"]

                    if (pose_dict["reprojection_error"] < 5 and pose_dict["positive_depth"] and CONNECT_MAVLINK):
                        # Calculate actual pipeline latency just for monitoring
                        pipeline_latency_ms = (int(time.monotonic() * 1e6) - image_capture_time_usec) / 1000.0
                        print(f"Sending ODOMETRY. Msg Time: {mavlink_timestamp} | Pipeline Latency: {pipeline_latency_ms:.3f}ms")
                        
                        msg = mavutil.mavlink.MAVLink_odometry_message(
                            mavlink_timestamp,
                            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                            mavutil.mavlink.MAV_FRAME_BODY_FRD,
                            *p,
                            [q[3], q[0], q[1], q[2]],  # w, x, y, z

                            # Velocities are unavialable / Ignored
                            float('nan'), float('nan'), float('nan'),
                            float('nan'), float('nan'), float('nan'),

                            # Corrected 21-value Upper-Triangle Pose Covariance
                            [
                                0.0009, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0009, 0.0, 0.0, 0.0, 0.0,
                                            0.0009, 0.0, 0.0, 0.0,
                                                0.01, 0.0, 0.0,
                                                        0.01, 0.0,
                                                            0.01
                            ],

                            # Tell EKF velocity variance is invalid since velocities are NaN
                            [-1.0] + [0.0]*20, 

                            100,  # quality
                            mavutil.mavlink.MAV_ESTIMATOR_TYPE_VISION
                        )
                        
                        m.mav.send(msg)


        elif markertype == 'aruco':
            marker_found = False
            image_undistorted_gray = cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2GRAY)

            print('Looking for Aruco')
            corners, ids, _ = detector.detectMarkers(image_undistorted_gray)
            print('Found ids', ids)
            if ids is not None:
                ids = ids.flatten()
                cv2.aruco.drawDetectedMarkers(image_undistorted, corners, ids)

                for i, marker_id in enumerate(ids):
                    if marker_id == target_id:
                        marker_found = True
                        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                            [corners[i]],
                            marker_size,
                            new_K,
                            np.zeros(4)
                        )

                        rvec = rvec[0][0]
                        tvec = tvec[0][0]

                        cv2.drawFrameAxes(image_undistorted, new_K, np.zeros(4), rvec, tvec, 0.05)

                        R_wld_to_cam, _ = cv2.Rodrigues(rvec)
                        T_wld_to_cam = np.eye(4)
                        T_wld_to_cam[:3, :3] = R_wld_to_cam
                        T_wld_to_cam[:3, 3] = tvec.flatten()

                        # Invert transform
                        T_cam_to_wld = np.linalg.inv(T_wld_to_cam)
                        #x, y, z = tvec[0], tvec[1], tvec[2]

            if (marker_found and CONNECT_MAVLINK):    
                R_wld_to_cam, _ = cv2.Rodrigues(rvec)
                #print('tvec', tvec)
                T_wld_to_cam = np.eye(4)
                T_wld_to_cam[:3, :3] = R_wld_to_cam
                T_wld_to_cam[:3, 3] = tvec.flatten()

                # Invert transform
                T_cam_to_wld = np.linalg.inv(T_wld_to_cam)
                T_cam_to_frd = np.array([
                  [0,  1, 0, 0],
                  [1,  0, 0, 0],
                  [0, 0, -1, 0],
                  [0,  0, 0, 1],
                ], dtype=float)

                #T_drone_to_wld = T_cam_to_wld 
                T_drone_to_wld = T_cam_to_frd @ T_cam_to_wld
                x, y, z = T_drone_to_wld[:3,3]
                text = f"ID {marker_id} X:{x:.2f} Y:{y:.2f} Z:{z:.2f} m"
                #print(text)
                cv2.putText(
                    image_undistorted,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

#            cv2.imshow("Original)", frame)
#            cv2.waitKey(1)
                cv2.imshow("Undistorted Image", image_undistorted)
                cv2.waitKey(1)

                p = cam_pos = T_drone_to_wld[:3, 3]
                q = cam_orient_quat = R.from_matrix(T_drone_to_wld[:3, :3]).as_quat()  # (x, y, z, w)
            
                # Calculate actual pipeline latency just for monitoring
                pipeline_latency_ms = (int(time.monotonic() * 1e6) - image_capture_time_usec) / 1000.0
                print(f"Sending ODOMETRY. Img Time: {image_capture_time_usec} | Pipeline Latency: {pipeline_latency_ms:.3f}ms")
                print("pose:", p)
                msg = mavutil.mavlink.MAVLink_odometry_message(
                    image_capture_time_usec,
                    mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                    mavutil.mavlink.MAV_FRAME_BODY_FRD,
                    *p,
                    [q[3], q[0], q[1], q[2]],  # w, x, y, z

                    # Velocities are unavialable / Ignored
                    float('nan'), float('nan'), float('nan'),
                    float('nan'), float('nan'), float('nan'),

                    # Corrected 21-value Upper-Triangle Pose Covariance
                    [
                        0.005, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.005, 0.0, 0.0, 0.0, 0.0,
                                    0.005, 0.0, 0.0, 0.0,
                                        0.005, 0.0, 0.0,
                                                0.005, 0.0,
                                                    0.005
                    ],

                    # Tell EKF velocity variance is invalid since velocities are NaN
                    [-1.0] + [0.0]*20, 

                    100,  # quality
                    mavutil.mavlink.MAV_ESTIMATOR_TYPE_VISION
                )
                
                m.mav.send(msg)

#        if cv2.waitKey(1) & 0xFF == ord("q"):
#            cv2.destroyAllWindows()
#            break  

    # finally:
    #     # -------------------------
    #     # Cleanup
    #     # -------------------------
    #     if cap is not None:
    #         cap.release()

    #     if picam2 is not None:
    #         picam2.stop()

    #     cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_lights_sendodometry(brightness_threshold)

latest_lighttarget_location = None
latest_lighttarget_pose = None
latest_arucotarget_location = None
latest_arucotarget_pose = None

def get_pose_from_lightmarker(stop_event,
    brightness_threshold,
    min_area: int = 1,
    max_area: int = 40000):
    # This function is similar to detect_lights_sendodometry but only returns the estimated pose without any MAVLink communication or visualization. It can be used for unit testing the pose estimation logic in isolation.

    global camera_matrix_rgb, dist_coeffs_rgb

    picam2 = None
    cap = None

    from picamera2 import Picamera2
    picam2 = Picamera2(0)
    config = picam2.create_video_configuration(
        main={"size": (int(s*1456), int(s*1088)), "format": "RGB888"},
        buffer_count=1,
        queue=False
    )
    picam2.configure(config)
    controls = {
    "ExposureTime": exposure_time_rgb,   # microseconds
    "AnalogueGain": 1.0}
    picam2.set_controls(controls)
    picam2.start()

    camera_matrix_rgb = scale_camera_matrix(camera_matrix_rgb, s)

    if (show_visualization):
       cv2.namedWindow("Thresholded", cv2.WINDOW_NORMAL)
       cv2.resizeWindow('Thresholded', 700, 700) 

       cv2.namedWindow("Annotated", cv2.WINDOW_NORMAL)
       cv2.resizeWindow('Annotated', 700, 700) 

       cv2.namedWindow("Annotated_colors", cv2.WINDOW_NORMAL)
       cv2.resizeWindow('Annotated_colors', 700, 700) 

    camera_matrix_rgb = rotate_intrinsics_180(camera_matrix_rgb, s*1456, s*1088) # because frame is rotated below
        
    while not stop_event.is_set():
    # while True:
#        time.sleep(1)
        # -------------------------
        # Read frame
        # -------------------------

        # Use monotonic clock, NOT time.time()
        image_capture_time_usec = int(time.monotonic() * 1e6)
        mavlink_timestamp = image_capture_time_usec
        frame = picam2.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        red = frame[:, :, 0]
        green = frame[:, :, 1]

        green = diff_image_RG = red.astype(np.int16) - green.astype(np.int16)

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix_rgb,
        dist_coeffs_rgb,
        (w, h),
        np.eye(3),
        balance=0.0)

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix_rgb,
            dist_coeffs_rgb,
            np.eye(3),
            new_K,
            (w, h),
            cv2.CV_16SC2,
        )
        
        image_undistorted = cv2.remap(
            frame, map1, map2, interpolation=cv2.INTER_LINEAR)

        green_undistorted = cv2.remap(
            green, map1, map2, interpolation=cv2.INTER_LINEAR)


#            print('Searching for LEDs...')
        blurred = cv2.GaussianBlur(image_undistorted, blur_window, 0)

        annotated = blurred.copy()
        annotated_colors = blurred.copy()
#            blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        blurred = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        
        # threshold for top 10%
        brightness_mask = np.percentile(blurred, 90)

        # select pixels above threshold
        top_pixels = blurred[blurred >= brightness_mask]

        avg_top_10_intensities = np.mean(top_pixels)
        # brightness_threshold = avg_top_10_intensities - 10

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

#            print('total circles', len(circles))
        # deterministic order
        order = np.lexsort((circles[:, 1], circles[:, 0]))
        circles = circles[order]
        
        # fitered_circles = circles
        fitered_circles = filter_circles_same_line_similar_radius(circles, radius_tol, line_tol, min_group_size, cross_ratio_tol) #cr: 0.015
        filteredcircles_avgcolor = []
#            print('filtered circles', len(fitered_circles))

        for circle in fitered_circles:    
            center = (int(circle[0]), int(circle[1]))
            radius = int(circle[2])
            h, w = annotated_colors.shape[:2]
            y, x = np.ogrid[:h, :w]
            inside_circle = (x-center[0])**2 + (y-center[1])**2 <= radius**2
            led_mean = int(green_undistorted[inside_circle].mean())
            filteredcircles_avgcolor.append(led_mean)

        filteredcircles_avgcolor_sorted = np.argsort(filteredcircles_avgcolor)
    
        if (show_visualization):
            led_count = 0
            for circle in fitered_circles:    
                # Draw annotation
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])
                h, w = annotated_colors.shape[:2]
                y, x = np.ogrid[:h, :w]
                inside_circle = (x-center[0])**2 + (y-center[1])**2 <= radius**2
                led_mean = int(green_undistorted[inside_circle].mean())
                led_type = np.where(filteredcircles_avgcolor_sorted==led_count)[0] <= 3  #first 4 LEDs based on min avg intensity
                ann_circle_color = (0,255,0) if (led_type==1) else (0,0,255)
                
                filteredcircles_avgcolor.append(led_mean)
                cv2.circle(annotated_colors, center, radius + 1, ann_circle_color, 2)
                cv2.putText(
                    annotated_colors,
                    f"Int {led_mean}",
                    (center[0] + 5, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    ann_circle_color,
                    1,
                    cv2.LINE_AA,
                    )

                led_count += 1

            ## Show images
            cv2.imshow("Thresholded", thresh)
            cv2.imshow("Annotated_colors", annotated_colors)
            cv2.waitKey(1)

#        if cv2.waitKey(1) & 0xFF == ord("q"):
#            cv2.destroyAllWindows()
#            break

        # print(len(fitered_circles), "circles after line/radius filtering")
        if (len(fitered_circles) == 8):
            # print("Attempting pose estimation with", len(fitered_circles), "circles...")
            image_points, object_points, pose_dict = pose_from_colored_leds(fitered_circles, filteredcircles_avgcolor_sorted, new_K, np.zeros((1, 4)))

            if (show_visualization):
                led_count = 0
                for image_point in image_points:    
                    # Draw annotation
                    center = (int(image_point[0]), int(image_point[1]))
                    #print('center', center)
                    cv2.circle(annotated, center, 10, (0, 255, 0), 2)
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

#                cv2.imshow("Annotated", annotated)              
#                print(f"Detected LEDs: {led_count}")

            if (len(image_points)%4 == 0) and (len(image_points) == len(object_points)):
                #pose_dict = estimate_planar_pose(object_points, image_points, new_K, np.zeros((1, 4)))
                # print('Reprojection error:', pose_dict["reprojection_error"])
                print('Reprojection error:', pose_dict["reprojection_error"])
                print('Positive depth', pose_dict["positive_depth"])
                x = pose_dict["camera_position"][0]
                y = pose_dict["camera_position"][1]
                z = pose_dict["camera_position"][2]
                text = f"Drone location: X:{x:.2f} Y:{y:.2f} Z:{z:.2f} m, {pose_dict["positive_depth"]}, {pose_dict["reprojection_error"]:.2f}"

                print("Estimated pose:", x, y, z) if (pose_dict["reprojection_error"] < reproj_threshold and pose_dict["positive_depth"]) else print("Pose estimation failed")

                if (show_visualization and pose_dict["reprojection_error"] < reproj_threshold):
                    cv2.putText(
                    annotated,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                    )

                    projected = pose_dict["projected_points"]
                    for p_img, p_proj in zip(image_points, projected):
                        cv2.circle(annotated, tuple(p_img.astype(int)), 10, (0,255,0), -1)
                        cv2.circle(annotated, tuple(p_proj.astype(int)), 10, (0,0,255), -1)
                        cv2.line(annotated,
                            tuple(p_img.astype(int)),
                            tuple(p_proj.astype(int)),
                            (255,0,0), 5)
                    cv2.imshow("Annotated", annotated)              
                    cv2.waitKey(1)

                cam_to_w_xyz = p = pose_dict["camera_position"]
                body_to_w_quat = q = pose_dict["camera_orientation"]
                global latest_lighttarget_location
                global latest_lighttarget_pose
                if (pose_dict["reprojection_error"] < 5 and pose_dict["positive_depth"]):
                    latest_lighttarget_location = p
                    latest_lighttarget_pose = q
                else:
                    latest_lighttarget_location = None
                    latest_lighttarget_pose = None

def get_pose_from_arucomarker():
    global camera_matrix_mono
    global dist_coeffs_mono
    
    picam2 = None
    cap = None

    from picamera2 import Picamera2
    picam2 = Picamera2(1)
    config = picam2.create_video_configuration(
        main={"size": (int(s*1456), int(s*1088)), "format": "RGB888"},
        buffer_count=1,
        queue=False
    )
    picam2.configure(config)
    controls = {
    "ExposureTime": exposure_time_mono,   # microseconds
    "AnalogueGain": 1.0}
    picam2.set_controls(controls)
    picam2.start()

    camera_matrix_mono = scale_camera_matrix(camera_matrix_mono, s)
    camera_matrix_mono = rotate_intrinsics_180(camera_matrix_mono, s*1456, s*1088) # because frame is rotated below
        
    while True:
#        time.sleep(1)
        # -------------------------
        # Read frame
        # -------------------------

        # Use monotonic clock, NOT time.time()
        image_capture_time_usec = int(time.monotonic() * 1e6)
        mavlink_timestamp = image_capture_time_usec
        frame = picam2.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        green = frame[:, :, 1]

        h, w = frame.shape[:2]        
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix_mono,
        dist_coeffs_mono,
        (w, h),
        np.eye(3),
        balance=0.0)

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix_mono,
            dist_coeffs_mono,
            np.eye(3),
            new_K,
            (w, h),
            cv2.CV_16SC2,
        )
        
        image_undistorted = cv2.remap(
            frame, map1, map2, interpolation=cv2.INTER_LINEAR)

        green_undistorted = cv2.remap(
            green, map1, map2, interpolation=cv2.INTER_LINEAR)

        ### ARUCO Detection
        aruco_marker_found = False
        print('Looking for Aruco')
        corners, ids, _ = detector.detectMarkers(image_undistorted)
        print('Found ids', ids)
        if ids is not None:
            ids = ids.flatten()
            cv2.aruco.drawDetectedMarkers(image_undistorted, corners, ids)

            for i, marker_id in enumerate(ids):
                if marker_id == target_id:
                    aruco_marker_found = True
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[i]],
                        marker_size,
                        new_K,
                        np.zeros(4)
                    )

                    rvec = rvec[0][0]
                    tvec = tvec[0][0]

                    if (show_visualization):
                        cv2.drawFrameAxes(image_undistorted, new_K, np.zeros(4), rvec, tvec, 0.05)

                    R_wld_to_cam, _ = cv2.Rodrigues(rvec)
                    T_wld_to_cam = np.eye(4)
                    T_wld_to_cam[:3, :3] = R_wld_to_cam
                    T_wld_to_cam[:3, 3] = tvec.flatten()

                    # Invert transform
                    T_cam_to_wld = np.linalg.inv(T_wld_to_cam)
    
                    x, y, z = tvec[0], tvec[1], tvec[2]
                    x, y, z = T_cam_to_wld[:3,3]
                    text = f"ID {marker_id} X:{x:.2f} Y:{y:.2f} Z:{z:.2f} m"
                    #print(text)
                    if (show_visualization):
                        cv2.putText(
                            image_undistorted,
                            text,
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

#            cv2.imshow("Original)", frame)
#            cv2.waitKey(1)
            if (show_visualization):
                cv2.imshow("Undistorted Image", image_undistorted)
                cv2.waitKey(1)

        if (aruco_marker_found):    
            R_wld_to_cam, _ = cv2.Rodrigues(rvec)
            #print('tvec', tvec)
            T_wld_to_cam = np.eye(4)
            T_wld_to_cam[:3, :3] = R_wld_to_cam
            T_wld_to_cam[:3, 3] = tvec.flatten()

            # Invert transform
            T_cam_to_wld = np.linalg.inv(T_wld_to_cam)
            T_cam_to_drone = np.array([
                [0,  1, 0, 0],
                [1,  0, 0, 0],
                [0, 0, -1, 0],
                [0,  0, 0, 1],
            ], dtype=float)

            #T_drone_to_wld = T_cam_to_wld 
            T_drone_to_wld = T_cam_to_drone @ T_cam_to_wld
            
            p = cam_pos = T_drone_to_wld[:3, 3]
            q = cam_orient_quat = R.from_matrix(T_drone_to_wld[:3, :3]).as_quat()  # (x, y, z, w)

            latest_arucotarget_location = p
            latest_arucotarget_pose = q
        
        else:
            latest_arucotarget_location = None
            latest_arucotarget_pose = None

def get_latest_lighttarget_location():
    global latest_lighttarget_location
    global latest_lighttarget_pose
    return latest_lighttarget_location, latest_lighttarget_pose

def get_latest_arucotarget_location():
    global latest_arucotarget_location
    global latest_arucotarget_pose
    return latest_arucotarget_location, latest_arucotarget_pose
