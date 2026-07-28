import threading
import time
import cv2
import numpy as np
from itertools import combinations
import math
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
import asyncio

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
MAVLINK_MULTIPLE_CONNECTIONS = True  # If we are also sending Mocap data to drone on serial then set this to True to avoid conflicts. Requires mavlink_routerd running on the pi.

if (not MAVLINK_MULTIPLE_CONNECTIONS):
    serial_ip = "/dev/ttyACM0"  # Serial port for MAVLink connection
else:
    serial_ip = "udp:127.0.0.1:14600"  # UDP port for MAVLink connection

rgb_cameratype = 'fisheye' # 'fisheye' or 'pinhole'
mono_cameratype = 'fisheye' # 'fisheye' or 'pinhole'

markertype = 'aruco'  # 'Lshape' or 'aruco'
show_visualization = False
drone_attitude_reliable = True

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

latest_attitude = None
latest_attitude_time = None
mavlink_lock = threading.Lock()

async def attitude_loop(drone):
    print('Starting attitude listener loop')
    global latest_attitude
    global latest_attitude_time

    await drone.telemetry.set_rate_attitude_quaternion(50)

    async for attitude in drone.telemetry.attitude_quaternion():
#        print('Awaiting attitude from drone')
        R_drone_to_ned = R.from_quat([
            attitude.x,
            attitude.y,
            attitude.z,
            attitude.w
        ]).as_matrix()
#        print('R_drone_to_ned', R_drone_to_ned)
        with mavlink_lock:
#            print('Setting latest_attitude', latest_attitude)
            latest_attitude = R_drone_to_ned
            latest_attitude_time = time.monotonic()
                    
        
def attitude_listener_mavlink(master):
    global latest_attitude
    global latest_attitude_time

    while True:
        with mavlink_lock:
            msg = master.recv_match(
                type="ATTITUDE_QUATERNION",
                blocking=True
            )

            if msg is not None:
                R_drone_to_ned = R.from_quat([
                    msg.q2,  # x
                    msg.q3,  # y
                    msg.q4,  # z
                    msg.q1   # w
                ]).as_matrix()

                latest_attitude = R_drone_to_ned
                latest_attitude_time = time.monotonic()
    
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
        threading.Thread(
            target=attitude_listener_mavlink,
            args=(m,),
            daemon=True
        ).start()

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
        with mavlink_lock:
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

    if (markertype == 'aruco'): # the mono camera is 180 degrees titled
        camera_matrix = rotate_intrinsics_180(camera_matrix, s*full_size_ar[0], s*full_size_ar[1]) # because frame is rotated below
        
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
            with mavlink_lock:
                rot_drone_to_ned = latest_attitude.copy()
                attitude_age = time.monotonic() - latest_attitude_time
   
            frame = picam2.capture_array()
        print('attitude_age', attitude_age)    
        height, width  = frame.shape[:2]
        frame = cv2.resize(frame, (int(s*width), int(s*height)))
        if (markertype == 'aruco'): # the mono camera is 180 degrees titled
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
                   led_type = np.where(filteredcircles_avgcolor_sorted==led_count)[0] <= 2  #first 3 LEDs based on min avg intensity
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
               cv2.waitKey(1)
               cv2.imshow("Annotated", annotated)
               cv2.waitKey(1)
               cv2.imshow("Annotated_colors", annotated_colors)
               cv2.waitKey(1)
               cv2.imshow("Undistorted Image", image_undistorted)
               cv2.waitKey(1)

    #        if cv2.waitKey(1) & 0xFF == ord("q"):
    #            cv2.destroyAllWindows()
    #            break

            # print(len(fitered_circles), "circles after line/radius filtering")
            if (len(fitered_circles) == 7):
                # print("Attempting pose estimation with", len(fitered_circles), "circles...")
                image_points, object_points, pose_dict = pose_from_colored_leds(fitered_circles, filteredcircles_avgcolor_sorted, new_K, np.zeros((1, 4)), drone_attitude_reliable, rot_drone_to_ned)

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

                if (len(image_points)%7 == 0) and (len(image_points) == len(object_points)):
                    #pose_dict = estimate_planar_pose(object_points, image_points, new_K, np.zeros((1, 4)))
                    # print('Reprojection error:', pose_dict["reprojection_error"])
#                    print('Reprojection error:', pose_dict["reprojection_error"])
#                    print('Positive depth', pose_dict["positive_depth"])
                    if (drone_attitude_reliable):
                        x = pose_dict["marker_position"][0]
                        y = pose_dict["marker_position"][1]
                        z = pose_dict["marker_position"][2]

                        cam_to_w_xyz = p = pose_dict["marker_position"]
                        body_to_w_quat = q = [float('nan'), float('nan'), float('nan'), float('nan')]

                    elif (not drone_attitude_reliable):
                        x = pose_dict["camera_position"][0]
                        y = pose_dict["camera_position"][1]
                        z = pose_dict["camera_position"][2]
                        
                        cam_to_w_xyz = p = pose_dict["camera_position"]
                        body_to_w_quat = q = pose_dict["camera_orientation"]
                        

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
                    print('p', p)
                    print('q', q)

                    if (pose_dict["reprojection_error"] < 5 and pose_dict["positive_depth"] and CONNECT_MAVLINK):
                        # Calculate actual pipeline latency just for monitoring
                        pipeline_latency_ms = (int(time.monotonic() * 1e6) - image_capture_time_usec) / 1000.0
#                        print(f"Sending ODOMETRY. Msg Time: {mavlink_timestamp} | Pipeline Latency: {pipeline_latency_ms:.3f}ms")
     
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
                        
#                        m.mav.send(msg)

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

            if (marker_found and CONNECT_MAVLINK):    
                if (drone_attitude_reliable):
                    R_wld_to_cam, _ = cv2.Rodrigues(rvec)
                    T_wld_to_cam = np.eye(4)
                    trans_marker_to_cam = tvec.flatten()
                    rot_cam_to_drone = np.array([
                        [ 0, -1,  0],
                        [ 1,  0,  0],
                        [ 0,  0,  1],
                    ])
            
                    trans_marker_to_drone = rot_cam_to_drone @ trans_marker_to_cam + CAMERA_MONO_TO_BODY_FRD_M
                    trans_marker_to_ned = rot_drone_to_ned @ trans_marker_to_drone

                    x, y, z = trans_marker_to_ned
                    cam_orient_quat = q = [float('nan'), float('nan'), float('nan'), float('nan')]
                    
                    cam_pos = p = trans_marker_to_ned
                    cam_orient_quat = q = rot_drone_to_ned @ rot_cam_to_drone @ R_wld_to_cam
                    
                else:
                    R_wld_to_cam, _ = cv2.Rodrigues(rvec)
                    #print('tvec', tvec)
                    T_wld_to_cam = np.eye(4)
                    T_wld_to_cam[:3, :3] = R_wld_to_cam
                    T_wld_to_cam[:3, 3] = tvec.flatten()

                    # Invert transform
                    T_cam_to_wld = np.linalg.inv(T_wld_to_cam)
                    basis_change = np.array([
                      [0,  1, 0, 0],
                      [1,  0, 0, 0],
                      [0, 0, -1, 0],
                      [0,  0, 0, 1],
                    ], dtype=float)

                    #T_drone_to_wld = T_cam_to_wld 
                    T_drone_to_wld = basis_change @ T_cam_to_wld
                    x, y, z = T_drone_to_wld[:3,3]
                    
                    cam_pos = p = T_drone_to_wld[:3,3]
                    cam_orient_quat = q =  R.from_matrix(T_drone_to_wld[:3, :3]).as_quat() 
                    text = f"ID {marker_id} DroneLoc X:{x:.2f} Y:{y:.2f} Z:{z:.2f} m"

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
                
#                m.mav.send(msg)

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
latest_lighttarget_orientation = None
latest_dronelocation_withlighttarget = None
latest_droneorientation_withlighttarget = None
latest_dronelocation_withlighttarget_timestamp = None
latest_dronelocation_withlighttarget_sequence = 0

latest_arucotarget_location = None
latest_arucotarget_orientation = None
latest_dronelocation_witharucotarget = None
latest_droneorientation_witharucotarget = None
latest_dronelocation_witharucotarget_timestamp = None
latest_dronelocation_witharucotarget_sequence = 0

vision_lock = threading.Lock()
latest_lighttarget_timestamp = None
latest_lighttarget_sequence = 0
latest_arucotarget_timestamp = None
latest_arucotarget_sequence = 0
MAX_ATTITUDE_AGE_S = 0.15
MAX_ARUCO_REPROJECTION_ERROR_PX = 3.0
MAX_ARUCO_RANGE_M = 6.0
CAMERA_RGB_TO_BODY_FRD_M = np.array([0.0, 0.0, 0.0])
CAMERA_MONO_TO_BODY_FRD_M = np.array([0.0, 0.0, 0.0])

def _publish_light_target(position, orientation, capture_time):
    global latest_lighttarget_location, latest_lighttarget_orientation, latest_lighttarget_timestamp, latest_lighttarget_sequence
    with vision_lock:
        latest_lighttarget_location = np.asarray(position, dtype=float).copy()
        latest_lighttarget_orientation = orientation
        latest_lighttarget_timestamp = capture_time
        latest_lighttarget_sequence += 1

def _clear_light_target():
    global latest_lighttarget_location, latest_lighttarget_orientation, latest_lighttarget_timestamp
    with vision_lock:
        latest_lighttarget_location = None
        latest_lighttarget_orientation = None
        latest_lighttarget_timestamp = None

def _publish_aruco_target(position, orientation, capture_time):
    global latest_arucotarget_location, latest_arucotarget_orientation, latest_arucotarget_timestamp, latest_arucotarget_sequence
    with vision_lock:
        latest_arucotarget_location = np.asarray(position, dtype=float).copy()
        latest_arucotarget_orientation = orientation
        latest_arucotarget_timestamp = capture_time
        latest_arucotarget_sequence += 1

def _clear_aruco_target():
    global latest_arucotarget_location, latest_arucotarget_orientation, latest_arucotarget_timestamp
    with vision_lock:
        latest_arucotarget_location = None
        latest_arucotarget_orientation = None
        latest_arucotarget_timestamp = None

from picamera2 import Picamera2
picam2 = Picamera2(0)  # Use camera port 0 (RGB camera) for L-shape
full_size = picam2.camera_properties["PixelArraySize"]
config = picam2.create_video_configuration(
            main={"size": full_size, "format": "RGB888"},
            buffer_count=1,
            queue=False)
            
picam2.configure(config)
controls = {
    "ScalerCrop": (0, 0, *full_size),
    "ExposureTime": exposure_time_rgb,   # microseconds
    "AnalogueGain": 1.0}
picam2.set_controls(controls)
picam2.start()

def get_pose_from_lightmarker(stop_event, pose_type, drone,
    brightness_threshold,
    min_area: int = 1,
    max_area: int = 40000):
    # This function is similar to detect_lights_sendodometry but only returns the estimated pose without any MAVLink communication or visualization. It can be used for unit testing the pose estimation logic in isolation.

    global camera_matrix_rgb, dist_coeffs_rgb
    global latest_lighttarget_location
    global latest_lighttarget_orientation
    global latest_dronelocation_withlighttarget
    global latest_droneorientation_withlighttarget
    global latest_dronelocation_withlighttarget_timestamp
    global latest_dronelocation_withlighttarget_sequence

    camera_matrix = camera_matrix_rgb.copy()
    dist_coeffs = dist_coeffs_rgb.copy()

    cap = None
    
    camera_matrix = scale_camera_matrix(camera_matrix, s)

    if (show_visualization):
       cv2.namedWindow("Thresholded", cv2.WINDOW_NORMAL)
       cv2.resizeWindow('Thresholded', 700, 700) 

       cv2.namedWindow("Annotated", cv2.WINDOW_NORMAL)
       cv2.resizeWindow('Annotated', 700, 700) 

       cv2.namedWindow("Annotated_colors", cv2.WINDOW_NORMAL)
       cv2.resizeWindow('Annotated_colors', 700, 700) 

    #camera_matrix = rotate_intrinsics_180(camera_matrix, s*1456, s*1088) # rotation was needed for mono camera, but not for rgb camera
        
    while not stop_event.is_set():
    # while True:
#        time.sleep(1)
        # -------------------------
        # Read frame
        # -------------------------

        # Use monotonic clock, NOT time.time()
        image_capture_time_usec = int(time.monotonic() * 1e6)
        mavlink_timestamp = image_capture_time_usec
        with mavlink_lock:
            rot_drone_to_ned = latest_attitude.copy()
            attitude_age = time.monotonic() - latest_attitude_time

        frame = picam2.capture_array()
        height, width  = frame.shape[:2]
        frame = cv2.resize(frame, (int(s*width), int(s*height)))
#        frame = cv2.rotate(frame, cv2.ROTATE_180) # rotation was needed for mono camera, but not for rgb camera
        height, width  = frame.shape[:2]

        red = frame[:, :, 0]
        green = frame[:, :, 1]

        green = diff_image_RG = red.astype(np.int16) - green.astype(np.int16)

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix,
        dist_coeffs,
        (width,height),
        np.eye(3),
        balance=0.0)

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            np.eye(3),
            new_K,
            (width, height),
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
        if (len(fitered_circles) == 7):
            if (pose_type == 'target'):
                drone_attitude_reliable = True # if we are just trying to estimate the location of the light marker, we can use the drone's attitude to help with pose estimation
            elif (pose_type == 'drone'):
                drone_attitude_reliable = False # if the drone's attitude is not reliable, we can still estimate the drone's pose relative to the light marker using the LEDs but it suffers from planar ambiguity

            image_points, object_points, pose_dict = pose_from_colored_leds(fitered_circles, filteredcircles_avgcolor_sorted, new_K, np.zeros((1, 4)),  drone_attitude_reliable,  rot_drone_to_ned)

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

            if (len(image_points)%7 == 0) and (len(image_points) == len(object_points)):
                #pose_dict = estimate_planar_pose(object_points, image_points, new_K, np.zeros((1, 4)))
                # print('Reprojection error:', pose_dict["reprojection_error"])
                print('Reprojection error:', pose_dict["reprojection_error"])
                print('Positive depth', pose_dict["positive_depth"])

                if (drone_attitude_reliable):
                    x = pose_dict["marker_position"][0]
                    y = pose_dict["marker_position"][1]
                    z = pose_dict["marker_position"][2]

                    cam_to_w_xyz = p = pose_dict["marker_position"]
                    body_to_w_quat = q = (rot_drone_to_ned @ np.array([[0,-1,0],[1,0,0],[0,0,1]]) @ pose_dict["R"])

                elif (not drone_attitude_reliable):
                    x = pose_dict["camera_position"][0]
                    y = pose_dict["camera_position"][1]
                    z = pose_dict["camera_position"][2]
                    
                    cam_to_w_xyz = p = pose_dict["camera_position"]
                    body_to_w_quat = q = pose_dict["camera_orientation"]

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

                if (pose_dict["reprojection_error"] < 5 and pose_dict["positive_depth"]):
                    if (pose_type == 'target'):
                        _publish_light_target(p, q, time.monotonic())
                        # latest_dronelocation_withlighttarget = None
                        # latest_droneorientation_withlighttarget = None

                    elif (pose_type == 'drone'):
                        with vision_lock:
                            latest_dronelocation_withlighttarget = np.asarray(p, dtype=float).copy()
                            latest_droneorientation_withlighttarget = q
                            latest_dronelocation_withlighttarget_timestamp = time.monotonic()
                            latest_dronelocation_withlighttarget_sequence += 1
                        # latest_lighttarget_location = None
                        # latest_lighttarget_orientation = None

                else:
                        latest_lighttarget_location = None
                        latest_lighttarget_orientation = None
                        latest_dronelocation_withlighttarget = None
                        latest_droneorientation_withlighttarget = None
                        latest_dronelocation_withlighttarget_timestamp = None
                        print('Setting light target and drone location to None, marker not found')

picam2_ar = Picamera2(1)
full_size_ar = picam2_ar.camera_properties["PixelArraySize"]
config_ar = picam2_ar.create_video_configuration(
            main={"size": full_size_ar, "format": "RGB888"},
            buffer_count=1,
            queue=False)

picam2_ar.configure(config_ar)
controls_ar = {
    "ScalerCrop": (0, 0, *full_size_ar),
    "ExposureTime": exposure_time_mono,   # microseconds
    "AnalogueGain": 1.0}
picam2_ar.set_controls(controls_ar)
picam2_ar.start()

def get_pose_from_arucomarker(pose_type, drone):
    global camera_matrix_mono
    global dist_coeffs_mono

    global latest_arucotarget_location
    global latest_arucotarget_orientation
    global latest_dronelocation_witharucotarget
    global latest_droneorientation_witharucotarget
    global latest_dronelocation_witharucotarget_timestamp
    global latest_dronelocation_witharucotarget_sequence

    camera_matrix = camera_matrix_mono.copy()
    dist_coeffs = dist_coeffs_mono.copy()

    cap = None

    camera_matrix = scale_camera_matrix(camera_matrix, s) # resizing the image below
    camera_matrix = rotate_intrinsics_180(camera_matrix, s*full_size_ar[0], s*full_size_ar[1]) # because frame is rotated below
        
    while True:
#        time.sleep(1)
        # -------------------------
        # Read frame
        # -------------------------

        # Use monotonic clock, NOT time.time()
        image_capture_time_usec = int(time.monotonic() * 1e6)
        mavlink_timestamp = image_capture_time_usec
        with mavlink_lock:
            rot_drone_to_ned = latest_attitude.copy()
            attitude_age = time.monotonic() - latest_attitude_time

        frame = picam2_ar.capture_array()
        height, width  = frame.shape[:2]
        frame = cv2.resize(frame, (int(s*width), int(s*height)))
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        green = frame[:, :, 1]

        h, w = frame.shape[:2]        
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix,
        dist_coeffs,
        (w, h),
        np.eye(3),
        balance=0.0)

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

#            cv2.imshow("Original)", frame)
#            cv2.waitKey(1)
            if (show_visualization):
                cv2.imshow("Undistorted Image", image_undistorted)
                cv2.waitKey(1)

            if (aruco_marker_found):    
                if (pose_type == 'target'):
                    R_wld_to_cam, _ = cv2.Rodrigues(rvec)
                    T_wld_to_cam = np.eye(4)
                    trans_marker_to_cam = tvec.flatten()
                    rot_cam_to_drone = np.array([
                        [ 0, -1,  0],
                        [ 1,  0,  0],
                        [ 0,  0,  1],
                    ])
            
                    trans_marker_to_drone = rot_cam_to_drone @ trans_marker_to_cam # + trans_cam_to_drone
                    trans_marker_to_ned = rot_drone_to_ned @ trans_marker_to_drone # + trans_drone_to_ned

                    x, y, z = trans_marker_to_ned
                    text = f"ID {marker_id} MarkerLoc X:{x:.2f} Y:{y:.2f} Z:{z:.2f} m"
                    
                    cam_pos = p = trans_marker_to_ned
                    cam_orient_quat = q = rot_drone_to_ned @ rot_cam_to_drone @ R_wld_to_cam

                    _publish_aruco_target(p, q, time.monotonic())
                    
                elif (pose_type == 'drone'):
                    R_wld_to_cam, _ = cv2.Rodrigues(rvec)
                    #print('tvec', tvec)
                    T_wld_to_cam = np.eye(4)
                    T_wld_to_cam[:3, :3] = R_wld_to_cam
                    T_wld_to_cam[:3, 3] = tvec.flatten()

                    # Invert transform
                    T_cam_to_wld = np.linalg.inv(T_wld_to_cam)
                    basis_change = np.array([
                      [0,  1, 0, 0],
                      [1,  0, 0, 0],
                      [0, 0, -1, 0],
                      [0,  0, 0, 1],
                    ], dtype=float)

                    #T_drone_to_wld = T_cam_to_wld 
                    T_drone_to_wld = basis_change @ T_cam_to_wld
                    x, y, z = T_drone_to_wld[:3,3]
                    
                    cam_pos = p = T_drone_to_wld[:3,3]
                    cam_orient_quat = q =  R.from_matrix(T_drone_to_wld[:3, :3]).as_quat() 
                    text = f"ID {marker_id} DroneLoc X:{x:.2f} Y:{y:.2f} Z:{z:.2f} m"

                    with vision_lock:
                        latest_dronelocation_witharucotarget = np.asarray(p, dtype=float).copy()
                        latest_droneorientation_witharucotarget = q
                        latest_dronelocation_witharucotarget_timestamp = time.monotonic()
                        latest_dronelocation_witharucotarget_sequence += 1

            else:
                latest_arucotarget_location = None
                latest_arucotarget_orientation = None
                latest_dronelocation_witharucotarget = None
                latest_droneorientation_witharucotarget = None
                latest_dronelocation_witharucotarget_timestamp = None

def get_latest_lighttarget_location():
    with vision_lock:
        p = None if latest_lighttarget_location is None else latest_lighttarget_location.copy()
        return p, latest_lighttarget_orientation

def get_latest_arucotarget_location():
    with vision_lock:
        p = None if latest_arucotarget_location is None else latest_arucotarget_location.copy()
        return p, latest_arucotarget_orientation

def get_latest_pose_from_lightmarker():
    global latest_dronelocation_withlighttarget
    global latest_droneorientation_withlighttarget
    return latest_dronelocation_withlighttarget, latest_droneorientation_withlighttarget

def get_latest_pose_from_arucomarker():
    global latest_dronelocation_witharucotarget
    global latest_droneorientation_witharucotarget
    return latest_dronelocation_witharucotarget, latest_droneorientation_witharucotarget

def get_latest_drone_measurement_from_lightmarker():
    with vision_lock:
        p = None if latest_dronelocation_withlighttarget is None else latest_dronelocation_withlighttarget.copy()
        return p, latest_droneorientation_withlighttarget, latest_dronelocation_withlighttarget_timestamp, latest_dronelocation_withlighttarget_sequence

def get_latest_drone_measurement_from_arucomarker():
    with vision_lock:
        p = None if latest_dronelocation_witharucotarget is None else latest_dronelocation_witharucotarget.copy()
        return p, latest_droneorientation_witharucotarget, latest_dronelocation_witharucotarget_timestamp, latest_dronelocation_witharucotarget_sequence

def get_latest_lighttarget_measurement():
    with vision_lock:
        p = None if latest_lighttarget_location is None else latest_lighttarget_location.copy()
        orientation = None if latest_lighttarget_orientation is None else np.asarray(latest_lighttarget_orientation, dtype=float).copy()
        return p, orientation, latest_lighttarget_timestamp, latest_lighttarget_sequence

def get_latest_arucotarget_measurement():
    with vision_lock:
        p = None if latest_arucotarget_location is None else latest_arucotarget_location.copy()
        orientation = None if latest_arucotarget_orientation is None else np.asarray(latest_arucotarget_orientation, dtype=float).copy()
        return p, orientation, latest_arucotarget_timestamp, latest_arucotarget_sequence
