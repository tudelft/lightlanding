import time
import cv2
import numpy as np
from itertools import combinations
import math
from scipy.spatial.transform import Rotation as R
import os
from scipy.spatial.distance import cdist

os.environ["MAVLINK20"] = "1"
os.environ["MAVLINK_DIALECT"] = "common"
from pymavlink import mavutil
import threading# =========================
# Input source configuration
# =========================
USE_VIDEO_FILE = False          # True = read from video, False = use RPi camera
VIDEO_PATH = "lightrecordingLshape.mp4" # Path to video file when USE_VIDEO_FILE=True
CONNECT_MAVLINK = True             # Whether to connect to MAVLink and send odometry messages

markertype = 'Lshape'  # 'Lshape' or 'aruco'
show_visualization = True

exposure_time = 3000 # microseconds
# L-shape marker setup
radius_tol=0.5 
line_tol=8.0
min_group_size=4 
cross_ratio_tol=0.025
reproj_threshold = 2.5
brightness_threshold = 60
# ArUco setup
marker_size = 0.1   # meters
target_id = 0
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# intrinsics and distortion parameters
camera_matrix = np.array(
 [[972.41752602,   0.,         719.86748972],
 [  0.,         970.82689346, 520.66180438],
 [  0.,           0.,           1.        ]])
dist_coeffs = np.array([-0.13573729,  0.03353202, -0.0345132,   0.01030255])

s=0.6 # scaling down the camera image and intrinsics for faster processing since we only care about large bright blobs (LEDs)
def scale_camera_matrix(K, s):
    K_new = K.copy().astype(float)
    K_new[0, 0] *= s
    K_new[1, 1] *= s
    K_new[0, 2] *= s
    K_new[1, 2] *= s
    return K_new

def get_fc_time_us(master):
    while True:
        msg = master.recv_match(
            blocking=True,
            timeout=1
        )

        if msg is None:
            continue

        if hasattr(msg, "time_boot_ms"):
            return msg.time_boot_ms * 1000

def detect_fiducials(
    brightness_threshold,
    min_area: int = 1,
    max_area: int = 40000,
) -> None:

    global camera_matrix
    global dist_coeffs
	
    if CONNECT_MAVLINK:
        m = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)
        
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
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (int(s*1456), int(s*1088)), "format": "RGB888"},
            buffer_count=1,
            queue=False
        )
        picam2.configure(config)
        controls = {
        "ExposureTime": exposure_time,   # microseconds
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

    camera_matrix = rotate_intrinsics_180(camera_matrix, s*1456, s*1088) # because frame is rotated below
		
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

        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # green = frame  # or frame[:, :, 1] if you want the green channel only

        # -------------------------
        # Undistort frame
        # -------------------------
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

#        green_undistorted = cv2.remap(
#            green, map1, map2, interpolation=cv2.INTER_LINEAR  )

        if (markertype == 'Lshape'):
            print('Searching for LEDs...')
            blurred = cv2.GaussianBlur(image_undistorted, (9, 9), 0)

            annotated = blurred.copy()
            annotated_colors = blurred.copy()
            blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            
            # threshold for top 10%
            brightness_mask = np.percentile(blurred, 90)

            # select pixels above threshold
            top_pixels = blurred[blurred >= brightness_mask]

            avg_top_10_intensities = np.mean(top_pixels)
            # brightness_threshold = avg_top_10_intensities - 10


            #green_undistorted = cv2.cvtColor(green_undistorted, cv2.COLOR_BGR2GRAY)

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
                led_mean = int(image_undistorted[inside_circle].mean())
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
                   led_mean = int(image_undistorted[inside_circle].mean())
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
               cv2.waitKey(1)

    #        if cv2.waitKey(1) & 0xFF == ord("q"):
    #            cv2.destroyAllWindows()
    #            break

            # print(len(fitered_circles), "circles after line/radius filtering")
            if (len(fitered_circles) == 8):
                # print("Attempting pose estimation with", len(fitered_circles), "circles...")
                image_points, object_points, pose_dict = pose_from_colored_leds(fitered_circles, filteredcircles_avgcolor_sorted, new_K, np.zeros((1, 4)))

                # image_points, object_points, info = order_l_shape_markers(fitered_circles)
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
                    # pose_dict = estimate_planar_pose(object_points, image_points, new_K, np.zeros((1, 4)))
                    # print('Reprojection error:', pose_dict["reprojection_error"])
                    print('Reprojection error:', pose_dict["reprojection_error"])
                    print('Positive depth', pose_dict["positive_depth"])
                    x = pose_dict["camera_position"][0]
                    y = pose_dict["camera_position"][1]
                    z = pose_dict["camera_position"][2]
                    print("Estimated pose:", x, y, z) if (pose_dict["reprojection_error"] < reproj_threshold and pose_dict["positive_depth"]) else print("Pose estimation failed")

                    text = f"Drone location: X:{x:.2f} Y:{y:.2f} Z:{z:.2f} m, {pose_dict["positive_depth"]}, {pose_dict["reprojection_error"]:.2f}"
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
                                0.001, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.001, 0.0, 0.0, 0.0, 0.0,
                                            0.001, 0.0, 0.0, 0.0,
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
            print('Looking for Aruco')
            corners, ids, _ = detector.detectMarkers(image_undistorted)
            print('Found ids', ids)
            if ids is not None:
                ids = ids.flatten()
#                cv2.aruco.drawDetectedMarkers(image_undistorted, corners, ids)

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

#                        cv2.drawFrameAxes(image_undistorted, new_K, np.zeros(4), rvec, tvec, 0.05)

#                        x, y, z = tvec
#                        text = f"ID {marker_id} X:{x:.2f} Y:{y:.2f} Z:{z:.2f} m"
                        #print(text)

#                        cv2.putText(
#                            frame,
#                            text,
#                            (20, 40),
#                            cv2.FONT_HERSHEY_SIMPLEX,
#                            0.8,
#                            (0, 255, 0),
#                            2
#                        )

#            cv2.imshow("Original)", frame)
#            cv2.waitKey(1)
#            cv2.imshow("Undistorted Image", image_undistorted)
#            cv2.waitKey(1)

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
                
                p = cam_pos = T_drone_to_wld[:3, 3]
                q = cam_orient_quat = R.from_matrix(T_drone_to_wld[:3, :3]).as_quat()  # (x, y, z, w)

                mavlink_timestamp = sync.get_autopilot_timestamp(image_capture_time_usec)
            
                # Calculate actual pipeline latency just for monitoring
                pipeline_latency_ms = (int(time.monotonic() * 1e6) - image_capture_time_usec) / 1000.0
                print(f"Sending ODOMETRY. Msg Time: {mavlink_timestamp} | Pipeline Latency: {pipeline_latency_ms:.3f}ms")
                print("pose:", p)
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

import numpy as np
from scipy.spatial.distance import cdist
import itertools

def get_endpoints_of_a_noisy_line(points):
    # PCA direction
    center = points.mean(axis=0)
    X = points - center

    _, _, vt = np.linalg.svd(X, full_matrices=False)
    direction = vt[0]

    # Position of each point along the arm
    proj = X @ direction

    # End circles
    idx1 = np.argmin(proj)
    idx2 = np.argmax(proj)

    end1 = points[idx1]
    end2 = points[idx2]

    return end1, end2

def pose_from_colored_leds(fitered_circles, filteredcircles_avgcolor_sorted, new_K, dist_coeffs):
    green_arm = fitered_circles[filteredcircles_avgcolor_sorted[:4]] #first 4 LEDs based on min avg intensity
    amber_arm = fitered_circles[filteredcircles_avgcolor_sorted[4:]]  #other 4 LEDs based on min avg intensity
    
    green_circles_indices = green_arm[:,:2]
    amber_circles_indices = amber_arm[:,:2]

    # green_circles_indices_lexsorted = green_circles_indices[np.lexsort((green_circles_indices[:,1], green_circles_indices[:,0]))]
    # green_corners = green_circles_indices_lexsorted[0], green_circles_indices_lexsorted[-1]

    # amber_circles_indices_lexsorted = amber_circles_indices[np.lexsort((amber_circles_indices[:,1], amber_circles_indices[:,0]))]
    # amber_corners = amber_circles_indices_lexsorted[0], amber_circles_indices_lexsorted[-1]

    green_corners = get_endpoints_of_a_noisy_line(green_circles_indices)
    amber_corners = get_endpoints_of_a_noisy_line(amber_circles_indices)
                                                  
#    image_points_perms = np.array(list(itertools.permutations(np.vstack((amber_edges, green_edges)))))
    image_points_perms = np.array([   #all 4 possible combinations of corner correspondences since we don't know apriori which is which
    [amber_corners[0], amber_corners[1], green_corners[0], green_corners[1]],
    [amber_corners[1], amber_corners[0], green_corners[0], green_corners[1]],
    [amber_corners[0], amber_corners[1], green_corners[1], green_corners[0]],
    [amber_corners[1], amber_corners[0], green_corners[1], green_corners[0]]
    ], dtype=np.float32) 
    
    # 3D object points in meters
    object_points = np.array([
        [0.0,    0.0,    -0.230],  # corner, long amber+green arm, amber led
        [0.375,   0.0,  -0.230],  # short arm, last amber led
        [0.0,  -0.255,    0.0],  # long amber+green arm, first green led
        [0.0,  -0.630,    0.0],  # long amber+green arm, last green led
    ], dtype=np.float32)

    min_reproj_error = float('inf')
    image_points_best_config = None
    rvec_best = None
    tvec_best = None
    projected_points_best = None
    positive_depth_best = None

    for image_points in image_points_perms:
        success, positive_depth, reproj_err, rvec, tvec, projected_points = estimate_pose_nonplanar(object_points, image_points, new_K, dist_coeffs)
        if success and positive_depth and reproj_err < min_reproj_error:
            image_points_best_config = image_points
            min_reproj_error = reproj_err
            rvec_best = rvec
            tvec_best = tvec
            projected_points_best = projected_points
            positive_depth_best = positive_depth

    if image_points_best_config is not None:
        R_wld_to_cam, _ = cv2.Rodrigues(rvec_best)
        #print('tvec', tvec)
        T_wld_to_cam = np.eye(4)
        T_wld_to_cam[:3, :3] = R_wld_to_cam
        T_wld_to_cam[:3, 3] = tvec_best.flatten()

        # Invert transform
        T_cam_to_wld = np.linalg.inv(T_wld_to_cam)

        T_cam_to_drone = np.array([
            [ 0, -1,  0, 0],
            [ 1,  0,  0, 0],
            [ 0,  0,  1, 0],
            [ 0,  0,  0, 1],
        ])

        T_drone_to_wld = T_cam_to_wld 
        # T_drone_to_wld = T_cam_to_wld @ np.linalg.inv(T_cam_to_drone)

        cam_pos = T_drone_to_wld[:3, 3]
        cam_orient_quat = R.from_matrix(T_drone_to_wld[:3, :3]).as_quat()  # (x, y, z, w)
        pose_dict = {
            "success": True,
            "rvec": rvec_best,
            "tvec": tvec_best,
            "R": R_wld_to_cam,
            "camera_position": cam_pos,
            "camera_orientation": cam_orient_quat,
            "reprojection_error": min_reproj_error,
            "projected_points": projected_points_best,
            "positive_depth": positive_depth_best,
        }

        return image_points_best_config, object_points, pose_dict
    
    else:
        return None, None, {"success": False}

def order_l_shape_markers(circles):
    pts = np.array(circles)[:, :2].astype(np.float32)
    dist_matrix = cdist(pts, pts)
    
    # --- 1. Find the Corner using Local Geometry ---
    # For each point, find its two closest neighbors and calculate the angle
    # The corner will have neighbors forming roughly a 90-degree angle.
    best_corner_idx = -1
    min_angle_diff = float('inf')
    
    for i in range(len(pts)):
        # Get indices of two closest points (excluding self)
        nearest_indices = np.argsort(dist_matrix[i])[1:3]
        p1, p2 = pts[nearest_indices[0]], pts[nearest_indices[1]]
        
        # Vectors from current point to neighbors
        v1 = p1 - pts[i]
        v2 = p2 - pts[i]
        
        # Calculate angle between vectors
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        angle = np.arccos(dot_product)
        
        # We want the angle closest to pi/2 (90 degrees)
        diff = abs(angle - np.pi/2)
        if diff < min_angle_diff:
            min_angle_diff = diff
            best_corner_idx = i

    corner = pts[best_corner_idx]
    # --- 2. Separate Arms ---
    other_indices = [i for i in range(8) if i != best_corner_idx]
    others = pts[other_indices]
    vectors = others - corner
    
    # Use the point furthest from the corner to define the "Long Arm" vector
    farthest_idx = np.argmax(np.linalg.norm(vectors, axis=1))
    long_vec_ref = vectors[farthest_idx]
    
    # Group points by checking alignment with the long_vec_ref
    # Points on the same arm will have a very high cosine similarity (near 1.0)
    cos_sims = np.dot(vectors, long_vec_ref) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(long_vec_ref))
    
    # The 4 points with the highest similarity belong to the long arm
    long_arm_mask = np.argsort(cos_sims)[-4:]
    short_arm_mask = np.argsort(cos_sims)[:3]
    
    long_indices = np.array(other_indices)[long_arm_mask]
    short_indices = np.array(other_indices)[short_arm_mask]
    
    # --- 3. Sort by distance from corner ---
    def sort_by_dist(idx_list):
        dists = np.linalg.norm(pts[idx_list] - corner, axis=1)
        return np.array(idx_list)[np.argsort(dists)]

    sorted_long = sort_by_dist(long_indices)
    sorted_short = sort_by_dist(short_indices)
    
    # Combine into final array [0=corner, 1-4=long, 5-7=short]
    final_indices = [best_corner_idx] + list(sorted_long) + list(sorted_short)
    
    image_points = pts[final_indices].astype(np.float32)
        
    # 3D object points in cm
    object_points = np.array([
        [0.0,  0.0,  0.230],   # corner
        [0.130, 0.0,  0.0],
        [0.255, 0.0,  0.0],
        [0.380, 0.0,  0.0],
        [0.505, 0.0,  0.0],   # long arm

        [0.0,  0.125, 0.230],
        [0.0,  0.250, 0.230],
        [0.0,  0.375, 0.230],   # short arm
    ], dtype=np.float32)

    info = {
    }

    return image_points, object_points, info    

def order_l_shape_markers_old(circles):
    """
    Orders 8 circles [x, y, r] into the L-shape convention.
    0: Corner
    1-4: Long arm (+X direction)
    5-7: Short arm (+Y direction)
    """
    # Extract only (x, y) coordinates
    pts = np.array(circles)[:, :2].astype(np.float32)
    
    # Calculate distance matrix between all pairs of points
    dist_matrix = cdist(pts, pts)
    
    # --- 1. Robust Corner Detection via 90-Degree Angle Analysis ---
    best_corner_idx = -1
    min_angle_diff = float('inf')
    
    for i in range(8):
        # Find the 2 nearest neighbors to point i (index 0 is the point itself)
        nearest_indices = np.argsort(dist_matrix[i])[1:3]
        p1, p2 = pts[nearest_indices[0]], pts[nearest_indices[1]]
        
        # Build vectors from point i to these two neighbors
        v1 = p1 - pts[i]
        v2 = p2 - pts[i]
        
        # Calculate the angle between these two vectors
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            continue
            
        unit_v1 = v1 / norm_v1
        unit_v2 = v2 / norm_v2
        
        dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        angle = np.arccos(dot_product)
        
        # We look for the point whose local neighbors form an angle closest to 90 deg (pi/2)
        angle_diff = abs(angle - np.pi / 2)
        if angle_diff < min_angle_diff:
            min_angle_diff = angle_diff
            best_corner_idx = i

    corner_idx = best_corner_idx
    corner = pts[corner_idx]
    
    # --- 2. Separate the Arms ---
    # Remaining 7 points
    others_mask = np.arange(8) != corner_idx
    others = pts[others_mask]
    other_indices = np.where(others_mask)[0]
    
    # Calculate vectors from corner to all other points
    vectors = others - corner
    
    # Find the point furthest from the corner. This MUST be the tip of the LONG arm.
    farthest_idx_in_others = np.argmax(np.linalg.norm(vectors, axis=1))
    long_arm_end_vec = vectors[farthest_idx_in_others]
    
    # Calculate alignment (cosine similarity) against the long arm vector
    # Points on the long arm will have a similarity close to 1.0; short arm will be near 0.0
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(long_arm_end_vec)
    # Avoid division by zero safely
    norms[norms == 0] = 1e-6
    cos_sim = np.dot(vectors, long_arm_end_vec) / norms
    
    # The 4 points most aligned with the long arm end vector go to the long arm
    long_arm_mask = np.argsort(cos_sim)[-4:]
    short_arm_mask = np.argsort(cos_sim)[:3]
    
    long_arm_pts_indices = other_indices[long_arm_mask]
    short_arm_pts_indices = other_indices[short_arm_mask]
    
    # --- 3. Sort points within arms by distance from corner ---
    long_arm_pts = pts[long_arm_pts_indices]
    dist_long = np.linalg.norm(long_arm_pts - corner, axis=1)
    sorted_long_indices = long_arm_pts_indices[np.argsort(dist_long)]
    
    short_arm_pts = pts[short_arm_pts_indices]
    dist_short = np.linalg.norm(short_arm_pts - corner, axis=1)
    sorted_short_indices = short_arm_pts_indices[np.argsort(dist_short)]
    
    # --- 4. Final Assembly ---
    final_indices = [corner_idx] + list(sorted_long_indices) + list(sorted_short_indices)
    image_points = pts[final_indices]
    
    # 3D object points in cm (or meters, as specified by your coordinates)
    object_points = np.array([
        [0.0,    0.0,    0.230],  # corner
        [0.130,  0.0,    0.0],
        [0.255,  0.0,    0.0],
        [0.380,  0.0,    0.0],
        [0.505,  0.0,    0.0],  # long arm

        [0.0,   0.125,  0.230],
        [0.0,   0.250,  0.230],
        [0.0,   0.375,  0.230],  # short arm
    ], dtype=np.float32)

    info = {
        "corner_index": int(corner_idx),
        "long_arm_indices": sorted_long_indices.tolist(),
        "short_arm_indices": sorted_short_indices.tolist(),
    }

    return image_points.astype(np.float32), object_points, info


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

def estimate_pose_nonplanar(object_points, image_points, K, dist_coeffs):
    object_points = np.ascontiguousarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.ascontiguousarray(image_points, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64)

    if object_points.shape[0] < 4:
        raise ValueError("Need at least 4 points")
    if image_points.shape[0] != object_points.shape[0]:
        raise ValueError("image_points and object_points must match in count")

    # success, rvec, tvec = cv2.solvePnP(
    #     object_points,
    #     image_points,
    #     K,
    #     None,
    #     flags=cv2.SOLVEPNP_EPNP
    # )

    # R_wld_to_cam, _ = cv2.Rodrigues(rvec)
    # if not success:
    #     return  False, False

    success, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(
        object_points,
        image_points,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_AP3P
    )

    if not success:
        return  False, False, None, None, None, None

    best_idx = np.argmin(
        [float(err) for err in reproj_errors]
    )

    rvec = rvecs[best_idx]
    tvec = tvecs[best_idx]

    rvec, tvec = cv2.solvePnPRefineLM(
        object_points,
        image_points,
        K,
        dist_coeffs,
        rvec,
        tvec
    )

    err, projected = reprojection_error(
        object_points, image_points, rvec, tvec, K, dist_coeffs
    )
             
    # cam_pos = camera_position_from_pose(R_mat, tvec)
    R_wld_to_cam, _ = cv2.Rodrigues(rvec)

    # Check that all points are in front of the camera
    pts_cam = (R_wld_to_cam @ object_points.T + tvec).T
    positive_depth = np.all(pts_cam[:, 2] > 0)

    return success, positive_depth, err, rvec, tvec, projected

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
        None,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

#    if not success:
#        # Fallback to iterative
#        success, rvec, tvec = cv2.solvePnP(
#            object_points,
#            image_points,
#            K,
#            None,
#            flags=cv2.SOLVEPNP_ITERATIVE
#        )

    if not success:
        return {"success": False}

    # Optional refinement
    #success, rvec, tvec = cv2.solvePnP(
    #    object_points,
    #    image_points,
    #    K,
    #    dist_coeffs,
    #    rvec=rvec,
    #    tvec=tvec,
    #    useExtrinsicGuess=True,
    #    flags=cv2.SOLVEPNP_ITERATIVE
    #)

    R_wld_to_cam, _ = cv2.Rodrigues(rvec)
    #print('tvec', tvec)
    T_wld_to_cam = np.eye(4)
    T_wld_to_cam[:3, :3] = R_wld_to_cam
    T_wld_to_cam[:3, 3] = tvec.flatten()

    # Invert transform
    T_cam_to_wld = np.linalg.inv(T_wld_to_cam)

    # R_mat, _ = cv2.Rodrigues(rvec)
    # R_cam_to_w = R_mat.T  # Camera orientation in world coordinates
    # cam_orient_quat = R.from_matrix(R_cam_to_w).as_quat()  # (x, y, z, w)

    err, projected = reprojection_error(
        object_points, image_points, rvec, tvec, K, None
    )
             
    # cam_pos = camera_position_from_pose(R_mat, tvec)

    # Check that all points are in front of the camera
    pts_cam = (R_wld_to_cam @ object_points.T + tvec).T
    positive_depth = np.all(pts_cam[:, 2] > 0)

    T_cam_to_drone = np.array([
        [ 0, -1,  0, 0],
        [ 1,  0,  0, 0],
        [ 0,  0,  1, 0],
        [ 0,  0,  0, 1],
    ])

    T_drone_to_wld = T_cam_to_wld 
    # T_drone_to_wld = T_cam_to_wld @ np.linalg.inv(T_cam_to_drone)

    cam_pos = T_drone_to_wld[:3, 3]
    cam_orient_quat = R.from_matrix(T_drone_to_wld[:3, :3]).as_quat()  # (x, y, z, w)
    # R_body_to_w = R_cam_to_w @ R_cam_to_body.T
    # R_body_to_w_quat =  R.from_matrix(R_body_to_w).as_quat()  # (x, y, z, w)

    return {
        "success": True,
        "rvec": rvec,
        "tvec": tvec,
        "R": R_wld_to_cam,
        "camera_position": cam_pos,
        "camera_orientation": cam_orient_quat,
        "reprojection_error": err,
        "projected_points": projected,
        "positive_depth": positive_depth,
        "points_camera_frame": pts_cam,
    }

def rotate_intrinsics_180(K, image_width, image_height):
    """
    Update camera intrinsics after rotating image by 180 degrees.

    Parameters
    ----------
    K : (3,3) ndarray
        Original intrinsic matrix.
    image_width : int
    image_height : int

    Returns
    -------
    K_rot : (3,3) ndarray
        Updated intrinsic matrix.
    """

    K_rot = K.copy().astype(np.float64)

    K_rot[0, 2] = image_width  - 1 - K[0, 2]   # cx
    K_rot[1, 2] = image_height - 1 - K[1, 2]   # cy

    return K_rot


if __name__ == "__main__":
    detect_fiducials(brightness_threshold)
