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

from helpers.utils import scale_camera_matrix, rotate_intrinsics_180
from helpers.poseestimation import pose_from_colored_leds, estimate_planar_pose, order_l_shape_markers
from helpers.led_detection import filter_circles_same_line_similar_radius, cross_ratio_1d   
from landing_config import *

# Runtime state; tunable values are imported from landing_config.py.
show_visualization = SHOW_VISUALIZATION
drone_attitude_reliable = True
radius_tol = RADIUS_TOLERANCE
line_tol = LINE_TOLERANCE
min_group_size = MIN_LED_GROUP_SIZE
cross_ratio_tol = CROSS_RATIO_TOLERANCE
RIGHT_ANGLE_TOL_DEG = RIGHT_ANGLE_TOLERANCE_DEG
blur_window = BLUR_WINDOW
brightness_threshold = BRIGHTNESS_THRESHOLD
reproj_threshold = LIGHT_REPROJECTION_THRESHOLD_PX
target_id = SMALL_ARUCO_MARKER_ID
marker_size = SMALL_ARUCO_MARKER_SIZE_M
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_ID)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
camera_matrix_mono = CAMERA_MATRIX_MONO.copy()
dist_coeffs_mono = DIST_COEFFS_MONO.copy()
camera_matrix_rgb = CAMERA_MATRIX_RGB.copy()
dist_coeffs_rgb = DIST_COEFFS_RGB.copy()
s_rgb = RGB_IMAGE_SCALE
s_aruco = ARUCO_IMAGE_SCALE
latest_attitude = None
latest_attitude_time = None
mavlink_lock = threading.Lock()
rgb_cameratype = RGB_CAMERA_TYPE
mono_cameratype = MONO_CAMERA_TYPE

_visualization_lock = threading.Lock()
_latest_visualizations = {}

def publish_visualization(name, image):
    if show_visualization and image is not None:
        with _visualization_lock:
            _latest_visualizations[name] = image

async def visualization_loop():
    """Render worker-produced frames from the main thread only."""
    shown_windows = set()
    while True:
        with _visualization_lock:
            images = _latest_visualizations.copy()
        for name, image in images.items():
            if name not in shown_windows:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(name, 700, 700)
                shown_windows.add(name)
            cv2.imshow(name, image)
        cv2.waitKey(1)
        await asyncio.sleep(0.03)

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
# marker_id -> (drone_position, drone_orientation, timestamp, sequence)
latest_drone_aruco_measurements = {}

vision_lock = threading.Lock()
latest_lighttarget_timestamp = None
latest_lighttarget_sequence = 0
latest_arucotarget_timestamp = None
latest_arucotarget_sequence = 0
latest_arucotarget_id = None
latest_arucotarget_measurements = {}  # marker_id -> (position, orientation, timestamp, sequence)

def _publish_light_target(position, orientation, capture_time):
    global latest_lighttarget_location, latest_lighttarget_orientation, latest_lighttarget_timestamp, latest_lighttarget_sequence
    with vision_lock:
        latest_lighttarget_location = np.asarray(position, dtype=float).copy()
        latest_lighttarget_orientation = orientation
        latest_lighttarget_timestamp = capture_time
        latest_lighttarget_sequence += 1
        flight_logging.flight_logger.log("light_pose", position=position)

def _clear_light_target():
    global latest_lighttarget_location, latest_lighttarget_orientation, latest_lighttarget_timestamp
    with vision_lock:
        latest_lighttarget_location = None
        latest_lighttarget_orientation = None
        latest_lighttarget_timestamp = None

def _publish_aruco_target(position, orientation, capture_time, marker_id=None):
    global latest_arucotarget_location, latest_arucotarget_orientation, latest_arucotarget_timestamp, latest_arucotarget_sequence, latest_arucotarget_id
    with vision_lock:
        position = np.asarray(position, dtype=float).copy()
        latest_arucotarget_sequence += 1
        latest_arucotarget_measurements[marker_id] = (
            position, orientation, capture_time, latest_arucotarget_sequence
        )
        # Preserve the legacy latest-marker interface for existing callers.
        latest_arucotarget_location = position
        latest_arucotarget_orientation = orientation
        latest_arucotarget_timestamp = capture_time
        latest_arucotarget_id = marker_id
        flight_logging.flight_logger.log("aruco_pose", position=position, marker_id=marker_id)

def _clear_aruco_target():
    global latest_arucotarget_location, latest_arucotarget_orientation, latest_arucotarget_timestamp, latest_arucotarget_id
    with vision_lock:
        latest_arucotarget_location = None
        latest_arucotarget_orientation = None
        latest_arucotarget_timestamp = None
        latest_arucotarget_id = None

def get_latest_arucotarget_id():
    with vision_lock:
        return latest_arucotarget_id

from picamera2 import Picamera2
import flight_logging
picam2 = Picamera2(RGB_CAMERA_PORT)  # Use camera port 0 (RGB camera) for L-shape
full_size = picam2.camera_properties["PixelArraySize"]
config = picam2.create_video_configuration(
            main={"size": full_size, "format": "RGB888"},
            buffer_count=1,
            queue=False)
            
picam2.configure(config)
controls = {
    "ScalerCrop": (0, 0, *full_size),    "ExposureTime": RGB_EXPOSURE_TIME_US,   # microseconds
    "AnalogueGain": RGB_ANALOGUE_GAIN}
picam2.set_controls(controls)
picam2.start()

def get_pose_from_lightmarker(stop_event, pose_type, drone,
    brightness_threshold,
    min_area: int = LIGHT_MIN_AREA,
    max_area: int = LIGHT_MAX_AREA):
    # This function is similar to detect_lights_sendodometry but only returns the estimated pose without any MAVLink communication or visualization. It can be used for unit testing the pose estimation logic in isolation.

    global camera_matrix_rgb, dist_coeffs_rgb
    global latest_lighttarget_location
    global latest_lighttarget_orientation
    global latest_dronelocation_withlighttarget
    global latest_droneorientation_withlighttarget
    global latest_dronelocation_withlighttarget_timestamp
    global latest_dronelocation_withlighttarget_sequence
    global s_rgb
    s = s_rgb

    camera_matrix = camera_matrix_rgb.copy()
    dist_coeffs = dist_coeffs_rgb.copy()

    cap = None
    
    camera_matrix = scale_camera_matrix(camera_matrix, s)

    #camera_matrix = rotate_intrinsics_180(camera_matrix, s*1456, s*1088) # rotation was needed for mono camera, but not for rgb camera

    # The camera configuration and scale are fixed for this worker, so the
    # undistortion maps only need to be built once per frame resolution.
    map_size = None
    new_K = None
    map1 = None
    map2 = None

    def local_circle_mean(image, center, radius, radius_scale=1.0):
        """Return a disk mean without allocating a mask the size of the frame."""
        center_x, center_y = center
        effective_radius = max(float(radius) * radius_scale, 0.5)
        extent = int(math.ceil(effective_radius))

        x0 = max(0, center_x - extent)
        x1 = min(image.shape[1], center_x + extent + 1)
        y0 = max(0, center_y - extent)
        y1 = min(image.shape[0], center_y + extent + 1)
        patch = image[y0:y1, x0:x1]

        patch_y, patch_x = np.ogrid[y0:y1, x0:x1]
        disk = ((patch_x - center_x) ** 2 + (patch_y - center_y) ** 2
                <= effective_radius ** 2)

        return int(np.median(patch[disk])) #.mean()
        
    while not stop_event.is_set():
    # while True:
#        time.sleep(1)
        # -------------------------
        # Read frame
        # -------------------------

        # Use monotonic clock, NOT time.time()
        start_time = time.time()
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

        if map_size != (width, height):
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                camera_matrix,
                dist_coeffs,
                (width, height),
                np.eye(3),
                balance=0.0,
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                camera_matrix,
                dist_coeffs,
                np.eye(3),
                new_K,
                (width, height),
                cv2.CV_16SC2,
            )
            map_size = (width, height)
        
        image_undistorted = cv2.remap(
            frame, map1, map2, interpolation=cv2.INTER_LINEAR)

        red = image_undistorted[:, :, 0]
        green = image_undistorted[:, :, 1]

        diff_image_RG = red.astype(np.int16) - green.astype(np.int16)
        lab = cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2LAB)
        lab_a = lab[:, : , 1]

#            print('Searching for LEDs...')
        blurred = cv2.GaussianBlur(image_undistorted, blur_window, 0)
        annotated = blurred.copy()
        annotated_colors = blurred.copy()
        blurred = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        
        # The fixed brightness threshold below is used instead of adaptive
        # top-10-percent statistics, so leave this expensive calculation off.
        # brightness_mask = np.percentile(blurred, 90)
        # top_pixels = blurred[blurred >= brightness_mask]
        # avg_top_10_intensities = np.mean(top_pixels)
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
        fitered_circles = filter_circles_same_line_similar_radius(
            circles, radius_tol, line_tol, min_group_size, cross_ratio_tol,
            RIGHT_ANGLE_TOL_DEG,
        )
        filteredcircles_avgcolor = []
#            print('filtered circles', len(fitered_circles))

        for circle in fitered_circles:    
            center = (int(circle[0]), int(circle[1]))
            radius = int(circle[2])
            led_mean = local_circle_mean(
                lab_a, center, radius, radius_scale=math.sqrt(0.8) # diff_image_RG or lab_a
            )
#            led_mean = green_undistorted[center]
            filteredcircles_avgcolor.append(led_mean)

        filteredcircles_avgcolor_sorted = np.argsort(filteredcircles_avgcolor)
    
        if (show_visualization):
            led_count = 0
            for circle in fitered_circles:    
                # Draw annotation
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])
                led_mean = local_circle_mean(lab_a, center, radius)
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

            publish_visualization("Thresholded", thresh)
            publish_visualization("Annotated colors", annotated_colors)

#        if cv2.waitKey(1) & 0xFF == ord("q"):
#            cv2.destroyAllWindows()
#            break

        # print(len(fitered_circles), "circles after line/radius filtering")
        flight_logging.flight_logger.save_image("light_leds", annotated_colors)
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

            if (len(image_points)%4 == 0) and (len(image_points) == len(object_points)):
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
                print("time_taken light detection loop:", time.time() - start_time)
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
                    publish_visualization("Annotated", annotated)

                print('pose_dict["reprojection_error"]', pose_dict["reprojection_error"])
                print('pose', p)

                if (pose_dict["reprojection_error"] < LIGHT_PUBLISH_REPROJECTION_THRESHOLD_PX and pose_dict["positive_depth"]):
                    if (pose_type == 'target'):
                        print('publsihing light target')
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
                    print('pose estimated by light detct:', p)
                else:
                        latest_lighttarget_location = None
                        latest_lighttarget_orientation = None
                        latest_dronelocation_withlighttarget = None
                        latest_droneorientation_withlighttarget = None
                        latest_dronelocation_withlighttarget_timestamp = None
                        print('Setting light target and drone location to None, marker not found')

import libcamera
picam2_ar = Picamera2(MONO_CAMERA_PORT)
print(picam2_ar.sensor_modes)  # Confirm that Y8 is supported
full_size_ar = picam2_ar.camera_properties["PixelArraySize"]
config_ar = picam2_ar.create_video_configuration(
    main={
        "size": full_size_ar,
        "format": "RGB888",
    },
    buffer_count=1,
    queue=False,
)

picam2_ar.configure(config_ar)
picam2_ar.set_controls({
    "ScalerCrop": (0, 0, *full_size_ar),

    "AeEnable": True,
    "AeExposureMode": libcamera.controls.AeExposureModeEnum.Short,
    "AeMeteringMode": libcamera.controls.AeMeteringModeEnum.CentreWeighted,
    # Allow AE, but prevent very long exposures.
    "FrameDurationLimits": (2_000, 10_000),
    # Bias darker to reduce white-cell saturation at close range.
    "ExposureValue": -1.0,
})

picam2_ar.start()

def get_pose_from_arucomarker(pose_type, drone, acquisition_marker_id=None, acquisition_marker_size_m=None):
    global camera_matrix_mono
    global dist_coeffs_mono

    global latest_arucotarget_location
    global latest_arucotarget_orientation
    global latest_dronelocation_witharucotarget
    global latest_droneorientation_witharucotarget
    global latest_dronelocation_witharucotarget_timestamp
    global latest_dronelocation_witharucotarget_sequence
    global s_aruco
    s = s_aruco

    camera_matrix = camera_matrix_mono.copy()
    dist_coeffs = dist_coeffs_mono.copy()

    cap = None

    camera_matrix = scale_camera_matrix(camera_matrix, s) # resizing the image below
#    camera_matrix = rotate_intrinsics_180(camera_matrix, s*full_size_ar[0], s*full_size_ar[1]) # because frame is rotated below
        
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

        frame_ar = picam2_ar.capture_array()
        height, width  = frame_ar.shape[:2]
        frame_ar = cv2.resize(frame_ar, (int(s*width), int(s*height)))
#        frame_ar = cv2.rotate(frame_ar, cv2.ROTATE_180)
        green_ar = frame_ar[:, :, 1]

        h, w = frame_ar.shape[:2]        
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
        
        image_undistorted_ar = cv2.remap(
            frame_ar, map1, map2, interpolation=cv2.INTER_LINEAR)

        green_undistorted_ar = cv2.remap(
            green_ar, map1, map2, interpolation=cv2.INTER_LINEAR)

        ### ARUCO Detection
        corners, ids, _ = detector.detectMarkers(image_undistorted_ar)
        published = False
        if ids is not None:
            ids = ids.flatten()
            marker_sizes = {target_id: marker_size}
            if acquisition_marker_id is not None:
                marker_sizes[acquisition_marker_id] = acquisition_marker_size_m

            for i, marker_id in enumerate(ids):
                if marker_id not in marker_sizes:
                    continue
                selected_marker_size = marker_sizes[marker_id]
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[i]], selected_marker_size, new_K, np.zeros(4)
                )
                rvec = rvec[0][0]
                tvec = tvec[0][0]
                published = True

                if show_visualization:
                    cv2.aruco.drawDetectedMarkers(image_undistorted_ar, corners, ids)
                    cv2.drawFrameAxes(image_undistorted_ar, new_K, np.zeros(4), rvec, tvec, 0.05)

                if pose_type == 'target':
                    R_wld_to_cam, _ = cv2.Rodrigues(rvec)
                    rot_cam_to_drone = np.array([
                        [0, -1, 0],
                        [1, 0, 0],
                        [0, 0, 1],
                    ])
                    trans_marker_to_drone = rot_cam_to_drone @ tvec.flatten()
                    p = rot_drone_to_ned @ trans_marker_to_drone
                    q = rot_drone_to_ned @ rot_cam_to_drone @ R_wld_to_cam
                    _publish_aruco_target(p, q, time.monotonic(), int(marker_id))
                elif pose_type == 'drone':
                    R_wld_to_cam, _ = cv2.Rodrigues(rvec)
                    T_wld_to_cam = np.eye(4)
                    T_wld_to_cam[:3, :3] = R_wld_to_cam
                    T_wld_to_cam[:3, 3] = tvec.flatten()
                    basis_change = np.array([
                        [0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1],
                    ], dtype=float)
                    T_drone_to_wld = basis_change @ np.linalg.inv(T_wld_to_cam)
                    with vision_lock:
                        capture_time = time.monotonic()
                        drone_position = T_drone_to_wld[:3, 3].copy()
                        drone_orientation = R.from_matrix(T_drone_to_wld[:3, :3]).as_quat()
                        latest_dronelocation_witharucotarget_sequence += 1
                        latest_drone_aruco_measurements[int(marker_id)] = (
                            drone_position, drone_orientation, capture_time,
                            latest_dronelocation_witharucotarget_sequence,
                        )
                        # Preserve the legacy no-ID drone-pose interface.
                        latest_dronelocation_witharucotarget = drone_position
                        latest_droneorientation_witharucotarget = drone_orientation
                        latest_dronelocation_witharucotarget_timestamp = capture_time

            if show_visualization and published:
                publish_visualization("ArUco", image_undistorted_ar)
                flight_logging.flight_logger.save_image("aruco", image_undistorted_ar)

        if not published:
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

def get_latest_drone_measurement_from_arucomarker(marker_id=None):
    with vision_lock:
        if marker_id is not None:
            measurement = latest_drone_aruco_measurements.get(marker_id)
            if measurement is None:
                return None, None, None, 0
            p, orientation, timestamp, sequence = measurement
            return p.copy(), np.asarray(orientation, dtype=float).copy(), timestamp, sequence
        p = None if latest_dronelocation_witharucotarget is None else latest_dronelocation_witharucotarget.copy()
        orientation = None if latest_droneorientation_witharucotarget is None else np.asarray(latest_droneorientation_witharucotarget, dtype=float).copy()
        return p, orientation, latest_dronelocation_witharucotarget_timestamp, latest_dronelocation_witharucotarget_sequence

def get_latest_lighttarget_measurement():
    with vision_lock:
        p = None if latest_lighttarget_location is None else latest_lighttarget_location.copy()
        orientation = None if latest_lighttarget_orientation is None else np.asarray(latest_lighttarget_orientation, dtype=float).copy()
        return p, orientation, latest_lighttarget_timestamp, latest_lighttarget_sequence

def get_latest_arucotarget_measurement(marker_id=None):
    with vision_lock:
        if marker_id is not None:
            measurement = latest_arucotarget_measurements.get(marker_id)
            if measurement is None:
                return None, None, None, 0, marker_id
            p, orientation, timestamp, sequence = measurement
            return p.copy(), np.asarray(orientation, dtype=float).copy(), timestamp, sequence, marker_id
        p = None if latest_arucotarget_location is None else latest_arucotarget_location.copy()
        orientation = None if latest_arucotarget_orientation is None else np.asarray(latest_arucotarget_orientation, dtype=float).copy()
        return p, orientation, latest_arucotarget_timestamp, latest_arucotarget_sequence, latest_arucotarget_id
