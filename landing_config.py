"""
Single place for all my landing and vision settings.

Both runtime scripts (light_detection*.py and autonomous_takeoff*.py) import this module. 
Units are included in parameter names or comments.
"""

import numpy as np

# Primary switches
ENABLE_AUTONOMY = True  # Enables arming/offboard flight in the autonomous landing script; keep False for bench tests.
ENABLE_LOGGING = True  # Enables JSONL telemetry and annotated-image logging during autonomous landing.
SHOW_VISUALIZATION = False  # Shows OpenCV debug windows produced by the vision workers.
CONSOLE_STATUS_PERIOD_S = 0.5  # [s] Refresh period for the compact SSH status line; does not affect control-loop timing.

# Autonomous flight and offboard handover
AUTONOMY_START_MODE = "rc_handover"  # "auto_takeoff" arms/takes off; "rc_handover" waits for a stable light lock before Offboard.
TAKEOFF_ALT = 4.5  # [m] NED takeoff altitude commanded in "auto_takeoff" mode. Not used in "rc_handover" mode.
TOTAL_TIMEOUT = 700.0  # [s] Maximum total autonomous-landing runtime before the script stops. Essentially mission timeout.
CONTROL_PERIOD = 0.05  # [s] Interval between offboard-controller updates, i.e., control loop frequency.
MAX_VISION_AGE_S = 0.50  # [s] Discards light/ArUco measurements older than this before control uses them.
OFFBOARD_TAKEOVER_RANGE_M = 6.5  # [m] Maximum marker range that allows RC-to-Offboard handover.
OFFBOARD_TAKEOVER_MAX_SPEED_M_S = 2.5  # [m/s] Maximum relative marker speed that allows handover.
OFFBOARD_TAKEOVER_STABLE_TIME = 0.4  # [s] Required continuous valid light lock before handover. With the current vision latency, means about 4-8 consecutive frames with light detection.
COMMAND_YAW_DEG = 0.0  # [deg] Fallback absolute NED yaw before RC handover captures the vehicle yaw.

# Connection, logging, and input
MAVLINK_MULTIPLE_CONNECTIONS = False  # False (outdoor testing) uses serial FC links with one route to QGC; True (e.g., for optitrack) uses the MAVLink-router UDP endpoints below.
CONNECT_MAVLINK = True  # Lets stand-alone light_detection_blobbased.py send MAVLink odometry.
MAVSDK_SERIAL_URL = "serial:///dev/ttyACM0:115200"  # MAVSDK endpoint selected when MAVLINK_MULTIPLE_CONNECTIONS is False.
MAVSDK_UDP_URL = "udpin://127.0.0.1:14600"  # MAVSDK endpoint selected when MAVLINK_MULTIPLE_CONNECTIONS is True.

# Visual-marker and camera selection
POSE_TYPE = "target"  # Pose worker mode: "target" estimates marker relative to drone; "drone" estimates drone relative to marker.
ENABLE_LIGHT_MARKER = False  # Autonomous landing: use light-marker acquisition before switching to precision ArUco.
RGB_CAMERA_PORT = 0  # Picamera2 device index used by the RGB/L-shape worker.
MONO_CAMERA_PORT = 1  # Picamera2 device index used by the monochrome ArUco worker.
RGB_CAMERA_TYPE = "fisheye"  # Selects fisheye or pinhole undistortion path for the RGB camera.
MONO_CAMERA_TYPE = "fisheye"  # Selects fisheye or pinhole undistortion path for the monochrome camera.
RGB_IMAGE_SCALE = 0.9  # Downscales RGB frames and intrinsics to trade accuracy for LED-processing speed.
ARUCO_IMAGE_SCALE = 0.65  # Downscales monochrome frames and intrinsics for ArUco detection.

# LED detector
BRIGHTNESS_THRESHOLD = 50  # [0-255] Grayscale threshold for pixels considered LED candidates.
LIGHT_MIN_AREA = 5  # [px²] Rejects LED contours smaller than this in the light-pose worker.
LIGHT_MAX_AREA = 3000  # [px²] Rejects LED contours larger than this in the light-pose worker.
RADIUS_TOLERANCE = 1.5  # [px] Maximum radius difference when grouping LED blobs into an L-shape.
LINE_TOLERANCE = 5.0  # [px] Maximum distance from a line when grouping LED blobs.
MIN_LED_GROUP_SIZE = 4  # Minimum number of aligned blobs required by the L-shape grouping filter.
CROSS_RATIO_TOLERANCE = 0.02  # Allowed cross-ratio error when validating LED spacing along a line.
RIGHT_ANGLE_TOLERANCE_DEG = 25.0  # [deg] Allowed L-shape corner-angle error; set None to disable this check.
BLUR_WINDOW = (19, 19)  # [px, px] Gaussian blur kernel applied before LED thresholding; values must be odd.
LIGHT_REPROJECTION_THRESHOLD_PX = 20.0  # [px] Light-pose error limit for debug display/annotation.
LIGHT_PUBLISH_REPROJECTION_THRESHOLD_PX = 15.0  # [px] Stricter light-pose error limit required before publishing a control measurement.

# ArUco detector and marker geometry
ARUCO_DICTIONARY_ID = 0  # OpenCV dictionary passed to getPredefinedDictionary; 0 is cv2.aruco.DICT_4X4_50.
SMALL_ARUCO_MARKER_ID = 0  # Marker ID of the precision landing ArUco marker.
SMALL_ARUCO_MARKER_SIZE_M = 0.120  # [m] Printed side length of the precision marker; used in pose scale estimation.
LARGE_ARUCO_MARKER_ID = 2  # Marker ID of the larger acquisition marker used before the precision handover.
LARGE_ARUCO_MARKER_SIZE_M = 1.0  # [m] Printed side length of the larger acquisition marker.
MAX_ATTITUDE_AGE_S = 0.15  # [s] Reserved: currently not checked by the vision/control code.
MAX_ARUCO_REPROJECTION_ERROR_PX = 5.0  # [px] Reserved: currently not checked by the vision/control code.
# MAX_ARUCO_RANGE_M = 6.0  # [m] Reserved disabled limit; currently no ArUco range gate is implemented.

# Camera setup
RGB_EXPOSURE_TIME_US = 5000  # [µs] Manual exposure used by the RGB camera in the stand-alone detector.
MONO_EXPOSURE_TIME_US = 20000  # [µs] Manual exposure setting retained for the mono camera; its active worker currently uses auto-exposure.
RGB_ANALOGUE_GAIN = 1.0  # RGB camera analogue gain applied when creating the camera configuration.

# Marker handover and horizontal control (NED vectors use [north, east, down])
LIGHT_TO_ARUCO_OFFSET_NED = np.array([0.0, 0.0, 0.0], dtype=float)  # [m] Fixed vector from light origin to ArUco landing origin; offsets light-tracking target.

# Light marker
# ARUCO_START_BOX = np.array([3.0, 3.0, 3.5], dtype=float)  # [m] Per-axis relative-position box that starts the ArUco worker during light tracking.
# ARUCO_HANDOFF_ENTRY_RANGE_M = 2.2  # [m] Enter ArUco tracking (without descent) below this marker range.
# ARUCO_HANDOFF_HOLD_RANGE_M = 2.5  # [m] Leave ArUco tracking only above this range; provides hysteresis.
# FINAL_DESCENT_ENTRY_RANGE_M = 2.2  # [m] Enter final descent below this ArUco marker range.
# FINAL_DESCENT_HOLD_RANGE_M = 2.5  # [m] Pause/leave final descent above this range; provides hysteresis.

# Aruco marker
ARUCO_START_BOX = np.array([3.0, 3.0, 4.5], dtype=float)  # [m] Per-axis relative-position box that starts the ArUco worker during light tracking.
ARUCO_HANDOFF_ENTRY_RANGE_M = 3.0  # [m] Enter ArUco tracking (without descent) below this marker range.
ARUCO_HANDOFF_HOLD_RANGE_M = 3.5  # [m] Leave ArUco tracking only above this range; provides hysteresis.
FINAL_DESCENT_ENTRY_RANGE_M = 3.0  # [m] Enter final descent below this ArUco marker range.
FINAL_DESCENT_HOLD_RANGE_M = 3.5  # [m] Pause/leave final descent above this range; provides hysteresis.

ARUCO_STABLE_TIME = 0.25  # [s] Continuous valid ArUco duration required before it may take control.
ARUCO_LIGHT_AGREEMENT_M = 1.5  # [m] Maximum light-vs-ArUco target disagreement allowed for handover.
KP_XY = 0.5  # Horizontal position-controller proportional gain.
KD_XY = 0.4  # Horizontal relative-velocity damping gain.
MAX_HORIZONTAL_SPEED = 2.0  # [m/s] Clamp on commanded north/east tracking speed.
POSE_FILTER_ALPHA = 0.75  # Low-pass weight for new relative-position measurements (higher = less smoothing).
VELOCITY_FILTER_ALPHA = 0.75  # Low-pass weight for estimated relative velocity (higher = less smoothing).
LIGHT_ACQUISITION_RANGE_M = 1.8  # [m] The height above the marker that drone will come down to and keep maintained in case of light tracking
LIGHT_DESCENT_ALIGN_RADIUS_M = 0.80  # [m] Lateral error limit for light-marker-based descent while tracking the light marker.

# Alignment, descent, touchdown, and vision-loss recovery
ALIGN_RADIUS_M = 0.80  # [m] Lateral ArUco error required to transition from ARUCO_TRACK to FINAL_DESCENT.
ALIGN_SPEED_M_S = 2.0  # [m/s] Maximum lateral marker speed allowed to begin/continue aligned descent.
ALIGN_HOLD_TIME = 0.5  # [s] Time the lateral alignment conditions must stay valid before descent.
MAX_LANDING_TILT_DEG = 365.0  # [deg] Maximum platform tilt allowed for landing; 365 effectively disables this gate.
ORIENTATION_HOLD_TIME = 0.75  # [s] Time the platform-orientation condition must stay valid before descent.
KP_LIGHT_Z = 0.45  # Light-marker vertical proportional gain during light-assisted descent.
MAX_LIGHT_DESCENT_SPEED = 0.5  # [m/s] Maximum downward speed while light-marker descent control is active.
TOUCHDOWN_RANGE_M = 0.35  # [m] Marker-relative range at which touchdown/predicted-landing logic may begin.
PREDICTED_LANDING_ENTRY_RANGE_M = 0.45  # [m] Enter predicted landing below this range, anticipating the marker may leave the camera view.
PREDICTED_LANDING_HOLD_RANGE_M = 0.55  # [m] Exit predicted landing above this range; provides hysteresis.
PREDICTED_LANDING_STABLE_TIME = 0.20  # [s] Required stable in-range time before predicted landing starts.
DESCENT_RATE_M_S = 0.2  # [m/s] Rate at which final-descent range reference is reduced.
KP_Z = 0.80  # Final-descent vertical position-controller proportional gain.
KD_Z = 0.20  # Final-descent vertical relative-velocity damping gain.
MAX_DESCENT_SPEED = 0.75  # [m/s] Clamp on final-descent downward speed.
MAX_CLIMB_SPEED = 0.25  # [m/s] Clamp on commanded upward speed during final descent/recovery.
LOST_MARKER_TIMEOUT = 0.5  # [s] Vision-loss duration that triggers final-descent loss handling.
ENABLE_VISION_LOSS_RECOVERY_CLIMB = False  # Enables climbing to reacquire vision after loss in final descent.
VISION_LOSS_RECOVERY_RANGE_M = 3.5  # [m] Marker-relative range targeted by the optional recovery climb.
VISION_LOSS_CLIMB_SPEED = 0.20  # [m/s] Upward speed used by the optional vision-loss recovery climb.
PREDICTED_LANDING_TIME_S = 1.5  # [s] Fixed blind-descent duration after the predicted-landing phase begins.
PREDICTED_LANDING_DESCENT_SPEED_M_S = 0.8  # [m/s] Downward speed commanded during predicted/blind landing.

# Calibrated intrinsics
CAMERA_RGB_TO_BODY_FRD_M = np.array([0.0, 0.0, 0.0])  # [m] Reserved RGB camera-to-body offset; not currently applied by the pose worker.
CAMERA_MONO_TO_BODY_FRD_M = np.array([0.0, 0.0, 0.0])  # [m] Mono camera-to-body FRD offset added when stand-alone ArUco odometry is sent.
CAMERA_MATRIX_MONO = np.array([  # Mono camera intrinsic matrix at its full calibrated resolution.
    [972.41752602, 0.0, 719.86748972],
    [0.0, 970.82689346, 520.66180438],
    [0.0, 0.0, 1.0],
])
DIST_COEFFS_MONO = np.array([-0.13573729, 0.03353202, -0.0345132, 0.01030255])  # Mono fisheye distortion coefficients matching CAMERA_MATRIX_MONO.
CAMERA_MATRIX_RGB = np.array([  # RGB camera intrinsic matrix at its full calibrated resolution.
    [1.28039260e3, 0.0, 1.25478995e3],
    [0.0, 1.27992166e3, 9.10815180e2],
    [0.0, 0.0, 1.0],
])
DIST_COEFFS_RGB = np.array([-1.99362880e-02, -1.10620253e-03, 1.52169132e-04, -4.47659636e-05])  # RGB fisheye distortion coefficients matching CAMERA_MATRIX_RGB.
