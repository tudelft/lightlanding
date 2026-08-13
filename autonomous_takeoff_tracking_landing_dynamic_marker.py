"""Two-stage visual landing on a moving light + ArUco marker.

The vision module supplies a *relative* marker displacement in NED axes:
    p = marker_position - drone_position
No global drone pose or global marker pose is used here.

The light marker is used for long-range acquisition/tracking.  The ArUco marker
is started before the handoff and must be stable before it becomes the control
source.  During final descent the ArUco controller remains active, so the drone
continues to follow lateral marker motion.

Set ENABLE_AUTONOMY=False for bench testing.  Tune every value below and verify
the NED signs with the vehicle restrained before enabling flight.
"""

import asyncio
import threading
import time

import numpy as np
from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.offboard import OffboardError, PositionNedYaw, VelocityNedYaw
from mavsdk.telemetry import LandedState
from scipy.spatial.transform import Rotation as R
from flight_logging import configure_flight_logger, log_drone_telemetry

from light_detection_blobbased import (
    attitude_loop,
    get_latest_arucotarget_id,
    get_latest_arucotarget_measurement,
    get_latest_drone_measurement_from_arucomarker,
    get_latest_drone_measurement_from_lightmarker,
    get_latest_lighttarget_measurement,
    get_pose_from_arucomarker,
    get_pose_from_lightmarker,
    show_visualization,
    visualization_loop,
)


# =========================
# CONFIGURATION
# =========================
MAVLINK_MULTIPLE_CONNECTIONS = False
ENABLE_AUTONOMY = False  # Change manually only after restrained/bench tests.
ENABLE_LOGGING = False  # Writes JSONL telemetry and 2 Hz annotated images to ~/logs/.
ENABLE_LIGHT_MARKER = True  # False: acquire with the large ArUco marker before switching to ID 0.
TAKEOFF_ALT = 4.5
TOTAL_TIMEOUT = 250.0
CONTROL_PERIOD = 0.1
MAX_VISION_AGE_S = 0.30

# "auto_takeoff": current behavior--the script arms, starts Offboard, and
# commands takeoff. "rc_handover": the pilot takes off and approaches in
# Position/Mission mode; Offboard starts only after a stable light-marker lock.
AUTONOMY_START_MODE = "rc_handover" # auto_takeoff or rc_handover
# Used only when AUTONOMY_START_MODE == "rc_handover". Values are relative
# marker-to-drone quantities, in metres and metres/second.
OFFBOARD_TAKEOVER_RANGE_M = 8.0
OFFBOARD_TAKEOVER_MAX_SPEED_M_S = 1.0
OFFBOARD_TAKEOVER_STABLE_TIME = 0.75

# VelocityNedYaw's yaw argument is an absolute NED yaw. This is only the
# fallback used before an RC-to-Offboard handover captures the current yaw.
COMMAND_YAW_DEG = 0.0
command_yaw_deg = COMMAND_YAW_DEG
BRIGHTNESS_THRESHOLD = 30
POSE_TYPE = "target"

# All position vectors below use NED components [north, east, down].
# This is p_aruco - p_light: vector from the light-marker origin to the ArUco
# landing origin.  Measure this for your assembly.  With no calibration, leave
# zero: ArUco takes over before final landing, so this only biases light tracking.
# A constant NED offset is valid only while the attached marker assembly does not
# yaw appreciably.  The current target-pose interfaces do not expose target yaw.
LIGHT_TO_ARUCO_OFFSET_NED = np.array([0.0, 0.0, 0.0], dtype=float) # optional, could be kept to [0,0,0] if not known

# Begin looking for ArUco while light tracking once the landing marker is in this
# box.  Do not hand over until ARUCO_STABLE_TIME has elapsed with valid ArUco data.
ARUCO_START_BOX = np.array([1.5, 1.5, 1.5], dtype=float)
ARUCO_STABLE_TIME = 0.25 # seconds: ArUco must be valid for this long before switching to ArUco control.
LARGE_ARUCO_MARKER_ID = 2  # Unique ID of the large daytime/acquisition marker.
LARGE_ARUCO_MARKER_SIZE_M = 1.0  # Measure and set the printed side length exactly.
SMALL_ARUCO_MARKER_ID = 0  # Precision landing marker.
ARUCO_LIGHT_AGREEMENT_M = 1.5  # meters: ArUco and light must agree within this distance to switch to ArUco control. If LIGHT_TO_ARUCO_OFFSET_NED is set, make this value lower (0.2-0.5).

# Horizontal tracking, applied to the desired ArUco landing origin.
KP_XY = 0.5
KD_XY = 0.4
MAX_HORIZONTAL_SPEED = 0.8
POSE_FILTER_ALPHA = 0.35
VELOCITY_FILTER_ALPHA = 0.25

# Do not descend until the marker is well centered and its relative lateral
# motion is manageable.  Keep following it whenever descent is paused.
ARUCO_TRACK_RANGE_M = 2.8 # meters: begin ArUco tracking when the marker is within this range.  Must be greater than LIGHT_ACQUISITION_RANGE_M.
LIGHT_ACQUISITION_RANGE_M = 2.0 # should be less than ARUCO_TRACK_RANGE_M, but not too small to avoid losing the light target before ArUco is acquired.
LIGHT_DESCENT_ALIGN_RADIUS_M = 0.80

ALIGN_RADIUS_M = 0.80
ALIGN_SPEED_M_S = 0.30
ALIGN_HOLD_TIME = 0.75
MAX_LANDING_TILT_DEG = 15.0
ORIENTATION_HOLD_TIME = 0.75
KP_LIGHT_Z = 0.45
MAX_LIGHT_DESCENT_SPEED = 0.5
TOUCHDOWN_RANGE_M = 0.5  # Must be validated against camera/landing-gear geometry.
DESCENT_RATE_M_S = 0.2 # When in FINAL_DESCENT, the range reference is decremented at this rate.  The controller will try to follow it, but will not descend while off-center.
KP_Z = 0.80
KD_Z = 0.20
MAX_DESCENT_SPEED = 0.75
MAX_CLIMB_SPEED = 0.25

# If vision is absent in final descent, never keep descending blind.
LOST_MARKER_TIMEOUT = 0.35
ABORT_CLIMB_SPEED = 0.20

# Once the marker reaches TOUCHDOWN_RANGE_M it can leave the camera field of
# view. Continue the last lateral tracking command for this short, fixed
# interval while descending, then let PX4 complete the landing.
PREDICTED_LANDING_TIME_S = 1.5
PREDICTED_LANDING_DESCENT_SPEED_M_S = 0.50

SERIAL_IP = "serial:///dev/ttyACM0:115200" if not MAVLINK_MULTIPLE_CONNECTIONS else "udpin://127.0.0.1:14600"
flight_logger = configure_flight_logger(ENABLE_LOGGING)

def clip(value, limit):
    return float(np.clip(value, -limit, limit))

class RelativePoseFilter:
    """Low-pass position and finite-difference relative-velocity estimator."""

    def __init__(self):
        self.position = None
        self.velocity = np.zeros(3, dtype=float)
        self.time = None

    def reset(self):
        self.position = None
        self.velocity[:] = 0.0
        self.time = None

    def update(self, measurement, now):
        measurement = np.asarray(measurement, dtype=float)
        if self.position is None:
            self.position = measurement.copy()
            self.time = now
            return self.position.copy(), self.velocity.copy()

        dt = now - self.time
        if dt <= 1e-3:
            return self.position.copy(), self.velocity.copy()

        filtered_position = (
            POSE_FILTER_ALPHA * measurement
            + (1.0 - POSE_FILTER_ALPHA) * self.position
        )
        raw_velocity = (filtered_position - self.position) / dt
        self.velocity = (
            VELOCITY_FILTER_ALPHA * raw_velocity
            + (1.0 - VELOCITY_FILTER_ALPHA) * self.velocity
        )
        self.position = filtered_position
        self.time = now
        return self.position.copy(), self.velocity.copy()


def latest_light_offset():
    if POSE_TYPE == "drone":
        p, orientation, timestamp, sequence = get_latest_drone_measurement_from_lightmarker()
        if p is None or timestamp is None or time.monotonic() - timestamp > MAX_VISION_AGE_S:
            return None
        # The drone-pose interface returns the vehicle pose in the marker frame.
        # Convert it back into the relative marker offset that the controller
        # expects.
        return -np.asarray(p, dtype=float), orientation, timestamp, sequence

    p, orientation, timestamp, sequence = get_latest_lighttarget_measurement()
    if p is None or timestamp is None or time.monotonic() - timestamp > MAX_VISION_AGE_S:
        return None
    return np.asarray(p, dtype=float), orientation, timestamp, sequence


def latest_aruco_offset(marker_id=None):
    if POSE_TYPE == "drone":
        p, orientation, timestamp, sequence = get_latest_drone_measurement_from_arucomarker(marker_id)
        if p is None or timestamp is None or time.monotonic() - timestamp > MAX_VISION_AGE_S:
            return None
        # The drone-pose interface returns the vehicle pose in the marker frame.
        # Convert it back into the relative marker offset that the controller
        # expects.
        return -np.asarray(p, dtype=float), orientation, timestamp, sequence, marker_id

    p, orientation, timestamp, sequence, detected_marker_id = get_latest_arucotarget_measurement(marker_id)
    if p is None or timestamp is None or time.monotonic() - timestamp > MAX_VISION_AGE_S:
        return None
    return np.asarray(p, dtype=float), orientation, timestamp, sequence, detected_marker_id


def platform_tilt_deg(marker_orientation, drone_to_ned):
    """Angle between platform normal and drone down, for both pose modes."""
    if marker_orientation is None or drone_to_ned is None:
        return None
    marker_orientation = np.asarray(marker_orientation, dtype=float)
    drone_to_ned = np.asarray(drone_to_ned, dtype=float)
    if drone_to_ned.shape != (3, 3):
        return None
    if not np.all(np.isfinite(marker_orientation)) or not np.all(np.isfinite(drone_to_ned)):
        return None
    if marker_orientation.shape == (4,):
        # drone mode publishes q for drone-to-marker.
        marker_to_ned = drone_to_ned @ R.from_quat(marker_orientation).as_matrix().T
    elif marker_orientation.shape == (3, 3):
        # target mode publishes marker-to-NED directly.
        marker_to_ned = marker_orientation
    else:
        return None
    cosine = float(np.dot(marker_to_ned[:, 2], drone_to_ned[:, 2]))
    return float(np.degrees(np.arccos(np.clip(abs(cosine), 0.0, 1.0))))


def latest_drone_to_ned():
    import light_detection_blobbased as vision
    if vision.latest_attitude is None:
        return None
    return np.asarray(vision.latest_attitude, dtype=float).copy()


def start_aruco_tracker(drone):
    threading.Thread(
        target=get_pose_from_arucomarker,
        kwargs={"pose_type": POSE_TYPE, "drone": drone, "acquisition_marker_id": LARGE_ARUCO_MARKER_ID, "acquisition_marker_size_m": LARGE_ARUCO_MARKER_SIZE_M},
        daemon=True,
    ).start()


async def connect_drone():
    print("Connecting...")
    drone = System()
    await drone.connect(system_address=SERIAL_IP)
    async for connection in drone.core.connection_state():
        if connection.is_connected:
            print("Connected")
            return drone
        await asyncio.sleep(1.0)


async def send_velocity(drone, velocity):
    velocity = np.asarray(velocity, dtype=float)
    print(
        f"command NED velocity: N={velocity[0]:+.2f}, "
        f"E={velocity[1]:+.2f}, D={velocity[2]:+.2f} m/s"
    )
    flight_logger.log("velocity_command_ned", velocity=velocity, yaw_deg=command_yaw_deg)
    if ENABLE_AUTONOMY:
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(float(velocity[0]), float(velocity[1]), float(velocity[2]), command_yaw_deg)
        )


def tracking_velocity(position, relative_velocity):
    """Velocity for a target displacement p = p_marker - p_drone in NED."""
    return np.array([
        clip(KP_XY * position[0] + KD_XY * relative_velocity[0], MAX_HORIZONTAL_SPEED),
        clip(KP_XY * position[1] + KD_XY * relative_velocity[1], MAX_HORIZONTAL_SPEED),
        0.0,
    ])


def light_tracking_velocity(position, relative_velocity):
    command = tracking_velocity(position, relative_velocity)
    if np.linalg.norm(position[:2]) <= LIGHT_DESCENT_ALIGN_RADIUS_M:
        command[2] = clip(KP_LIGHT_Z * (position[2] - LIGHT_ACQUISITION_RANGE_M), MAX_LIGHT_DESCENT_SPEED)
    return command


def is_centered(position, relative_velocity):
    lateral_distance = float(np.linalg.norm(position[:2]))
    lateral_speed = float(np.linalg.norm(relative_velocity[:2]))
    return lateral_distance <= ALIGN_RADIUS_M and lateral_speed <= ALIGN_SPEED_M_S


async def prepare_offboard_and_takeoff(drone):
    if not ENABLE_AUTONOMY:
        return

    # PX4 needs an initial setpoint before Offboard can start.
    for _ in range(10):
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, COMMAND_YAW_DEG))
        await asyncio.sleep(0.1)

    try:
        await drone.action.arm()
        await drone.offboard.start()
    except (ActionError, OffboardError) as error:
        flight_logger.log("takeoff_or_offboard_failure", error=repr(error))
        raise RuntimeError(f"Could not arm/start Offboard: {error}") from error

    for _ in range(200):
        await drone.offboard.set_position_ned(
            PositionNedYaw(0.0, 0.0, -TAKEOFF_ALT, COMMAND_YAW_DEG)
        )
        await asyncio.sleep(0.1)


async def start_offboard_control(drone):
    """Request Offboard without arming or commanding takeoff."""
    global command_yaw_deg

    if not ENABLE_AUTONOMY:
        return

    # VelocityNedYaw uses an *absolute* yaw setpoint. Freeze the current yaw
    # before sending the pre-Offboard setpoints, so the handover does not
    # command a turn toward COMMAND_YAW_DEG (north).
    async for attitude in drone.telemetry.attitude_euler():
        if np.isfinite(attitude.yaw_deg):
            command_yaw_deg = float(attitude.yaw_deg)
            print(f"Preserving current heading for Offboard: {command_yaw_deg:.1f} deg")
            break

    # PX4 requires a >2 Hz setpoint stream before accepting Offboard. Until
    # offboard.start() succeeds these setpoints do not override the RC mode.
    for _ in range(20):
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(0.0, 0.0, 0.0, command_yaw_deg)
        )
        await asyncio.sleep(0.1)

    try:
        await drone.offboard.start()
    except OffboardError as error:
        flight_logger.log("offboard_start_failure", error=repr(error))
        raise RuntimeError(f"Could not start Offboard: {error}") from error


async def finish_landing(drone):
    if not ENABLE_AUTONOMY:
        return

    print("Final contact range reached; handing landing to the flight controller.")
    try:
        await drone.action.land()
        async for landed_state in drone.telemetry.landed_state():
            if landed_state == LandedState.ON_GROUND:
                break
    finally:
        try:
            await drone.offboard.stop()
        except OffboardError:
            pass


async def run_mission(light_stop_event, drone):
    if AUTONOMY_START_MODE == "auto_takeoff":
        await prepare_offboard_and_takeoff(drone)
        offboard_active = ENABLE_AUTONOMY
    elif AUTONOMY_START_MODE == "rc_handover":
        offboard_active = False
    else:
        raise ValueError(
            "AUTONOMY_START_MODE must be 'auto_takeoff' or 'rc_handover'"
        )

    await send_velocity(drone, np.zeros(3))

    state = "HOVER"
    mission_start = time.monotonic()
    last_aruco_time = None
    last_acquisition_time = None
    aruco_started = not ENABLE_LIGHT_MARKER
    aruco_valid_since = None
    aligned_since = None
    range_reference = None
    light_stable_since = None
    predicted_landing_since = None
    predicted_landing_command = None

    # These filters are deliberately independent: light/large ArUco is the
    # acquisition source, while ID 0 is the precision landing source.
    acquisition_filter = RelativePoseFilter()
    small_aruco_filter = RelativePoseFilter()

    while True:
        print('state', state)
        now = time.monotonic()
        if now - mission_start > TOTAL_TIMEOUT:
            print("Mission timeout; aborting visual descent.")
            await send_velocity(drone, np.zeros(3))
            return

        light_raw = latest_light_offset() if ENABLE_LIGHT_MARKER else None
        large_aruco_raw = (
            latest_aruco_offset(LARGE_ARUCO_MARKER_ID)
            if aruco_started and not ENABLE_LIGHT_MARKER else None
        )
        small_aruco_raw = (
            latest_aruco_offset(SMALL_ARUCO_MARKER_ID)
            if aruco_started else None
        )

        # The configured acquisition source is either the light marker or the
        # large ArUco. ID 0 is always handled by its own precision filter.
        light_position = light_velocity = None
        if light_raw is not None:
            light_measurement, light_orientation, light_timestamp, _ = light_raw
            light_position, light_velocity = acquisition_filter.update(
                light_measurement + LIGHT_TO_ARUCO_OFFSET_NED, light_timestamp
            )
            last_acquisition_time = light_timestamp

        aruco_position = aruco_velocity = None
        aruco_orientation = None
        if large_aruco_raw is not None:
            large_measurement, _, large_timestamp, _, _ = large_aruco_raw
            light_position, light_velocity = acquisition_filter.update(
                large_measurement, large_timestamp
            )
            last_acquisition_time = large_timestamp

        if small_aruco_raw is not None:
            small_measurement, aruco_orientation, aruco_timestamp, _, _ = small_aruco_raw
            aruco_position, aruco_velocity = small_aruco_filter.update(
                small_measurement, aruco_timestamp
            )
            last_aruco_time = aruco_timestamp

        # In large-ArUco acquisition mode the detector may publish ID 0 on one
        # frame and ID 2 on the next. Keep the last fresh acquisition estimate
        # available during the stable-ID-0 handoff, without mixing filters.
        if (
            light_position is None
            and acquisition_filter.position is not None
            and last_acquisition_time is not None
            and now - last_acquisition_time <= MAX_VISION_AGE_S
        ):
            light_position = acquisition_filter.position.copy()
            light_velocity = acquisition_filter.velocity.copy()

        print('light_position status', light_position)
        flight_logger.log("control_sample", state=state, light_position=light_position, light_velocity=light_velocity, aruco_position=aruco_position, aruco_velocity=aruco_velocity)

        if state == "HOVER":
            await send_velocity(drone, np.zeros(3))
            if light_position is not None:
                state = "LIGHT_TRACK"
                print("Light target acquired.")

        if state == "LIGHT_TRACK":
            if light_position is None:
                light_stable_since = None
                state = "HOVER"
                await send_velocity(drone, np.zeros(3))
            else:
                if AUTONOMY_START_MODE == "rc_handover" and not offboard_active:
                    # Do not take control merely because the marker appears in
                    # one frame. It must remain near and have low relative
                    # motion for the entire hold period.
                    light_is_stable = (
                        np.linalg.norm(light_position) <= OFFBOARD_TAKEOVER_RANGE_M
                        and np.linalg.norm(light_velocity) <= OFFBOARD_TAKEOVER_MAX_SPEED_M_S
                    )
                    light_stable_since = (
                        (light_stable_since or now) if light_is_stable else None
                    )
                    if (
                        light_stable_since is not None
                        and now - light_stable_since >= OFFBOARD_TAKEOVER_STABLE_TIME
                    ):
                        print("Stable light lock: requesting Offboard control.")
                        await start_offboard_control(drone)
                        offboard_active = ENABLE_AUTONOMY

                # Light tracking is lateral only: it follows the platform but
                # preserves a safe height until ArUco supplies precision range.
                await send_velocity(drone, light_tracking_velocity(light_position, light_velocity))

                if ENABLE_LIGHT_MARKER and not aruco_started and np.all(np.abs(light_position) <= ARUCO_START_BOX):
                    print("Starting ArUco tracker while retaining light control.")
                    start_aruco_tracker(drone)
                    aruco_started = True

                # Both acquisition modes use the same handoff: the small
                # precision marker must be continuously visible and stable.
                if (
                    aruco_position is not None
                    and aruco_position[2] <= ARUCO_TRACK_RANGE_M
                ):
                    agreement = float(np.linalg.norm(aruco_position - light_position))
                    if agreement <= ARUCO_LIGHT_AGREEMENT_M:
                        aruco_valid_since = aruco_valid_since or now
                        if now - aruco_valid_since >= ARUCO_STABLE_TIME:
                            state = "ARUCO_TRACK"
                            aligned_since = None
                            print("Stable small ArUco lock: switching control source.")
                    else:
                        aruco_valid_since = None
                else:
                    aruco_valid_since = None

        elif state == "ARUCO_TRACK":
            if aruco_position is None:
                # Return to the configured acquisition source (light or large
                # ArUco); the large marker must not drive precision tracking.
                if light_position is not None:
                    print("Small ArUco lost: returning to acquisition tracking.")
                    state = "LIGHT_TRACK"
                    aruco_valid_since = None
                else:
                    state = "HOVER"
                    await send_velocity(drone, np.zeros(3))
            else:
                await send_velocity(drone, tracking_velocity(aruco_position, aruco_velocity))
                aruco_tilt = platform_tilt_deg(aruco_orientation, latest_drone_to_ned())
                orientation_ok = aruco_tilt is not None and aruco_tilt <= MAX_LANDING_TILT_DEG
                print('orientation_ok', orientation_ok)
                print(is_centered(aruco_position, aruco_velocity))
                orientation_ok = True
                if aruco_position[2] <= ARUCO_TRACK_RANGE_M and is_centered(aruco_position, aruco_velocity) and orientation_ok:
                    aligned_since = aligned_since or now
                    if now - aligned_since >= max(ALIGN_HOLD_TIME, ORIENTATION_HOLD_TIME):
                        range_reference = float(aruco_position[2])
                        state = "FINAL_DESCENT"
                        print(f"ArUco alignment held at tilt {aruco_tilt:.1f} deg: beginning tracked final descent.")
                else:
                    aligned_since = None

        elif state == "PREDICTED_LANDING":
            # The close-range marker may now be outside the camera field of
            # view. Assume the platform keeps the last observed velocity and
            # direction, but only for this strictly bounded final interval.
            if now - predicted_landing_since >= PREDICTED_LANDING_TIME_S:
                await finish_landing(drone)
                return
            await send_velocity(drone, predicted_landing_command)

        elif state == "FINAL_DESCENT":
            if aruco_position is None:
                # Cancel descent immediately when the small precision marker is
                # lost. Resume the selected acquisition mode when it is visible.
                await send_velocity(drone, np.zeros(3))
                if light_position is not None:
                    print("Small ArUco lost: returning to acquisition tracking.")
                    state = "LIGHT_TRACK"
                    aruco_valid_since = None
                    aligned_since = None
                elif (
                    last_aruco_time is None
                    or now - last_aruco_time > LOST_MARKER_TIMEOUT
                ):
                    print("Vision lost during final descent: climbing and aborting.")
                    await send_velocity(drone, np.array([0.0, 0.0, -ABORT_CLIMB_SPEED]))
                    return
            else:
                command = tracking_velocity(aruco_position, aruco_velocity)

                if is_centered(aruco_position, aruco_velocity):
                    range_reference = max(
                        TOUCHDOWN_RANGE_M,
                        range_reference - DESCENT_RATE_M_S * CONTROL_PERIOD,
                    )
                # If the target moves laterally, hold vertical range while the
                # horizontal controller catches up; never descend while off-center.
                vertical_error = aruco_position[2] - range_reference
                command[2] = float(np.clip(
                    KP_Z * vertical_error + KD_Z * aruco_velocity[2],
                    -MAX_CLIMB_SPEED,
                    MAX_DESCENT_SPEED,
                ))
                if not is_centered(aruco_position, aruco_velocity):
                    command[2] = 0.0
                aruco_tilt = platform_tilt_deg(aruco_orientation, latest_drone_to_ned())
                print('aruco_tilt', aruco_tilt)
                if aruco_tilt is None or aruco_tilt > MAX_LANDING_TILT_DEG:
                    command[2] = 0.0

                await send_velocity(drone, command)

                if (
                    aruco_position[2] <= TOUCHDOWN_RANGE_M
                    and is_centered(aruco_position, aruco_velocity)
                    and aruco_tilt is not None
                    and aruco_tilt <= MAX_LANDING_TILT_DEG
                ):
                    predicted_landing_command = command.copy()
                    predicted_landing_command[2] = PREDICTED_LANDING_DESCENT_SPEED_M_S
                    predicted_landing_since = now
                    state = "PREDICTED_LANDING"
                    print("Close-range marker lock: continuing predicted tracking before landing.")

        await asyncio.sleep(CONTROL_PERIOD)


async def main():
    light_stop_event = threading.Event()
    drone = await connect_drone()
    asyncio.create_task(log_drone_telemetry(flight_logger, drone))
    asyncio.create_task(attitude_loop(drone))
    if show_visualization:
        asyncio.create_task(visualization_loop())
    await asyncio.sleep(4)  # Wait for attitude needed by pose_type="target".
    if not ENABLE_LIGHT_MARKER:
        start_aruco_tracker(drone)
    else:
        threading.Thread(
           target=get_pose_from_lightmarker,
           kwargs={
            "stop_event": light_stop_event,
            "pose_type": POSE_TYPE,
            "drone": drone,
            "brightness_threshold": BRIGHTNESS_THRESHOLD,
            },
          daemon=True,
        ).start()

    try:
        await run_mission(light_stop_event, drone)
    finally:
        light_stop_event.set()
        if ENABLE_AUTONOMY:
            try:
                await drone.offboard.stop()
            except OffboardError:
                pass
            try:
                await drone.action.land()
            except ActionError:
                pass
        flight_logger.close()


if __name__ == "__main__":
    asyncio.run(main())
