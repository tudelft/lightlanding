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

from light_detection_blobbased import (
    attitude_loop,
    get_latest_arucotarget_measurement,
    get_latest_drone_measurement_from_arucomarker,
    get_latest_drone_measurement_from_lightmarker,
    get_latest_lighttarget_measurement,
    get_pose_from_arucomarker,
    get_pose_from_lightmarker,
)


# =========================
# CONFIGURATION
# =========================
MAVLINK_MULTIPLE_CONNECTIONS = True
ENABLE_AUTONOMY = False  # Change manually only after restrained/bench tests.
TAKEOFF_ALT = 3.5
TOTAL_TIMEOUT = 120.0
CONTROL_PERIOD = 0.05
MAX_VISION_AGE_S = 0.20

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
ARUCO_STABLE_TIME = 0.75
ARUCO_LIGHT_AGREEMENT_M = 1.0  # meters: ArUco and light must agree within this distance to switch to ArUco control. If LIGHT_TO_ARUCO_OFFSET_NED is set, make this value lower (0.2-0.5).

# Horizontal tracking, applied to the desired ArUco landing origin.
KP_XY = 0.55
KD_XY = 0.35
MAX_HORIZONTAL_SPEED = 0.65
POSE_FILTER_ALPHA = 0.35
VELOCITY_FILTER_ALPHA = 0.25

# Do not descend until the marker is well centered and its relative lateral
# motion is manageable.  Keep following it whenever descent is paused.
ALIGN_RADIUS_M = 0.20
ALIGN_SPEED_M_S = 0.30
ALIGN_HOLD_TIME = 0.75
MAX_LANDING_TILT_DEG = 10.0
ORIENTATION_HOLD_TIME = 0.75
ARUCO_TRACK_RANGE_M = 1.40
FINAL_RANGE_START_M = 1.20
LIGHT_ACQUISITION_RANGE_M = 1.25
LIGHT_DESCENT_ALIGN_RADIUS_M = 0.60
KP_LIGHT_Z = 0.45
MAX_LIGHT_DESCENT_SPEED = 0.25
TOUCHDOWN_RANGE_M = 0.4  # Must be validated against camera/landing-gear geometry.
DESCENT_RATE_M_S = 0.2 # When in FINAL_DESCENT, the range reference is decremented at this rate.  The controller will try to follow it, but will not descend while off-center.
KP_Z = 0.70
KD_Z = 0.20
MAX_DESCENT_SPEED = 0.25
MAX_CLIMB_SPEED = 0.20

# If vision is absent in final descent, never keep descending blind.
LOST_MARKER_TIMEOUT = 0.35
ABORT_CLIMB_SPEED = 0.20

# VelocityNedYaw's yaw argument is an absolute NED yaw.  Keep this explicit:
# use the heading you have validated for this vehicle, rather than assuming that
# the marker orientation is available from the current target-pose interface.
COMMAND_YAW_DEG = 0.0
BRIGHTNESS_THRESHOLD = 35
POSE_TYPE = "target"

SERIAL_IP = "/dev/ttyACM0" if not MAVLINK_MULTIPLE_CONNECTIONS else "udpin://127.0.0.1:14600"


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


def latest_aruco_offset():
    if POSE_TYPE == "drone":
        p, orientation, timestamp, sequence = get_latest_drone_measurement_from_arucomarker()
        if p is None or timestamp is None or time.monotonic() - timestamp > MAX_VISION_AGE_S:
            return None
        # The drone-pose interface returns the vehicle pose in the marker frame.
        # Convert it back into the relative marker offset that the controller
        # expects.
        return -np.asarray(p, dtype=float), orientation, timestamp, sequence

    p, orientation, timestamp, sequence = get_latest_arucotarget_measurement()
    if p is None or timestamp is None or time.monotonic() - timestamp > MAX_VISION_AGE_S:
        return None
    return np.asarray(p, dtype=float), orientation, timestamp, sequence


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
        kwargs={"pose_type": POSE_TYPE, "drone": drone},
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
    if ENABLE_AUTONOMY:
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(float(velocity[0]), float(velocity[1]), float(velocity[2]), COMMAND_YAW_DEG)
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
        raise RuntimeError(f"Could not arm/start Offboard: {error}") from error

    for _ in range(100):
        await drone.offboard.set_position_ned(
            PositionNedYaw(0.0, 0.0, -TAKEOFF_ALT, COMMAND_YAW_DEG)
        )
        await asyncio.sleep(0.1)


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
    await prepare_offboard_and_takeoff(drone)
    await send_velocity(drone, np.zeros(3))

    state = "HOVER"
    mission_start = time.monotonic()
    last_aruco_time = None
    aruco_started = False
    aruco_valid_since = None
    aligned_since = None
    range_reference = None

    light_filter = RelativePoseFilter()
    aruco_filter = RelativePoseFilter()

    while True:
        now = time.monotonic()
        if now - mission_start > TOTAL_TIMEOUT:
            print("Mission timeout; aborting visual descent.")
            await send_velocity(drone, np.zeros(3))
            return

        light_raw = latest_light_offset()
        aruco_raw = latest_aruco_offset() if aruco_started else None

        # The light measurement is shifted onto the ArUco landing origin before
        # it is used for control.  p_aruco = p_light + (aruco - light).
        light_position = light_velocity = None
        if light_raw is not None:
            light_measurement, light_orientation, light_timestamp, _ = light_raw
            light_position, light_velocity = light_filter.update(
                light_measurement + LIGHT_TO_ARUCO_OFFSET_NED, light_timestamp
            )

        aruco_position = aruco_velocity = None
        if aruco_raw is not None:
            aruco_measurement, aruco_orientation, aruco_timestamp, _ = aruco_raw
            aruco_position, aruco_velocity = aruco_filter.update(aruco_measurement, aruco_timestamp)
            last_aruco_time = aruco_timestamp

        if state == "HOVER":
            await send_velocity(drone, np.zeros(3))
            if light_position is not None:
                state = "LIGHT_TRACK"
                print("Light target acquired.")

        if state == "LIGHT_TRACK":
            if light_position is None:
                state = "HOVER"
                await send_velocity(drone, np.zeros(3))
            else:
                # Light tracking is lateral only: it follows the platform but
                # preserves a safe height until ArUco supplies precision range.
                await send_velocity(drone, light_tracking_velocity(light_position, light_velocity))

                if not aruco_started and np.all(np.abs(light_position) <= ARUCO_START_BOX):
                    print("Starting ArUco tracker while retaining light control.")
                    start_aruco_tracker(drone)
                    aruco_started = True

                if aruco_position is not None:
                    agreement = float(np.linalg.norm(aruco_position - light_position))
                    if agreement <= ARUCO_LIGHT_AGREEMENT_M:
                        aruco_valid_since = aruco_valid_since or now
                        if now - aruco_valid_since >= ARUCO_STABLE_TIME:
                            state = "ARUCO_TRACK"
                            aligned_since = None
                            print("Stable ArUco lock: switching control source.")
                    else:
                        aruco_valid_since = None

        elif state == "ARUCO_TRACK":
            if aruco_position is None:
                # Light stays alive until final descent and is a safe fallback.
                if light_position is not None:
                    print("ArUco lost: falling back to light tracking.")
                    state = "LIGHT_TRACK"
                    aruco_valid_since = None
                else:
                    state = "HOVER"
                    await send_velocity(drone, np.zeros(3))
            else:
                await send_velocity(drone, tracking_velocity(aruco_position, aruco_velocity))
                aruco_tilt = platform_tilt_deg(aruco_orientation, latest_drone_to_ned())
                orientation_ok = aruco_tilt is not None and aruco_tilt <= MAX_LANDING_TILT_DEG
                if aruco_position[2] <= ARUCO_TRACK_RANGE_M and is_centered(aruco_position, aruco_velocity) and orientation_ok:
                    aligned_since = aligned_since or now
                    if now - aligned_since >= max(ALIGN_HOLD_TIME, ORIENTATION_HOLD_TIME):
                        range_reference = min(float(aruco_position[2]), FINAL_RANGE_START_M)
                        state = "FINAL_DESCENT"
                        print(f"ArUco alignment held at tilt {aruco_tilt:.1f} deg: beginning tracked final descent.")
                else:
                    aligned_since = None

        elif state == "FINAL_DESCENT":
            if aruco_position is None:
                # Cancel the last descent command immediately. Light detections
                # must not keep a final ArUco-guided descent alive.
                await send_velocity(drone, np.zeros(3))
                if (
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
                if aruco_tilt is None or aruco_tilt > MAX_LANDING_TILT_DEG:
                    command[2] = 0.0

                await send_velocity(drone, command)

                if (
                    aruco_position[2] <= TOUCHDOWN_RANGE_M
                    and is_centered(aruco_position, aruco_velocity)
                    and aruco_tilt is not None
                    and aruco_tilt <= MAX_LANDING_TILT_DEG
                ):
                    await finish_landing(drone)
                    return

        await asyncio.sleep(CONTROL_PERIOD)


async def main():
    light_stop_event = threading.Event()
    drone = await connect_drone()
    asyncio.create_task(attitude_loop(drone))
    await asyncio.sleep(2.0)  # Wait for attitude needed by pose_type="target".

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


if __name__ == "__main__":
    asyncio.run(main())
