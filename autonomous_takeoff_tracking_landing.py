import asyncio
import time
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw, PositionNedYaw
from light_detection_blobbased import get_latest_lighttarget_location, get_latest_arucotarget_location, get_pose_from_arucomarker, get_pose_from_lightmarker
import threading
# =========================
# CONFIG
# =========================
MAVLINK_MULTIPLE_CONNECTIONS = True  # If we are also sending Mocap data to drone on serial then set this to True to avoid conflicts. Requires mavlink_routerd running on the pi.
ENABLE_AUTONOMY = True   # MUST be set True manually
TAKEOFF_ALT = 6.5        # meters (keep low for testing)
MAX_VEL = 0.5             # m/s safety cap
LOST_MARKER_TIMEOUT = 1.0 # seconds
TOTAL_TIMEOUT = 60        # seconds max mission time
ARUCO_SWITCH_CRITERIA = [1.0, 1.0, 1.5]    # meters (distance to marker to switch from light-based to aruco-based TRACKING)
ARUCO_LANDING_THRESHOLD = 0.8  # meters (vertical distance to aruco marker to initiate landing)
kpx = 0.1  # simple P controller gains
kpy = 0.1
kpz = 0.2

if (not MAVLINK_MULTIPLE_CONNECTIONS):
    serial_ip = "/dev/ttyACM0"  # Serial port for MAVLink connection
else:
    serial_ip = "udpin://127.0.0.1:14600"  # UDP port for MAVLink connection

def start_aruco_tracker():
    """
    Start the aruco tracker in a separate thread.
    """
    threading.Thread(target=get_pose_from_arucomarker, 
    kwargs={},
    daemon=True).start()

def check_aruco_startcriteria_met(marker):
    """
    Check if the marker is close enough to start aruco-based landing mode.
    """
    mx, my, mz = marker
    if (abs(mx) < ARUCO_SWITCH_CRITERIA[0] + 1 and abs(my) < ARUCO_SWITCH_CRITERIA[1] + 1 and abs(mz) < ARUCO_SWITCH_CRITERIA[2] + 1):
        print("Aruco start criteria met → starting aruco tracker")
        return True
    else:
        return False

def check_aruco_switchcriteria_met(marker):
    """
    Check if the marker is close enough to switch to aruco-based landing mode.
    """
    mx, my, mz = marker
    if (abs(mx) < ARUCO_SWITCH_CRITERIA[0] and abs(my) < ARUCO_SWITCH_CRITERIA[1] and abs(mz) < ARUCO_SWITCH_CRITERIA[2]):
        print("Aruco switch criteria met → switching to aruco tracker")
        return True
    else:
        return False
                            
async def print_drone_position(drone):
    async for pos in drone.telemetry.position_velocity_ned():
        print("Drone position (NED):")
        print(f"x: {pos.position.north_m_s:.2f} m")
        print(f"y: {pos.position.east_m_s:.2f} m")
        print(f"z: {pos.position.down_m_s:.2f} m")
        print("---")
    
def get_lightmarker_offset():
    """
    return:
        (x, y, z) in meters (drone frame or NED-consistent frame)
        OR None if not detected
    """
    p, q = get_latest_lighttarget_location()

    t = time.time()
    if (p is not None):
        x = -1.0 * p[0]
        y = -1.0 * p[1]
        z = -1.0 * p[2]

        return (x, y, z)
    else:
        return None

def get_arucomarker_offset():
    """
    return:
        (x, y, z) in meters (drone frame or NED-consistent frame)
        OR None if not detected
    """
    p, q = get_latest_arucotarget_location()

    t = time.time()
    if (p is not None):
        x = -1.0 * p[0]
        y = -1.0 * p[1]
        z = -1.0 * p[2]

        return (x, y, z)
    else:
        return None

# =========================
# MAIN
# =========================
async def run():
    drone = System()
    await drone.connect(system_address=serial_ip)

    print("Connecting...")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected")
            break

    if not ENABLE_AUTONOMY:
        print("AUTONOMY DISABLED (safety switch)")
        return

    print("-- Sending holding setpoints")
    for i in range(20):
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))
        print("Sending setpoints to PX4 before we can switch to offboard mode")
        await asyncio.sleep(0.1)

    print("Waiting for health checks...")
    async for health in drone.telemetry.health():
          print("HEALTH STATUS")
          print(health)
          break

    # =========================
    # ARM
    # =========================
    print("Arming...")
    await drone.action.arm()

    try:
       await drone.action.arm()
       print("Armed")

    except ActionError as e:
       print(f"Arming failed: {e}")
       return

    # =========================
    # TAKEOFF
    # =========================
    print("SWITCHING TO OFFBOARD MODE!")
    try:
        await drone.offboard.start()

    except OffboardError as e:
        print(f"Offboard start failed: {e._result.result}")
        return

    print("Taking off...")
    # Command takeoff to 5 m
    await drone.offboard.set_position_ned(
        PositionNedYaw(0.0, 0.0, -1 * TAKEOFF_ALT, 0.0)
    )
    # await drone.action.set_takeoff_altitude(TAKEOFF_ALT)
    # await drone.action.takeoff()
    await print_drone_position(drone)
    await asyncio.sleep(10)

    # =========================
    # START OFFBOARD (HOVER FIRST)
    # MUST send initial setpoint before start
    # =========================
    print("Starting offboard hover...")
    await print_drone_position(drone)
    await drone.offboard.set_velocity_ned(
            VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
        )

    # =========================
    # STATE MACHINE
    # =========================
    state = "HOVER"
    last_seen = time.time()
    start_time = time.time()

    try:
        while True:
            print(f"STATE: {state}")
            await print_drone_position(drone)
            now = time.time()
            # safety timeout
            if now - start_time > TOTAL_TIMEOUT:
                print("Mission timeout → landing")
                state = "LAND"

            # =========================
            # LAND STATE
            # =========================
            if state == "LAND":
                break

            light_marker = get_lightmarker_offset()
            if light_marker is not None:
                mx, my, mz = light_marker
                last_seen = now

                if state == "HOVER":
                    print("Light Marker detected → TRACK")
                    state = "TRACK"

                if state == "TRACK":
                    if check_aruco_startcriteria_met(light_marker):
                        start_aruco_tracker()

                    if check_aruco_switchcriteria_met(light_marker): 
                        print("Aruco switch criteria met → ARUCO_TRACK")
                        state = "ARUCO_TRACK"
                    else:
                        # velocity control (capped to MAX_VEL)
                        vx = kpx * mx
                        vy = kpy * my
                        vz = kpz * mz if (abs(mz) >= 2.5) else 0.0

                        vx = max(min(vx, MAX_VEL), -MAX_VEL)
                        vy = max(min(vy, MAX_VEL), -MAX_VEL)
                        vz = max(min(vz, MAX_VEL), -MAX_VEL)
                        await drone.offboard.set_velocity_ned(
                            VelocityNedYaw(vx, vy, vz, 0.0)
                        )

                        print(f"Light-based TRACK vx={vx:.2f} vy={vy:.2f} vz={vz:.2f}")
                
                if state == "ARUCO_TRACK":
                    stop_event.set()  # Stop the light marker detection thread
                    light_marker = None  # Clear the light marker variable
                    aruco_marker = get_arucomarker_offset()

                    if (aruco_marker is not None):
                        mx, my, mz = aruco_marker
                        print("Aruco marker detected → TRACKING")
                        if (abs(mz) < ARUCO_LANDING_THRESHOLD):
                            print("Aruco marker close enough → LANDING")
                            state = "LAND"
                            break

                        else:
                            # velocity control (capped to MAX_VEL)
                            vx = kpx * mx
                            vy = kpy * my
                            vz = kpz * mz 

                            vx = max(min(vx, MAX_VEL), -MAX_VEL)
                            vy = max(min(vy, MAX_VEL), -MAX_VEL)
                            vz = max(min(vz, MAX_VEL), -MAX_VEL)
                            await drone.offboard.set_velocity_ned(
                                VelocityNedYaw(vx, vy, vz, 0.0)
                            )

                    else:
                        print("Aruco marker lost → HOVER")
                        state = "HOVER"
                        await drone.offboard.set_velocity_ned(
                            VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
                        )

                    print(f"TRACK vx={vx:.2f} vy={vy:.2f} vz={vz:.2f}")


            else:
                # =========================
                # MARKER LOST → SAFE HOVER
                # =========================
                if now - last_seen > LOST_MARKER_TIMEOUT:
                    await drone.offboard.set_velocity_ned(
                        VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
                    )

                    print("Marker lost → HOLD (hover)")
                    state = "HOVER"

            await asyncio.sleep(0.05)  # 20 Hz control loop

    except Exception as e:
        print(f"Error: {e}")

    # =========================
    # SAFE LANDING
    # =========================
    print("Landing...")
    try:
        await drone.offboard.stop()
    except:
        pass

    await drone.action.land()

    await asyncio.sleep(3)

    print("Disarming...")
    await drone.action.disarm()
    print("Disarmed safely.")

if __name__ == "__main__":
    stop_event = threading.Event()
    threading.Thread(target=get_pose_from_lightmarker, 
    kwargs={"stop_event": stop_event, "brightness_threshold": 35},
    daemon=True).start()

    asyncio.run(run(stop_event))

