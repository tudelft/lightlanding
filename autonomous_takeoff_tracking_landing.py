import asyncio
import time
from mavsdk import System
from mavsdk.telemetry import LandedState
from mavsdk.offboard import OffboardError, VelocityNedYaw, PositionNedYaw
from mavsdk.action import ActionError
from light_detection_blobbased import get_latest_lighttarget_location, get_latest_arucotarget_location, get_latest_pose_from_lightmarker, get_latest_pose_from_arucomarker , get_pose_from_arucomarker, get_pose_from_lightmarker, attitude_loop
import threading

# =========================
# CONFIG
# =========================
MAVLINK_MULTIPLE_CONNECTIONS = True  # If we are also sending Mocap data to drone on serial then set this to True to avoid conflicts. Requires mavlink_routerd running on the pi.
ENABLE_AUTONOMY = False   # MUST be set to True manually. When False the drone will not move due to offboard control, but the code will still run and print debug info. This is useful for testing the vision system without risking a crash.
TAKEOFF_ALT = 3.5       # meters (keep low for testing)
MAX_VEL = 0.5             # m/s safety cap
LOST_MARKER_TIMEOUT = 1.0 # seconds
TOTAL_TIMEOUT = 120        # seconds max mission time
ARUCO_SWITCH_CRITERIA = [1.5, 1.5, 1.5]    # meters (distance to marker to switch from light-based to aruco-based TRACKING)
ARUCO_LANDING_THRESHOLD = 0.5  # meters (vertical distance to aruco marker to initiate landing)
kpx = 0.25  # simple P controller gains
kpy = 0.25
kpz = 0.25

pose_type = "target"  # "drone" or "target"; 'drone' when the drone's attitude is not reliable, 'target' when the drone's attitude is reliable. The former will suffer from planar ambiguity, the latter will not. The drone's attitude is reliable when the drone is in stable flight and not being disturbed by other forces.

if (not MAVLINK_MULTIPLE_CONNECTIONS):
    serial_ip = "/dev/ttyACM0"  # Serial port for MAVLink connection
else:
    serial_ip = "udpin://127.0.0.1:14600"  # UDP port for MAVLink connection


async def connect_drone():
    print("Connecting...")
    drone = System()
    await drone.connect(system_address=serial_ip)
    print("Connecting...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected")
            return drone
        else:
            print("Waiting for connection...")
            await asyncio.sleep(1)

def start_aruco_tracker(pose_type, drone):
    """
    Start the aruco tracker in a separate thread.
    """
    threading.Thread(target=get_pose_from_arucomarker, 
    kwargs={"pose_type": pose_type, "drone": drone},
    daemon=True).start()

aruco_started = 0
def check_aruco_startcriteria_met(marker):
    """
    Check if the marker is close enough to start aruco-based landing mode.
    """
    mx, my, mz = marker
    if (abs(mx) < ARUCO_SWITCH_CRITERIA[0] + 1 and abs(my) < ARUCO_SWITCH_CRITERIA[1] + 1 and abs(mz) < ARUCO_SWITCH_CRITERIA[2] + 1):
        print("Aruco start criteria met → starting aruco tracker")
        global aruco_started
        aruco_started = 1
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
        print(f"x: {pos.position.north_m:.2f} m")
        print(f"y: {pos.position.east_m:.2f} m")
        print(f"z: {pos.position.down_m:.2f} m")
        print("---")

async def get_current_position(drone):
     async for position in drone.telemetry.position_velocity_ned():
           return position.position    
    
def get_lightmarker_offset(pose_type):
    """
    return:
        (x, y, z) in meters (drone frame or NED-consistent frame)
        OR None if not detected
    """
    
    if (pose_type == 'drone'): 
        p, q = get_latest_pose_from_lightmarker()
        if (p is not None):
            x = -1.0 * p[0]
            y = -1.0 * p[1]
            z = -1.0 * p[2]

            return (x, y, z)
        else:
            return None
        
    elif (pose_type == 'target'):
        p, q = get_latest_lighttarget_location()
        if (p is not None):
            x = p[0]
            y = p[1]
            z = p[2]

            return (x, y, z)
        
        else:
            return None
            print('Light marker offset is None')

def get_arucomarker_offset(pose_type):
    """
    return:
        (x, y, z) in meters (drone frame or NED-consistent frame)
        OR None if not detected
    """

    if (pose_type == 'drone'):
        p, q = get_latest_pose_from_arucomarker()
        if (p is not None):
            x = -1.0 * p[0]
            y = -1.0 * p[1]
            z = -1.0 * p[2]

            return (x, y, z)
        
        else:
            return None
        
    elif (pose_type == 'target'):
        p, q = get_latest_arucotarget_location()
        if (p is not None):
            x = p[0]
            y = p[1]
            z = p[2]

            return (x, y, z)
        
        else:
            return None

# =========================
# MAIN
# =========================
async def stream_setpoints(drone, stop_signal):
    """Background loop to ensure PX4 never sees a gap in offboard data"""
    while not stop_signal.is_set():
        try:
            await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))
        except Exception:
            pass
        await asyncio.sleep(0.4) # Must be > 2Hz (0.5s max gap)

async def run(stop_event, drone):
    print("Waiting for local and home position health checks...")
#    async for health in drone.telemetry.health():
#        if health.is_local_position_ok and health.is_home_position_ok:
#            break
#        await asyncio.sleep(1)

    if ENABLE_AUTONOMY:
        # Start continuous background setpoint streaming
        stop_streaming = asyncio.Event()
        stream_task = asyncio.create_task(stream_setpoints(drone, stop_streaming))
    
    # Give PX4 a moment to register the initial stream
    await asyncio.sleep(0.5)

    if ENABLE_AUTONOMY:
        # 1. ARMING
        print("Arming...")
        try:
            await drone.action.arm()
            print("Armed.")
        except ActionError as e:
            print(f"Arming failed: {e}")
            stop_streaming.set()
            await stream_task
            return

        # 2. Switch to offboard
        print("SWITCHING TO OFFBOARD MODE!")
        try:
            await drone.offboard.start()
            print("Successfully in Offboard mode!")
        except OffboardError as e:
            print(f"Offboard start failed: {e._result.result}")
            # Stop background loop and clean up safety if it fails
            stop_streaming.set()
            await stream_task
            await drone.action.disarm()
            return

        # 3. TAKEOFF
        print("Taking off...")
        stop_streaming.set() # Stop the baseline holding stream
        await stream_task
        # Command takeoff to 5 m
        for i in range(150):
            await drone.offboard.set_position_ned(
                PositionNedYaw(0.0, 0.0, -1 * TAKEOFF_ALT, 0.0)
            )
            # await drone.action.set_takeoff_altitude(TAKEOFF_ALT)
            # await drone.action.takeoff()
            await asyncio.sleep(0.1)
        
    # 4. Transition to actual flight loop
    print("Beginning flight plan...")

    # =========================
    # START OFFBOARD (HOVER FIRST)
    # MUST send initial setpoint before start
    # =========================
    print("Starting offboard hover...")
#    await print_drone_position(drone)
    if ENABLE_AUTONOMY:
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
            now = time.time()

            # safety timeout
            if now - start_time > TOTAL_TIMEOUT:
                print("Mission timeout → landing")
                state = "LAND"

            if state == "LAND":
                break

            elif state == "HOVER":
                print('State HOVER, sending zero velocities')
                if ENABLE_AUTONOMY:
                    await drone.offboard.set_velocity_ned(
                        VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
                    )

            light_marker = get_lightmarker_offset(pose_type)
            if light_marker is not None:
                mx, my, mz = light_marker
                last_seen = now
                print("Light Marker detected → TRACK")
                state = "LIGHT_TRACK"
            
            elif light_marker is None and state != "ARUCO_TRACK":
                state = "HOVER"

            if state == "LIGHT_TRACK":
                global aruco_started
                if check_aruco_startcriteria_met(light_marker) and not aruco_started:
                    start_aruco_tracker(pose_type, drone)

                if check_aruco_switchcriteria_met(light_marker): 
                    print("Aruco switch criteria met → ARUCO_TRACK")
                    state = "ARUCO_TRACK"

                elif not check_aruco_switchcriteria_met(light_marker):
                    # velocity control (capped to MAX_VEL)
                    vx = kpx * mx
                    vy = kpy * my
                    vz = kpz * mz if (abs(mz) >= 2.5) else 0.0

                    vx = max(min(vx, MAX_VEL), -MAX_VEL)
                    vy = max(min(vy, MAX_VEL), -MAX_VEL)
                    vz = max(min(vz, MAX_VEL), -MAX_VEL)

                    if ENABLE_AUTONOMY:
                        await drone.offboard.set_velocity_ned(
                            VelocityNedYaw(vx, vy, vz, 0.0)
                        )

                    print(f"Light-based TRACK vx={vx:.2f} vy={vy:.2f} vz={vz:.2f}")
                
            if state == "ARUCO_TRACK":
                print('ARUCO_TRACK')
                stop_event.set()  # Stop the light marker detection thread
                light_marker = None  # Clear the light marker variable
                aruco_marker = get_arucomarker_offset(pose_type)

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

                        if ENABLE_AUTONOMY:
                            await drone.offboard.set_velocity_ned(
                                VelocityNedYaw(vx, vy, vz, 0.0)
                            )
                        print(f" Aruco TRACK vx={vx:.2f} vy={vy:.2f} vz={vz:.2f}")

                else:
                    print("Aruco marker lost → HOVER")
                    state = "HOVER"
                    if ENABLE_AUTONOMY:
                        await drone.offboard.set_velocity_ned(
                            VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
                        )

            await asyncio.sleep(0.05)  # 20 Hz control loop

    except Exception as e:
        print(f"Error: {e}")

    # =========================
    # LANDING SEQUENCE
    # =========================
    if ENABLE_AUTONOMY:
        # =========================
        # LANDING SEQUENCE
        # =========================
        print("Landing...")
        try:
            await drone.action.land()
            print("Landing command sent.")
        except Exception as e:
            print(f"Landing command failed: {e}")
            return

        # Wait until the drone is actually on the ground
        print("Waiting for touchdown...")
        try:
            async for state in drone.telemetry.landed_state():
                print(f"Landed state: {state}")

                if state == LandedState.ON_GROUND:
                    print("Drone is on the ground.")
                    break

        except Exception as e:
            print(f"Landing state monitoring failed: {e}")
            return

        # Now it is safe to leave Offboard mode
        print("Stopping offboard...")
        try:
            await drone.offboard.stop()
            print("Offboard stopped.")
        except Exception as e:
            print(f"Offboard stop failed: {e}")

        # Disarm motors
        print("Disarming...")

        try:
            await drone.action.disarm()
            print("Disarmed safely.")
        except Exception as e:
            print(f"Disarming failed: {e}")

async def main():
    stop_event = threading.Event()
    drone = await connect_drone()
    await   asyncio.sleep(2)
    # Run attitude loop in the background
    asyncio.create_task(attitude_loop(drone))
    await  asyncio.sleep(2) # get some attitude first before  starting light tracker
    # Start OpenCV thread
    threading.Thread(
        target=get_pose_from_lightmarker,
        kwargs={
            "stop_event": stop_event,
            "pose_type": pose_type,
            "drone": drone,
            "brightness_threshold": 35,
        },
        daemon=True,
    ).start()

    # Main autonomous mission
    await run(stop_event, drone)

if __name__ == "__main__":
    asyncio.run(main())
