import asyncio
import time
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw

# =========================
# SAFETY CONFIG
# =========================
ENABLE_AUTONOMY = False   # MUST be set True manually
TAKEOFF_ALT = 1.5         # meters (keep low for testing)
MAX_VEL = 0.3             # m/s safety cap
LOST_MARKER_TIMEOUT = 1.0 # seconds
TOTAL_TIMEOUT = 60        # seconds max mission time

# =========================
# FAKE MARKER DETECTOR (replace with your CV)
# return None if marker lost
# =========================
def get_marker_offset():
    """
    Replace this with your real CV output:

    return:
        (x, y, z) in meters (drone frame or NED-consistent frame)
        OR None if not detected
    """

    # Example simulation: marker oscillates
    t = time.time()
    x = 1.0 * (0.5)
    y = 1.0 * (0.2)
    z = 1.0

    return (x, y, z)

# =========================
# MAIN
# =========================
async def run():

    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Connecting...")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected")
            break

    if not ENABLE_AUTONOMY:
        print("AUTONOMY DISABLED (safety switch)")
        return

    # =========================
    # ARM
    # =========================
    print("Arming...")
    await drone.action.arm()

    # =========================
    # TAKEOFF
    # =========================
    print("Taking off...")
    await drone.action.set_takeoff_altitude(TAKEOFF_ALT)
    await drone.action.takeoff()

    await asyncio.sleep(5)

    # =========================
    # START OFFBOARD (HOVER FIRST)
    # MUST send initial setpoint before start
    # =========================
    print("Starting offboard hover...")

    try:
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
        )
        await drone.offboard.start()

    except OffboardError as e:
        print(f"Offboard start failed: {e._result.result}")
        await drone.action.land()
        return

    # =========================
    # STATE MACHINE
    # =========================
    state = "HOVER"
    last_seen = time.time()
    start_time = time.time()

    kp = 0.4  # simple P controller

    try:
        while True:

            now = time.time()

            # safety timeout
            if now - start_time > TOTAL_TIMEOUT:
                print("Mission timeout → landing")
                state = "LAND"

            marker = get_marker_offset()

            # =========================
            # MARKER TRACKING LOGIC
            # =========================
            if marker is not None:
                mx, my, mz = marker
                last_seen = now

                if state == "HOVER":
                    print("Marker detected → TRACKING")
                    state = "TRACK"

                if state == "TRACK":

                    # velocity control (IMPORTANT: capped)
                    vx = kp * mx
                    vy = kp * my
                    vz = 0.0

                    vx = max(min(vx, MAX_VEL), -MAX_VEL)
                    vy = max(min(vy, MAX_VEL), -MAX_VEL)

                    await drone.offboard.set_velocity_ned(
                        VelocityNedYaw(vx, vy, vz, 0.0)
                    )

                    print(f"TRACK vx={vx:.2f} vy={vy:.2f}")

            else:
                # =========================
                # MARKER LOST → SAFE HOVER
                # =========================
                if now - last_seen > LOST_MARKER_TIMEOUT:
                    print("Marker lost → HOLD (hover)")

                    state = "HOVER"

                    await drone.offboard.set_velocity_ned(
                        VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
                    )

            # =========================
            # LAND STATE
            # =========================
            if state == "LAND":
                break

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

    await asyncio.sleep(5)

    print("Disarming...")
    await drone.action.disarm()

    print("Done safely.")

if __name__ == "__main__":
    asyncio.run(run())