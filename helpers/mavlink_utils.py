import time
import math
import os

os.environ["MAVLINK20"] = "1"
os.environ["MAVLINK_DIALECT"] = "common"
from pymavlink import mavutil

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
