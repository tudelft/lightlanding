"""Low-overhead, asynchronous flight logging utilities."""

import asyncio
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


class FlightLogger:
    def __init__(self, enabled, image_period_s=0.5):
        self.enabled = enabled
        self.image_period_s = image_period_s
        self._queue = queue.Queue(maxsize=512)
        self._last_image_time = {}
        self._thread = None
        self.run_dir = None
        self.log_path = None
        self.image_dir = None

        if enabled:
            stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
            self.run_dir = Path.home() / "logs"
            self.log_path = self.run_dir / f"flight_{stamp}.jsonl"
            self.image_dir = self.run_dir / f"images_{stamp}"
            self.image_dir.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(target=self._write_loop, daemon=True)
            self._thread.start()
            self.log("logger_started", log_file=str(self.log_path), image_directory=str(self.image_dir))

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if hasattr(value, "__dict__"):
            return value.__dict__
        return str(value)

    def log(self, event, **fields):
        if not self.enabled:
            return
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "monotonic_s": time.monotonic(),
            "event": event,
            **fields,
        }
        try:
            self._queue.put_nowait(("record", record))
        except queue.Full:
            pass

    def save_image(self, source, image):
        """Queue at most one annotated image per source every image_period_s."""
        if not self.enabled or image is None:
            return
        now = time.monotonic()
        if now - self._last_image_time.get(source, 0.0) < self.image_period_s:
            return
        self._last_image_time[source] = now
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        try:
            self._queue.put_nowait(("image", source, stamp, image.copy()))
        except queue.Full:
            pass

    def _write_loop(self):
        import cv2

        with self.log_path.open("a", encoding="utf-8", buffering=1) as stream:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                if item[0] == "record":
                    stream.write(json.dumps(item[1], default=self._json_default) + "\n")
                else:
                    _, source, stamp, image = item
                    cv2.imwrite(str(self.image_dir / f"{stamp}_{source}.jpg"), image)

    def close(self):
        if self.enabled and self._thread is not None:
            self.log("logger_stopped")
            self._queue.put(None)
            self._thread.join(timeout=2.0)


flight_logger = FlightLogger(False)


def configure_flight_logger(enabled):
    global flight_logger
    flight_logger.close()
    flight_logger = FlightLogger(enabled)
    return flight_logger


async def log_drone_telemetry(logger, drone):
    """Log state changes and position/velocity without changing telemetry rates."""
    if not logger.enabled:
        return

    async def landed_state_loop():
        async for value in drone.telemetry.landed_state():
            logger.log("drone_landed_state", value=str(value))

    async def flight_mode_loop():
        async for value in drone.telemetry.flight_mode():
            logger.log("drone_flight_mode", value=str(value))

    async def position_loop():
        last_log = 0.0
        async for value in drone.telemetry.position_velocity_ned():
            now = time.monotonic()
            if now - last_log >= 0.2:
                last_log = now
                logger.log("drone_position_velocity_ned", position=value.position, velocity=value.velocity)

    async def status_text_loop():
        async for value in drone.telemetry.status_text():
            logger.log("drone_status_text", type=str(value.type), text=value.text)

    results = await asyncio.gather(
        landed_state_loop(), flight_mode_loop(), position_loop(), status_text_loop(),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, Exception):
            logger.log("telemetry_logger_failure", error=repr(result))
