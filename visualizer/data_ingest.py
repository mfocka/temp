from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from .log_parser import StreamingLogParser, ParsedRecord


@dataclass
class IngestCallbacks:
    on_angles: Callable[[float, float, float, float, str], None]
    on_quat: Optional[Callable[[float, float, float, float, float], None]] = None
    on_raw: Optional[Callable[[float, float, float, float, float, float, float], None]] = None
    on_event: Optional[Callable[[float, str, str, str], None]] = None


def parse_payload_csv(payload: bytes) -> list[str]:
    # Split payload by commas without decoding errors
    text = payload.decode(errors="replace")
    return [part.strip() for part in text.split(",")]


def ingest_bytes_stream(byte_iterable: Iterable[bytes], cb: IngestCallbacks) -> None:
    parser = StreamingLogParser()
    for chunk in byte_iterable:
        for rec in parser.feed(chunk):
            topic = rec.topic.upper()
            parts = parse_payload_csv(rec.payload)
            try:
                if topic == "ANGLES":
                    # [device_id, roll, pitch, yaw, state]
                    if len(parts) >= 5:
                        t = rec.time_sec
                        roll = float(parts[1])
                        pitch = float(parts[2])
                        yaw = float(parts[3])
                        state = parts[4]
                        cb.on_angles(t, roll, pitch, yaw, state)
                elif topic == "QUAT" and cb.on_quat is not None:
                    if len(parts) >= 5:
                        t = rec.time_sec
                        qx = float(parts[1])
                        qy = float(parts[2])
                        qz = float(parts[3])
                        qw = float(parts[4])
                        cb.on_quat(t, qx, qy, qz, qw)
                elif topic == "RAW_DATA" and cb.on_raw is not None:
                    if len(parts) >= 8:
                        t = rec.time_sec
                        ax = float(parts[1])
                        ay = float(parts[2])
                        az = float(parts[3])
                        gx = float(parts[4])
                        gy = float(parts[5])
                        gz = float(parts[6])
                        cb.on_raw(t, ax, ay, az, gx, gy, gz)
                elif topic.endswith("_EVENT") and cb.on_event is not None:
                    # Example: MOTION_EVENT, device_id, RAISED/CLEARED, details
                    if len(parts) >= 4:
                        t = rec.time_sec
                        name = topic
                        action = parts[2]
                        details = parts[3]
                        cb.on_event(t, name, action, details)
            except Exception:
                # Drop invalid records without printing to console
                continue


def ingest_file(file_path: str, cb: IngestCallbacks) -> None:
    # Read raw lines from file and pass as bytes stream
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            for rec in StreamingLogParser().feed(chunk):
                parts = parse_payload_csv(rec.payload)
                topic = rec.topic.upper()
                try:
                    if topic == "ANGLES" and len(parts) >= 5:
                        cb.on_angles(rec.time_sec, float(parts[1]), float(parts[2]), float(parts[3]), parts[4])
                    elif topic == "QUAT" and cb.on_quat is not None and len(parts) >= 5:
                        cb.on_quat(rec.time_sec, float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                    elif topic == "RAW_DATA" and cb.on_raw is not None and len(parts) >= 8:
                        cb.on_raw(rec.time_sec, float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6]))
                    elif topic.endswith("_EVENT") and cb.on_event is not None and len(parts) >= 4:
                        cb.on_event(rec.time_sec, topic, parts[2], parts[3])
                except Exception:
                    continue

