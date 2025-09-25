from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional


TIMESTAMP_PREFIX_RE = re.compile(rb"^\[(?P<sec>[0-9]+(?:\.[0-9]+)?)s\]\s+")


@dataclass
class ParsedRecord:
    time_sec: float
    topic: str
    payload: bytes


class StreamingLogParser:
    """
    Robust streaming parser that handles partial lines and mixed binary/utf8.
    - Accepts arbitrary byte chunks via feed()
    - Yields complete ParsedRecord as they are recognized
    - Tolerates truncated chunks like b"[1042\n" and resumes on next data
    """

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> Iterator[ParsedRecord]:
        # Append new bytes
        self._buffer.extend(data)
        start = 0
        buflen = len(self._buffer)

        # Process line by line without splitting on newlines prematurely
        while True:
            # Find newline for a complete record
            newline_index = self._buffer.find(b"\n", start)
            if newline_index == -1:
                # Keep remaining partial data for next feed
                break

            line = bytes(self._buffer[start:newline_index]).strip()
            start = newline_index + 1

            if not line:
                continue

            # Example line: b"[10420.300s] ANGLES,2404433308,0.443,0.029,2.066,MONITORING"
            # Or without leading b prefix once decoded. We operate on bytes for robustness.
            # Remove timestamp prefix
            m = TIMESTAMP_PREFIX_RE.match(line)
            if not m:
                # Skip malformed line but continue parsing
                continue
            ts_sec = float(m.group("sec"))
            rest = line[m.end():]

            # Topic is before the first comma
            comma_index = rest.find(b",")
            if comma_index == -1:
                # No payload; treat the remainder as topic only
                topic = rest.decode(errors="replace").strip()
                payload = b""
            else:
                topic = rest[:comma_index].decode(errors="replace").strip()
                payload = rest[comma_index + 1 :]

            yield ParsedRecord(time_sec=ts_sec, topic=topic, payload=payload)

        # Compact buffer by removing processed prefix
        if start > 0:
            del self._buffer[:start]

