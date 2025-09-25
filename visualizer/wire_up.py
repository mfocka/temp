from __future__ import annotations

from typing import Optional

from PySide6 import QtCore

from .main_window import MainWindow
from .data_ingest import IngestCallbacks, ingest_file


def connect_log_to_ui(window: MainWindow, log_path: str) -> None:
    # Periodically feed data from file to UI without blocking the event loop
    def on_angles(t: float, roll: float, pitch: float, yaw: float, state: str) -> None:
        window.filter_output_tab.add_angles(t, roll, pitch, yaw)

    def on_quat(t: float, qx: float, qy: float, qz: float, qw: float) -> None:
        # Optional: extend UI later
        return

    def on_raw(t: float, ax: float, ay: float, az: float, gx: float, gy: float, gz: float) -> None:
        # Optional: extend UI later
        return

    def on_event(t: float, name: str, action: str, details: str) -> None:
        window.events_tab.add_event(t, name, action, details)

    cb = IngestCallbacks(on_angles=on_angles, on_quat=on_quat, on_raw=on_raw, on_event=on_event)

    # Run ingestion in a worker thread so large files do not block UI
    worker = _IngestWorker(log_path, cb)
    thread = QtCore.QThread(window)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()


class _IngestWorker(QtCore.QObject):
    finished = QtCore.Signal()

    def __init__(self, path: str, cb: IngestCallbacks) -> None:
        super().__init__()
        self._path = path
        self._cb = cb

    @QtCore.Slot()
    def run(self) -> None:
        try:
            ingest_file(self._path, self._cb)
        finally:
            self.finished.emit()

