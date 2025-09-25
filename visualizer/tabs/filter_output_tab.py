from typing import List, Optional, Tuple
from PySide6 import QtWidgets, QtCore
import numpy as np
import pyqtgraph as pg


class FilterOutputTab(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        # Fast pyqtgraph configuration
        pg.setConfigOptions(antialias=False, useOpenGL=False, background="#111", foreground="#DDD")

        layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(layout)

        # Controls bar
        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        self.clear_button = QtWidgets.QPushButton("Clear")
        controls.addWidget(self.clear_button)
        controls.addStretch(1)

        # Graphs container
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget, 1)

        # Existing graphs (placeholders for roll/pitch/yaw/quaternion etc.)
        self.angles_plot = self.plot_widget.addPlot(row=0, col=0, title="Angles (deg)")
        self.angles_plot.showGrid(x=True, y=True, alpha=0.3)

        # New: Resulting angle graph (fast) on second row
        self.resulting_angle_plot = self.plot_widget.addPlot(row=1, col=0, title="Resulting Angle (deg)")
        self.resulting_angle_plot.showGrid(x=True, y=True, alpha=0.3)

        # Curves
        self.angle_curve_roll = self.angles_plot.plot(pen=pg.mkPen("#3fa7ff", width=2))
        self.angle_curve_pitch = self.angles_plot.plot(pen=pg.mkPen("#ff7f50", width=2))
        self.angle_curve_yaw = self.angles_plot.plot(pen=pg.mkPen("#a0e75a", width=2))

        self.resulting_angle_curve = self.resulting_angle_plot.plot(pen=pg.mkPen("#e94be8", width=2))

        # Buffers
        self.max_points: int = 3000
        self.time_buffer: List[float] = []
        self.roll_buffer: List[float] = []
        self.pitch_buffer: List[float] = []
        self.yaw_buffer: List[float] = []
        self.resulting_angle_buffer: List[float] = []

        self.clear_button.clicked.connect(self._clear)

        # Update timer for batched rendering (avoid per-sample repaints)
        self._pending_data: List[Tuple[float, float, float, float]] = []
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._flush_pending_render)
        self._timer.start(33)  # ~30 FPS

    def _clear(self) -> None:
        self.time_buffer.clear()
        self.roll_buffer.clear()
        self.pitch_buffer.clear()
        self.yaw_buffer.clear()
        self.resulting_angle_buffer.clear()
        self._apply_data()

    def add_angles(self, t_sec: float, roll_deg: float, pitch_deg: float, yaw_deg: float) -> None:
        # Queue data for batched UI update
        self._pending_data.append((t_sec, roll_deg, pitch_deg, yaw_deg))

    def _flush_pending_render(self) -> None:
        if not self._pending_data:
            return
        # Extend buffers
        for (t, r, p, y) in self._pending_data:
            self.time_buffer.append(t)
            self.roll_buffer.append(r)
            self.pitch_buffer.append(p)
            self.yaw_buffer.append(y)
            # Compute resulting angle magnitude in deg (roll/pitch/yaw treated as vector)
            self.resulting_angle_buffer.append(float(np.sqrt(r*r + p*p + y*y)))
        self._pending_data.clear()
        # Enforce max buffer size
        if len(self.time_buffer) > self.max_points:
            start = len(self.time_buffer) - self.max_points
            self.time_buffer = self.time_buffer[start:]
            self.roll_buffer = self.roll_buffer[start:]
            self.pitch_buffer = self.pitch_buffer[start:]
            self.yaw_buffer = self.yaw_buffer[start:]
            self.resulting_angle_buffer = self.resulting_angle_buffer[start:]
        self._apply_data()

    def _apply_data(self) -> None:
        if not self.time_buffer:
            for curve in (self.angle_curve_roll, self.angle_curve_pitch, self.angle_curve_yaw, self.resulting_angle_curve):
                curve.setData([], [])
            return
        t = np.asarray(self.time_buffer, dtype=float)
        self.angle_curve_roll.setData(t, np.asarray(self.roll_buffer, dtype=float))
        self.angle_curve_pitch.setData(t, np.asarray(self.pitch_buffer, dtype=float))
        self.angle_curve_yaw.setData(t, np.asarray(self.yaw_buffer, dtype=float))
        self.resulting_angle_curve.setData(t, np.asarray(self.resulting_angle_buffer, dtype=float))

