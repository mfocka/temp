from typing import List, Optional, Tuple
from PySide6 import QtWidgets
import pyqtgraph as pg


class EventsTab(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(layout)

        # Table for raised/cleared events
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Time (s)", "Event", "Action", "Details"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        # Timeline plot for visual tracking
        self.timeline = pg.PlotWidget(title="Events Timeline")
        self.timeline.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.timeline, 1)

        self._events: List[Tuple[float, str, str, str]] = []  # (t, name, action, details)

    def add_event(self, t_sec: float, name: str, action: str, details: str = "") -> None:
        self._events.append((t_sec, name, action, details))
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{t_sec:.3f}"))
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(name))
        self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(action))
        self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(details))
        self.table.scrollToBottom()

        # Update timeline: encode action as y value (raised=1, cleared=0)
        xs = [e[0] for e in self._events]
        ys = [1.0 if e[2].lower().startswith("raise") else 0.0 for e in self._events]
        self.timeline.clear()
        self.timeline.plot(xs, ys, pen=None, symbol="o", symbolSize=6, symbolBrush="#f4c430")

