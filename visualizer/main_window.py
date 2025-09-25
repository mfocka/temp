from typing import Optional
from PySide6 import QtWidgets

from .tabs.filter_output_tab import FilterOutputTab
from .tabs.events_tab import EventsTab


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MEMS Visualizer")
        self.resize(1400, 900)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self.filter_output_tab = FilterOutputTab()
        self.events_tab = EventsTab()

        self.tabs.addTab(self.filter_output_tab, "Filter Output")
        self.tabs.addTab(self.events_tab, "Events")

