import sys
import os
from PySide6 import QtWidgets
from .main_window import MainWindow
from .wire_up import connect_log_to_ui


def run_app() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # Optional: pass a log file path as first CLI argument
    if len(sys.argv) > 1:
        candidate_path = sys.argv[1]
        if os.path.isfile(candidate_path):
            connect_log_to_ui(window, candidate_path)
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()

