2025-09-25 visualizer fixes (existing Tk-based app)
- Added resulting angle to Filter Outputs page (computed from Fused pitch/yaw/roll).
- Quieted console prints in `serial_interface.py` and `main.py` (no stdout spam).
- Improved logger file handle tracking to avoid header duplication.
- Documented Events tab derived from ANGLES state transitions.
MEMS Visualizer Tooling Updates
===============================

- Added fast GUI visualizer using PySide6 + pyqtgraph (`visualizer/`).
- Filter Output now includes a new graph for resulting angle magnitude.
- New Events tab shows raised/cleared state table and a timeline plot.
- Implemented robust streaming log parser to fix partial-line issues and missing data.
- Removed console printing from ingest/parsing; UI updates are batched (~30 FPS) for speed.
- Added `requirements.txt` and `visualizer/README.md` for setup and usage.

