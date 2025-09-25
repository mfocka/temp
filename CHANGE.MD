MEMS Visualizer Tooling Updates
===============================

- Added fast GUI visualizer using PySide6 + pyqtgraph (`visualizer/`).
- Filter Output now includes a new graph for resulting angle magnitude.
- New Events tab shows raised/cleared state table and a timeline plot.
- Implemented robust streaming log parser to fix partial-line issues and missing data.
- Removed console printing from ingest/parsing; UI updates are batched (~30 FPS) for speed.
- Added `requirements.txt` and `visualizer/README.md` for setup and usage.

