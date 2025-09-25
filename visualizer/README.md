MEMS Visualizer
================

Overview
--------
This GUI visualizer renders real-time sensor data with high-throughput plotting, shows a resulting angle graph in the Filter Output view, provides an Events tab to track raised/cleared states, and includes a robust log loader that correctly parses partial chunks without printing to console.

Install
-------
1. Create a virtual environment and install dependencies:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run
---
```
python -m visualizer.app
```

Usage
-----
- Filter Output: displays roll, pitch, yaw, and the resulting angle magnitude.
- Events: table and timeline of raised/cleared events.

Logs
----
Use `visualizer.data_ingest.ingest_file(path, callbacks)` to populate the UI from a recorded log in the expected line format:

```
[10420.300s] ANGLES,2404433308,0.443,0.029,2.066,MONITORING
[10421.750s] RAW_DATA,2405971708,-110.000,960.000,-147.000,0.419,-0.629,-0.629
```

The parser tolerates partial chunks like `b"[1042\n"` and resumes when the next bytes arrive.

