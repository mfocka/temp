from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


DI_COMPLETE_RE = re.compile(r"MotionDI calibration complete", re.IGNORECASE)
ME_COMPLETE_RE = re.compile(r"MotionEstimator\s+calibration complete", re.IGNORECASE)
GYRO_RE = re.compile(r"Final gyro bias:\s*\[([^\]]+)\]\s*dps", re.IGNORECASE)
ACC_RE = re.compile(r"Final acc bias:\s*\[([^\]]+)\]\s*mg", re.IGNORECASE)


def _parse_vector(text: str) -> Optional[List[float]]:
    try:
        nums = [float(x.strip()) for x in text.split(",")]
        if len(nums) != 3:
            return None
        return nums
    except Exception:
        return None


def parse_log(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    data = {
        "gyro": {"MotionDI": [], "MotionEstimator": []},
        "acc": {"MotionEstimator": []},
    }

    last_ctx: Optional[str] = None  # 'MotionDI' or 'MotionEstimator'

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if DI_COMPLETE_RE.search(line):
                last_ctx = "MotionDI"
                continue
            if ME_COMPLETE_RE.search(line):
                last_ctx = "MotionEstimator"
                continue

            gmatch = GYRO_RE.search(line)
            if gmatch:
                vec = _parse_vector(gmatch.group(1))
                if vec is not None:
                    ctx = last_ctx or "MotionEstimator"  # default to ME if unknown
                    if ctx not in data["gyro"]:
                        data["gyro"][ctx] = []
                    data["gyro"][ctx].append(vec)
                continue

            amatch = ACC_RE.search(line)
            if amatch:
                vec = _parse_vector(amatch.group(1))
                if vec is not None:
                    # Acc bias only expected after MotionEstimator complete
                    data["acc"]["MotionEstimator"].append(vec)
                continue

    # Convert lists to numpy arrays
    for ctx in list(data["gyro"].keys()):
        arr = np.array(data["gyro"][ctx], dtype=float) if data["gyro"][ctx] else np.empty((0, 3))
        data["gyro"][ctx] = arr

    arr = np.array(data["acc"]["MotionEstimator"], dtype=float) if data["acc"]["MotionEstimator"] else np.empty((0, 3))
    data["acc"]["MotionEstimator"] = arr

    return data


def compute_stats(arr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute per-axis stats for an (N,3) array.
    Returns dict with keys: count, mean, std, min, max
    """
    if arr.size == 0:
        return {
            "count": np.array([0, 0, 0]),
            "mean": np.array([np.nan, np.nan, np.nan]),
            "std": np.array([np.nan, np.nan, np.nan]),
            "min": np.array([np.nan, np.nan, np.nan]),
            "max": np.array([np.nan, np.nan, np.nan]),
        }

    return {
        "count": np.array([arr.shape[0]] * 3),
        "mean": np.nanmean(arr, axis=0),
        "std": np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(3),
        "min": np.nanmin(arr, axis=0),
        "max": np.nanmax(arr, axis=0),
    }


def print_stats(name: str, arr: np.ndarray, unit: str) -> None:
    s = compute_stats(arr)
    print(f"{name} ({unit}):")
    print(f"  count: {int(s['count'][0])}")
    print(
        "  mean: [{:.3f}, {:.3f}, {:.3f}]".format(*s["mean"])
    )
    print(
        "  std:  [{:.3f}, {:.3f}, {:.3f}]".format(*s["std"])
    )
    print(
        "  min:  [{:.3f}, {:.3f}, {:.3f}]".format(*s["min"])
    )
    print(
        "  max:  [{:.3f}, {:.3f}, {:.3f}]".format(*s["max"])
    )
    # Also print mean vector magnitude for a quick sanity check
    if arr.size > 0:
        mags = np.linalg.norm(arr, axis=1)
        print("  |bias| mean: {:.3f} {}".format(np.mean(mags), unit))
    print("")


def plot_biases(data: Dict[str, Dict[str, np.ndarray]], save: Optional[str], show: bool) -> None:
    # Gyro plot: compare MotionDI vs MotionEstimator
    gy_di = data["gyro"].get("MotionDI", np.empty((0, 3)))
    gy_me = data["gyro"].get("MotionEstimator", np.empty((0, 3)))

    fig1, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    axes_names = ["X", "Y", "Z"]
    for i, ax in enumerate(axes):
        if gy_di.size > 0:
            ax.plot(range(len(gy_di)), gy_di[:, i], "o", label="MotionDI", alpha=0.8)
        if gy_me.size > 0:
            ax.plot(range(len(gy_me)), gy_me[:, i], "s", label="MotionEstimator", alpha=0.8)
        ax.set_title(f"Gyro bias {axes_names[i]} (dps)")
        ax.set_xlabel("Run index")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Bias (dps)")
    axes[0].legend()

    # Acc plot: MotionEstimator only
    acc_me = data["acc"].get("MotionEstimator", np.empty((0, 3)))
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for i, ax in enumerate(axes2):
        if acc_me.size > 0:
            ax.plot(range(len(acc_me)), acc_me[:, i], "o", label="Acc (ME)", alpha=0.8)
        ax.set_title(f"Acc bias {axes_names[i]} (mg)")
        ax.set_xlabel("Run index")
        ax.grid(True, alpha=0.3)
    axes2[0].set_ylabel("Bias (mg)")
    if acc_me.size > 0:
        axes2[0].legend()

    if save:
        # If a single filename provided, append suffixes for multiple figs
        base = Path(save)
        if base.suffix:
            p1 = base.with_name(f"{base.stem}_gyro{base.suffix}")
            p2 = base.with_name(f"{base.stem}_acc{base.suffix}")
        else:
            p1 = base.with_suffix(".gyro.png")
            p2 = base.with_suffix(".acc.png")
        fig1.savefig(p1, dpi=150)
        fig2.savefig(p2, dpi=150)
        print(f"Saved plots to: {p1} and {p2}")

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and analyze ISM330DHCX bias data from logs.")
    parser.add_argument("logfile", type=str, help="Path to log file")
    parser.add_argument("--save", type=str, default=None, help="Path prefix or filename to save plots")
    parser.add_argument("--no-show", action="store_true", help="Do not display plots interactively")
    args = parser.parse_args()

    path = Path(args.logfile)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    data = parse_log(path)

    # Print stats
    print_stats("Gyro bias (MotionDI)", data["gyro"]["MotionDI"], "dps")
    print_stats("Gyro bias (MotionEstimator)", data["gyro"]["MotionEstimator"], "dps")
    print_stats("Acc bias (MotionEstimator)", data["acc"]["MotionEstimator"], "mg")

    plot_biases(data, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
