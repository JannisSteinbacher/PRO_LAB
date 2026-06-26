#!/usr/bin/env python3
"""Plot Euclidean distance error over time for KF, EKF, PF on one run."""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

GT_DIR = "ground truth"
RUN = "20260626_161754"  # which run/timestamp to plot

COLORS = {"kf": "red", "ekf": "green", "pf": "blue"}
LABELS = {"kf": "KF", "ekf": "EKF", "pf": "PF"}


def load(path):
    t, est_x, est_y, gt_x, gt_y = [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t.append(float(row["time_s"]))
            est_x.append(float(row["est_x"]))
            est_y.append(float(row["est_y"]))
            gt_x.append(float(row["gt_x"]))
            gt_y.append(float(row["gt_y"]))
    t = np.array(t)
    err = np.hypot(np.array(est_x) - np.array(gt_x), np.array(est_y) - np.array(gt_y))
    return t, err


fig, ax = plt.subplots(figsize=(11, 6))
for filt in ("kf", "ekf", "pf"):
    path = os.path.join(GT_DIR, f"{filt}_eval_{RUN}.csv")
    t, err = load(path)
    ax.plot(t, err, color=COLORS[filt], label=f"{LABELS[filt]} (mean {err.mean():.3f} m)", lw=1.4)

ax.set_yscale("log")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Euclidean position error [m] (log scale)")
ax.set_title(f"Euclidean Distance Error over Time — run {RUN}")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()

out = os.path.join(GT_DIR, f"euclidean_error_{RUN}.png")
fig.savefig(out, dpi=150)
print("saved", out)
