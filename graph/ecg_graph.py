from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from biosppy.signals import ecg as ecgprocessing
from scipy.io import loadmat


DEFAULT_INPUT = Path("data/training2017/A00001.mat")
DEFAULT_OUTPUT = Path("graph/output")
DEFAULT_FS = 300
DEFAULT_SCALE = 1000.0


ecg = loadmat(DEFAULT_INPUT)["val"][0] / DEFAULT_SCALE
# ecg = loadmat(DEFAULT_INPUT)["val"][0] 
ts, filt_ecg, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecgprocessing.ecg(ecg, DEFAULT_FS, show=False, interactive=False)
print(f"R-peaks: {rpeaks}")

def plot_ecg():
    
    plt.figure(figsize=(8, 4), dpi=300)

    ymin = np.min(filt_ecg)
    ymax = np.max(filt_ecg)

    
    plt.vlines(
        ts[rpeaks],
        ymin=ymin,
        ymax=ymax,
        color="m",
        linewidth=0.8,
        label="R-peaks",
        zorder=1
    )
    plt.plot(ts, filt_ecg, label="Filtered ECG", linewidth=1.0, zorder=2)

    # plt.scatter(ts[rpeaks], filt_ecg[rpeaks], color="red", label="R-peaks", s=10)
    plt.title("ECG Signal with R-peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.legend(loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(DEFAULT_OUTPUT / "ecg_plot.png", dpi=600, bbox_inches="tight")
    plt.show()

def plot_heartbeat():
    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(templates_ts, templates.T, color="blue", alpha=0.25, label="Heartbeats")
    median_template = np.median(templates, axis=0)
    plt.plot(templates_ts, median_template, color="black", linewidth=2.0, label="Median Heartbeat")
    plt.title("Extracted Heartbeats")
    # plt.xlabel("Time (s)")
    plt.xlabel("Time relative to R-peak (s)")
    plt.ylabel("Amplitude (mV)")
    # plt.legend(loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(DEFAULT_OUTPUT / "heartbeat_plot.png", dpi=600, bbox_inches="tight")
    plt.show()
    

if __name__ == "__main__":
    # plot_ecg()
    plot_heartbeat()