from __future__ import annotations
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

# ts : Signal time axis reference (seconds).
# filtered : Filtered ECG signal.
# rpeaks : R-peak location indices.
# templates_ts : Templates time axis reference (seconds).
# templates : Extracted heartbeat templates.
# heart_rate_ts : Heart rate time axis reference (seconds).
# heart_rate : Instantaneous heart rate (bpm).
ts, filt_ecg, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecgprocessing.ecg(ecg, DEFAULT_FS, show=False, interactive=False)

def plot_ecg(ecg=ecg, ts=ts, filt_ecg=filt_ecg, rpeaks=rpeaks, dot=False, save=DEFAULT_OUTPUT / "ecg_plot.png", show=False):
    
    plt.figure(figsize=(8, 4), dpi=300)

    ymin = np.min(filt_ecg)
    ymax = np.max(filt_ecg)

    # r-peaks vertical lines
    plt.plot(ts, filt_ecg, label="Filtered ECG", linewidth=1.0, zorder=2)

    if dot:
        plt.scatter(ts[rpeaks], filt_ecg[rpeaks], color="red", label="R-peaks", s=10)
    else:
        plt.vlines(
            ts[rpeaks],
            ymin=ymin,
            ymax=ymax,
            color="m",
            linewidth=0.8,
            label="R-peaks",
            zorder=1
        )

    plt.title("ECG Signal with R-peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.legend(loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(save, dpi=600, bbox_inches="tight")
    if show:
        plt.show()

def plot_heartbeat(templates_ts=templates_ts, templates=templates, save=DEFAULT_OUTPUT / "heartbeat_plot.png", show=False):
    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(templates_ts, templates.T, color="blue", alpha=0.3, label="Heartbeats")
    median_template = np.median(templates, axis=0)
    plt.plot(templates_ts, median_template, color="black", linewidth=2.0, label="Median Heartbeat")
    plt.title("Extracted Heartbeats")
    # plt.xlabel("Time (s)")
    plt.xlabel("Time relative to R-peak (s)")
    plt.ylabel("Amplitude (mV)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(save, dpi=600, bbox_inches="tight")
    if show:
        plt.show()
    

if __name__ == "__main__":
    plot_ecg(save=DEFAULT_OUTPUT / "ecg_plot.png", show=True)
    plot_heartbeat(save=DEFAULT_OUTPUT / "heartbeat_plot.png", show=True)