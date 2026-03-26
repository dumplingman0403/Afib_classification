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

    # plt.title("ECG Signal with R-peaks")
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
    plt.plot(templates_ts, templates.T, color="blue", alpha=0.3)
    median_template = np.median(templates, axis=0)
    plt.plot(templates_ts, median_template, color="black", linewidth=2.0, label="Median Heartbeat")
    # plt.title("Extracted Heartbeats")
    # plt.xlabel("Time (s)")
    plt.xlabel("Time relative to R-peak (s)")
    plt.ylabel("Amplitude (mV)")
    plt.legend(loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(save, dpi=600, bbox_inches="tight")
    if show:
        plt.show()

def plot_heartbeat_offset():
    plt.figure(figsize=(8, 6), dpi=300)
    n_beats = templates.shape[0]

    amp_range = np.max(templates) - np.min(templates)
    offset_step = amp_range * 1.2

    for i in range(n_beats):
        offset = i * offset_step
        plt.plot(
            templates_ts,
            templates[i] + offset,
            color="blue",
            linewidth=1.0,
            alpha=0.9,
        )

    plt.xlabel("Time relative to R-peak (s)")
    plt.ylabel("Amplitude + beat offset (mV)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(DEFAULT_OUTPUT / "heartbeat_plot_v2.png", dpi=600, bbox_inches="tight")
    plt.show()

def plot_heartbeat_sequential_individual():
    plt.figure(figsize=(10, 4), dpi=300)
    n_beats, n_samples = templates.shape
    beat_duration = templates_ts[-1] - templates_ts[0]
    gap = 0.05
    

    for i in range(n_beats):
        x_offset = i * (beat_duration + gap)
        x = templates_ts - templates_ts[0] + x_offset
        plt.plot(
            x,
            templates[i],
            color="blue",
            linewidth=1.0,
            alpha=0.9,
        )

    plt.xlabel("Sequential heartbeat segments")
    plt.ylabel("Amplitude (mV)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(DEFAULT_OUTPUT / "heartbeat_plot_v3.png", dpi=600, bbox_inches="tight")
    plt.show()

def plot_ecg_cells(ts, filtered, rpeaks, fs,
                   before_sec=0.2,
                   after_sec=0.4):

    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    # ECG waveform
    ax.plot(ts, filtered, color="C0", linewidth=1.0, label="Filtered ECG")
    ymin = np.min(filtered)
    ymax = np.max(filtered)

    ax.vlines(
        ts[rpeaks],
        ymin,
        ymax,
        color="m",
        linewidth=0.8,
        alpha=0.7,
        label="R-peaks"
    )

    # window 
    before = int(before_sec * fs)
    after = int(after_sec * fs)

    for r in rpeaks:

        start_idx = r - before
        end_idx = r + after

        if start_idx < 0 or end_idx >= len(ts):
            continue

        start_t = ts[start_idx]
        end_t = ts[end_idx]


        ax.axvspan(
            start_t,
            end_t,
            color="gray",
            alpha=0.15
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.grid()

    ax.legend(loc="upper left")

    plt.tight_layout()

    plt.savefig(
        DEFAULT_OUTPUT / "ecg_cells.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()

def plot_median_heartbeat(templates_ts, templates):
    plt.figure(figsize=(6, 3), dpi=300)
    median_beat = np.median(templates, axis=0)

    plt.plot(
        templates_ts,
        median_beat,
        color="blue",
        linewidth=1.5
    )

    # mark r-peak
    plt.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.2
    )

    plt.xlabel("Time relative to R-peak (s)")
    plt.ylabel("Amplitude (mV)")
    plt.grid()

    plt.tight_layout()

    plt.savefig(
        DEFAULT_OUTPUT / "median_heartbeat.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()


def plot_single_heartbeat(
    templates_ts,
    templates,
    beat_index=0,
    show=True,
    save_path=None,
    x_axis=None,
    xlabel=None,
    rpeak_x=None,
    rl=None,
):
    templates_ts = np.asarray(templates_ts).reshape(-1)
    templates = np.asarray(templates)

    n_ts = len(templates_ts)

    if templates.ndim == 1:
        if len(templates) != n_ts:
            raise ValueError(
                f"templates length {len(templates)} does not match templates_ts length {n_ts}"
            )
        beat = templates

    elif templates.ndim == 2:
        if templates.shape[1] == n_ts:
            beat = templates[beat_index]
        elif templates.shape[0] == n_ts:
            beat = templates[:, beat_index]
        else:
            raise ValueError(
                f"templates shape {templates.shape} does not match templates_ts length {n_ts}"
            )
    else:
        raise ValueError("templates must be 1D or 2D")

    if x_axis is None:
        x_axis = templates_ts
    else:
        x_axis = np.asarray(x_axis).reshape(-1)
        if len(x_axis) != len(beat):
            raise ValueError(
                f"x_axis length {len(x_axis)} does not match beat length {len(beat)}"
            )

    if xlabel is None:
        xlabel = "Time relative to R-peak (s)"

    plt.figure(figsize=(6, 3), dpi=300)
    plt.plot(x_axis, beat, color="blue", linewidth=1.5)

    if rpeak_x is None:
        rpeak_x = 0.0

    if rl is not None:
        plt.axvline(
            rpeak_x,
            color="black",
            linestyle="--",
            linewidth=1.2
        )

    plt.xlabel(xlabel)
    plt.ylabel("Amplitude (mV)")
    plt.grid()
    plt.tight_layout()

    if save_path is None:
        save_path = DEFAULT_OUTPUT / "single_heartbeat.png"
    else:
        save_path = Path(save_path)
        if not save_path.parent.exists():
            raise FileNotFoundError(f"Parent directory does not exist: {save_path.parent}")

    plt.savefig(save_path, dpi=600, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()

def plot_nine_baselines():
    n_show = min(9, templates.shape[0])
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), dpi=300, sharex=True, sharey=True)
    axes = axes.flatten()
    

    for i in range(9):
        ax = axes[i]

        if i < n_show:
            beat = templates[i]
            ax.plot(templates_ts, beat, color="blue", linewidth=1.2)
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.grid()
            ax.set_title(f"Beat {i+1}", fontsize=10)
        else:
            ax.axis("off")

    for ax in axes[6:9]:
        ax.set_xlabel("Time relative to R-peak (s)")
    for ax in axes[0::3]:
        ax.set_ylabel("Amplitude (mV)")

    plt.tight_layout()
    plt.savefig(DEFAULT_OUTPUT / "first_nine_baselines_grid.png", dpi=600, bbox_inches="tight")
    plt.show()


def save_extracted_heartbeats(
    templates_ts,
    templates,
    rpeaks,
    fs,
    save_dir,
    version="1",
    plot_func=None,
    rl=True,
):
    if plot_func is None:
        raise ValueError("plot_func must be provided")

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    templates_ts = np.asarray(templates_ts).reshape(-1)
    templates = np.asarray(templates)
    rpeaks = np.asarray(rpeaks).reshape(-1)

    n_ts = len(templates_ts)

    if templates.ndim != 2:
        raise ValueError("templates must be 2D")

    # normalize to (n_beats, n_samples)
    if templates.shape[1] == n_ts:
        templates_2d = templates
    elif templates.shape[0] == n_ts:
        templates_2d = templates.T
    else:
        raise ValueError(
            f"templates shape {templates.shape} does not match templates_ts length {n_ts}"
        )

    n_beats = templates_2d.shape[0]

    if len(rpeaks) != n_beats:
        raise ValueError(
            f"len(rpeaks)={len(rpeaks)} does not match number of beats={n_beats}"
        )

    saved_paths = []

    rel_start = float(templates_ts[0])
    rel_end = float(templates_ts[-1])

    for i in range(n_beats):
        rpeak = rpeaks[i]
        rpeak_time = rpeak / fs

        real_x = templates_ts + rpeak_time
        start_time = real_x[0]
        end_time = real_x[-1]

        if version == "1":
            filename = (
                f"heartbeat_{i+1:03d}"
                f"_start_{start_time:.3f}"
                f"_end_{end_time:.3f}.png"
            )
            x_axis = real_x
            xlabel = "Time (s)"
            rpeak_x = rpeak_time

        elif version == "2":
            filename = f"heartbeat_{i+1:03d}.png"
            x_axis = templates_ts
            xlabel = "Time relative to R-peak (s)"
            rpeak_x = 0.0

        else:
            raise ValueError("version must be '1' or '2'")

        save_path = save_dir / filename

        plot_func(
            templates_ts=templates_ts,
            templates=templates_2d,
            beat_index=i,
            show=False,
            save_path=save_path,
            x_axis=x_axis,
            xlabel=xlabel,
            rpeak_x=rpeak_x,
            rl=rl
        )

        saved_paths.append(save_path)

    print(f"Saved {len(saved_paths)} heartbeat images to {save_dir}")
    return saved_paths


if __name__ == "__main__":
    
    
    plot_ecg(save=DEFAULT_OUTPUT / "ecg_plot.png", show=True)
    

    # generate version 1 with real time axis
    save_extracted_heartbeats(
        templates_ts=templates_ts,
        templates=templates,
        rpeaks=rpeaks,
        fs=DEFAULT_FS,
        save_dir=DEFAULT_OUTPUT / "heartbeats_v1",
        version="1",
        plot_func=plot_single_heartbeat,
        rl=True
    )

    # generate version 2 with relative time axis and no time info in filename
    save_extracted_heartbeats(
        templates_ts=templates_ts,
        templates=templates,
        rpeaks=rpeaks,
        fs=DEFAULT_FS,
        save_dir=DEFAULT_OUTPUT / "heartbeats_v2",
        version="2",
        plot_func=plot_single_heartbeat,
        rl=True
    )