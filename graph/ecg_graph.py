from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from biosppy.signals import ecg as ecgprocessing
from scipy.io import loadmat
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd

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
    plt.close()

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
    plt.close()

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
    plt.close()

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
    plt.close()

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
    plt.close(fig)

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
    plt.close()

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
    plt.close(fig)

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

def plot_ecg_2d_waterfall(
    templates_ts,
    templates,
    beat_indices=None,
    offset=0.6,
    save_path=None,
    show_dpi=300,
    save_dpi=600,
    show_rpeak=True
):
    """
    2D waterfall plot for beat-to-beat variation

    templates_ts : 1D array
        relative time axis (e.g. -0.2 ~ 0.4 sec)

    templates : 2D array (n_beats, n_samples)
        extracted beats aligned at R peak

    offset : float
        vertical spacing between beats
    """
    n_beats = len(templates)
    colors = cm.viridis(np.linspace(0, 1, n_beats))
    if beat_indices is None:
        beat_indices = np.arange(n_beats)
    else:
        beat_indices = np.asarray(beat_indices)
        if len(beat_indices) != n_beats:
            raise ValueError("len(beat_indices) must match len(templates)")

    fig = plt.figure(figsize=(6, 4), dpi=show_dpi)
    ax = fig.add_subplot(111)

    for i, beat in enumerate(templates):

        ax.plot(
            templates_ts,
            beat + i * offset,
            color=colors[i],
            linewidth=1.2
        )

    # R peak reference line
    if show_rpeak:
        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
            color="black"
        )

    ax.set_xlabel("Time relative to R-peak (s)")
    ax.set_ylabel("Beat index (offset)")

    tick_step = max(1, n_beats // 5)
    tick_idx = np.arange(0, n_beats, tick_step)
    ax.set_yticks(tick_idx * offset)
    ax.set_yticklabels(beat_indices[tick_idx])

    # ax.set_title("Beat-to-beat ECG variation (waterfall view)")
    ax.grid(alpha=0.3)

    plt.tight_layout()


    if save_path:
        plt.savefig(
            save_path,
            dpi=save_dpi,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()

def plot_ecg_3d_waterfall(
    templates_ts,
    templates,
    beat_indices=None,
    save_path=None,
    show_dpi=300,
    save_dpi=600,
    elev=25,
    azim=-60,
    linewidth=1.2
):
    """
    3D waterfall plot for ECG beat-to-beat variation

    parameters
    ----------
    templates_ts : 1D array
        relative time axis (e.g. -0.2 ~ 0.4 sec)

    templates : 2D array (n_beats, n_samples)
        extracted ECG beats aligned at R peak

    elev : int
        vertical viewing angle

    azim : int
        horizontal rotation angle
    """
    fig = plt.figure(figsize=(8, 6), dpi=show_dpi)
    ax = fig.add_subplot(111, projection="3d")

    n_beats = len(templates)
    if beat_indices is None:
        beat_indices = np.arange(n_beats)
    else:
        beat_indices = np.asarray(beat_indices)

    colors = cm.viridis(np.linspace(0, 1, n_beats))
    for i, beat in enumerate(templates):
        ax.plot(
            templates_ts,
            np.full(len(templates_ts), i),
            beat,
            color=colors[i],
            linewidth=linewidth
        )

    ax.set_xlabel("Time relative to R-peak (s)", labelpad=5)
    ax.set_ylabel("Beat index", labelpad=5)
    ax.set_zlabel("Amplitude (mV)", labelpad=5)

    tick_step = max(1, n_beats // 5)
    tick_idx = np.arange(0, n_beats, tick_step)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels(beat_indices[tick_idx])

    ax.view_init(elev=elev, azim=azim)

    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)

    if save_path:
        plt.savefig(
            save_path,
            dpi=save_dpi,
            bbox_inches="tight"
        )

    plt.show()

    plt.close(fig)

def plot_ecg_surface_plotly(templates_ts, templates, save_path=None, show=True):

    X, Y = np.meshgrid(
        templates_ts,
        np.arange(len(templates))
    )

    Z = templates

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale="Viridis"
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Time relative to R peak (s)",
            yaxis_title="Beat index",
            zaxis_title="Amplitude (mV)"
        )
    )

    if save_path:
        fig.write_html(str(save_path))

    if show:
        fig.show()

def plot_ecg_surface_plotly_2(
    templates_ts,
    templates,
    beat_indices=None,
    max_beats=25,
    save_html=None,
    save_png=None
):

    # 避免 surface 太密
    if len(templates) > max_beats:

        idx = np.linspace(
            0,
            len(templates)-1,
            max_beats,
            dtype=int
        )

        templates = templates[idx]

        if beat_indices is not None:
            beat_indices = beat_indices[idx]
        else:
            beat_indices = idx

    else:

        if beat_indices is None:
            beat_indices = np.arange(len(templates))


    X, Y = np.meshgrid(
        templates_ts,
        beat_indices
    )

    Z = templates


    fig = go.Figure(
        data=[
            go.Surface(

                x=X,
                y=Y,
                z=Z,

                colorscale="Viridis",

                showscale=True,

                colorbar=dict(
                    title="Amplitude (mV)"
                ),

                contours = {
                    "z": dict(
                        show=True,
                        usecolormap=True,
                        highlightwidth=1
                    )
                }
            )
        ]
    )


    fig.update_layout(

        width=800,
        height=500,

        scene=dict(

            xaxis_title="Time relative to R-peak (s)",

            yaxis_title="Beat index (temporal order)",

            zaxis_title="Amplitude (mV)",

            xaxis=dict(
                backgroundcolor="white"
            ),

            yaxis=dict(
                backgroundcolor="white"
            ),

            zaxis=dict(
                backgroundcolor="white"
            ),

            aspectratio=dict(
                x=1.6,
                y=1,
                z=0.7
            ),

            camera=dict(
                eye=dict(
                    x=1.6,
                    y=1.4,
                    z=0.8
                )
            )
        ),

        font=dict(
            size=14
        ),

        margin=dict(
            l=0,
            r=0,
            b=0,
            t=30
        )
    )


    if save_html:

        fig.write_html(
            save_html,
            include_plotlyjs="cdn"
        )


    if save_png:

        fig.write_image(
            save_png,
            scale=3
        )


    return fig

def plot_ecg_ridgeplot(
    templates_ts,
    templates,
    beat_indices=None,
    max_beats=20,
    save_path=None,
    dpi=600
):
    if len(templates) > max_beats:
        idx = np.linspace(0, len(templates) - 1, max_beats, dtype=int)
        templates = templates[idx]
        beat_indices = idx if beat_indices is None else beat_indices[idx]
    else:
        if beat_indices is None:
            beat_indices = np.arange(len(templates))

    beat_labels = [str(int(b)) for b in beat_indices]

    df = pd.concat(
        [
            pd.DataFrame({"time": templates_ts, "amplitude": beat, "beat": label})
            for beat, label in zip(templates, beat_labels)
        ],
        ignore_index=True
    )

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    palette = sns.cubehelix_palette(len(beat_labels), rot=-.25, light=.7)

    g = sns.FacetGrid(
        df,
        row="beat",
        row_order=beat_labels,
        hue="beat",
        hue_order=beat_labels,
        aspect=6,
        height=0.5,
        palette=palette
    )

    def fill_ecg(time, amplitude, color, **kwargs):
        plt.gca().fill_between(time, amplitude, alpha=1, color=color)

    g.map(fill_ecg, "time", "amplitude")
    g.map(sns.lineplot, "time", "amplitude", color="w", linewidth=2, clip_on=False)
    g.map(sns.lineplot, "time", "amplitude", linewidth=1.2, clip_on=False)

    def label_beat(time, amplitude, color, label):
        plt.gca().text(
            0.02, 0.2, label,
            fontweight="bold", color=color,
            ha="left", va="center",
            transform=plt.gca().transAxes
        )

    g.map(label_beat, "time", "amplitude")

    for ax in g.axes.flat:
        ax.axvline(0, linestyle="--", linewidth=0.8, color="gray", clip_on=False)

    g.figure.subplots_adjust(hspace=-0.5)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    for ax in g.axes.flat[:-1]:
        ax.set_xlabel("")
    g.axes.flat[-1].set_xlabel("Time relative to R-peak (s)")

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close()

if __name__ == "__main__":

    plot_ecg(
        ecg=ecg,
        ts=ts,
        filt_ecg=filt_ecg,
        rpeaks=rpeaks,
        dot=False,
        save=DEFAULT_OUTPUT / "ecg_plot.png",
        show=True
    )
    
    plot_ecg_2d_waterfall(
        templates_ts,
        templates[:15],
        beat_indices=np.arange(15),
        offset=0.6,
        save_path=DEFAULT_OUTPUT / "ecg_waterfall_2d_15.png"
    )

    
    plot_ecg_3d_waterfall(
        templates_ts,
        templates[:15],
        beat_indices=np.arange(15),
        save_path=DEFAULT_OUTPUT / "ecg_waterfall_3d_15.png",
        elev=30,
        azim=-60,
        linewidth=1.0
    )


    plot_ecg_surface_plotly(
        templates_ts,
        templates,
        save_path=DEFAULT_OUTPUT / "ecg_surface.html"
    )

