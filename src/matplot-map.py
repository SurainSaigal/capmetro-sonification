import argparse
import json
import os
import subprocess
import tempfile

import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from pyproj import Transformer
from scipy.io import wavfile

# WGS-84 → Web Mercator (EPSG:3857)
_to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

_base_map = ctx.providers.CartoDB.PositronNoLabels

# Fixed map view: Austin city center
_AUSTIN_X, _AUSTIN_Y = _to_mercator.transform(-97.7431, 30.2672)
_HALF_EXTENT = 15_000  # radius of map view from city center in meters


def load_data(date: str, bin_seconds: int = 15) -> pd.DataFrame:
    with open(f"../vehicle_positions_data/{date}/processed.json", "r") as f:
        raw_data = json.load(f)

    flat_data = []
    for key, route_info in raw_data.items():
        route_id = route_info.get("routeId")
        for trip in route_info.get("trips", []):
            trip_id = trip.get("tripId")
            vehicle_ids = trip.get("vehicleIds", [])
            vehicle_id = vehicle_ids[0] if vehicle_ids else "Unknown"
            for pos in trip.get("vehiclePositions", []):
                flat_data.append({
                    "route_id": route_id,
                    "trip_id": trip_id,
                    "vehicle_id": vehicle_id,
                    "latitude": pos.get("latitude"),
                    "longitude": pos.get("longitude"),
                    "speed": pos.get("speed"),
                    "bearing": pos.get("bearing"),
                    "raw_timestamp": pos.get("timestamp"),
                })

    df = pd.DataFrame(flat_data)
    df["timestamp"] = (
        pd.to_datetime(df["raw_timestamp"], unit="s")
        .dt.tz_localize("UTC")
        .dt.tz_convert("America/Chicago")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    df = df.sort_values(by="timestamp", ascending=True).reset_index(drop=True)

    # Reproject coordinates to Web Mercator for contextily
    df["x"], df["y"] = _to_mercator.transform(
        df["longitude"].values, df["latitude"].values
    )

    # Bin to configurable interval for animation frames
    freq = f"{bin_seconds}s"
    df["minute"] = (
        pd.to_datetime(df["timestamp"])
        .dt.floor(freq)
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    return df


# --- Sonification helpers ---
_C2_HZ = 65.41   # lowest pitch (fewest buses)
_C6_HZ = 1046.5  # highest pitch (peak buses)
_SAMPLE_RATE = 44100


def _freq_from_count(count: int, min_count: int, max_count: int) -> float:
    """Map active bus count to frequency on a log (octave) scale."""
    if max_count <= min_count:
        return _C2_HZ
    t = np.clip((count - min_count) / (max_count - min_count), 0.0, 1.0)
    return _C2_HZ * (2 ** (t * np.log2(_C6_HZ / _C2_HZ)))


def _build_freq_array(
    sonification: str,
    df: pd.DataFrame,
    frames: list,
    counts_series: pd.Series,
    global_counts_series: pd.Series | None = None,
) -> np.ndarray:
    """Build a per-frame frequency array for the given sonification mode.

    global_counts_series, if provided, is used to compute the min/max range so
    that frequency scaling is consistent across previews (--max-frames) and full runs.
    """
    if sonification == "buscount":
        scale_series = global_counts_series if global_counts_series is not None else counts_series
        min_buses = int(scale_series.min())
        max_buses = int(scale_series.max())
        print(
            f"Bus count range (scale): {min_buses} – {max_buses} active trips/frame")
        return np.array(
            [_freq_from_count(int(counts_series.get(f, min_buses)), min_buses, max_buses)
             for f in frames],
            dtype=np.float64,
        )
    raise ValueError(f"Unknown sonification mode: {sonification!r}")


def build_matplotlib_map(
    df: pd.DataFrame,
    interval_ms: int = 200,
    trail_minutes: int = 5,
    gradient: bool = False,
    sonify: bool = True,
    sonification: str = "buscount",
    prerender: bool = False,
    output_path: str | None = None,
    global_counts_series: pd.Series | None = None,
) -> None:
    # --- Color map: one colour per unique route_id ---
    route_ids = sorted(df["route_id"].dropna().unique())
    cmap = plt.get_cmap("tab20", len(route_ids))
    route_color = {rid: cmap(i) for i, rid in enumerate(route_ids)}

    # --- Bin timestamps by minute for smoother, manageable animation ---
    frames = sorted(df["minute"].unique())
    frame_index = {f: i for i, f in enumerate(frames)}

    # Pre-index rows by frame for fast lookup in _update()
    df_by_minute = {m: grp.reset_index(drop=True)
                    for m, grp in df.groupby("minute")}

    counts_series = df.groupby("minute")["trip_id"].nunique()
    freq_array = _build_freq_array(
        sonification, df, frames, counts_series, global_counts_series)
    current_frame_idx = np.zeros(1, dtype=np.int64)

    # Fixed bounding box centered on Austin
    x_min, x_max = _AUSTIN_X - _HALF_EXTENT, _AUSTIN_X + _HALF_EXTENT
    y_min, y_max = _AUSTIN_Y - _HALF_EXTENT, _AUSTIN_Y + _HALF_EXTENT

    # --- Figure size: fixed for prerender, screen-fitted for live ---
    _DPI = 150
    if prerender:
        _fig_size = 8.0
    else:
        try:
            import tkinter as _tk
            _root = _tk.Tk()
            _root.withdraw()
            _screen_h_in = _root.winfo_screenheight() / _DPI * 0.90
            _root.destroy()
        except Exception:
            _screen_h_in = 7.0
        _fig_size = max(4.0, _screen_h_in)

    # --- Figure / axes: square window matches the square extent ---
    fig, ax = plt.subplots(figsize=(_fig_size, _fig_size), dpi=_DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout(pad=0)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # CartoDB Positron: clean, minimal, white basemap
    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=_base_map,
            zoom=13,
        )
    except Exception:
        ctx.add_basemap(ax, crs="EPSG:3857", zoom=13)

    # --- Trail LineCollection (starts empty) ---
    lc = LineCollection([], linewidths=1.5, zorder=5)
    ax.add_collection(lc)

    # --- Timestamp HUD ---
    timestamp_text = ax.text(
        0.01, 0.97, "",
        transform=ax.transAxes,
        color="#111111",
        fontsize=11,
        fontfamily="monospace",
        va="top",
        zorder=6,
    )

    # --- Legend (route colours) ---
    # legend_handles = [
    #     Line2D([0], [0], marker="o", color="none",
    #            markerfacecolor=route_color[rid], markersize=7, label=str(rid))
    #     for rid in route_ids
    # ]
    # legend = ax.legend(
    #     handles=legend_handles,
    #     title="Route",
    #     loc="lower right",
    #     fontsize=5,
    #     title_fontsize=6,
    #     framealpha=0.7,
    #     facecolor="white",
    #     labelcolor="#111111",
    # )
    # legend.get_title().set_color("#111111")

    # --- Live audio stream (skipped in prerender mode) ---
    if sonify and not prerender:
        _phase = np.zeros(1, dtype=np.float64)

        def _audio_callback(outdata, frames_count, time_info, status):
            freq = freq_array[current_frame_idx[0]]
            phases = _phase[0] + 2 * np.pi * \
                np.arange(1, frames_count + 1) * freq / _SAMPLE_RATE
            _phase[0] = phases[-1] % (2 * np.pi)
            outdata[:] = (np.sin(phases) *
                          0.3).astype(np.float32).reshape(-1, 1)

        audio_stream = sd.OutputStream(
            samplerate=_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=4096,
            callback=_audio_callback,
        )
    else:
        audio_stream = None

    def _update(frame_minute: str):
        fi = frame_index[frame_minute]
        window = frames[max(0, fi - trail_minutes + 1): fi + 1]
        parts = [df_by_minute[f] for f in window if f in df_by_minute]
        trail_df = pd.concat(
            parts, ignore_index=True) if parts else df.iloc[:0]

        segments, seg_colors = [], []
        _MIN_ALPHA = 0.05

        for trip_id, grp in trail_df.groupby("trip_id", sort=False):
            grp = grp.sort_values("timestamp")
            xs, ys = grp["x"].values, grp["y"].values
            color = route_color.get(grp["route_id"].iloc[0], (1, 1, 1, 1))
            r, g, b = color[:3]

            n = len(xs)
            if n >= 2:
                if gradient:
                    for i in range(n - 1):
                        alpha = _MIN_ALPHA + \
                            (1.0 - _MIN_ALPHA) * (i / (n - 2) if n > 2 else 1.0)
                        segments.append(
                            [(xs[i], ys[i]), (xs[i + 1], ys[i + 1])])
                        seg_colors.append((r, g, b, alpha))
                else:
                    segments.append(list(zip(xs, ys)))
                    seg_colors.append(color)

        lc.set_segments(segments)
        lc.set_color(seg_colors)

        if sonify and not prerender:
            prev_fi = int(current_frame_idx[0])
            current_frame_idx[0] = fi
            if fi == 0 or freq_array[fi] != freq_array[prev_fi]:
                active_count = counts_series.get(frame_minute, 0)
                print(
                    f"[{frame_minute}] buses: {active_count:3d}  →  {freq_array[fi]:.1f} Hz"
                )

        timestamp_text.set_text(frame_minute)
        return lc, timestamp_text

    anim = FuncAnimation(
        fig,
        _update,
        frames=frames,
        interval=interval_ms,
        blit=True,
        repeat=not prerender,
    )

    plt.title("CapMetro Bus Positions — Austin TX", color="#111111", pad=8)

    if prerender:
        _render_to_file(
            anim=anim,
            freq_array=freq_array,
            interval_ms=interval_ms,
            output_stem=output_path,
            sonify=sonify,
            dpi=_DPI,
        )
    else:
        try:
            if audio_stream:
                audio_stream.start()
            plt.show(block=True)
        finally:
            if audio_stream:
                audio_stream.stop()
                audio_stream.close()

    _ = anim  # prevent GC before show() / save() returns


def _render_to_file(
    anim: FuncAnimation,
    freq_array: np.ndarray,
    interval_ms: int,
    output_stem: str,
    sonify: bool,
    dpi: int,
) -> None:
    """Render animation and synthesized audio, merging into a single MP4.

    Also saves the WAV separately alongside the MP4.
    Sync guarantee: both tracks are derived from the same freq_array and interval_ms.
    Video FPS = 1000 / interval_ms, so each frame = interval_ms ms exactly.
    Audio: frame i gets round(interval_ms * SAMPLE_RATE / 1000) samples at freq_array[i],
    with phase carried across boundaries — total durations are identical by construction.
    """
    fps = 1000.0 / interval_ms
    samples_per_frame = round(interval_ms * _SAMPLE_RATE / 1000)
    n_frames = len(freq_array)
    mp4_path = f"{output_stem}.mp4"

    print(f"Rendering video ({n_frames} frames at {fps:.1f} fps)…")
    _render_start = [0.0]

    def _progress(current, total):
        import time
        if current == 0:
            _render_start[0] = time.time()
            return
        elapsed = time.time() - _render_start[0]
        fps_actual = current / elapsed
        remaining = (total - current) / fps_actual if fps_actual > 0 else 0
        pct = current / total * 100
        print(
            f"  frame {current:>{len(str(total))}}/{total}  "
            f"({pct:5.1f}%)  "
            f"elapsed {elapsed:5.1f}s  "
            f"ETA {remaining:5.1f}s",
            end="\r",
        )
        if current == total:
            print()

    with tempfile.TemporaryDirectory() as tmpdir:
        silent_video_path = os.path.join(tmpdir, "silent.mp4")
        anim.save(silent_video_path, writer="ffmpeg", fps=fps, dpi=dpi,
                  extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
                  progress_callback=_progress)

        if sonify:
            print("Synthesizing audio…")
            audio_chunks = []
            phase = 0.0
            for i, freq in enumerate(freq_array, 1):
                t = phase + 2 * np.pi * \
                    np.arange(1, samples_per_frame + 1) * freq / _SAMPLE_RATE
                audio_chunks.append((np.sin(t) * 0.3).astype(np.float32))
                phase = t[-1] % (2 * np.pi)
                print(
                    f"  frame {i:>{len(str(n_frames))}}/{n_frames}", end="\r")
            print()
            audio_int16 = (np.concatenate(audio_chunks)
                           * 32767).astype(np.int16)

            wav_path = os.path.join(tmpdir, "audio.wav")
            wavfile.write(wav_path, _SAMPLE_RATE, audio_int16)

            print(f"Merging → {mp4_path}")
            subprocess.run(
                ["ffmpeg", "-i", silent_video_path, "-i", wav_path,
                 "-c:v", "copy", "-c:a", "aac", mp4_path],
                check=True, capture_output=True,
            )
        else:
            import shutil
            shutil.copy(silent_video_path, mp4_path)

    print(f"Done → {mp4_path}")


def _safe_output_stem(directory: str, stem: str) -> str:
    """Return a path stem that doesn't overwrite an existing file.

    If {stem}.mp4 already exists, tries {stem} (2).mp4, (3).mp4, etc.
    """
    candidate = os.path.join(directory, stem)
    if not os.path.exists(f"{candidate}.mp4"):
        return candidate
    counter = 2
    while os.path.exists(f"{os.path.join(directory, stem)} ({counter}).mp4"):
        counter += 1
    return os.path.join(directory, f"{stem} ({counter})")


def main():
    parser = argparse.ArgumentParser(
        description="Native matplotlib map of CapMetro bus positions."
    )
    parser.add_argument(
        "date", help="Date string in YYYY_MM_DD format (e.g. 2026_03_04)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=200,
        help="Animation frame interval in milliseconds (default: 200)",
    )
    parser.add_argument(
        "--bin",
        type=int,
        default=30,
        help="Bin size in seconds for animation frames (default: 30)",
    )
    parser.add_argument(
        "--trail",
        type=int,
        default=150,
        help="Trail length in seconds (default: 150 = 2.5 minutes)",
    )
    parser.add_argument(
        "--routes",
        default=None,
        help="Comma-separated list of route IDs to render (default: all routes)",
    )
    parser.add_argument(
        "--gradient",
        action="store_true",
        help="Fade trail from transparent at tail to opaque at head (default: solid)",
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Disable audio sonification",
    )
    parser.add_argument(
        "--sonification",
        default="buscount",
        choices=["buscount"],
        help="Sonification mode (default: buscount)",
    )
    parser.add_argument(
        "--prerender",
        action="store_true",
        help="Render to an MP4 file instead of displaying on screen",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit rendering to the first N frames (useful for quick previews)",
    )
    parser.add_argument(
        "--output-dir",
        default="../renders",
        help="Output directory for prerendered MP4 files (default: ../renders)",
    )
    args = parser.parse_args()

    df = load_data(args.date, bin_seconds=args.bin)
    if args.routes:
        route_filter = [r.strip() for r in args.routes.split(",")]
        df = df[df["route_id"].astype(str).isin(route_filter)]
        if df.empty:
            print(f"No data found for routes: {route_filter}")
            return
        print(f"Filtering to routes: {route_filter}")

    # Compute global frequency scale before any frame truncation
    global_counts_series = df.groupby("minute")["trip_id"].nunique()

    trail_frames = max(1, args.trail // args.bin)

    if args.max_frames is not None:
        frames_all = sorted(df["minute"].unique())
        cutoff = frames_all[min(args.max_frames, len(frames_all)) - 1]
        df = df[df["minute"] <= cutoff]
        print(
            f"Preview mode: capped at {args.max_frames} frames (up to {cutoff})")

    print(
        f"Loaded {len(df)} vehicle position records across {df['minute'].nunique()} "
        f"{args.bin}s-bins with a trail of {args.trail} seconds."
    )

    output_path = None
    if args.prerender:
        os.makedirs(args.output_dir, exist_ok=True)
        stem = f"{args.date}_{args.sonification}_{args.bin}s"
        output_path = _safe_output_stem(args.output_dir, stem)

    build_matplotlib_map(
        df,
        interval_ms=args.interval,
        trail_minutes=trail_frames,
        gradient=args.gradient,
        sonify=not args.no_sound,
        sonification=args.sonification,
        prerender=args.prerender,
        output_path=output_path,
        global_counts_series=global_counts_series,
    )


if __name__ == "__main__":
    main()
