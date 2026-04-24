from audio_synth import (
    SAMPLE_RATE as _SAMPLE_RATE,
    BPM_MIN as _BPM_MIN,
    BPM_MAX as _BPM_MAX,
    build_freq_array as _build_freq_array,
    ChordProgressionSynth,
    GridBlipSynth,
    GRID_CELL_SIZE_DEFAULT as _GRID_CELL_SIZE_DEFAULT,
    CrosspathBlipSynth,
    CROSSPATH_PROXIMITY_DEFAULT as _CROSSPATH_PROXIMITY_DEFAULT,
)
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
from matplotlib.patches import Circle, Rectangle
from pyproj import Transformer
from scipy.io import wavfile

# WGS-84 → Web Mercator (EPSG:3857)
_to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

_base_map = ctx.providers.CartoDB.PositronNoLabels

# Fixed map view: Austin city center
_AUSTIN_X, _AUSTIN_Y = _to_mercator.transform(-97.7431, 30.2672)
_HALF_EXTENT = 15_000  # radius of map view from city center in meters
# metres in Web Mercator for proximity flash
_CROSSPATH_FLASH_RADIUS: float = 300.0

_GRIDBLIP_PASTELS: list[tuple[float, float, float]] = [
    (1.00, 0.71, 0.76),  # pink
    (0.80, 0.73, 0.92),  # lavender
    (0.71, 0.93, 0.80),  # mint
    (1.00, 0.85, 0.71),  # peach
    (0.71, 0.87, 0.97),  # sky blue
    (0.95, 0.95, 0.71),  # pale yellow
    (0.97, 0.76, 0.71),  # salmon
    (0.71, 0.95, 0.95),  # aqua
]


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


def _compute_avgspeed_series(df: pd.DataFrame) -> pd.Series:
    """Return mean speed (m/s) per bin for moving buses (speed > 0)."""
    moving = df[df["speed"] > 0]
    return moving.groupby("minute")["speed"].mean()


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
    avgspeed_series: pd.Series | None = None,
    global_avgspeed_series: pd.Series | None = None,
    grid_size: int = _GRID_CELL_SIZE_DEFAULT,
    proximity_radius: int = _CROSSPATH_PROXIMITY_DEFAULT,
) -> None:
    # --- Color map: one colour per route_id; grey for gridblip / crosspath ---
    route_ids = sorted(df["route_id"].dropna().unique())
    if sonification in ("gridblip", "crosspath"):
        route_color = {rid: (0.55, 0.55, 0.55, 1.0) for rid in route_ids}
    else:
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
    samples_per_frame = round(interval_ms * _SAMPLE_RATE / 1000)

    # Speed normalisation bounds for avgspeed mode (use global series so
    # --max-frames previews stay consistent with full renders)
    _spd_scale = global_avgspeed_series if global_avgspeed_series is not None else avgspeed_series
    if _spd_scale is not None and not _spd_scale.empty:
        _spd_min = float(_spd_scale.min())
        _spd_max = float(_spd_scale.max())
    else:
        _spd_min, _spd_max = 0.0, 1.0

    # Precompute normalised speed array aligned to frames (for prerender)
    if sonification == "avgspeed" and avgspeed_series is not None:
        _spd_range = _spd_max - _spd_min + 1e-9
        normalized_speeds = np.array(
            [float(np.clip((avgspeed_series.get(f, _spd_min) - _spd_min) / _spd_range, 0, 1))
             for f in frames],
            dtype=np.float32,
        )
    else:
        normalized_speeds = None

    # Precompute voice count array (1–12) for avgspeed mode using bus count
    if sonification == "avgspeed":
        _cnt_scale = global_counts_series if global_counts_series is not None else counts_series
        _cnt_min = int(
            _cnt_scale.min()) if _cnt_scale is not None and not _cnt_scale.empty else 0
        _cnt_max = int(
            _cnt_scale.max()) if _cnt_scale is not None and not _cnt_scale.empty else 1
        _cnt_range = max(_cnt_max - _cnt_min, 1)
        n_voices_array = np.array(
            [int(np.clip(round(1 + 11 * (counts_series.get(f, _cnt_min) - _cnt_min) / _cnt_range), 1, 12))
             for f in frames],
            dtype=np.int32,
        )
        # BPM per frame: 70% bus count, 30% avg speed
        _cnt_norm_arr = np.array(
            [(counts_series.get(f, _cnt_min) - _cnt_min) /
             _cnt_range for f in frames],
            dtype=np.float64,
        )
        bpm_array = _BPM_MIN + (_BPM_MAX - _BPM_MIN) * (
            0.7 * _cnt_norm_arr + 0.3 * normalized_speeds
        )
    else:
        n_voices_array = None
        bpm_array = None

    # Create ChordProgressionSynth for avgspeed live playback
    synth = (ChordProgressionSynth()
             if sonification == "avgspeed" and sonify
             else None)

    # Fixed bounding box centered on Austin
    x_min, x_max = _AUSTIN_X - _HALF_EXTENT, _AUSTIN_X + _HALF_EXTENT
    y_min, y_max = _AUSTIN_Y - _HALF_EXTENT, _AUSTIN_Y + _HALF_EXTENT

    # --- gridblip geometry and state (computed always; used only in gridblip) ---
    cell_size = float(grid_size)
    n_rows_grid = int(np.ceil((y_max - y_min) / cell_size))
    n_cols_grid = int(np.ceil((x_max - x_min) / cell_size))
    _flash_patches: list = []
    _blip_audio_buf: list = [np.zeros(0, dtype=np.float32)]
    _blip_buf_pos = np.zeros(1, dtype=np.int64)
    if sonification == "gridblip":
        grid_blip_synth: GridBlipSynth | None = GridBlipSynth(
            cell_size=cell_size,
            origin_x=x_min,
            origin_y=y_min,
            n_rows=n_rows_grid,
        )
    else:
        grid_blip_synth = None

    if sonification == "crosspath":
        crosspath_blip_synth: CrosspathBlipSynth | None = CrosspathBlipSynth(
            proximity=proximity_radius,
        )
    else:
        crosspath_blip_synth = None

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

    # --- gridblip: semi-transparent grid overlay ---
    if sonification == "gridblip":
        h_segs = [
            [(x_min, y_min + j * cell_size), (x_max, y_min + j * cell_size)]
            for j in range(n_rows_grid + 1)
        ]
        v_segs = [
            [(x_min + i * cell_size, y_min), (x_min + i * cell_size, y_max)]
            for i in range(n_cols_grid + 1)
        ]
        ax.add_collection(LineCollection(
            h_segs + v_segs,
            colors=[(0.4, 0.4, 0.4, 0.15)],
            linewidths=0.4,
            zorder=3,
        ))

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

    # --- Stats HUD (bottom-left): avg speed + bus count ---
    stats_text = ax.text(
        0.01, 0.03, "",
        transform=ax.transAxes,
        color="#111111",
        fontsize=11,
        fontfamily="monospace",
        va="bottom",
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
        if synth is not None:
            # avgspeed: chord progression with speed-driven detuning + count-driven voices
            def _audio_callback(outdata, frames_count, time_info, status):
                fi = int(current_frame_idx[0])
                speed_norm = float(
                    normalized_speeds[fi]) if normalized_speeds is not None else 0.0
                n_v = int(n_voices_array[fi]
                          ) if n_voices_array is not None else 1
                synth.set_speed(speed_norm)
                synth.set_count(n_v)
                if bpm_array is not None:
                    synth.set_tempo(float(bpm_array[fi]))
                chunk = synth.generate(frames_count)
                outdata[:] = chunk.reshape(-1, 1)
        elif grid_blip_synth is not None or crosspath_blip_synth is not None:
            # gridblip / crosspath: stream from pre-rendered per-frame blip buffer
            def _audio_callback(outdata, frames_count, time_info, status):
                buf = _blip_audio_buf[0]
                pos = int(_blip_buf_pos[0])
                end = pos + frames_count
                available = len(buf)
                if end <= available:
                    outdata[:, 0] = buf[pos:end]
                    _blip_buf_pos[0] = end
                elif pos < available:
                    n_avail = available - pos
                    outdata[:n_avail, 0] = buf[pos:]
                    outdata[n_avail:, 0] = 0.0
                    _blip_buf_pos[0] = available
                else:
                    outdata[:, 0] = 0.0
        else:
            # buscount: single sine tone
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

        # --- gridblip: cell flash visuals + live audio ---
        crossing_count = 0
        if sonification == "gridblip" and grid_blip_synth is not None:
            for p in _flash_patches:
                p.remove()
            _flash_patches.clear()

            current_frame_df = df_by_minute.get(frame_minute, df.iloc[:0])
            events = grid_blip_synth.feed_frame(
                current_frame_df, samples_per_frame)
            crossing_count = len(events)

            for _, _, row, col in events:
                color = _GRIDBLIP_PASTELS[(
                    row * 31 + col) % len(_GRIDBLIP_PASTELS)]
                rect = Rectangle(
                    (x_min + col * cell_size, y_min + row * cell_size),
                    cell_size, cell_size,
                    facecolor=color, alpha=0.65, zorder=4, linewidth=0,
                )
                ax.add_patch(rect)
                _flash_patches.append(rect)

            if sonify and not prerender:
                blip_audio = grid_blip_synth.render_blips(
                    events, samples_per_frame)
                _blip_audio_buf[0] = blip_audio
                _blip_buf_pos[0] = 0
                current_frame_idx[0] = fi

        elif sonification == "crosspath" and crosspath_blip_synth is not None:
            for p in _flash_patches:
                p.remove()
            _flash_patches.clear()

            current_frame_df = df_by_minute.get(frame_minute, df.iloc[:0])
            events = crosspath_blip_synth.feed_frame(
                current_frame_df, samples_per_frame)
            crossing_count = len(events)

            for evt_idx, (_, _, mid_x, mid_y) in enumerate(events):
                color = _GRIDBLIP_PASTELS[evt_idx % len(_GRIDBLIP_PASTELS)]
                circle = Circle(
                    (mid_x, mid_y),
                    radius=_CROSSPATH_FLASH_RADIUS,
                    facecolor=color, alpha=0.65, zorder=4, linewidth=0,
                )
                ax.add_patch(circle)
                _flash_patches.append(circle)

            if sonify and not prerender:
                blip_audio = crosspath_blip_synth.render_blips(
                    events, samples_per_frame)
                _blip_audio_buf[0] = blip_audio
                _blip_buf_pos[0] = 0
                current_frame_idx[0] = fi

        if sonification == "avgspeed" and avgspeed_series is not None:
            avg_ms = avgspeed_series.get(frame_minute, float("nan"))
            moving_count = int(
                (df_by_minute.get(frame_minute, df.iloc[:0])["speed"] > 0).sum())
            avg_mph = avg_ms * 2.23694

        timestamp_text.set_text(frame_minute)

        # --- Bottom-left stats HUD ---
        bus_count = int(counts_series.get(frame_minute, 0))
        if sonification == "buscount":
            stats_text.set_text(
                f"Frequency: {freq_array[fi]:.1f} Hz\nBuses: {bus_count}")
        elif sonification in ("gridblip", "crosspath"):
            stats_text.set_text(
                f"crossings: {crossing_count}\nbuses: {bus_count}")
        elif avgspeed_series is not None:
            avg_ms = avgspeed_series.get(frame_minute, float("nan"))
            if avg_ms == avg_ms:  # not NaN
                avg_mph_str = f"{avg_ms * 2.23694:.1f} mph"
            else:
                avg_mph_str = "-- mph"
            stats_text.set_text(
                f"avg speed: {avg_mph_str}\nbuses: {bus_count}")
        else:
            stats_text.set_text(f"buses: {bus_count}")

        return lc, timestamp_text, stats_text

    anim = FuncAnimation(
        fig,
        _update,
        frames=frames,
        interval=interval_ms,
        blit=(sonification not in ("gridblip", "crosspath")),
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
            synth=synth,
            normalized_speeds=normalized_speeds,
            n_voices_array=n_voices_array,
            bpm_array=bpm_array,
            grid_blip_synth=grid_blip_synth,
            crosspath_blip_synth=crosspath_blip_synth,
            frames=frames,
            df_by_minute=df_by_minute,
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
    synth: "ChordProgressionSynth | None" = None,
    normalized_speeds: "np.ndarray | None" = None,
    n_voices_array: "np.ndarray | None" = None,
    bpm_array: "np.ndarray | None" = None,
    grid_blip_synth: "GridBlipSynth | None" = None,
    crosspath_blip_synth: "CrosspathBlipSynth | None" = None,
    frames: "list | None" = None,
    df_by_minute: "dict | None" = None,
) -> None:
    """Render animation and synthesized audio, merging into a single MP4.

    Also saves the WAV separately alongside the MP4.
    Sync guarantee: both tracks are derived from the same freq_array and interval_ms.
    Video FPS = 1000 / interval_ms, so each frame = interval_ms ms exactly.
    Audio: frame i gets round(interval_ms * SAMPLE_RATE / 1000) samples,
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

            if synth is not None and normalized_speeds is not None:
                # avgspeed: chord progression synth with per-frame speed + voice count
                for i in range(n_frames):
                    synth.set_speed(float(normalized_speeds[i]))
                    if n_voices_array is not None:
                        synth.set_count(int(n_voices_array[i]))
                    if bpm_array is not None:
                        synth.set_tempo(float(bpm_array[i]))
                    audio_chunks.append(synth.generate(samples_per_frame))
                    print(
                        f"  frame {i + 1:>{len(str(n_frames))}}/{n_frames}", end="\r")
            elif (
                grid_blip_synth is not None
                and frames is not None
                and df_by_minute is not None
            ):
                # gridblip: marimba blips for each cell crossing
                grid_blip_synth.reset()
                for i, frame_minute in enumerate(frames[:n_frames]):
                    frame_df = df_by_minute.get(frame_minute, pd.DataFrame())
                    events = grid_blip_synth.feed_frame(
                        frame_df, samples_per_frame, seed=i)
                    audio_chunks.append(
                        grid_blip_synth.render_blips(events, samples_per_frame))
                    print(
                        f"  frame {i + 1:>{len(str(n_frames))}}/{n_frames}", end="\r")
            elif (
                crosspath_blip_synth is not None
                and frames is not None
                and df_by_minute is not None
            ):
                # crosspath: soft pad blips for proximity crossings
                crosspath_blip_synth.reset()
                for i, frame_minute in enumerate(frames[:n_frames]):
                    frame_df = df_by_minute.get(frame_minute, pd.DataFrame())
                    events = crosspath_blip_synth.feed_frame(
                        frame_df, samples_per_frame, seed=i)
                    audio_chunks.append(
                        crosspath_blip_synth.render_blips(events, samples_per_frame))
                    print(
                        f"  frame {i + 1:>{len(str(n_frames))}}/{n_frames}", end="\r")
            else:
                # buscount: simple sine tone
                phase = 0.0
                for i, freq in enumerate(freq_array, 1):
                    t = phase + 2 * np.pi * \
                        np.arange(1, samples_per_frame + 1) * \
                        freq / _SAMPLE_RATE
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
        choices=["buscount", "avgspeed", "gridblip", "crosspath"],
        help="Sonification mode (default: buscount)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=_GRID_CELL_SIZE_DEFAULT,
        help=f"Grid cell size in metres for gridblip mode (default: {_GRID_CELL_SIZE_DEFAULT})",
    )
    parser.add_argument(
        "--proximity-radius",
        type=int,
        default=_CROSSPATH_PROXIMITY_DEFAULT,
        help=f"Proximity radius in metres for crosspath mode (default: {_CROSSPATH_PROXIMITY_DEFAULT})",
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
    global_counts_series = df.groupby(
        "minute")["trip_id"].nunique()  # bus count per bin/frame

    # average speed per bin/frame (for avgspeed mode only)
    avgspeed_series = _compute_avgspeed_series(
        df) if args.sonification == "avgspeed" else None

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
        avgspeed_series=avgspeed_series,
        global_avgspeed_series=avgspeed_series,
        grid_size=args.grid_size,
        proximity_radius=args.proximity_radius,
    )


if __name__ == "__main__":
    main()
