"""audio_synth.py — All audio synthesis logic for the CapMetro sonification.

Four sonification modes are supported:

  buscount   : single sine tone, pitch mapped logarithmically to active bus count.
  avgspeed   : looping piano-chord progression (6 bars, variable 60–100 BPM) where:
               - bus count  → number of active voices (1–12, bass-up stacking)
               - avg speed  → per-voice LFO pitch detuning amplitude
               Progression: Am | D7 Dm | Cmaj7 | Dm G | A | Cmaj7 E
               Higher speed + more buses = dense, chaotic sound.
  gridblip   : event-driven marimba blips — each bus crossing into a new grid cell
               triggers a short pitched blip (C-major pentatonic, latitude → pitch).
               Auto-ducking keeps dense rush-hour frames clean.
  crosspath  : event-driven soft-pad blips — each pair of buses entering within a
               configurable proximity radius of each other (for the first time) triggers
               a gentle pad tone (C-major pentatonic, avg speed of the pair → pitch).
               Edge-triggered: fires once per pair on entry, not on every close frame.
               Auto-ducking keeps dense rush-hour frames clean.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 44100

# ---------------------------------------------------------------------------
# buscount helpers
# ---------------------------------------------------------------------------

_C2_HZ: float = 65.41    # lowest pitch  (fewest buses)
_C6_HZ: float = 1046.5   # highest pitch (peak buses)


def freq_from_count(count: int, min_count: int, max_count: int) -> float:
    """Map active bus count to frequency on a log (octave) scale."""
    if max_count <= min_count:
        return _C2_HZ
    t = np.clip((count - min_count) / (max_count - min_count), 0.0, 1.0)
    return _C2_HZ * (2.0 ** (t * np.log2(_C6_HZ / _C2_HZ)))


def build_freq_array(
    sonification: str,
    df,
    frames: list,
    counts_series,
    global_counts_series=None,
) -> np.ndarray:
    """Build a per-frame frequency array.

    For avgspeed mode the array is all-zeros; audio is handled separately by
    ChordProgressionSynth, not by freq_array.

    global_counts_series, if provided, sets the min/max scale so that
    --max-frames previews stay consistent with full renders.
    """
    if sonification == "buscount":
        scale = global_counts_series if global_counts_series is not None else counts_series
        min_buses = int(scale.min())
        max_buses = int(scale.max())
        print(
            f"Bus count range (scale): {min_buses} \u2013 {max_buses} active trips/frame")
        return np.array(
            [freq_from_count(int(counts_series.get(f, min_buses)), min_buses, max_buses)
             for f in frames],
            dtype=np.float64,
        )
    if sonification in ("avgspeed", "gridblip", "crosspath"):
        return np.zeros(len(frames), dtype=np.float64)
    raise ValueError(f"Unknown sonification mode: {sonification!r}")


# ---------------------------------------------------------------------------
# avgspeed: chord progression synth
# ---------------------------------------------------------------------------

BPM: float = 80.0                   # default / reference tempo
# slowest tempo (fewest buses / lowest speed)
BPM_MIN: float = 60.0
# fastest tempo (most buses / highest speed)
BPM_MAX: float = 120.0
# reference beat duration (s) — used for CHORD_SCHEDULE only
_BEAT: float = 60.0 / BPM
# reference loop duration at BPM (6 bars × 4 beats)
LOOP_DURATION: float = _BEAT * 24

# Beat positions of each chord within the 6-bar loop (BPM-independent).
# These are converted to seconds per-chunk using the current live BPM.
_CHORD_BEAT_POSITIONS: list[int] = [0, 4, 6, 8, 12, 14, 16, 20, 22]

# Note frequencies (Hz) for A440 — two octaves covered
_NOTE = {
    # octave 2
    "D2":  73.42,
    "E2":  82.41,
    "F2":  87.31,
    "Fs2": 92.50,   # F#2
    "A2":  110.00,
    "C3":  130.81,
    "D3":  146.83,
    "E3":  164.81,
    "F3":  174.61,
    "Fs3": 185.00,  # F#3
    # octave 3/4
    "A3":  220.00,
    "C4":  261.63,
    "D4":  293.66,
    "E4":  329.63,
    "F4":  349.23,
    "Fs4": 369.99,  # F#4
    # octave 4/5
    "A4":  440.00,
    "C5":  523.25,
    "D5":  587.33,
    "E5":  659.26,
    "F5":  698.46,
    "Fs5": 739.99,  # F#5
    # octave 5/6
    "A5":  880.00,
    "C6":  1046.50,
    "E6":  1318.51,
    # G and B notes (Cmaj7, G major)
    "G2":  98.00,
    "B2":  123.47,
    "G3":  196.00,
    "B3":  246.94,
    "G4":  392.00,
    "B4":  493.88,
    "G5":  783.99,
    "B5":  987.77,
    "D6":  1174.66,
    # G# / Ab notes (E major)
    "Gs2": 103.83,
    "Gs3": 207.65,
    "Gs4": 415.30,
    "Gs5": 830.61,
    # C# / Db notes (A major)
    "Cs3": 138.59,
    "Cs4": 277.18,
    "Cs5": 554.37,
    "Cs6": 1108.73,
}

# CHORD_SCHEDULE: list of (onset_seconds, [12 voice frequencies, bass-up])
# Bar 1: Am     (4 beats)  beat  0
# Bar 2: D7     (2 beats)  beat  4  |  Dm    (2 beats)  beat  6
# Bar 3: Cmaj7  (4 beats)  beat  8
# Bar 4: Dm     (2 beats)  beat 12  |  G     (2 beats)  beat 14
# Bar 5: A      (4 beats)  beat 16
# Bar 6: Cmaj7  (2 beats)  beat 20  |  E     (2 beats)  beat 22
# Total loop: 24 beats = 18.0 s at 80 BPM
#
# Voices ordered low→high so 1–12 active voices stack the chord from the bass.
CHORD_SCHEDULE: list[tuple[float, list[float]]] = [
    (0.0, [              # Am  —  E A C E  A C E A  C E A C
        _NOTE["E2"], _NOTE["A2"], _NOTE["C3"], _NOTE["E3"],
        _NOTE["A3"], _NOTE["C4"], _NOTE["E4"], _NOTE["A4"],
        _NOTE["C5"], _NOTE["E5"], _NOTE["A5"], _NOTE["C6"],
    ]),
    (_BEAT * 4, [        # D7  —  D A C D  Fs A C D  Fs A C Fs
        _NOTE["D2"], _NOTE["A2"], _NOTE["C3"], _NOTE["D3"],
        _NOTE["Fs3"], _NOTE["A3"], _NOTE["C4"], _NOTE["D4"],
        _NOTE["Fs4"], _NOTE["A4"], _NOTE["C5"], _NOTE["Fs5"],
    ]),
    (_BEAT * 6, [        # Dm  —  D A C D  F  A C D  F  A C F
        _NOTE["D2"], _NOTE["A2"], _NOTE["C3"], _NOTE["D3"],
        _NOTE["F3"], _NOTE["A3"], _NOTE["C4"], _NOTE["D4"],
        _NOTE["F4"], _NOTE["A4"], _NOTE["C5"], _NOTE["F5"],
    ]),
    (_BEAT * 8, [        # Cmaj7 — E G B C  E G B C  E G B C
        _NOTE["E2"], _NOTE["G2"], _NOTE["B2"], _NOTE["C3"],
        _NOTE["E3"], _NOTE["G3"], _NOTE["B3"], _NOTE["C4"],
        _NOTE["E4"], _NOTE["G4"], _NOTE["B4"], _NOTE["C5"],
    ]),
    (_BEAT * 12, [       # Dm  (repeat)
        _NOTE["D2"], _NOTE["A2"], _NOTE["C3"], _NOTE["D3"],
        _NOTE["F3"], _NOTE["A3"], _NOTE["C4"], _NOTE["D4"],
        _NOTE["F4"], _NOTE["A4"], _NOTE["C5"], _NOTE["F5"],
    ]),
    (_BEAT * 14, [       # G major Sus — G B D A  G B D A  G B D A
        _NOTE["G2"], _NOTE["B2"], _NOTE["D3"], _NOTE["A3"],
        _NOTE["G3"], _NOTE["B3"], _NOTE["D4"], _NOTE["A4"],
        _NOTE["G4"], _NOTE["B4"], _NOTE["D5"], _NOTE["A5"],
    ]),
    (_BEAT * 16, [       # A major sus — A Cs E Fs  A Cs E Fs  A Cs E Fs
        _NOTE["A2"], _NOTE["Cs3"], _NOTE["E3"], _NOTE["Fs3"],
        _NOTE["A3"], _NOTE["Cs4"], _NOTE["E4"], _NOTE["Fs4"],
        _NOTE["A4"], _NOTE["Cs5"], _NOTE["E5"], _NOTE["Fs5"],
    ]),
    (_BEAT * 20, [       # Cmaj7 (repeat)
        _NOTE["E2"], _NOTE["G2"], _NOTE["B2"], _NOTE["C3"],
        _NOTE["E3"], _NOTE["G3"], _NOTE["B3"], _NOTE["C4"],
        _NOTE["E4"], _NOTE["G4"], _NOTE["B4"], _NOTE["C5"],
    ]),
    (_BEAT * 22, [       # E major — E Gs B E  Gs B E Gs  B E Gs B
        _NOTE["E2"], _NOTE["Gs2"], _NOTE["B2"], _NOTE["E3"],
        _NOTE["Gs3"], _NOTE["B3"], _NOTE["E4"], _NOTE["Gs4"],
        _NOTE["B4"], _NOTE["E5"], _NOTE["Gs5"], _NOTE["B5"],
    ]),
]

# Voice frequency table: shape (N_CHORDS, N_VOICES)
_CHORD_FREQS: np.ndarray = np.array(
    [c[1] for c in CHORD_SCHEDULE], dtype=np.float64)
_CHORD_BEAT_POS: np.ndarray = np.array(_CHORD_BEAT_POSITIONS, dtype=np.float64)

_N_VOICES: int = 12   # maximum voices (all phase accumulators always run)
_N_HARMONICS: int = 4
_HARMONIC_AMPS: np.ndarray = np.array([1.0, 0.5, 0.25, 0.12])  # partials 1–4
_DECAY_RATE: float = 0.5  # amplitude envelope decay rate (s⁻¹)

# max pitch wobble (±0.5 semitone at full speed)
_MAX_DETUNE_CENTS: float = 50.0

# Fixed LFO parameters for all 12 voices — deterministic/reproducible renders.
_LFO_RATES: np.ndarray = np.array(
    [0.08, 0.12, 0.17, 0.23, 0.07, 0.14, 0.19, 0.25, 0.09, 0.11, 0.16, 0.21])
_LFO_PHASES: np.ndarray = np.array(
    [0.0,  1.1,  2.3,  4.7,  0.8,  1.9,  3.1,  5.2,  0.4,  2.7,  4.1,  6.0])

# Output gain fixed at 12-voice ceiling so amplitude grows naturally with count.
_HARMONIC_SUM: float = float(_HARMONIC_AMPS.sum())  # 1.87
_MASTER_GAIN: float = 0.3 / (_N_VOICES * _HARMONIC_SUM)


class ChordProgressionSynth:
    """Stateful, chunk-based chord-progression synthesiser.

    Two continuous control inputs, updated before each generate() call:
      set_speed(speed_norm)  — 0=in tune, 1=max detuning (±50 cents per voice)
      set_count(n_voices)    — 1–12: how many bass-up voices are active

    All 12 phase accumulators always advance so voice activation/deactivation
    never causes phase discontinuities.
    """

    def __init__(self) -> None:
        self._sample_pos: int = 0
        # [0, 1) — fractional position within the loop
        self._loop_frac: float = 0.0
        # Phase accumulators: shape (N_VOICES=12, N_HARMONICS=4)
        self._osc_phases = np.zeros(
            (_N_VOICES, _N_HARMONICS), dtype=np.float64)
        self._speed_norm: float = 0.0
        self._n_active: int = 1
        self._bpm: float = BPM

    def set_speed(self, speed_normalized: float) -> None:
        self._speed_norm = float(np.clip(speed_normalized, 0.0, 1.0))

    def set_count(self, n_voices: int) -> None:
        self._n_active = int(np.clip(n_voices, 1, _N_VOICES))

    def set_tempo(self, bpm: float) -> None:
        """Set playback tempo (clamped to BPM_MIN..BPM_MAX)."""
        self._bpm = float(np.clip(bpm, BPM_MIN, BPM_MAX))

    def generate(self, n_samples: int) -> np.ndarray:
        """Synthesise n_samples of audio and advance internal state.

        Returns a float32 array of length n_samples.
        All 12 phase accumulators are advanced even for inactive voices so that
        re-activating a voice later produces a seamless continuation.
        """
        # Absolute time for every sample — used only for LFO (BPM-independent)
        t = (np.arange(n_samples, dtype=np.float64) +
             self._sample_pos) / SAMPLE_RATE

        # Tempo-dependent loop tracking using _loop_frac [0, 1)
        beat_dur = 60.0 / self._bpm
        loop_dur = 24.0 * beat_dur          # 24 beats per loop (6 bars)
        chord_onsets_s = _CHORD_BEAT_POS * beat_dur

        # Per-sample fractional loop position, then convert to seconds
        frac = self._loop_frac + \
            np.arange(n_samples, dtype=np.float64) / (SAMPLE_RATE * loop_dur)
        frac_wrapped = frac % 1.0
        loop_pos = frac_wrapped * loop_dur  # seconds within current loop

        # Advance the loop fraction accumulator (avoids ever-growing float)
        self._loop_frac = float(frac[-1] % 1.0)

        # Chord lookup: which chord is active and how long since it started
        chord_idx = np.searchsorted(chord_onsets_s, loop_pos, side="right") - 1
        t_since_onset = loop_pos - chord_onsets_s[chord_idx]

        # Amplitude envelope shared across all voices (decays from each chord onset)
        amp_env = np.exp(-_DECAY_RATE * t_since_onset)  # shape (n_samples,)

        # Per-voice LFO detuning in cents — shape (n_voices, n_samples)
        lfo_arg = (
            2.0 * np.pi * _LFO_RATES[:, None] * t[None, :]
            + _LFO_PHASES[:, None]
        )
        detune_cents = _MAX_DETUNE_CENTS * self._speed_norm * np.sin(lfo_arg)

        output = np.zeros(n_samples, dtype=np.float64)

        for v in range(_N_VOICES):
            # Base frequency per sample (steps at chord boundaries)
            base_freq = _CHORD_FREQS[chord_idx, v]  # shape (n_samples,)

            # Apply LFO detuning: f_inst = f_base * 2^(cents/1200)
            inst_freq = base_freq * (2.0 ** (detune_cents[v] / 1200.0))

            voice_signal = np.zeros(n_samples, dtype=np.float64)
            for h in range(_N_HARMONICS):
                harmonic_num = h + 1
                # Each harmonic has its own phase accumulator to avoid
                # inter-chunk phase discontinuities at chunk boundaries.
                delta_phase = 2.0 * np.pi * inst_freq * harmonic_num / SAMPLE_RATE
                phases = self._osc_phases[v, h] + np.cumsum(delta_phase)
                self._osc_phases[v, h] = phases[-1] % (2.0 * np.pi)
                voice_signal += _HARMONIC_AMPS[h] * np.sin(phases)

            # Only mix active voices into output; all accumulators still advanced
            if v < self._n_active:
                output += voice_signal * amp_env

        self._sample_pos += n_samples
        return (output * _MASTER_GAIN).astype(np.float32)


# ---------------------------------------------------------------------------
# gridblip: grid-crossing marimba-blip synthesiser
# ---------------------------------------------------------------------------

GRID_CELL_SIZE_DEFAULT: int = 500  # metres (Web Mercator)

# C-major pentatonic, C4 – A5: any combination of these notes is consonant.
_PENTA_FREQS: np.ndarray = np.array([
    261.63,  # C4
    293.66,  # D4
    329.63,  # E4
    392.00,  # G4
    440.00,  # A4
    523.25,  # C5
    587.33,  # D5
    659.26,  # E5
    783.99,  # G5
    880.00,  # A5
], dtype=np.float64)

_BLIP_ATTACK_S: float = 0.001    # 1 ms linear attack
_BLIP_DECAY_RATE: float = 50.0   # exponential decay (s⁻¹); ~80 ms audible tail
_BLIP_HARMONICS: np.ndarray = np.array([1.0, 0.5, 0.25], dtype=np.float64)
_BLIP_MASTER_GAIN: float = 0.4   # per-blip peak gain before auto-duck scaling


def grid_cell(
    x: float,
    y: float,
    cell_size: float,
    origin_x: float,
    origin_y: float,
) -> tuple[int, int]:
    """Convert Web Mercator (x, y) to (row, col) grid indices."""
    col = int(np.floor((x - origin_x) / cell_size))
    row = int(np.floor((y - origin_y) / cell_size))
    return row, col


def pitch_for_row(row: int, n_rows: int) -> float:
    """Map a grid row to a C-major pentatonic frequency (south=low, north=high)."""
    n_notes = len(_PENTA_FREQS)
    idx = int(np.clip(
        round(row / max(n_rows - 1, 1) * (n_notes - 1)),
        0, n_notes - 1,
    ))
    return float(_PENTA_FREQS[idx])


class GridBlipSynth:
    """Stateful synthesiser for grid-crossing blip events.

    Per-frame usage::

        events = synth.feed_frame(frame_df, n_samples)
        audio  = synth.render_blips(events, n_samples)

    ``events`` is a list of ``(pitch_hz, onset_offset_samples, row, col)``.
    A bus's first appearance in the data counts as a crossing.
    """

    def __init__(
        self,
        cell_size: float = GRID_CELL_SIZE_DEFAULT,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        n_rows: int = 60,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self._cell_size = float(cell_size)
        self._origin_x = float(origin_x)
        self._origin_y = float(origin_y)
        self._n_rows = int(n_rows)
        self._sr = int(sample_rate)
        self._prev_cells: dict[str, tuple[int, int]] = {}

    def reset(self) -> None:
        """Clear all tracked bus positions (restart from the first frame)."""
        self._prev_cells.clear()

    def feed_frame(
        self,
        frame_df,
        n_samples: int,
        seed: "int | None" = None,
    ) -> list[tuple[float, int, int, int]]:
        """Detect grid crossings and return blip events for this frame.

        ``frame_df`` must have columns ``trip_id``, ``x``, ``y``.
        Returns a list of ``(pitch_hz, onset_offset_samples, row, col)``.
        """
        if frame_df is None or frame_df.empty or "trip_id" not in frame_df.columns:
            return []

        events: list[tuple[float, int, int, int]] = []
        seen: set[str] = set()
        rng = np.random.default_rng(seed)

        for _, bus in frame_df.iterrows():
            trip_id = str(bus["trip_id"])
            x, y = float(bus["x"]), float(bus["y"])
            seen.add(trip_id)

            row, col = grid_cell(x, y, self._cell_size, self._origin_x, self._origin_y)
            prev = self._prev_cells.get(trip_id)

            if prev is None or prev != (row, col):
                pitch = pitch_for_row(row, self._n_rows)
                onset = int(rng.integers(0, max(n_samples, 1)))
                events.append((pitch, onset, row, col))

            self._prev_cells[trip_id] = (row, col)

        for trip_id in list(self._prev_cells):
            if trip_id not in seen:
                del self._prev_cells[trip_id]

        return events

    def render_blips(
        self,
        events: list[tuple[float, int, int, int]],
        n_samples: int,
    ) -> np.ndarray:
        """Mix blip events into a float32 array of length ``n_samples``.

        Auto-ducking: gain scales as ``1 / sqrt(n_events)`` to prevent
        clipping during dense rush-hour frames.
        """
        output = np.zeros(n_samples, dtype=np.float64)
        n = len(events)
        if n == 0:
            return output.astype(np.float32)

        gain = _BLIP_MASTER_GAIN / np.sqrt(n)
        attack_samples = max(1, round(self._sr * _BLIP_ATTACK_S))
        blip_cap = round(self._sr * 0.1)  # cap each blip at 100 ms

        for pitch_hz, onset, _row, _col in events:
            end = min(onset + blip_cap, n_samples)
            length = end - onset
            if length <= 0:
                continue

            t = np.arange(length, dtype=np.float64) / self._sr
            atk = min(attack_samples, length)
            env = np.empty(length, dtype=np.float64)
            env[:atk] = np.linspace(0.0, 1.0, atk)
            if atk < length:
                env[atk:] = np.exp(
                    -_BLIP_DECAY_RATE
                    * np.arange(length - atk, dtype=np.float64)
                    / self._sr
                )

            blip = np.zeros(length, dtype=np.float64)
            for h_idx, amp in enumerate(_BLIP_HARMONICS):
                blip += amp * np.sin(2.0 * np.pi * pitch_hz * (h_idx + 1) * t)

            output[onset:end] += env * blip * gain

        return output.astype(np.float32)


        self._sample_pos += n_samples
        return (output * _MASTER_GAIN).astype(np.float32)


# ---------------------------------------------------------------------------
# crosspath: proximity-crossing soft-pad synthesiser
# ---------------------------------------------------------------------------

CROSSPATH_PROXIMITY_DEFAULT: int = 200  # metres (Web Mercator)

_CROSSPATH_MAX_SPEED: float = 20.0  # m/s cap for pitch mapping

_PAD_ATTACK_S: float = 0.05     # 50 ms linear attack
_PAD_DECAY_RATE: float = 3.0    # exponential decay (s⁻¹); ~1 s audible tail
_PAD_HARMONICS: np.ndarray = np.array([1.0, 0.3], dtype=np.float64)
_PAD_MASTER_GAIN: float = 0.35  # per-blip peak gain before auto-duck scaling


def _pitch_for_speed(speed_ms: float) -> float:
    """Map average bus speed (m/s, clamped 0–20) to a C-major pentatonic frequency."""
    n_notes = len(_PENTA_FREQS)
    t = float(np.clip(speed_ms / _CROSSPATH_MAX_SPEED, 0.0, 1.0))
    idx = int(round(t * (n_notes - 1)))
    return float(_PENTA_FREQS[idx])


class CrosspathBlipSynth:
    """Stateful synthesiser for bus proximity-crossing events (crosspath mode).

    Two buses entering within ``proximity`` metres of each other (Web Mercator)
    triggers a soft-pad blip.  Events are edge-triggered: a pair fires once when
    it *enters* proximity, not on every subsequent frame it remains close.

    Per-frame usage::

        events = synth.feed_frame(frame_df, n_samples)
        audio  = synth.render_blips(events, n_samples)

    ``events`` is a list of ``(pitch_hz, onset_offset_samples, mid_x, mid_y)``.
    """

    def __init__(
        self,
        proximity: float = CROSSPATH_PROXIMITY_DEFAULT,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self._proximity = float(proximity)
        self._sr = int(sample_rate)
        self._prev_pairs: set[frozenset] = set()

    def reset(self) -> None:
        """Clear all tracked bus pairs (restart from the first frame)."""
        self._prev_pairs.clear()

    def feed_frame(
        self,
        frame_df,
        n_samples: int,
        seed: "int | None" = None,
    ) -> list[tuple[float, int, float, float]]:
        """Detect new bus-pair proximity crossings and return blip events.

        ``frame_df`` must have columns ``trip_id``, ``x``, ``y``, ``speed``.
        Returns a list of ``(pitch_hz, onset_offset_samples, mid_x, mid_y)``.
        If the frame is empty, returns [] without resetting pair state.
        """
        if frame_df is None or frame_df.empty or len(frame_df) < 2:
            return []

        # Deduplicate: one position per trip (latest raw_timestamp in the bin)
        sort_col = "raw_timestamp" if "raw_timestamp" in frame_df.columns else "timestamp"
        deduped = (
            frame_df.sort_values(sort_col)
            .groupby("trip_id", sort=False)
            .last()
            .reset_index()
        )

        if len(deduped) < 2:
            return []

        trip_ids = deduped["trip_id"].values.astype(str)
        x_arr = deduped["x"].values.astype(np.float64)
        y_arr = deduped["y"].values.astype(np.float64)
        speed_arr = deduped["speed"].values.astype(np.float64)
        speed_arr = np.where(np.isfinite(speed_arr), speed_arr, 0.0)

        dx = x_arr[:, None] - x_arr[None, :]
        dy = y_arr[:, None] - y_arr[None, :]
        dist_sq = dx * dx + dy * dy
        prox_sq = self._proximity * self._proximity

        i_arr, j_arr = np.where(np.triu(dist_sq < prox_sq, k=1))

        current_pairs: set[frozenset] = set()
        events: list[tuple[float, int, float, float]] = []
        rng = np.random.default_rng(seed)

        for i, j in zip(i_arr, j_arr):
            pair: frozenset = frozenset((trip_ids[i], trip_ids[j]))
            current_pairs.add(pair)
            if pair not in self._prev_pairs:
                avg_speed = (speed_arr[i] + speed_arr[j]) / 2.0
                pitch = _pitch_for_speed(avg_speed)
                onset = int(rng.integers(0, max(n_samples, 1)))
                mid_x = (x_arr[i] + x_arr[j]) / 2.0
                mid_y = (y_arr[i] + y_arr[j]) / 2.0
                events.append((pitch, onset, mid_x, mid_y))

        self._prev_pairs = current_pairs
        return events

    def render_blips(
        self,
        events: list[tuple[float, int, float, float]],
        n_samples: int,
    ) -> np.ndarray:
        """Mix soft-pad events into a float32 array of length ``n_samples``.

        Timbre: 50 ms linear attack, exponential decay (3 s⁻¹ ≈ 1 s tail),
        2 harmonics (fundamental + 2nd at 0.3 amp).
        Auto-ducking: gain scales as ``1 / sqrt(n_events)``.
        """
        output = np.zeros(n_samples, dtype=np.float64)
        n = len(events)
        if n == 0:
            return output.astype(np.float32)

        gain = _PAD_MASTER_GAIN / np.sqrt(n)
        attack_samples = max(1, round(self._sr * _PAD_ATTACK_S))

        for pitch_hz, onset, _mx, _my in events:
            length = n_samples - onset
            if length <= 0:
                continue

            t = np.arange(length, dtype=np.float64) / self._sr
            atk = min(attack_samples, length)
            env = np.empty(length, dtype=np.float64)
            env[:atk] = np.linspace(0.0, 1.0, atk)
            if atk < length:
                env[atk:] = np.exp(
                    -_PAD_DECAY_RATE
                    * np.arange(length - atk, dtype=np.float64)
                    / self._sr
                )

            blip = np.zeros(length, dtype=np.float64)
            for h_idx, amp in enumerate(_PAD_HARMONICS):
                blip += amp * np.sin(2.0 * np.pi * pitch_hz * (h_idx + 1) * t)

            output[onset:] += env * blip * gain

        # Fade the last 5 ms to zero to eliminate the pop caused by abrupt
        # truncation at the frame boundary.
        taper_len = min(n_samples, round(self._sr * 0.005))
        if taper_len > 0:
            output[-taper_len:] *= np.linspace(1.0, 0.0, taper_len)

        return output.astype(np.float32)

