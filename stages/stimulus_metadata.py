from __future__ import annotations

import csv
import json
import re

import numpy as np
import polars as pl

from pipeline_config import PipelineConfig
from stages.common import StageLogger, require_files


def _load_sequence(cfg: PipelineConfig) -> tuple[pl.DataFrame, str]:
    paths = [
        cfg.stim_cam_path / "stimulus_trials.csv",
        cfg.stim_cam_path / "somatosensory_stimulation" / "run_001_sequence.csv",
    ]
    sequence_path = next((path for path in paths if path.exists()), None)
    if sequence_path is not None:
        sequence = pl.read_csv(sequence_path)
    else:
        sequence_path = cfg.stim_cam_path / "session_snapshot.json"
        if not sequence_path.exists():
            raise FileNotFoundError("No stimulus sequence found in stimulus_trials.csv, run_001_sequence.csv, or session_snapshot.json.")
        resolved_plan = json.loads(sequence_path.read_text(encoding="utf-8"))["resolved_plan"]
        rows = []
        previous_position = None
        for index, step in enumerate(resolved_plan):
            kind = str(step.get("kind", "movement"))
            position_label = str(step.get("position_label", ""))
            position_cm = float(step.get("position_cm", 0.0))
            rotary_label = str(step.get("rotary_label", ""))
            repeat_index = int(step.get("repeat_index", 0))
            is_recalibration = kind == "recalibration"
            switching_required = is_recalibration or previous_position is None or position_cm != previous_position
            previous_position = None if is_recalibration else position_cm
            rows.append(
                {
                    **{key: value for key, value in step.items() if key != "rotation_duration_s"},
                    "global_step": int(step.get("global_index", index + 1)),
                    "stimulus_condition": kind,
                    "move_label": "" if is_recalibration else f"R{repeat_index} · {position_label} · {rotary_label}",
                    "pos_label": position_label,
                    "hardness": position_label,
                    "pos_cm": position_cm,
                    "rot_token": rotary_label,
                    "planned_rotation_duration_s": float(step.get("rotation_duration_s", 0.0)),
                    "rotation_event_expected": not is_recalibration and float(step.get("rotary_speed_cm_s", 0.0)) != 0.0,
                    "switching_required": switching_required,
                }
            )
        sequence = pl.DataFrame(rows)
    if "hardness" in sequence.columns and "pos_label" not in sequence.columns:
        sequence = sequence.with_columns(pl.col("hardness").alias("pos_label"))
    if "pos_label" in sequence.columns and "hardness" not in sequence.columns:
        sequence = sequence.with_columns(pl.col("pos_label").alias("hardness"))
    return sequence, str(sequence_path)


def _load_event_times(path) -> np.ndarray:
    if not path.read_text(encoding="utf-8", errors="replace").strip():
        return np.array([], dtype=float)
    times = np.loadtxt(path, ndmin=1).astype(float)
    times = times[np.isfinite(times)]
    return np.sort(times)


def _load_pulse_events(
    start_path,
    end_path,
    minimum_width_ms: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], np.ndarray, np.ndarray]:
    starts = _load_event_times(start_path)
    ends = _load_event_times(end_path)
    minimum_width_s = float(minimum_width_ms) / 1000.0
    kept_starts = []
    kept_ends = []
    recorded_starts = []
    recorded_ends = []
    rejected_short = 0
    orphan_ends = 0
    end_i = 0

    for start in starts:
        while end_i < len(ends) and ends[end_i] <= start:
            orphan_ends += 1
            end_i += 1
        if end_i >= len(ends):
            break
        end = ends[end_i]
        end_i += 1
        recorded_starts.append(float(start))
        recorded_ends.append(float(end))
        if end - start >= minimum_width_s:
            kept_starts.append(float(start))
            kept_ends.append(float(end))
        else:
            rejected_short += 1

    kept_starts_array = np.asarray(kept_starts, dtype=float)
    kept_ends_array = np.asarray(kept_ends, dtype=float)
    stats = {
        "input_starts": int(len(starts)),
        "input_ends": int(len(ends)),
        "kept_before_drop": int(len(kept_starts_array)),
        "rejected_short": int(rejected_short),
        "orphan_starts": int(len(starts) - len(kept_starts_array) - rejected_short),
        "orphan_ends": int(orphan_ends + len(ends) - end_i),
    }
    return (
        kept_starts_array,
        kept_ends_array,
        stats,
        np.asarray(recorded_starts, dtype=float),
        np.asarray(recorded_ends, dtype=float),
    )


def _rotation_trial_mask(sequence: pl.DataFrame) -> np.ndarray:
    if "stimulus_condition" not in sequence.columns:
        return np.ones(sequence.height, dtype=bool)
    return np.asarray(
        [str(value).strip().lower() != "recalibration" for value in sequence["stimulus_condition"].to_list()],
        dtype=bool,
    )


def _completed_move_trial_indices(sequence: pl.DataFrame, events_path) -> np.ndarray | None:
    if not events_path.exists():
        return None

    start_labels = []
    done_labels = []
    start_steps = []
    done_steps = []
    with events_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            event_type = row.get("event_type", "")
            message = row.get("message", "")
            if event_type == "movement_start" and row.get("index", "").strip():
                start_steps.append(int(row["index"]))
            elif event_type == "movement_done" and row.get("index", "").strip():
                done_steps.append(int(row["index"]))
            elif event_type == "move_start":
                match = re.search(r"\bMOVE_START\s+\d+/\d+\s+(.+?)\s*$", message)
                if match:
                    start_labels.append(match.group(1))
            elif event_type == "move_done":
                match = re.search(r"\bMOVE_DONE\s+\d+\s+(.+?)\s*$", message)
                if match:
                    done_labels.append(match.group(1))

    if start_steps or done_steps:
        if start_steps and done_steps and start_steps != done_steps:
            raise RuntimeError("events.csv movement_start and movement_done indices do not match; stimulus assignment is ambiguous.")
        if "global_step" not in sequence.columns:
            raise RuntimeError("Current events.csv format requires global_step in the stimulus sequence.")
        completed_steps = done_steps or start_steps
        sequence_steps = {int(value): index for index, value in enumerate(sequence["global_step"].to_list())}
        missing = [step for step in completed_steps if step not in sequence_steps]
        if missing:
            raise RuntimeError(f"Completed movement indices are absent from the stimulus sequence: {missing[:10]}")
        return np.asarray([sequence_steps[step] for step in completed_steps], dtype=int)

    if "move_label" not in sequence.columns:
        return None
    if start_labels and done_labels and start_labels != done_labels:
        raise RuntimeError("events.csv MOVE_START and MOVE_DONE labels do not match; stimulus assignment is ambiguous.")
    completed_labels = done_labels or start_labels
    if not completed_labels:
        return None

    sequence_labels = [str(value) for value in sequence["move_label"].to_list()]
    trial_indices = []
    search_from = 0
    for label in completed_labels:
        matches = [index for index in range(search_from, len(sequence_labels)) if sequence_labels[index] == label]
        if not matches:
            raise RuntimeError(f"Completed move {label!r} from events.csv has no matching later row in stimulus_trials.csv.")
        trial_index = matches[0]
        trial_indices.append(trial_index)
        search_from = trial_index + 1
    return np.asarray(trial_indices, dtype=int)


def _completed_calibration_counts(events_path) -> tuple[int, int] | None:
    if not events_path.exists():
        return None

    initial_calibrations = 0
    recalibrations = 0
    pending = None
    home_events_found = False
    with events_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("event_type", "") != "home_ok":
                continue
            home_events_found = True
            message = row.get("message", "")
            if "waiting HOME_OK (Recalibration)" in message:
                pending = "recalibration"
            elif "waiting HOME_OK (Calibration)" in message:
                pending = "initial"
            elif "ARDUINO: HOME_OK" in message:
                if pending == "recalibration":
                    recalibrations += 1
                elif pending == "initial":
                    initial_calibrations += 1
                pending = None
    return (initial_calibrations, recalibrations) if home_events_found else None


def _match_rotation_events(
    sequence: pl.DataFrame,
    completed_trials: np.ndarray,
    events_path,
    rotation_starts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    if "rotation_event_expected" in sequence.columns:
        expected_mask = sequence["rotation_event_expected"].to_numpy()
        expected_trials = completed_trials[expected_mask[completed_trials]]
    else:
        expected_trials = completed_trials
    if len(rotation_starts) == len(expected_trials):
        return expected_trials, np.arange(len(rotation_starts)), len(expected_trials)

    required_columns = {"global_step", "interval_before_rotation_s"}
    if not required_columns.issubset(sequence.columns) or not events_path.exists():
        raise RuntimeError(
            "Rotation-pulse assignment is not one-to-one and the sequence lacks current-GUI timing metadata."
        )

    movement_start_times = {}
    with events_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("event_type") == "movement_start" and row.get("index", "").strip():
                movement_start_times[int(row["index"])] = np.datetime64(row["timestamp"]).astype("datetime64[ms]").astype(np.int64) / 1000.0

    predicted_times = []
    for trial in expected_trials:
        global_step = int(sequence["global_step"][int(trial)])
        if global_step not in movement_start_times:
            raise RuntimeError(f"events.csv has no movement_start timestamp for global step {global_step}.")
        predicted_times.append(
            movement_start_times[global_step] + float(sequence["interval_before_rotation_s"][int(trial)])
        )
    predicted_times = np.asarray(predicted_times)
    if predicted_times.size == 0 or rotation_starts.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), len(expected_trials)

    tolerance_s = 5.0

    def pairs_for_offset(offset: float) -> tuple[list[tuple[int, int]], float]:
        predicted_i = 0
        event_i = 0
        pairs = []
        error = 0.0
        while predicted_i < len(predicted_times) and event_i < len(rotation_starts):
            delta = rotation_starts[event_i] + offset - predicted_times[predicted_i]
            if abs(delta) <= tolerance_s:
                pairs.append((predicted_i, event_i))
                error += abs(float(delta))
                predicted_i += 1
                event_i += 1
            elif delta < 0:
                event_i += 1
            else:
                predicted_i += 1
        return pairs, error

    candidates = (
        predicted_times[: min(10, len(predicted_times)), None]
        - rotation_starts[None, : min(10, len(rotation_starts))]
    ).ravel()
    best_pairs = []
    best_score = (-1, -np.inf)
    for offset in candidates:
        pairs, error = pairs_for_offset(float(offset))
        score = (len(pairs), -error)
        if score > best_score:
            best_pairs = pairs
            best_score = score

    predicted_indices = np.asarray([pair[0] for pair in best_pairs], dtype=int)
    event_indices = np.asarray([pair[1] for pair in best_pairs], dtype=int)
    return expected_trials[predicted_indices], event_indices, len(expected_trials)


def _movement_done_times(events_path) -> dict[int, float]:
    times = {}
    if not events_path.exists():
        return times
    with events_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("event_type") == "movement_done" and row.get("index", "").strip():
                times[int(row["index"])] = (
                    np.datetime64(row["timestamp"])
                    .astype("datetime64[us]")
                    .astype(np.int64)
                    / 1_000_000.0
                )
    return times


def _add_controller_reconstructed_timing(
    sequence: pl.DataFrame,
    timing: pl.DataFrame,
    events_path,
    logger: StageLogger,
) -> pl.DataFrame:
    speed_column = "rotary_speed_cm_s" if "rotary_speed_cm_s" in sequence.columns else "rot_cm_s"
    required = {"global_step", speed_column, "planned_rotation_duration_s"}
    done_times = _movement_done_times(events_path)
    if not required.issubset(sequence.columns) or not done_times:
        return timing.with_columns(
            stimulus_active_start_s=pl.col("rotation_start_s"),
            stimulus_active_end_s=pl.col("rotation_end_s"),
            stimulus_active_duration_s=pl.col("rotation_end_s") - pl.col("rotation_start_s"),
            stimulus_timing_source=pl.when(pl.col("rotation_start_s").is_finite())
            .then(pl.lit("nidq_ttl"))
            .otherwise(pl.lit("")),
            stimulus_timing_reconstructed=pl.lit(False),
            stimulus_timing_note=pl.lit(""),
        )

    n = sequence.height
    steps = np.asarray(sequence["global_step"].to_list(), dtype=int)
    speed = np.asarray(sequence[speed_column].cast(pl.Float64).to_list(), dtype=float)
    planned_duration = np.asarray(
        sequence["planned_rotation_duration_s"].cast(pl.Float64).to_list(), dtype=float
    )
    wall_done = np.asarray([done_times.get(step, np.nan) for step in steps], dtype=float)
    raw_start = np.asarray(timing["rotation_start_s"].to_list(), dtype=float)
    raw_end = np.asarray(timing["rotation_end_s"].to_list(), dtype=float)
    measured = np.isfinite(wall_done) & np.isfinite(raw_end)
    if measured.sum() < 2:
        return timing.with_columns(
            stimulus_active_start_s=pl.col("rotation_start_s"),
            stimulus_active_end_s=pl.col("rotation_end_s"),
            stimulus_active_duration_s=pl.col("rotation_end_s") - pl.col("rotation_start_s"),
            stimulus_timing_source=pl.when(pl.col("rotation_start_s").is_finite())
            .then(pl.lit("nidq_ttl"))
            .otherwise(pl.lit("")),
            stimulus_timing_reconstructed=pl.lit(False),
            stimulus_timing_note=pl.lit(""),
        )

    origin = float(np.nanmin(wall_done[measured]))
    fit_mask = measured.copy()
    for _ in range(4):
        slope, intercept = np.polyfit(wall_done[fit_mask] - origin, raw_end[fit_mask], 1)
        residual = raw_end - (slope * (wall_done - origin) + intercept)
        center = float(np.nanmedian(residual[fit_mask]))
        mad = float(np.nanmedian(np.abs(residual[fit_mask] - center)))
        next_mask = measured & (np.abs(residual - center) <= max(0.01, 8.0 * 1.4826 * mad))
        if next_mask.sum() < 2 or np.array_equal(next_mask, fit_mask):
            break
        fit_mask = next_mask
    slope, intercept = np.polyfit(wall_done[fit_mask] - origin, raw_end[fit_mask], 1)
    predicted_end = slope * (wall_done - origin) + intercept
    residual = raw_end[fit_mask] - predicted_end[fit_mask]

    raw_duration = raw_end - raw_start
    expected_duration = planned_duration.copy()
    for value in np.unique(speed[np.isfinite(speed) & (speed > 0)]):
        group = measured & np.isclose(speed, value) & np.isfinite(raw_duration) & np.isfinite(planned_duration)
        if group.any():
            expected_duration[np.isclose(speed, value)] += float(
                np.nanmedian(raw_duration[group] - planned_duration[group])
            )

    duration_tolerance = np.maximum(0.05, 0.05 * expected_duration)
    complete_ttl = (
        np.isfinite(raw_start)
        & np.isfinite(raw_end)
        & np.isfinite(expected_duration)
        & (np.abs(raw_duration - expected_duration) <= duration_tolerance)
    )
    active_start = raw_start.copy()
    active_end = raw_end.copy()
    source = np.full(n, "", dtype=object)
    source[complete_ttl] = "nidq_ttl"
    reconstructed = np.zeros(n, dtype=bool)
    note = np.full(n, "", dtype=object)
    missing_steps = []
    truncated_steps = []
    silent_steps = []

    for index in np.flatnonzero(np.isfinite(wall_done)):
        if speed[index] <= 0 and planned_duration[index] > 0:
            active_end[index] = predicted_end[index]
            active_start[index] = predicted_end[index] - planned_duration[index]
            source[index] = "controller_aligned_silent_control"
            reconstructed[index] = True
            note[index] = "Intentional zero-speed window reconstructed from controller completion time."
            silent_steps.append(int(steps[index]))
        elif speed[index] > 0 and not complete_ttl[index] and np.isfinite(expected_duration[index]):
            if np.isfinite(raw_end[index]) and abs(raw_end[index] - predicted_end[index]) <= 0.1:
                active_end[index] = raw_end[index]
                source[index] = "controller_reconstructed_truncated_ttl"
                note[index] = "Rotary TTL onset was truncated; onset reconstructed from controller-aligned end and normal speed-specific duration."
                truncated_steps.append(int(steps[index]))
            else:
                active_end[index] = predicted_end[index]
                source[index] = "controller_reconstructed_missing_ttl"
                note[index] = "Rotary TTL was absent; window reconstructed from controller completion time and normal speed-specific duration."
                missing_steps.append(int(steps[index]))
            active_start[index] = active_end[index] - expected_duration[index]
            reconstructed[index] = True

    logger.output(
        "controller-to-probe timing reconstruction: "
        f"{int(fit_mask.sum())} measured endpoint(s), residual SD {np.std(residual) * 1000.0:.2f} ms; "
        f"{len(missing_steps)} absent TTL, {len(truncated_steps)} truncated TTL, "
        f"{len(silent_steps)} intentional silent window(s) reconstructed"
    )
    if missing_steps or truncated_steps:
        logger.output(
            "reconstructed positive-speed global step(s): "
            f"absent={missing_steps or 'none'}, truncated={truncated_steps or 'none'}"
        )

    return timing.with_columns(
        pl.Series("time_start_s", active_start),
        pl.Series("time_end_s", active_end),
        pl.Series("total_duration_s", active_end - active_start),
        pl.Series("stimulus_active_start_s", active_start),
        pl.Series("stimulus_active_end_s", active_end),
        pl.Series("stimulus_active_duration_s", active_end - active_start),
        pl.Series("expected_rotation_ttl_duration_s", expected_duration),
        pl.Series("stimulus_timing_source", source),
        pl.Series("stimulus_timing_reconstructed", reconstructed),
        pl.Series("stimulus_timing_note", note),
    )


def _assign_switching_pulses_by_rotation_gaps(
    n_trials: int,
    switching_expected: np.ndarray,
    rotation_trial: np.ndarray,
    rotation_starts: np.ndarray,
    rotation_ends: np.ndarray,
    switching_starts: np.ndarray,
    switching_ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    assigned_starts = np.full(n_trials, np.nan)
    assigned_ends = np.full(n_trials, np.nan)
    used = np.zeros(len(switching_starts), dtype=bool)
    rejected_sequence_incompatible = 0
    startup_calibration_pulses = 0
    previous_trial = -1
    previous_rotation_end = -np.inf

    for move_index, trial_index in enumerate(rotation_trial):
        candidates = np.flatnonzero(
            (~used)
            & (switching_starts >= previous_rotation_end)
            & (switching_starts < rotation_starts[move_index])
        )
        expected = np.flatnonzero(switching_expected[previous_trial + 1 : trial_index + 1]) + previous_trial + 1

        if move_index == 0 and len(candidates) >= len(expected):
            if len(expected):
                selected = candidates[-len(expected) :]
                assigned_starts[expected] = switching_starts[selected]
                assigned_ends[expected] = switching_ends[selected]
            startup = candidates[: len(candidates) - len(expected)]
            startup_calibration_pulses += len(startup)
            used[candidates] = True
        elif len(expected) == 0:
            used[candidates] = True
            rejected_sequence_incompatible += len(candidates)
        elif len(candidates) != len(expected):
            used[candidates] = True
            rejected_sequence_incompatible += len(candidates)
        else:
            assigned_starts[expected] = switching_starts[candidates]
            assigned_ends[expected] = switching_ends[candidates]
            used[candidates] = True

        previous_trial = int(trial_index)
        previous_rotation_end = float(rotation_ends[move_index])

    trailing_expected = np.flatnonzero(switching_expected[previous_trial + 1 :]) + previous_trial + 1
    trailing_candidates = np.flatnonzero((~used) & (switching_starts >= previous_rotation_end))
    if len(trailing_candidates) == len(trailing_expected):
        assigned_starts[trailing_expected] = switching_starts[trailing_candidates]
        assigned_ends[trailing_expected] = switching_ends[trailing_candidates]
        used[trailing_candidates] = True
    else:
        used[trailing_candidates] = True
        rejected_sequence_incompatible += len(trailing_candidates)

    rejected_sequence_incompatible += int((~used).sum())
    return assigned_starts, assigned_ends, {
        "matched_starts": int(np.isfinite(assigned_starts).sum()),
        "matched_ends": int(np.isfinite(assigned_ends).sum()),
        "missing_starts": int(switching_expected.sum() - np.isfinite(assigned_starts).sum()),
        "missing_ends": int(switching_expected.sum() - np.isfinite(assigned_ends).sum()),
        "unused_starts": int(rejected_sequence_incompatible),
        "unused_ends": int(rejected_sequence_incompatible),
        "startup_calibration_pulses": int(startup_calibration_pulses),
    }


def _align_start_end_events_to_trials(
    n_trials: int,
    start_events: np.ndarray,
    end_events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    starts = np.full(n_trials, np.nan)
    ends = np.full(n_trials, np.nan)
    events = sorted(
        [(float(t), "start") for t in start_events]
        + [(float(t), "end") for t in end_events],
        key=lambda item: (item[0], 0 if item[1] == "start" else 1),
    )

    event_i = 0
    for trial_i in range(n_trials):
        if event_i >= len(events):
            break

        t, kind = events[event_i]
        if kind == "start":
            starts[trial_i] = t
            event_i += 1
            if event_i < len(events) and events[event_i][1] == "end":
                ends[trial_i] = events[event_i][0]
                event_i += 1
        else:
            ends[trial_i] = t
            event_i += 1

    unused = events[event_i:]
    stats = {
        "start_events": int(len(start_events)),
        "end_events": int(len(end_events)),
        "matched_starts": int(np.isfinite(starts).sum()),
        "matched_ends": int(np.isfinite(ends).sum()),
        "missing_starts": int(np.isnan(starts).sum()),
        "missing_ends": int(np.isnan(ends).sum()),
        "unused_starts": int(sum(kind == "start" for _t, kind in unused)),
        "unused_ends": int(sum(kind == "end" for _t, kind in unused)),
    }
    return starts, ends, stats


def _camera_trigger_groups(camera_times: np.ndarray, gap_s: float = 0.5) -> list[np.ndarray]:
    if camera_times.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(camera_times) > gap_s) + 1
    groups = np.split(camera_times, breaks)
    return [group for group in groups if group.size >= 2]


def _load_repaired_move_windows(cfg: PipelineConfig, tprime, sequence: pl.DataFrame, logger: StageLogger) -> pl.DataFrame | None:
    moves_path = cfg.stim_cam_path / "completed_moves_repaired.csv"
    if not moves_path.exists():
        return None

    camera_path = tprime / "camera_trigger_probe_time.txt"
    require_files([camera_path], "Repaired stim/cam input needs camera trigger times for window-based TTL assignment.")

    moves = pl.read_csv(moves_path)
    required_columns = {
        "source_segment_index",
        "source_start_relative_to_first_camera_frame_s",
        "source_end_relative_to_first_camera_frame_s",
    }
    missing = required_columns.difference(moves.columns)
    if missing:
        raise ValueError(f"{moves_path.name} is missing required columns: {sorted(missing)}")
    match_key = "repair_trial_index" if "repair_trial_index" in moves.columns and "repair_trial_index" in sequence.columns else "global_step"
    if match_key not in moves.columns or match_key not in sequence.columns:
        raise ValueError(f"Repaired stim/cam input requires {match_key} in both stimulus_trials.csv and {moves_path.name}.")

    camera_times = _load_event_times(camera_path, 0)
    groups = _camera_trigger_groups(camera_times)
    segment_ids = sorted(int(value) for value in moves["source_segment_index"].unique().to_list())
    if len(groups) < len(segment_ids):
        raise RuntimeError(
            "Could not map repaired source segments to camera trigger sessions. "
            f"Found {len(groups)} camera sessions but repaired metadata has {len(segment_ids)} source segments."
        )

    session_start_by_segment = {segment_id: float(groups[index][0]) for index, segment_id in enumerate(segment_ids)}
    rows = []
    for row in moves.iter_rows(named=True):
        segment_id = int(row["source_segment_index"])
        start_rel = float(row["source_start_relative_to_first_camera_frame_s"])
        end_rel = float(row["source_end_relative_to_first_camera_frame_s"])
        session_start = session_start_by_segment[segment_id]
        rows.append(
            {
                "match_key": int(row[match_key]),
                "move_window_start_s": session_start + start_rel,
                "move_window_end_s": session_start + end_rel,
                "source_segment_index": segment_id,
                "source_move_index": int(row["source_move_index"]) if row.get("source_move_index") not in (None, "") else None,
            }
        )

    logger.output(
        "repair metadata detected: assigning TTLs by completed move windows "
        f"from {moves_path.name} and {camera_path.name}"
    )
    return pl.DataFrame(rows)


def _first_event_pair_in_window(
    start_events: np.ndarray,
    end_events: np.ndarray,
    window_start: float,
    window_end: float,
    *,
    pad_s: float = 0.25,
) -> tuple[float, float]:
    lo = window_start - pad_s
    hi = window_end + pad_s
    starts = start_events[(start_events >= lo) & (start_events <= hi)]
    ends = end_events[(end_events >= lo) & (end_events <= hi)]
    start = float(starts[0]) if starts.size else np.nan
    if ends.size:
        if np.isfinite(start):
            after_start = ends[ends >= start]
            end = float(after_start[0]) if after_start.size else float(ends[0])
        else:
            end = float(ends[0])
    else:
        end = np.nan
    return start, end


def _window_based_timing(
    sequence: pl.DataFrame,
    windows: pl.DataFrame,
    rotation_start_events: np.ndarray,
    rotation_end_events: np.ndarray,
    switching_start_events: np.ndarray,
    switching_end_events: np.ndarray,
) -> tuple[pl.DataFrame, dict[str, int]]:
    if "switching_required" in sequence.columns:
        position_change = sequence["switching_required"].cast(pl.Boolean).to_numpy()
    else:
        pos = sequence["pos_cm"].to_numpy()
        position_change = np.r_[False, pos[1:] != pos[:-1]]

    n = sequence.height
    match_key = "repair_trial_index" if "repair_trial_index" in sequence.columns else "global_step"
    window_by_key = {int(row["match_key"]): row for row in windows.iter_rows(named=True)}
    rotation_start_s = np.full(n, np.nan)
    rotation_end_s = np.full(n, np.nan)
    switching_start = np.full(n, np.nan)
    switching_end = np.full(n, np.nan)
    window_start = np.full(n, np.nan)
    window_end = np.full(n, np.nan)

    for trial_i, row in enumerate(sequence.iter_rows(named=True)):
        window = window_by_key.get(int(row[match_key]))
        if window is None:
            continue
        start_s = float(window["move_window_start_s"])
        end_s = float(window["move_window_end_s"])
        window_start[trial_i] = start_s
        window_end[trial_i] = end_s
        rotation_start_s[trial_i], rotation_end_s[trial_i] = _first_event_pair_in_window(
            rotation_start_events,
            rotation_end_events,
            start_s,
            end_s,
        )
        switching_start[trial_i], switching_end[trial_i] = _first_event_pair_in_window(
            switching_start_events,
            switching_end_events,
            start_s,
            end_s,
        )

    time_start_s = rotation_start_s.copy()
    time_end_s = rotation_end_s.copy()

    timing = pl.DataFrame(
        {
            "trial_index": np.arange(n),
            "position_change": position_change,
            "time_start_s": time_start_s,
            "time_end_s": time_end_s,
            "switching_start_s": switching_start,
            "switching_end_s": switching_end,
            "rotation_start_s": rotation_start_s,
            "rotation_end_s": rotation_end_s,
            "move_window_start_s": window_start,
            "move_window_end_s": window_end,
            "rotation_start_matched": np.isfinite(rotation_start_s),
            "rotation_end_matched": np.isfinite(rotation_end_s),
            "switching_start_matched": np.isfinite(switching_start),
            "switching_end_matched": np.isfinite(switching_end),
            "repair_window_matched": np.isfinite(window_start) & np.isfinite(window_end),
        }
    ).with_columns(
        switching_duration_s=pl.col("switching_end_s") - pl.col("switching_start_s"),
        rotation_duration_s=pl.col("rotation_end_s") - pl.col("rotation_start_s"),
        total_duration_s=pl.col("time_end_s") - pl.col("time_start_s"),
    )
    stats = {
        "window_matched": int(np.isfinite(window_start).sum()),
        "rotation_start_matched": int(np.isfinite(rotation_start_s).sum()),
        "rotation_end_matched": int(np.isfinite(rotation_end_s).sum()),
        "switching_start_matched": int(np.isfinite(switching_start).sum()),
        "switching_end_matched": int(np.isfinite(switching_end).sum()),
    }
    return timing, stats


def build_stimulus_metadata(cfg: PipelineConfig, logger: StageLogger) -> None:
    tprime = cfg.catgt_root / "tprime"
    required_tprime_files = [
        tprime / "rotation_start_probe_time.txt",
        tprime / "rotation_end_probe_time.txt",
        tprime / "switching_start_probe_time.txt",
        tprime / "switching_end_probe_time.txt",
    ]
    if (cfg.stim_cam_path / "completed_moves_repaired.csv").exists():
        required_tprime_files.append(tprime / "camera_trigger_probe_time.txt")
    require_files(required_tprime_files, "Missing TPrime event files. Re-run TPrime first.")

    logger.log("Building stimulus metadata table")
    sequence, sequence_path = _load_sequence(cfg)
    logger.output(f"sequence file: {sequence_path}")
    sequence.write_csv(tprime / "stimulus_trials.csv")

    n = sequence.height
    repaired_windows = _load_repaired_move_windows(cfg, tprime, sequence, logger)
    events_path = cfg.stim_cam_path / "events.csv"
    rotation_start_events, rotation_end_events, rotation_filter_stats, recorded_rotation_starts, recorded_rotation_ends = _load_pulse_events(
        tprime / "rotation_start_probe_time.txt",
        tprime / "rotation_end_probe_time.txt",
        cfg.minimum_rotation_pulse_ms,
    )
    switching_start_events, switching_end_events, switching_filter_stats, recorded_switching_starts, recorded_switching_ends = _load_pulse_events(
        tprime / "switching_start_probe_time.txt",
        tprime / "switching_end_probe_time.txt",
        cfg.minimum_switching_pulse_ms,
    )
    logger.output(
        "rotation pulse data: "
        f"{rotation_filter_stats['kept_before_drop']} pair(s) at/above the configured "
        f"{cfg.minimum_rotation_pulse_ms:.1f} ms analysis threshold; "
        f"all {len(recorded_rotation_starts)} recorded pair(s) packaged"
    )
    logger.output(
        "switching pulse data: "
        f"{switching_filter_stats['kept_before_drop']} pair(s) at/above the configured "
        f"{cfg.minimum_switching_pulse_ms:.1f} ms analysis threshold; "
        f"all {len(recorded_switching_starts)} recorded pair(s) packaged"
    )

    if repaired_windows is not None:
        timing, window_stats = _window_based_timing(
            sequence,
            repaired_windows,
            rotation_start_events,
            rotation_end_events,
            switching_start_events,
            switching_end_events,
        )
        total_duration_s = timing["total_duration_s"].to_numpy()
    else:
        if "switching_required" in sequence.columns:
            position_change = sequence["switching_required"].cast(pl.Boolean).to_numpy()
        else:
            pos = sequence["pos_cm"].to_numpy()
            position_change = np.r_[False, pos[1:] != pos[:-1]]

        completed_trials = _completed_move_trial_indices(sequence, events_path)
        if completed_trials is None:
            completed_trials = np.flatnonzero(_rotation_trial_mask(sequence))
            rotation_assignment_source = "stimulus condition sequence"
        else:
            rotation_assignment_source = "completed movements in events.csv"
        rotation_trial, rotation_event_indices, expected_rotation_count = _match_rotation_events(
            sequence,
            completed_trials,
            events_path,
            rotation_start_events,
        )
        matched_rotation_starts = rotation_start_events[rotation_event_indices]
        matched_rotation_ends = rotation_end_events[rotation_event_indices]
        logger.output(f"rotation assignment source: {rotation_assignment_source}")
        rotation_start_s = np.full(n, np.nan)
        rotation_end_s = np.full(n, np.nan)
        rotation_start_s[rotation_trial] = matched_rotation_starts
        rotation_end_s[rotation_trial] = matched_rotation_ends
        rotation_stats = {
            "matched_starts": len(rotation_trial),
            "matched_ends": len(rotation_trial),
            "missing_starts": expected_rotation_count - len(rotation_trial),
            "missing_ends": expected_rotation_count - len(rotation_trial),
            "unused_starts": len(rotation_start_events) - len(rotation_event_indices),
            "unused_ends": len(rotation_end_events) - len(rotation_event_indices),
        }

        recalibration_rows = np.flatnonzero(~_rotation_trial_mask(sequence))
        calibration_counts = _completed_calibration_counts(events_path)
        if calibration_counts is None:
            initial_calibrations = 0
            executed_recalibration_rows = recalibration_rows[recalibration_rows < completed_trials[-1]]
            calibration_source = "sequence rows before the last completed move"
        else:
            initial_calibrations, completed_recalibrations = calibration_counts
            if completed_recalibrations > len(recalibration_rows):
                raise RuntimeError(
                    f"events.csv records {completed_recalibrations} completed recalibrations, "
                    f"but stimulus_trials.csv contains only {len(recalibration_rows)} recalibration rows."
                )
            executed_recalibration_rows = recalibration_rows[:completed_recalibrations]
            calibration_source = "completed HOME_OK events"

        switching_expected = np.zeros(n, dtype=bool)
        switching_expected[completed_trials] = position_change[completed_trials]
        switching_expected[executed_recalibration_rows] = True
        recalibration_executed = np.zeros(n, dtype=bool)
        recalibration_executed[executed_recalibration_rows] = True
        sequence_row_executed = np.zeros(n, dtype=bool)
        sequence_row_executed[completed_trials] = True
        sequence_row_executed[executed_recalibration_rows] = True
        logger.output(
            "calibration assignment: "
            f"{initial_calibrations} initial calibration(s), "
            f"{len(executed_recalibration_rows)}/{len(recalibration_rows)} planned recalibration(s) completed; "
            f"source={calibration_source}"
        )

        switching_start, switching_end, switch_stats = _assign_switching_pulses_by_rotation_gaps(
            n,
            switching_expected,
            rotation_trial,
            matched_rotation_starts,
            matched_rotation_ends,
            switching_start_events,
            switching_end_events,
        )

        time_start_s = rotation_start_s.copy()
        time_end_s = rotation_end_s.copy()
        total_duration_s = rotation_end_s - time_start_s
        timing = pl.DataFrame(
            {
                "trial_index": np.arange(n),
                "position_change": position_change,
                "sequence_row_executed": sequence_row_executed,
                "switching_event_expected": switching_expected,
                "recalibration_executed": recalibration_executed,
                "time_start_s": time_start_s,
                "time_end_s": time_end_s,
                "switching_start_s": switching_start,
                "switching_end_s": switching_end,
                "rotation_start_s": rotation_start_s,
                "rotation_end_s": rotation_end_s,
                "rotation_start_matched": np.isfinite(rotation_start_s),
                "rotation_end_matched": np.isfinite(rotation_end_s),
                "switching_start_matched": np.isfinite(switching_start),
                "switching_end_matched": np.isfinite(switching_end),
            }
        ).with_columns(
            switching_duration_s=pl.col("switching_end_s") - pl.col("switching_start_s"),
            rotation_duration_s=pl.col("rotation_end_s") - pl.col("rotation_start_s"),
            total_duration_s=pl.col("time_end_s") - pl.col("time_start_s"),
        )

    timing = _add_controller_reconstructed_timing(sequence, timing, events_path, logger)

    assignment_maps = {}
    for event_type, column in [("rotation", "rotation_start_s"), ("switching", "switching_start_s")]:
        assignment_maps[event_type] = {
            float(value): trial_index
            for trial_index, value in enumerate(timing[column].to_list())
            if value is not None and np.isfinite(value)
        }
    event_rows = []
    for event_type, starts, ends, analysis_starts in [
        ("rotation", recorded_rotation_starts, recorded_rotation_ends, rotation_start_events),
        ("switching", recorded_switching_starts, recorded_switching_ends, switching_start_events),
    ]:
        analysis_start_set = set(analysis_starts.tolist())
        for event_index, (start_s, end_s) in enumerate(zip(starts, ends)):
            sequence_row = assignment_maps[event_type].get(float(start_s))
            event_rows.append(
                {
                    "event_type": event_type,
                    "event_index": event_index,
                    "start_s": float(start_s),
                    "end_s": float(end_s),
                    "duration_s": float(end_s - start_s),
                    "included_in_analysis": float(start_s) in analysis_start_set,
                    "sequence_row": sequence_row,
                    "global_step": (
                        int(sequence["global_step"][sequence_row])
                        if sequence_row is not None and "global_step" in sequence.columns
                        else None
                    ),
                    "move_label": (
                        str(sequence["move_label"][sequence_row])
                        if sequence_row is not None and "move_label" in sequence.columns
                        else None
                    ),
                }
            )
    stimulus_events = pl.DataFrame(event_rows)
    stimulus_events.write_csv(tprime / "stimulus_event_table.csv")
    stimulus_events.write_parquet(tprime / "stimulus_event_table.parquet")

    bad_duration = np.isfinite(total_duration_s) & (total_duration_s < 0)
    if bad_duration.any():
        bad_trials = np.flatnonzero(bad_duration)[:10].tolist()
        raise RuntimeError(
            f"Stimulus metadata has {int(bad_duration.sum())} trial(s) with negative total duration. "
            f"First bad trials: {bad_trials}"
        )

    stimulus_metadata = pl.concat(
        [sequence.with_columns(pl.col("rot_token").alias("rot_taken")), timing],
        how="horizontal",
    )

    stimulus_metadata.write_csv(tprime / "stimulus_metadata_table.csv")
    stimulus_metadata.write_parquet(tprime / "stimulus_metadata_table.parquet")
    logger.output(f"sequence rows: {sequence.height}")
    if repaired_windows is not None:
        logger.output(f"repair move windows: {window_stats['window_matched']}/{n} matched")
        logger.output(f"rotation starts in repaired windows: {window_stats['rotation_start_matched']}/{n} matched")
        logger.output(f"rotation ends in repaired windows: {window_stats['rotation_end_matched']}/{n} matched")
        logger.output(f"switching starts in repaired windows: {window_stats['switching_start_matched']}/{n} matched")
        logger.output(f"switching ends in repaired windows: {window_stats['switching_end_matched']}/{n} matched")
    else:
        logger.output(
            "rotation timing: "
            f"assigned {rotation_stats['matched_starts']} recorded pulse pair(s) to sequence rows; "
            f"packaged all {len(rotation_start_events)} analysis pulse pair(s)"
        )
        logger.output(
            "switching timing: "
            f"assigned {switch_stats['matched_starts']} recorded pulse pair(s) to sequence rows; "
            f"packaged all {len(switching_start_events)} analysis pulse pair(s)"
        )
        logger.output(
            f"startup/calibration pulses before first stimulus: {switch_stats['startup_calibration_pulses']}"
        )
        logger.output(f"position-change trials: {int(position_change.sum())}")
    logger.log(f"Stimulus metadata saved: {tprime / 'stimulus_metadata_table.parquet'}")
