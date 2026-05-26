from __future__ import annotations

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
    if sequence_path is None:
        raise FileNotFoundError("No stimulus sequence file found in new or old GUI format.")

    sequence = pl.read_csv(sequence_path)
    if "hardness" in sequence.columns and "pos_label" not in sequence.columns:
        sequence = sequence.with_columns(pl.col("hardness").alias("pos_label"))
    if "pos_label" in sequence.columns and "hardness" not in sequence.columns:
        sequence = sequence.with_columns(pl.col("pos_label").alias("hardness"))
    return sequence, str(sequence_path)


def _load_event_times(path, drop_first: int) -> np.ndarray:
    if not path.read_text(encoding="utf-8", errors="replace").strip():
        return np.array([], dtype=float)
    times = np.loadtxt(path, ndmin=1).astype(float)
    times = times[np.isfinite(times)]
    return np.sort(times[int(drop_first) :])


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

    time_start_s = np.where(
        np.isfinite(switching_start),
        switching_start,
        np.where(np.isfinite(rotation_start_s), rotation_start_s, window_start),
    )
    time_end_s = np.where(
        np.isfinite(rotation_end_s),
        rotation_end_s,
        np.where(np.isfinite(switching_end), switching_end, window_end),
    )

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

    n = sequence.height
    repaired_windows = _load_repaired_move_windows(cfg, tprime, sequence, logger)

    if repaired_windows is not None:
        rotation_start_events = _load_event_times(tprime / "rotation_start_probe_time.txt", 0)
        rotation_end_events = _load_event_times(tprime / "rotation_end_probe_time.txt", 0)
        switching_start_events = _load_event_times(tprime / "switching_start_probe_time.txt", 0)
        switching_end_events = _load_event_times(tprime / "switching_end_probe_time.txt", 0)
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
        rotation_start_events = _load_event_times(
            tprime / "rotation_start_probe_time.txt",
            cfg.drop_first_rotation_intervals,
        )
        rotation_end_events = _load_event_times(
            tprime / "rotation_end_probe_time.txt",
            cfg.drop_first_rotation_intervals,
        )
        switching_start_events = _load_event_times(
            tprime / "switching_start_probe_time.txt",
            cfg.drop_first_switching_intervals,
        )
        switching_end_events = _load_event_times(
            tprime / "switching_end_probe_time.txt",
            cfg.drop_first_switching_intervals,
        )

        if "switching_required" in sequence.columns:
            position_change = sequence["switching_required"].cast(pl.Boolean).to_numpy()
        else:
            pos = sequence["pos_cm"].to_numpy()
            position_change = np.r_[False, pos[1:] != pos[:-1]]

        rotation_start_s, rotation_end_s, rotation_stats = _align_start_end_events_to_trials(
            n,
            rotation_start_events,
            rotation_end_events,
        )

        switch_trial = np.flatnonzero(position_change)
        switch_starts, switch_ends, switch_stats = _align_start_end_events_to_trials(
            len(switch_trial),
            switching_start_events,
            switching_end_events,
        )
        switching_start = np.full(n, np.nan)
        switching_end = np.full(n, np.nan)
        switching_start[switch_trial] = switch_starts
        switching_end[switch_trial] = switch_ends

        time_start_s = np.where(position_change, switching_start, rotation_start_s)
        total_duration_s = rotation_end_s - time_start_s
        timing = pl.DataFrame(
            {
                "trial_index": np.arange(n),
                "position_change": position_change,
                "time_start_s": time_start_s,
                "time_end_s": rotation_end_s,
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
            "rotation starts: "
            f"{rotation_stats['matched_starts']}/{n} matched, "
            f"{rotation_stats['missing_starts']} missing, "
            f"{rotation_stats['unused_starts']} unused"
        )
        logger.output(
            "rotation ends: "
            f"{rotation_stats['matched_ends']}/{n} matched, "
            f"{rotation_stats['missing_ends']} missing, "
            f"{rotation_stats['unused_ends']} unused"
        )
        logger.output(
            "switching starts: "
            f"{switch_stats['matched_starts']}/{len(switch_trial)} matched, "
            f"{switch_stats['missing_starts']} missing, "
            f"{switch_stats['unused_starts']} unused"
        )
        logger.output(
            "switching ends: "
            f"{switch_stats['matched_ends']}/{len(switch_trial)} matched, "
            f"{switch_stats['missing_ends']} missing, "
            f"{switch_stats['unused_ends']} unused"
        )
        logger.output(f"position-change trials: {int(position_change.sum())}")
    logger.log(f"Stimulus metadata saved: {tprime / 'stimulus_metadata_table.parquet'}")
