from __future__ import annotations

import shutil
import json
from pathlib import Path

import numpy as np
import polars as pl

from pipeline_config import PipelineConfig
from stages.common import StageLogger, require_files


def copy_file(src: Path, dst: Path, copied: list[dict[str, str]]) -> None:
    src = Path(src)
    dst = Path(dst)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append({"source": str(src), "destination": str(dst)})


def write_numeric_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_arrays = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.dtype == object:
            raise TypeError(f"Refusing to save object array in curated NPZ: {key}")
        safe_arrays[key] = array
    np.savez_compressed(path, **safe_arrays)


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def copy_randomization_provenance(cfg: PipelineConfig, out_dir: Path, copied: list[dict[str, str]]) -> None:
    stimulus_dir = out_dir / "stimulus"
    snapshot = cfg.stim_cam_path / "session_snapshot.json"
    if snapshot.exists():
        copy_file(snapshot, stimulus_dir / "session_snapshot.json", copied)
        return

    legacy_root = Path(__file__).resolve().parents[2] / "Experiment_running_GUI"
    metadata = legacy_root / "last_run_recording_metadata.json"
    if not metadata.exists():
        return
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    recorded_session = Path(str(payload.get("session_dir", "")))
    if str(recorded_session).casefold() != str(cfg.stim_cam_path).casefold():
        return
    copy_file(metadata, stimulus_dir / "legacy_recording_metadata.json", copied)
    settings = Path(str(payload.get("settings_file", "")))
    copy_file(settings, stimulus_dir / "legacy_gui_settings.json", copied)


def read_event_times(path: Path) -> np.ndarray:
    times: list[float] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                times.append(float(text.split()[0]))
            except ValueError:
                continue
    return np.asarray(times, dtype=float)


def write_event_times(path: Path, times: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if times.size:
        np.savetxt(path, times, fmt="%.12f")
    else:
        path.write_text("", encoding="utf-8")


def copy_event_file(
    src: Path,
    dst: Path,
    copied: list[dict[str, str]],
    crop_bounds: tuple[float, float] | None,
) -> None:
    src = Path(src)
    if not src.exists():
        return
    if crop_bounds is None:
        copy_file(src, dst, copied)
        return

    start_s, end_s = crop_bounds
    times = read_event_times(src)
    cropped = times[(times >= start_s) & (times < end_s)]
    write_event_times(dst, cropped)
    copied.append({"source": str(src), "destination": str(dst)})


def _normalise_tprime_stream(value: str) -> str:
    stream = str(value).strip().lower()
    if stream == "probe0":
        return "imec0"
    if stream == "nidq":
        return "ni"
    return stream


def _stream_probe_id(stream: str) -> int | None:
    if stream.startswith("imec"):
        suffix = stream.removeprefix("imec")
    elif stream.startswith("probe"):
        suffix = stream.removeprefix("probe")
    else:
        return None
    return int(suffix) if suffix.isdigit() else None


def _ap_sync_candidates(cfg: PipelineConfig, probe_id: int) -> list[Path]:
    run, gate = cfg.run_and_gate
    probe_dir = cfg.catgt_root / f"{run}_g{gate}_imec{probe_id}"
    pattern = f"*.imec{probe_id}.ap.xd_{cfg.ap_sync_word}_{cfg.ap_sync_bit}_{cfg.sync_threshold}.txt"
    exact = probe_dir / f"{run}_g{gate}_tcat.imec{probe_id}.ap.xd_{cfg.ap_sync_word}_{cfg.ap_sync_bit}_{cfg.sync_threshold}.txt"
    candidates = [exact]
    candidates.extend(sorted(probe_dir.glob(pattern)))
    candidates.extend(sorted(cfg.catgt_root.rglob(pattern)))
    return candidates


def _nidq_sync_candidates(cfg: PipelineConfig) -> list[Path]:
    run, gate = cfg.run_and_gate
    pattern = f"*nidq.xd_{cfg.ni_word}_{cfg.sync_bit}_{cfg.sync_threshold}.txt"
    exact = cfg.catgt_root / f"{run}_g{gate}_tcat.nidq.xd_{cfg.ni_word}_{cfg.sync_bit}_{cfg.sync_threshold}.txt"
    candidates = [exact]
    candidates.extend(sorted(cfg.catgt_root.glob(pattern)))
    candidates.extend(sorted(cfg.catgt_root.rglob(pattern)))
    return candidates


def sync_crop_bounds_for_packaged_outputs(cfg: PipelineConfig, logger: StageLogger) -> tuple[float, float] | None:
    if not cfg.sync_crop_enabled:
        return None

    start_index = int(cfg.sync_crop_start_index)
    end_index = int(cfg.sync_crop_end_index)
    stream = _normalise_tprime_stream(cfg.tprime_reference_stream)

    if stream == "ni":
        sync_file = first_existing(_nidq_sync_candidates(cfg))
        stream_label = "NIDQ"
    else:
        probe_id = _stream_probe_id(stream)
        if probe_id is None:
            raise ValueError(
                "Sync crop needs TPrime reference to be 'ni', 'nidq', 'imecN', or 'probeN'; "
                f"got {cfg.tprime_reference_stream!r}."
            )
        sync_file = first_existing(_ap_sync_candidates(cfg, probe_id))
        stream_label = f"imec{probe_id}"

    if sync_file is None:
        raise FileNotFoundError(
            "Sync crop is enabled, but no matching sync pulse file was found for "
            f"TPrime reference stream {cfg.tprime_reference_stream!r}."
        )

    sync_times = read_event_times(sync_file)
    if end_index >= sync_times.size:
        raise IndexError(
            f"Sync crop end edge index {end_index} is outside {sync_file}, "
            f"which has {sync_times.size} rising edges."
        )

    start_s = float(sync_times[start_index])
    end_s = float(sync_times[end_index])
    if end_s <= start_s:
        raise ValueError(f"Invalid sync crop bounds from {sync_file}: start={start_s}, end={end_s}")

    logger.output(
        "sync-index crop for packaged outputs: "
        f"{stream_label} sync edges [{start_index}:{end_index}] -> "
        f"{start_s:.6f}s to {end_s:.6f}s"
    )
    return start_s, end_s


def _nullable_float_col(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Float64, strict=False).fill_nan(None)


def copy_stimulus_metadata(
    src: Path,
    dst: Path,
    copied: list[dict[str, str]],
    crop_bounds: tuple[float, float] | None,
) -> None:
    src = Path(src)
    if not src.exists():
        return
    if crop_bounds is None:
        copy_file(src, dst, copied)
        return

    if src.suffix.lower() == ".parquet":
        metadata = pl.read_parquet(src)
    else:
        metadata = pl.read_csv(src)

    if {"time_start_s", "time_end_s"}.issubset(set(metadata.columns)):
        start_s, end_s = crop_bounds
        start_col = _nullable_float_col("time_start_s")
        end_col = _nullable_float_col("time_end_s")
        metadata = metadata.filter(
            (start_col.fill_null(end_col) >= start_s)
            & (end_col.fill_null(start_col) < end_s)
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() == ".parquet":
        metadata.write_parquet(dst)
    else:
        metadata.write_csv(dst)
    copied.append({"source": str(src), "destination": str(dst)})


def sort_output_dir(cfg: PipelineConfig, run: str, gate: int, probe_id: int) -> Path:
    package_dir = cfg.probe_sort_dir(probe_id)
    if package_dir.exists():
        return package_dir
    legacy_dir = cfg.catgt_root / f"{run}_g{gate}_imec{probe_id}" / "kilosort4"
    return legacy_dir


def load_aligned_spike_seconds(ks_dir: Path) -> np.ndarray:
    return np.load(ks_dir / "spike_times_sec_adj.npy", allow_pickle=False).squeeze().astype(float)


def load_unit_templates_uV(ks_dir: Path, n_channels: int) -> np.ndarray:
    templates = np.load(ks_dir / "mean_waveforms.npy", allow_pickle=False)
    if templates.ndim != 3:
        raise ValueError(f"mean_waveforms.npy must have shape units x channels x samples: {ks_dir / 'mean_waveforms.npy'}")
    if templates.shape[1] > n_channels:
        templates = templates[:, :n_channels, :]
    if templates.shape[1] != n_channels:
        raise ValueError(
            "mean_waveforms.npy channel count does not match channel_positions.npy.\n"
            f"  mean_waveforms shape: {templates.shape}\n"
            f"  channel_positions channels: {n_channels}"
        )
    return templates.astype(np.float32, copy=False)


def read_quality_metrics(ks_dir: Path, probe_id: int) -> pl.DataFrame:
    path = ks_dir / "metrics.csv"
    quality = pl.read_csv(path)
    rename = {}
    if "cluster_id" in quality.columns and "unit_id" not in quality.columns:
        rename["cluster_id"] = "unit_id"
    if rename:
        quality = quality.rename(rename)
    if "probe" not in quality.columns:
        quality = quality.with_columns(pl.lit(int(probe_id)).alias("probe"))
    return quality


def build_preprocessing_output(cfg: PipelineConfig, logger: StageLogger) -> None:
    run, gate = cfg.run_and_gate
    catgt_root = cfg.catgt_root
    out_dir = cfg.preprocessed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    probe_ids = cfg.normalized_probe_ids()
    combined_unit_metadata: list[pl.DataFrame] = []
    spike_times_by_unit: dict[str, np.ndarray] = {}
    templates_by_unit: dict[str, np.ndarray] = {}
    channel_rows: list[dict[str, float | int]] = []

    logger.log(f"Building preprocessing output: {out_dir}")
    crop_bounds = sync_crop_bounds_for_packaged_outputs(cfg, logger)
    sequence_paths = [
        catgt_root / "tprime" / "stimulus_trials.csv",
        cfg.stim_cam_path / "stimulus_trials.csv",
        cfg.stim_cam_path / "somatosensory_stimulation" / "run_001_sequence.csv",
    ]
    sequence_path = next((path for path in sequence_paths if path.exists()), None)
    if sequence_path is not None:
        copy_file(sequence_path, out_dir / "stimulus" / "stimulus_trials.csv", copied)
    copy_randomization_provenance(cfg, out_dir, copied)
    for sidecar_name in [
        "completed_moves_repaired.csv",
        "excluded_moves_repaired.csv",
        "repair_manifest.json",
    ]:
        copy_file(cfg.stim_cam_path / sidecar_name, out_dir / "stimulus" / sidecar_name, copied)

    for src in sorted((catgt_root / "tprime").glob("*.txt")):
        copy_event_file(src, out_dir / "events" / src.name, copied, crop_bounds)
    for name in [
        "stimulus_metadata_table.csv",
        "stimulus_metadata_table.parquet",
        "stimulus_event_table.csv",
        "stimulus_event_table.parquet",
    ]:
        copy_stimulus_metadata(catgt_root / "tprime" / name, out_dir / "stimulus" / name, copied, crop_bounds)

    for probe_id in probe_ids:
        ks_dir = sort_output_dir(cfg, run, gate, probe_id)
        aligned_spike_times = ks_dir / "spike_times_sec_adj.npy"
        require_files(
            [
                aligned_spike_times,
                ks_dir / "spike_clusters.npy",
                ks_dir / "channel_positions.npy",
                ks_dir / "mean_waveforms.npy",
                ks_dir / "metrics.csv",
            ],
            f"Missing files needed to build probe {probe_id} output.",
        )

        spike_seconds = load_aligned_spike_seconds(ks_dir)
        spike_clusters = np.load(ks_dir / "spike_clusters.npy", allow_pickle=False)
        channel_positions = np.load(ks_dir / "channel_positions.npy", allow_pickle=False)
        channel_map_path = ks_dir / "channel_map.npy"
        channel_map = np.load(channel_map_path, allow_pickle=False) if channel_map_path.exists() else np.arange(channel_positions.shape[0])
        unit_templates = load_unit_templates_uV(ks_dir, int(channel_positions.shape[0]))
        quality = read_quality_metrics(ks_dir, probe_id)

        if spike_seconds.size == 0:
            logger.output(f"probe {probe_id}: no spikes to package")
            continue

        for channel_index, position in enumerate(channel_positions):
            channel_rows.append(
                {
                    "probe": int(probe_id),
                    "channel_index": int(channel_index),
                    "ks_channel_id": int(channel_map[channel_index]) if channel_index < len(channel_map) else int(channel_index),
                    "x_um": float(position[0]),
                    "y_um": float(position[1]),
                }
            )

        rows = []

        for unit_id in np.unique(spike_clusters):
            idx = spike_clusters == unit_id
            t = np.sort(spike_seconds[idx])
            unit_key = f"probe{int(probe_id)}_unit{int(unit_id)}"
            if int(unit_id) >= unit_templates.shape[0]:
                raise ValueError(
                    f"Missing mean waveform for unit {unit_id} in {ks_dir / 'mean_waveforms.npy'} "
                    f"with shape {unit_templates.shape}"
                )
            spike_times_by_unit[unit_key] = t.astype(np.float64, copy=False)
            templates_by_unit[unit_key] = unit_templates[int(unit_id), :, :]
            rows.append(
                {
                    "probe": int(probe_id),
                    "unit_id": int(unit_id),
                    "unit_key": unit_key,
                }
            )

        unit_metadata = pl.DataFrame(rows)
        if {"probe", "unit_id"}.issubset(set(quality.columns)) and quality.height > 0:
            unit_metadata = unit_metadata.join(quality, on=["probe", "unit_id"], how="left")
        combined_unit_metadata.append(unit_metadata)
        logger.output(f"probe {probe_id}: {unit_metadata.height} units packaged")

    if combined_unit_metadata:
        root_unit_metadata = pl.concat(combined_unit_metadata).sort(["probe", "unit_id"])
        root_unit_metadata.write_csv(out_dir / "units.csv")
        copied.append({"source": "generated", "destination": str(out_dir / "units.csv")})

    if channel_rows:
        pl.DataFrame(channel_rows).write_csv(out_dir / "channels.csv")
        copied.append({"source": "generated", "destination": str(out_dir / "channels.csv")})

    if spike_times_by_unit:
        write_numeric_npz(out_dir / "spike_times_by_unit.npz", spike_times_by_unit)
        copied.append({"source": "generated", "destination": str(out_dir / "spike_times_by_unit.npz")})

    if templates_by_unit:
        write_numeric_npz(out_dir / "unit_templates_uV_by_unit.npz", templates_by_unit)
        copied.append({"source": "generated", "destination": str(out_dir / "unit_templates_uV_by_unit.npz")})

    manifest = pl.DataFrame(copied)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(cfg.logs_dir / "preprocessed_data_manifest.csv")
    logger.output(f"copied files: {manifest.height}")
    logger.log(f"Preprocessing output ready: {out_dir}")
