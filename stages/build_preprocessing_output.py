from __future__ import annotations

import shutil
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.io import loadmat

from pipeline_config import PipelineConfig
from stages.common import StageLogger, require_files


CHANNEL_KEY_COLUMNS = ["probe_name", "probe_channel_number"]
CHANNEL_GEOMETRY_COLUMNS = [
    "probe_horizontal_position",
    "probe_vertical_position",
    "probe_shank",
]
ANATOMY_COLUMNS = [
    "structure_id",
    "structure_name",
    "structure_acronym",
    "ccf_ap_index",
    "ccf_dv_index",
    "ccf_ml_index",
    "atlas_region_id",
    "atlas_region",
    "atlas_acronym",
    "atlas_ap",
    "atlas_dv",
    "atlas_ml",
    "stereotaxic_ap_um",
    "stereotaxic_dv_um",
    "stereotaxic_ml_um",
    "trajectory_distance_um",
    "probe_type",
    "anatomy_source",
    "anatomy_assignment_method",
    "anatomy_mapped_at",
]


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


def _integer_vector(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    rounded = np.rint(values).astype(np.int64)
    if not np.allclose(values, rounded):
        raise ValueError(f"{label} contains non-integer channel identifiers")
    if np.unique(rounded).size != rounded.size:
        raise ValueError(f"{label} contains duplicate channel identifiers")
    return rounded


def load_probe_channels(ks_dir: Path, probe_id: int) -> tuple[pl.DataFrame, np.ndarray]:
    active_positions = np.load(ks_dir / "channel_positions.npy", allow_pickle=False)
    channel_map_path = ks_dir / "channel_map.npy"
    active_map = (
        _integer_vector(np.load(channel_map_path, allow_pickle=False), str(channel_map_path))
        if channel_map_path.exists()
        else np.arange(active_positions.shape[0], dtype=np.int64)
    )
    if active_positions.ndim != 2 or active_positions.shape[1] < 2 or len(active_map) != len(active_positions):
        raise ValueError(
            f"channel_map/channel_positions mismatch in {ks_dir}: "
            f"map={active_map.shape}, positions={active_positions.shape}"
        )

    chanmap_paths = sorted(ks_dir.glob("*chanMap.mat"))
    if chanmap_paths:
        chanmap = loadmat(chanmap_paths[0], squeeze_me=True)
        required = {"chanMap0ind", "xcoords", "ycoords"}
        missing = sorted(required.difference(chanmap))
        if missing:
            raise ValueError(f"Missing {missing} in {chanmap_paths[0]}")
        channel_numbers = _integer_vector(chanmap["chanMap0ind"], f"{chanmap_paths[0]}:chanMap0ind")
        horizontal = np.asarray(chanmap["xcoords"], dtype=float).reshape(-1)
        vertical = np.asarray(chanmap["ycoords"], dtype=float).reshape(-1)
        connected = np.asarray(chanmap.get("connected", np.ones(len(channel_numbers))), dtype=bool).reshape(-1)
        raw_shanks = np.asarray(
            chanmap.get("kcoords", np.ones(len(channel_numbers))), dtype=float
        ).reshape(-1)
        shanks = np.rint(raw_shanks).astype(np.int64)
        if not np.allclose(raw_shanks, shanks):
            raise ValueError(f"Non-integer kcoords in {chanmap_paths[0]}")
        shanks -= shanks.min()
        if not (len(horizontal) == len(vertical) == len(connected) == len(shanks) == len(channel_numbers)):
            raise ValueError(f"Inconsistent channel geometry lengths in {chanmap_paths[0]}")
    else:
        channel_numbers = active_map.copy()
        horizontal = active_positions[:, 0].astype(float)
        vertical = active_positions[:, 1].astype(float)
        connected = np.ones(len(channel_numbers), dtype=bool)
        shanks = np.zeros(len(channel_numbers), dtype=np.int64)

    active_index = {int(channel): int(index) for index, channel in enumerate(active_map)}
    probe_name = f"imec{int(probe_id)}"
    rows = [
        {
            "probe": int(probe_id),
            "probe_name": probe_name,
            "channel_index": active_index.get(int(channel)),
            "ks_channel_id": int(channel),
            "probe_channel_number": int(channel),
            "x_um": float(x),
            "y_um": float(y),
            "probe_horizontal_position": float(x),
            "probe_vertical_position": float(y),
            "probe_shank": int(round(float(shank))),
            "channel_connected": bool(is_connected),
            "used_for_sorting": int(channel) in active_index,
        }
        for channel, x, y, shank, is_connected in zip(
            channel_numbers, horizontal, vertical, shanks, connected, strict=True
        )
    ]
    return pl.DataFrame(rows), active_map


def load_unit_templates_uV(
    ks_dir: Path,
    probe_channel_numbers: np.ndarray,
    active_channel_map: np.ndarray,
) -> np.ndarray:
    templates = np.load(ks_dir / "mean_waveforms.npy", allow_pickle=False)
    if templates.ndim != 3:
        raise ValueError(f"mean_waveforms.npy must have shape units x channels x samples: {ks_dir / 'mean_waveforms.npy'}")
    channel_numbers = _integer_vector(probe_channel_numbers, "probe_channel_numbers")
    n_physical_channels = int(channel_numbers.max()) + 1
    if templates.shape[1] >= n_physical_channels:
        templates = templates[:, :n_physical_channels, :]
    elif templates.shape[1] == len(active_channel_map):
        expanded = np.full(
            (templates.shape[0], n_physical_channels, templates.shape[2]),
            np.nan,
            dtype=np.float32,
        )
        expanded[:, active_channel_map, :] = templates
        templates = expanded
    else:
        raise ValueError(
            "mean_waveforms.npy cannot be aligned to physical probe-channel numbers.\n"
            f"  mean_waveforms shape: {templates.shape}\n"
            f"  physical channel count: {n_physical_channels}\n"
            f"  sorting channel count: {len(active_channel_map)}"
        )
    return templates.astype(np.float32, copy=False)


def measured_peak_probe_channels(
    unit_ids: np.ndarray,
    templates: np.ndarray,
    probe_channels: pl.DataFrame,
) -> np.ndarray:
    connected_channels = probe_channels.filter(pl.col("channel_connected"))["probe_channel_number"].to_numpy()
    peaks: list[int] = []
    for unit_id in unit_ids:
        unit_id = int(unit_id)
        if unit_id < 0 or unit_id >= templates.shape[0]:
            raise ValueError(f"Missing mean waveform for unit {unit_id} with templates shape {templates.shape}")
        waveforms = templates[unit_id, connected_channels, :]
        has_samples = np.isfinite(waveforms).any(axis=1)
        if not has_samples.any():
            raise ValueError(f"Mean waveform for unit {unit_id} has no finite connected-channel samples")
        amplitude = np.full(len(connected_channels), -np.inf, dtype=float)
        amplitude[has_samples] = (
            np.nanmax(waveforms[has_samples], axis=1)
            - np.nanmin(waveforms[has_samples], axis=1)
        )
        peaks.append(int(connected_channels[int(np.argmax(amplitude))]))
    return np.asarray(peaks, dtype=np.int64)


def _canonical_anatomy_table(table: pl.DataFrame) -> pl.DataFrame | None:
    columns = set(table.columns)
    if "probe_name" not in columns and "probe" in columns:
        table = table.with_columns(
            pl.format("imec{}", pl.col("probe").cast(pl.Int64)).alias("probe_name")
        )
    if "probe_channel_number" not in columns:
        source = "ks_channel_id" if "ks_channel_id" in columns else None
        if source is None:
            return None
        table = table.with_columns(pl.col(source).cast(pl.Int64).alias("probe_channel_number"))
    if "probe_horizontal_position" not in table.columns and "x_um" in table.columns:
        table = table.with_columns(
            pl.col("x_um").cast(pl.Float64).alias("probe_horizontal_position")
        )
    if "probe_vertical_position" not in table.columns and "y_um" in table.columns:
        table = table.with_columns(
            pl.col("y_um").cast(pl.Float64).alias("probe_vertical_position")
        )
    if "structure_acronym" not in table.columns and "atlas_acronym" in table.columns:
        table = table.with_columns(pl.col("atlas_acronym").alias("structure_acronym"))
    keep = (
        CHANNEL_KEY_COLUMNS
        + [name for name in CHANNEL_GEOMETRY_COLUMNS if name in table.columns]
        + [name for name in ANATOMY_COLUMNS if name in table.columns]
    )
    if "structure_acronym" not in keep:
        return None
    anatomy = table.select(keep)
    duplicates = anatomy.group_by(CHANNEL_KEY_COLUMNS).len().filter(pl.col("len") != 1)
    if duplicates.height:
        raise ValueError("Saved anatomy contains duplicate (probe_name, probe_channel_number) keys")
    return anatomy


def load_preserved_channel_anatomy(out_dir: Path) -> pl.DataFrame | None:
    sidecar = out_dir / "anatomy" / "channel_brain_regions.csv"
    if sidecar.exists():
        return _canonical_anatomy_table(pl.read_csv(sidecar))
    channels_path = out_dir / "channels.csv"
    if channels_path.exists():
        return _canonical_anatomy_table(pl.read_csv(channels_path))
    return None


def validate_preserved_anatomy_geometry(
    generated: pl.DataFrame,
    preserved: pl.DataFrame,
) -> pl.DataFrame:
    geometry = [name for name in CHANNEL_GEOMETRY_COLUMNS if name in preserved.columns]
    if not {"probe_horizontal_position", "probe_vertical_position"}.issubset(geometry):
        raise ValueError(
            "Saved anatomy has no peak-contact geometry and cannot be safely matched to this rebuild"
        )
    comparison = generated.select(CHANNEL_KEY_COLUMNS + geometry).join(
        preserved.select(CHANNEL_KEY_COLUMNS + geometry),
        on=CHANNEL_KEY_COLUMNS,
        how="inner",
        suffix="_saved",
        validate="1:1",
    )
    mismatch = pl.lit(False)
    for name in geometry:
        if name == "probe_shank":
            mismatch = mismatch | (pl.col(name) != pl.col(f"{name}_saved"))
        else:
            mismatch = mismatch | (
                (pl.col(name) - pl.col(f"{name}_saved")).abs() > 1e-6
            )
    bad = comparison.filter(mismatch)
    if bad.height:
        keys = bad.select(CHANNEL_KEY_COLUMNS).head(10).to_dicts()
        raise ValueError(
            "Saved anatomy geometry does not match the current probe channel map; "
            f"refusing to reuse stale assignments: {keys}"
        )
    return preserved.drop(geometry)


def write_metadata_schema(out_dir: Path, copied: list[dict[str, str]]) -> None:
    path = out_dir / "metadata_schema.json"
    schema = {
        "schema_version": 1,
        "channel_identity": ["probe_name", "probe_channel_number"],
        "unit_identity": "unit_key",
        "columns": {
            "probe_name": "SpikeGLX probe stream name, for example imec0 or imec1",
            "probe_channel_number": (
                "physical SpikeGLX AP channel number; in units.csv this is the channel where the unit's "
                "mean waveform has maximum peak-to-peak amplitude"
            ),
            "peak_channel_index": "row of the peak channel in the Kilosort active-channel arrays",
            "quality_peak_channel": "original peak_channel value reported by metrics.csv",
            "probe_horizontal_position": "across-shank contact position in micrometres",
            "probe_vertical_position": (
                "along-shank contact coordinate in micrometres relative to the chanMap y=0 contact"
            ),
            "probe_shank": "zero-based shank index normalized from chanMap kcoords",
            "structure_acronym": (
                "Allen atlas acronym written by the proprietary HERBS tracker for the peak channel; "
                "not generated by spike sorting"
            ),
        },
        "unit_anatomy_assignment": (
            "units.csv inherits structure_acronym from channels.csv by the exact composite key "
            "(probe_name, probe_channel_number)"
        ),
    }
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    copied.append({"source": "generated", "destination": str(path)})


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
    probe_channel_tables: list[pl.DataFrame] = []
    preserved_anatomy = load_preserved_channel_anatomy(out_dir)

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
        probe_channels, active_channel_map = load_probe_channels(ks_dir, probe_id)
        probe_channel_tables.append(probe_channels)
        probe_channel_numbers = probe_channels["probe_channel_number"].to_numpy()
        unit_templates = load_unit_templates_uV(ks_dir, probe_channel_numbers, active_channel_map)
        quality = read_quality_metrics(ks_dir, probe_id)

        if spike_seconds.size == 0:
            logger.output(f"probe {probe_id}: no spikes to package")
            continue

        rows = []
        unit_ids = np.unique(spike_clusters)
        measured_peaks = measured_peak_probe_channels(
            unit_ids,
            unit_templates,
            probe_channels,
        )

        for unit_id, measured_peak in zip(unit_ids, measured_peaks, strict=True):
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
                    "probe_channel_number": int(measured_peak),
                }
            )

        unit_metadata = pl.DataFrame(rows)
        if {"probe", "unit_id"}.issubset(set(quality.columns)) and quality.height > 0:
            unit_metadata = unit_metadata.join(quality, on=["probe", "unit_id"], how="left")
        if "peak_channel" in unit_metadata.columns:
            unit_metadata = unit_metadata.rename({"peak_channel": "quality_peak_channel"})
            mismatch_count = unit_metadata.filter(
                pl.col("quality_peak_channel").is_not_null()
                & (
                    pl.col("quality_peak_channel").cast(pl.Int64, strict=False)
                    != pl.col("probe_channel_number")
                )
            ).height
            if mismatch_count:
                logger.output(
                    f"probe {probe_id}: {mismatch_count} metrics.csv peak-channel value(s) differed from "
                    "the packaged mean-waveform maximum; using the measured waveform channel"
                )
        unit_metadata = unit_metadata.with_columns(
            pl.lit(f"imec{int(probe_id)}").alias("probe_name"),
            pl.col("probe_channel_number").alias("peak_channel"),
        )
        combined_unit_metadata.append(unit_metadata)
        logger.output(f"probe {probe_id}: {unit_metadata.height} units packaged")

    root_channels: pl.DataFrame | None = None
    if probe_channel_tables:
        root_channels = pl.concat(probe_channel_tables).sort(["probe", "probe_channel_number"])
        duplicates = root_channels.group_by(CHANNEL_KEY_COLUMNS).len().filter(pl.col("len") != 1)
        if duplicates.height:
            raise ValueError("Generated channels contain duplicate (probe_name, probe_channel_number) keys")
        if preserved_anatomy is not None:
            anatomy_mapping = validate_preserved_anatomy_geometry(
                root_channels,
                preserved_anatomy,
            )
            root_channels = root_channels.join(
                anatomy_mapping,
                on=CHANNEL_KEY_COLUMNS,
                how="left",
                validate="1:1",
            )
            anatomy_dir = out_dir / "anatomy"
            anatomy_dir.mkdir(exist_ok=True)
            channel_anatomy_columns = [
                *CHANNEL_KEY_COLUMNS,
                "probe_horizontal_position",
                "probe_vertical_position",
                "probe_shank",
                *[name for name in ANATOMY_COLUMNS if name in root_channels.columns],
            ]
            root_channels.select(channel_anatomy_columns).write_csv(
                anatomy_dir / "channel_brain_regions.csv"
            )
            copied.append(
                {
                    "source": "preserved proprietary HERBS mapping",
                    "destination": str(anatomy_dir / "channel_brain_regions.csv"),
                }
            )
        root_channels.write_csv(out_dir / "channels.csv")
        copied.append({"source": "generated", "destination": str(out_dir / "channels.csv")})

    if combined_unit_metadata:
        if root_channels is None:
            raise ValueError("Cannot attach peak-channel metadata without channels")
        root_unit_metadata = pl.concat(combined_unit_metadata).sort(["probe", "unit_id"])
        if root_unit_metadata["unit_key"].n_unique() != root_unit_metadata.height:
            raise ValueError("Generated units contain duplicate unit_key values")
        channel_copy_columns = [
            pl.col("channel_index").alias("peak_channel_index"),
            pl.col("probe_horizontal_position"),
            pl.col("probe_vertical_position"),
            pl.col("probe_shank"),
        ]
        channel_copy_columns.extend(pl.col(name) for name in ANATOMY_COLUMNS if name in root_channels.columns)
        unit_count = root_unit_metadata.height
        root_unit_metadata = root_unit_metadata.join(
            root_channels.select(CHANNEL_KEY_COLUMNS + channel_copy_columns),
            on=CHANNEL_KEY_COLUMNS,
            how="left",
            validate="m:1",
        )
        if root_unit_metadata.height != unit_count:
            raise ValueError("Peak-channel metadata join changed the number of units")
        unresolved = root_unit_metadata.filter(
            pl.col("probe_horizontal_position").is_null()
            | pl.col("probe_vertical_position").is_null()
        )
        if unresolved.height:
            keys = unresolved.select(["unit_key", "probe_name", "probe_channel_number"]).head(10).to_dicts()
            raise ValueError(f"Units have peak channels absent from channels.csv: {keys}")
        root_unit_metadata.write_csv(out_dir / "units.csv")
        copied.append({"source": "generated", "destination": str(out_dir / "units.csv")})
        if preserved_anatomy is not None:
            unit_anatomy_columns = [
                "unit_key",
                "unit_id",
                *CHANNEL_KEY_COLUMNS,
                "peak_channel_index",
                "probe_horizontal_position",
                "probe_vertical_position",
                "probe_shank",
                *[name for name in ANATOMY_COLUMNS if name in root_unit_metadata.columns],
            ]
            assignment_path = out_dir / "anatomy" / "unit_brain_region_assignments.csv"
            root_unit_metadata.select(unit_anatomy_columns).write_csv(assignment_path)
            copied.append(
                {
                    "source": "regenerated from preserved proprietary HERBS mapping",
                    "destination": str(assignment_path),
                }
            )

    if spike_times_by_unit:
        write_numeric_npz(out_dir / "spike_times_by_unit.npz", spike_times_by_unit)
        copied.append({"source": "generated", "destination": str(out_dir / "spike_times_by_unit.npz")})

    if templates_by_unit:
        write_numeric_npz(out_dir / "unit_templates_uV_by_unit.npz", templates_by_unit)
        copied.append({"source": "generated", "destination": str(out_dir / "unit_templates_uV_by_unit.npz")})

    write_metadata_schema(out_dir, copied)
    manifest = pl.DataFrame(copied)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(cfg.logs_dir / "preprocessed_data_manifest.csv")
    logger.output(f"copied files: {manifest.height}")
    logger.log(f"Preprocessing output ready: {out_dir}")
