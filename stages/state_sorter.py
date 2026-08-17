from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from pipeline_config import PipelineConfig
from stages.common import StageLogger, read_meta


def _prepend_path(path: Path) -> None:
    text = str(path.resolve())
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)
    current = os.environ.get("PYTHONPATH", "")
    parts = [part for part in current.split(os.pathsep) if part and part != text]
    os.environ["PYTHONPATH"] = os.pathsep.join([text, *parts])


def _clear_sorter_imports() -> None:
    for name in list(sys.modules):
        if name == "kilosort" or name.startswith("kilosort.") or name == "state_sorter" or name.startswith("state_sorter."):
            del sys.modules[name]


def _catgt_or_raw_ap_file(cfg: PipelineConfig, probe_id: int) -> Path:
    run, gate = cfg.run_and_gate
    catgt_probe_dir = cfg.catgt_root / f"{run}_g{gate}_imec{probe_id}"
    catgt_expected = catgt_probe_dir / f"{run}_g{gate}_tcat.imec{probe_id}.ap.bin"
    if catgt_expected.exists():
        return catgt_expected
    catgt_matches = sorted(catgt_probe_dir.glob(f"*_tcat.imec{probe_id}.ap.bin"))
    if catgt_matches:
        return catgt_matches[0]

    raw_probe_dir = cfg.spikeglx_path / f"{run}_g{gate}_imec{probe_id}"
    raw_expected = raw_probe_dir / f"{run}_g{gate}_t{cfg.trial_start}.imec{probe_id}.ap.bin"
    if raw_expected.exists():
        return raw_expected
    raw_matches = sorted(raw_probe_dir.glob(f"*_t*.imec{probe_id}.ap.bin"))
    if raw_matches:
        return raw_matches[0]

    raise FileNotFoundError(f"Missing AP binary for StateSorter probe {probe_id}: {catgt_expected}")


def _ecephys_repo_dir(cfg: PipelineConfig) -> Path:
    if cfg.ecephys_directory.strip():
        path = Path(cfg.ecephys_directory)
        return path.parent if path.name == "ecephys_spike_sorting" else path
    return Path(__file__).resolve().parent.parent / "ecephys_spike_sorting_LNE"


def _ap_channel_count(meta: dict[str, str]) -> int:
    return int(meta.get("snsApLfSy", f"{meta['nSavedChans']},0,0").split(",")[0])


def _single_shank_probe_json(meta: dict[str, str], output_path: Path) -> Path:
    n_ap = _ap_channel_count(meta)
    chan = np.arange(n_ap, dtype=np.int32)
    probe = {
        "chanMap": chan.tolist(),
        "xc": ((chan % 2) * 32).astype(float).tolist(),
        "yc": ((chan // 2) * 20).astype(float).tolist(),
        "kcoords": np.ones(n_ap, dtype=np.float32).tolist(),
        "n_chan": int(n_ap),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(probe), encoding="utf-8")
    return output_path


def _probe_path(cfg: PipelineConfig, probe_id: int, meta_path: Path, meta: dict[str, str]) -> Path:
    if cfg.probe_geometry_mode == "custom_json":
        return Path(cfg.custom_probe_geometry)
    if cfg.probe_geometry_mode == "single_shank":
        return _single_shank_probe_json(meta, cfg.json_dir / f"{cfg.run_label}_imec{probe_id}_state_sorter_probe.json")

    _prepend_path(_ecephys_repo_dir(cfg))
    from ecephys_spike_sorting.common.SGLXMetaToCoords import MetaToCoords

    chanmap_path = cfg.json_dir / f"{cfg.run_label}_imec{probe_id}_state_sorter_chanMap.mat"
    chanmap_path.parent.mkdir(parents=True, exist_ok=True)
    MetaToCoords(metaFullPath=meta_path, outType=1, destFullPath=str(chanmap_path))
    return chanmap_path


def _thresholds(cfg: PipelineConfig, probe_index: int) -> tuple[float, float]:
    regions = cfg.normalized_brain_regions()
    region = regions[probe_index] if probe_index < len(regions) else "default"
    text = cfg.ks_th4_by_region.get(region, cfg.ks_th4_by_region.get("default", "[8,9]")).strip()
    universal, learned = [part.strip() for part in text.strip("[]").split(",")]
    return float(universal), float(learned)


def run_state_sorter(cfg: PipelineConfig, logger: StageLogger) -> None:
    sorter_repo = cfg.state_sorter_repo_dir
    if not (sorter_repo / "state_sorter").is_dir():
        raise FileNotFoundError(f"StateSorter repository does not contain a state_sorter package: {sorter_repo}")

    _prepend_path(sorter_repo)
    _clear_sorter_imports()
    from state_sorter import run_state_sorter as run_sorter

    for probe_index, probe_id in enumerate(cfg.normalized_probe_ids()):
        input_file = _catgt_or_raw_ap_file(cfg, probe_id)
        meta_path = input_file.with_suffix(".meta")
        meta = read_meta(meta_path)
        th_universal, th_learned = _thresholds(cfg, probe_index)
        output_dir = cfg.state_sorter_probe_output_dir(probe_id)
        probe_file = _probe_path(cfg, probe_id, meta_path, meta)
        settings = {
            "n_chan_bin": int(meta["nSavedChans"]),
            "fs": float(meta["imSampRate"]),
            "tmin": float(cfg.ks_tmin),
            "tmax": np.inf if float(cfg.ks_tmax) < 0 else float(cfg.ks_tmax),
            "Th_universal": th_universal,
            "Th_learned": th_learned,
            "duplicate_spike_ms": float(cfg.ks4_duplicate_spike_ms),
            "nblocks": int(cfg.ks_nblocks),
            "min_template_size": float(cfg.ks4_min_template_size_um),
            "cluster_init_seed": int(cfg.ks_CSBseed),
            "probe_path": str(probe_file),
            "state_n_clusters": int(cfg.state_sorter_n_states),
            "state_n_components": int(cfg.state_sorter_n_components),
            "use_drift": bool(cfg.state_sorter_use_drift),
        }

        logger.log(f"Running StateSorter for probe {probe_id}")
        logger.output(f"StateSorter repository: {sorter_repo}")
        logger.output(f"StateSorter input: {input_file}")
        logger.output(f"StateSorter output: {output_dir}")
        metadata, mean_waveforms, state_event_coordinates = run_sorter(
            settings,
            filename=input_file,
            results_dir=output_dir,
            data_dtype="int16",
            do_CAR=bool(cfg.ks_CAR),
            clear_cache=False,
        )
        logger.output(f"StateSorter events: {state_event_coordinates.shape[1]}")
        logger.output(f"StateSorter states: {mean_waveforms.shape[0]}")
        logger.output(f"StateSorter metadata rows: {metadata.shape[0]}")
