from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Callable

from pipeline_config import PipelineConfig
from stages.common import StageLogger


class _EmitBuffer(io.TextIOBase):
    def __init__(self, emit: Callable[[str], None]) -> None:
        self.emit = emit
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.emit(line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self.emit(self._buffer.rstrip())
        self._buffer = ""


def _tool_dir(path_text: str) -> str:
    text = path_text.strip()
    if not text:
        return ""
    path = Path(text)
    if path.suffix.lower() in {".exe", ".bat", ".cmd"}:
        return str(path.parent)
    return str(path)


def _normalise_tprime_stream(value: str) -> str:
    stream = value.strip().lower()
    if stream == "probe0":
        return "imec0"
    if stream == "nidq":
        return "ni"
    return stream


def _settings_from_config(cfg: PipelineConfig) -> dict:
    run, gate = cfg.run_and_gate
    probes = cfg.normalized_probe_ids()
    trigger_string = f"{cfg.trial_start},{cfg.trial_end}"
    probe_string = ",".join(str(probe) for probe in probes)

    settings = {
        "ks_ver": cfg.ks_ver,
        "refPerMS_dict": cfg.ref_per_ms_by_region,
        "ksTh2_dict": cfg.ks_th2_by_region,
        "ksTh3_dict": cfg.ks_th3_by_region,
        "ksTh4_dict": cfg.ks_th4_by_region,
        "logName": cfg.log_name,
        "npx_directory": str(cfg.spikeglx_data_dir),
        "run_specs": [[run, str(gate), trigger_string, probe_string, cfg.onebox_streams, cfg.normalized_brain_regions()]],
        "catGT_dest": str(cfg.catgt_output_dir),
        "run_CatGT": cfg.run_catgt,
        "car_mode": cfg.car_mode,
        "loccar_min": cfg.loccar_min,
        "loccar_max": cfg.loccar_max,
        "process_lf": cfg.process_lf,
        "catGT_cmd_string": cfg.catgt_cmd_string,
        "obx_present": cfg.obx_present,
        "ni_present": cfg.ni_present,
        "ni_obx_extract_string": cfg.ni_obx_extract_string,
        "ks_remDup": cfg.ks_remDup,
        "ks_saveRez": cfg.ks_saveRez,
        "ks_copy_fproc": cfg.ks_copy_fproc,
        "ks_templateRadius_um": cfg.ks_templateRadius_um,
        "ks_whiteningRadius_um": cfg.ks_whiteningRadius_um,
        "ks_minfr_goodchannels": cfg.ks_minfr_goodchannels,
        "ks_CAR": cfg.ks_CAR,
        "ks_nblocks": cfg.ks_nblocks,
        "ks4_duplicate_spike_ms": cfg.ks4_duplicate_spike_ms,
        "ks4_min_template_size_um": cfg.ks4_min_template_size_um,
        "ks4_det": cfg.ks4_det,
        "ks_tmin": cfg.ks_tmin,
        "ks_tmax": cfg.ks_tmax,
        "ks_CSBseed": cfg.ks_CSBseed,
        "ks_LTseed": cfg.ks_LTseed,
        "ks_helper_noise_threshold": cfg.ks_helper_noise_threshold,
        "ks_doFilter": cfg.ks_doFilter,
        "c_Waves_snr_um": cfg.c_waves_snr_um,
        "c_Waves_calc_half": cfg.c_waves_calc_half,
        "runTPrime": cfg.run_tprime,
        "sync_period": cfg.tprime_syncperiod_s,
        "toStream_sync_params": _normalise_tprime_stream(cfg.tprime_reference_stream),
        "sync_crop_enabled": cfg.sync_crop_enabled,
        "sync_crop_start_index": cfg.sync_crop_start_index,
        "sync_crop_end_index": cfg.sync_crop_end_index,
        "sync_crop_ap_sync_word": cfg.ap_sync_word,
        "sync_crop_ap_sync_bit": cfg.ap_sync_bit,
        "sync_crop_ni_word": cfg.ni_word,
        "sync_crop_ni_sync_bit": cfg.sync_bit,
        "sync_crop_sync_threshold": cfg.sync_threshold,
        "modules": cfg.modules,
        "json_directory": str(cfg.json_dir),
        "noise_template_use_rf": cfg.noise_template_use_rf,
        "include_pc_metrics": cfg.include_pc_metrics,
        "ecephys_directory": str(cfg.ecephys_package_dir),
        "kilosort_repository": cfg.kilosort_repository,
        "kilosort20_repository": cfg.kilosort20_repository,
        "kilosort25_repository": cfg.kilosort25_repository,
        "kilosort30_repository": cfg.kilosort30_repository,
        "npy_matlab_repository": cfg.npy_matlab_repository,
        "catGTPath": _tool_dir(cfg.catgt_exe),
        "tPrime_path": _tool_dir(cfg.tprime_exe),
        "cWaves_path": _tool_dir(cfg.cwaves_path),
        "kilosort_output_tmp": str(cfg.kilosort_tmp_dir),
    }
    return settings


def _prepare_import_path(cfg: PipelineConfig) -> None:
    repo_dir = str(cfg.ecephys_repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    current = os.environ.get("PYTHONPATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if repo_dir not in parts:
        os.environ["PYTHONPATH"] = repo_dir + (os.pathsep + current if current else "")


def _extract_threshold(extract_string: str, edge_type: str, bit: int, fallback: int | float) -> str:
    pattern = re.compile(rf"-{edge_type}=([^\s]+)")
    for match in pattern.finditer(extract_string):
        parts = [part.strip() for part in match.group(1).split(",")]
        if len(parts) >= 5 and parts[3] == str(bit):
            return parts[4]
    return str(fallback)


def _event_candidates(cfg: PipelineConfig, edge_type: str, bit: int, threshold: str) -> list[Path]:
    run, gate = cfg.run_and_gate
    stem = f"{run}_g{gate}_tcat.nidq.{edge_type}_{cfg.ni_word}_{bit}_{threshold}"
    candidates = [
        cfg.catgt_root / f"{stem}.adj.txt",
        cfg.catgt_root / f"{stem}.txt",
    ]
    candidates.extend(sorted(cfg.catgt_root.glob(f"*nidq.{edge_type}_*_{bit}_{threshold}.adj.txt")))
    candidates.extend(sorted(cfg.catgt_root.glob(f"*nidq.{edge_type}_*_{bit}_{threshold}.txt")))
    return candidates


def _copy_first_existing(candidates: list[Path], destination: Path, logger: StageLogger) -> None:
    for candidate in candidates:
        if candidate.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
            logger.output(f"{candidate.name} -> {destination.name}")
            return
    logger.output(f"missing TPrime event file for {destination.name}")


def _collect_tprime_event_files(cfg: PipelineConfig, logger: StageLogger) -> None:
    tprime_dir = cfg.catgt_root / "tprime"
    event_specs = [
        ("rotation_start_probe_time.txt", "xd", cfg.rotation_bit),
        ("rotation_end_probe_time.txt", "xid", cfg.rotation_bit),
        ("switching_start_probe_time.txt", "xd", cfg.switching_bit),
        ("switching_end_probe_time.txt", "xid", cfg.switching_bit),
        ("camera_trigger_probe_time.txt", "xd", cfg.camera_bit),
    ]
    for filename, edge_type, bit in event_specs:
        threshold = _extract_threshold(cfg.ni_obx_extract_string, edge_type, bit, cfg.event_threshold)
        _copy_first_existing(_event_candidates(cfg, edge_type, bit, threshold), tprime_dir / filename, logger)


def run_ecephys_spike_sorting(cfg: PipelineConfig, logger: StageLogger) -> None:
    os.environ["MPLBACKEND"] = "Agg"
    _prepare_import_path(cfg)
    cfg.catgt_output_dir.mkdir(parents=True, exist_ok=True)
    cfg.json_dir.mkdir(parents=True, exist_ok=True)

    settings = _settings_from_config(cfg)
    settings_path = cfg.logs_dir / "ecephys_pipeline_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    logger.log(f"Running ecephys_spike_sorting_LNE pipeline with settings: {settings_path}")

    module = importlib.import_module("ecephys_spike_sorting.scripts.sglx_multi_run_pipeline_mycopy")
    buffer = _EmitBuffer(logger.output)
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        module.run_pipeline(settings)
    buffer.flush()

    if cfg.run_tprime:
        logger.log("Collecting TPrime event files for stimulus metadata")
        _collect_tprime_event_files(cfg, logger)

    logger.log(f"ecephys_spike_sorting_LNE pipeline finished: {cfg.catgt_root}")


def run_catgt_only(cfg: PipelineConfig, logger: StageLogger) -> None:
    os.environ["MPLBACKEND"] = "Agg"
    _prepare_import_path(cfg)
    cfg.catgt_output_dir.mkdir(parents=True, exist_ok=True)
    cfg.json_dir.mkdir(parents=True, exist_ok=True)

    settings = _settings_from_config(cfg)
    settings["run_CatGT"] = True
    settings["runTPrime"] = False
    settings["modules"] = []
    settings_path = cfg.logs_dir / "catgt_only_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    logger.log(f"Running CatGT only with settings: {settings_path}")

    module = importlib.import_module("ecephys_spike_sorting.scripts.sglx_multi_run_pipeline_mycopy")
    buffer = _EmitBuffer(logger.output)
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        module.run_pipeline(settings)
    buffer.flush()

    logger.log(f"CatGT only finished: {cfg.catgt_root}")
