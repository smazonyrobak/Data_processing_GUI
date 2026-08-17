from __future__ import annotations

import datetime as dt
from pathlib import Path

from pipeline_config import PipelineConfig
from stages.build_preprocessing_output import build_preprocessing_output
from stages.custom_ks4 import run_custom_ks4
from stages.ecephys_pipeline import run_catgt_only, run_ecephys_spike_sorting
from stages.state_sorter import run_state_sorter
from stages.stimulus_metadata import build_stimulus_metadata


STAGES = [
    ("catgt", "CatGT only", run_catgt_only),
    ("ecephys_pipeline", "ecephys_spike_sorting_LNE pipeline", run_ecephys_spike_sorting),
    ("custom_ks4", "Kilosort4S residual states", run_custom_ks4),
    ("state_sorter", "StateSorter waveform states", run_state_sorter),
    ("stimulus_metadata", "Stimulus metadata", build_stimulus_metadata),
    ("preprocessing_output", "Build preprocessing output", build_preprocessing_output),
]


class FileLogger:
    def __init__(self, cfg: PipelineConfig, emit_log=None, emit_output=None) -> None:
        self.cfg = cfg
        self.emit_log = emit_log
        self.emit_output = emit_output
        cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_path = cfg.logs_dir / "pipeline_run_log.txt"
        self.console_path = cfg.logs_dir / "pipeline_console_output.txt"

    def _append(self, path: Path, message: str) -> None:
        with path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(message + "\n")

    def log(self, message: str) -> None:
        line = f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}"
        self._append(self.run_log_path, line)
        if self.emit_log:
            self.emit_log(line)

    def output(self, message: str) -> None:
        self._append(self.console_path, message)
        if self.emit_output:
            self.emit_output(message)


def run_selected_stages(
    cfg: PipelineConfig,
    selected_stage_keys: list[str],
    logger: FileLogger,
    *,
    allow_existing_output: bool = False,
) -> None:
    cfg.validate_for_run(selected_stage_keys, allow_existing_output=allow_existing_output)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(cfg.logs_dir / "pipeline_config.json")
    selected = set(selected_stage_keys)
    for key, label, func in STAGES:
        if key not in selected:
            continue
        logger.log(f"Starting stage: {label}")
        func(cfg, logger)
        logger.log(f"Finished stage: {label}")
