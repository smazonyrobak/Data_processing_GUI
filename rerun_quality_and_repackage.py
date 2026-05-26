from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


GUI_ROOT = Path(__file__).resolve().parent


def emit_console(message: str) -> None:
    sys.__stdout__.write(str(message) + "\n")
    sys.__stdout__.flush()


def existing_config_path(target: Path, explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            target / "processing_work" / "logs" / "pipeline_config.json",
            target / "logs" / "pipeline_config.json",
            GUI_ROOT / "pipeline_config.json",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find pipeline_config.json. Checked:\n"
        + "\n".join(str(path) for path in candidates)
    )


def sort_dir(root: Path, run: str, gate: int, probe_id: int, ks_tag: str) -> Path | None:
    probe_dir = root / f"{run}_g{gate}_imec{probe_id}"
    candidates = [
        probe_dir / f"imec{probe_id}_{ks_tag}",
        probe_dir / "kilosort4",
    ]
    return next((path for path in candidates if (path / "spike_clusters.npy").exists()), None)


def find_catgt_root(
    cfg,
    config_data: dict,
    target: Path,
    explicit: str | None,
) -> Path:
    run, gate = cfg.run_and_gate
    probes = cfg.normalized_probe_ids()
    root_name = f"catgt_{run}_g{gate}"
    candidates: list[Path] = []

    if explicit:
        candidates.append(Path(explicit))
    if str(config_data.get("catgt_dest", "")).strip():
        candidates.append(Path(config_data["catgt_dest"]) / root_name)

    candidates.extend(
        [
            cfg.catgt_root,
            target / "processing_work" / "CatGT_runs" / root_name,
            cfg.spikeglx_data_dir / f"catgt_{run}_output" / root_name,
        ]
    )

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique_candidates.append(path)
            seen.add(key)

    for root in unique_candidates:
        if all(sort_dir(root, run, gate, probe, cfg.ks_output_tag) is not None for probe in probes):
            return root

    raise FileNotFoundError(
        "Could not find CatGT/Kilosort output root. Checked:\n"
        + "\n".join(str(path) for path in unique_candidates)
    )


def latest_versioned_path(base: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    if base.exists():
        candidates.append((0, base))
    pattern = re.compile(rf"^{re.escape(base.stem)}_(\d+){re.escape(base.suffix)}$")
    for path in base.parent.glob(f"{base.stem}_*{base.suffix}"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def promote_latest_outputs(ks_dir: Path) -> None:
    for name in ["metrics.csv", "waveform_metrics.csv", "mean_waveforms.npy", "cluster_snr.npy"]:
        base = ks_dir / name
        latest = latest_versioned_path(base)
        if latest is not None and latest != base:
            shutil.copy2(latest, base)
            print(f"promoted {latest.name} -> {base.name}")


class FileLogger:
    def __init__(self, cfg, emit_log=print, emit_output=print) -> None:
        self.emit_log = emit_log
        self.emit_output = emit_output
        cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_path = cfg.logs_dir / "pipeline_run_log.txt"
        self.console_path = cfg.logs_dir / "pipeline_console_output.txt"

    def _append(self, path: Path, message: str) -> None:
        with path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(message + "\n")

    def log(self, message: str) -> None:
        self._append(self.run_log_path, message)
        if self.emit_log:
            self.emit_log(message)

    def output(self, message: str) -> None:
        self._append(self.console_path, message)
        if self.emit_output:
            self.emit_output(message)


def make_rerun_config(config_data: dict, target: Path, catgt_root: Path, pipeline_config_cls):
    class RerunConfig(pipeline_config_cls):
        @property
        def run_output_dir(self) -> Path:
            return target

        @property
        def preprocessed_dir(self) -> Path:
            return target

        @property
        def processing_work_dir(self) -> Path:
            return target / "processing_work"

        @property
        def logs_dir(self) -> Path:
            return target / "logs"

        @property
        def catgt_output_dir(self) -> Path:
            return catgt_root.parent

        @property
        def catgt_root(self) -> Path:
            return catgt_root

        @property
        def json_dir(self) -> Path:
            return target / "logs" / "rerun_ecephys_json"

        @property
        def kilosort_tmp_dir(self) -> Path:
            return target / "logs" / "kilosort_tmp"

    return RerunConfig.from_dict(config_data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Existing preprocessed run directory to overwrite.")
    parser.add_argument("--config", help="Existing pipeline_config.json. If omitted, common locations are searched.")
    parser.add_argument("--catgt-root", help="Existing catgt_<run>_g<gate> directory. If omitted, common locations are searched.")
    parser.add_argument(
        "--rerun-mean-waveforms",
        action="store_true",
        help="Also rerun mean_waveforms before quality_metrics. By default only quality_metrics is rerun.",
    )
    args = parser.parse_args()

    if str(GUI_ROOT) not in sys.path:
        sys.path.insert(0, str(GUI_ROOT))

    from pipeline_config import PipelineConfig
    from stages.build_preprocessing_output import build_preprocessing_output
    from stages.ecephys_pipeline import run_ecephys_spike_sorting

    target = Path(args.target)
    if not target.exists():
        raise FileNotFoundError(f"Target preprocessed directory does not exist: {target}")

    config_path = existing_config_path(target, args.config)
    config_data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    base_cfg = PipelineConfig.from_dict(config_data)
    catgt_root = find_catgt_root(base_cfg, config_data, target, args.catgt_root)

    cfg = make_rerun_config(config_data, target, catgt_root, PipelineConfig)
    cfg.ecephys_directory = str(GUI_ROOT / "ecephys_spike_sorting_LNE" / "ecephys_spike_sorting")
    cfg.run_catgt = False
    cfg.run_tprime = False
    cfg.modules = ["quality_metrics"]
    if args.rerun_mean_waveforms:
        cfg.modules.insert(0, "mean_waveforms")

    target.joinpath("logs").mkdir(parents=True, exist_ok=True)
    cfg.save(target / "logs" / "rerun_quality_config.json")

    logger = FileLogger(cfg, emit_log=emit_console, emit_output=emit_console)
    emit_console(f"config: {config_path}")
    emit_console(f"target: {target}")
    emit_console(f"catgt root: {catgt_root}")
    emit_console(f"modules: {cfg.modules}")

    run_ecephys_spike_sorting(cfg, logger)

    run, gate = cfg.run_and_gate
    for probe_id in cfg.normalized_probe_ids():
        ks_dir = sort_dir(catgt_root, run, gate, probe_id, cfg.ks_output_tag)
        if ks_dir is None:
            raise FileNotFoundError(f"Missing Kilosort output for probe {probe_id} under {catgt_root}")
        promote_latest_outputs(ks_dir)

    build_preprocessing_output(cfg, logger)
    emit_console("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
