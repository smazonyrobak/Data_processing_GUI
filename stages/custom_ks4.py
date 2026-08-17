from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipeline_config import PipelineConfig
from stages.common import StageLogger


def _prepend_path(path: Path) -> None:
    text = str(path.resolve())
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)
    current = os.environ.get("PYTHONPATH", "")
    parts = [part for part in current.split(os.pathsep) if part and part != text]
    os.environ["PYTHONPATH"] = os.pathsep.join([text, *parts])


def _clear_kilosort_imports() -> None:
    for name in list(sys.modules):
        if name == "kilosort" or name.startswith("kilosort."):
            del sys.modules[name]
    sys.modules.pop("ecephys_spike_sorting.modules.ks4_helper.__main__", None)


def _catgt_ap_file(cfg: PipelineConfig, probe_id: int) -> Path:
    run, gate = cfg.run_and_gate
    probe_dir = cfg.catgt_root / f"{run}_g{gate}_imec{probe_id}"
    expected = probe_dir / f"{run}_g{gate}_tcat.imec{probe_id}.ap.bin"
    if expected.exists():
        return expected
    matches = sorted(probe_dir.glob(f"*_tcat.imec{probe_id}.ap.bin"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Missing CatGT AP binary for custom KS4. Run CatGT first or select the "
        f"ecephys_pipeline stage with Run CatGT enabled. Checked: {expected}"
    )


def _threshold_text_for_probe(cfg: PipelineConfig, probe_index: int) -> str:
    regions = cfg.normalized_brain_regions()
    region = regions[probe_index] if probe_index < len(regions) else "default"
    return cfg.ks_th4_by_region.get(region, cfg.ks_th4_by_region.get("default", "[8,9]"))


def _validate_threshold_text(text: str) -> None:
    values = [part.strip() for part in text.strip()[1:-1].split(",")]
    if len(values) != 2:
        raise ValueError(f"KS4 threshold entry must look like [universal,learned], got {text!r}")
    float(values[0])
    float(values[1])


def _enable_residual_or_state_sidecar(args: dict[str, Any]) -> str:
    defaults = importlib.import_module("kilosort.parameters").DEFAULT_SETTINGS
    ks4_params = args["ks4_helper_params"]["ks4_params"]
    if "save_residual_states" in defaults:
        ks4_params["save_residual_states"] = True
        ks4_params["nearest_chans"] = 15
        return "Kilosort4S residual-state sidecar"
    if "state_compat_enabled" in defaults:
        ks4_params["state_compat_enabled"] = True
        return "legacy KS4 state-compat sidecar"
    raise RuntimeError("Selected KS4 repository does not expose a residual/state sidecar setting.")


def _ks4_args(
    cfg: PipelineConfig,
    probe_id: int,
    probe_index: int,
    input_file: Path,
    output_dir: Path,
    args_path: Path,
    create_input_json,
) -> dict[str, Any]:
    th_text = _threshold_text_for_probe(cfg, probe_index)
    _validate_threshold_text(th_text)
    run, gate = cfg.run_and_gate
    args = create_input_json(
        str(args_path),
        npx_directory=str(cfg.spikeglx_data_dir),
        continuous_file=str(input_file),
        input_meta_path=str(input_file.with_suffix(".meta")),
        kilosort_output_directory=str(output_dir),
        extracted_data_directory=str(input_file.parent),
        catGT_run_name=f"{run}_imec{probe_id}",
        gate_string=str(gate),
        trigger_string=f"{cfg.trial_start},{cfg.trial_end}",
        probe_string=str(probe_id),
        ks_ver="4",
        ks_remDup=cfg.ks_remDup,
        ks_finalSplits=1,
        ks_labelGood=1,
        ks_saveRez=cfg.ks_saveRez,
        ks_copy_fproc=cfg.ks_copy_fproc,
        ks_helper_noise_threshold=cfg.ks_helper_noise_threshold,
        ks_minfr_goodchannels=cfg.ks_minfr_goodchannels,
        ks_whiteningRadius_um=cfg.ks_whiteningRadius_um,
        ks_doFilter=cfg.ks_doFilter,
        ks_Th=th_text,
        ks_CSBseed=cfg.ks_CSBseed,
        ks_LTseed=cfg.ks_LTseed,
        ks_templateRadius_um=cfg.ks_templateRadius_um,
        ks_nblocks=cfg.ks_nblocks,
        ks_CAR=cfg.ks_CAR,
        ks_tmin=cfg.ks_tmin,
        ks_tmax=cfg.ks_tmax,
        ks4_det=cfg.ks4_det,
        ks_nNeighbors_sites_fix=0,
        ks4_duplicate_spike_ms=cfg.ks4_duplicate_spike_ms,
        ks4_min_template_size_um=cfg.ks4_min_template_size_um,
        include_pc_metrics=cfg.include_pc_metrics,
        c_Waves_snr_um=cfg.c_waves_snr_um,
        c_Waves_calc_half=cfg.c_waves_calc_half,
        ecephys_directory=str(cfg.ecephys_package_dir),
        kilosort_repository=cfg.kilosort_repository,
        kilosort20_repository=cfg.kilosort20_repository,
        kilosort25_repository=cfg.kilosort25_repository,
        kilosort30_repository=cfg.kilosort30_repository,
        npy_matlab_repository=cfg.npy_matlab_repository,
        catGTPath="",
        tPrime_path="",
        cWaves_path=cfg.cwaves_path,
        kilosort_output_tmp=str(cfg.kilosort_tmp_dir),
        ks_make_copy=False,
    )
    args["ks4_helper_params"]["save_extra_vars"] = True
    args["ks4_helper_params"]["ks_make_copy"] = False
    sidecar_name = _enable_residual_or_state_sidecar(args)
    args_path.write_text(json.dumps(args, indent=2), encoding="utf-8")
    args["_custom_ks4_sidecar_name"] = sidecar_name
    return args


TOP_LEVEL_CURATED_FILES = (
    "spike_times.npy",
    "spike_clusters.npy",
    "spike_templates.npy",
    "spike_detection_templates.npy",
    "amplitudes.npy",
    "templates.npy",
    "templates_ind.npy",
    "similar_templates.npy",
    "channel_map.npy",
    "channel_positions.npy",
    "channel_shanks.npy",
    "pc_features.npy",
    "pc_feature_ind.npy",
    "whitening_mat.npy",
    "whitening_mat_inv.npy",
    "params.py",
    "ops.npy",
    "kept_spikes.npy",
    "metrics.csv",
    "waveform_metrics.csv",
    "mean_waveforms.npy",
    "cluster_group.tsv",
    "cluster_KSLabel.tsv",
    "cluster_ContamPct.tsv",
    "cluster_Amplitude.tsv",
    "spike_event_ids.npy",
    "full_event_ids.npy",
    "residual_states.h5",
    "residual_spike_states.csv",
    "state_compat_enabled.json",
    "spike_local_template_id.npy",
    "spike_parent_template_id.npy",
    "spike_template_window_id.npy",
    "spike_match_score.npy",
    "spike_residual_energy.npy",
    "spike_residual_energy_normed.npy",
    "local_template_table.tsv",
    "local_template_parent_id.npy",
    "local_template_window_id.npy",
    "local_template_valid_start_sample.npy",
    "local_template_valid_end_sample.npy",
    "feature_scale_spikes.npy",
    "feature_scale_template.npy",
    "graph_features_scaled.npy",
    "cluster_anisotropy.tsv",
    "cluster_state_axis.npy",
    "cluster_feature_mean.npy",
    "spike_state_coord.npy",
)


def _copy_if_exists(src: Path, dst: Path, manifest: list[dict[str, str]]) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append({"source": str(src), "destination": str(dst)})


def _copy_custom_results_to_preprocessed(
    cfg: PipelineConfig,
    probe_id: int,
    output_dir: Path,
    logger: StageLogger,
) -> Path:
    curated_dir = cfg.preprocessed_dir / "custom_ks4_state_compat" / f"imec{probe_id}"
    if curated_dir.exists():
        shutil.rmtree(curated_dir)
    copied: list[dict[str, str]] = []

    for name in TOP_LEVEL_CURATED_FILES:
        _copy_if_exists(output_dir / name, curated_dir / name, copied)

    manifest_path = curated_dir / "custom_ks4_state_compat_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(copied, indent=2), encoding="utf-8")
    logger.output(f"Modified KS4 curated copy: {curated_dir}")
    logger.output(f"Modified KS4 curated files copied: {len(copied)}")
    return curated_dir


def _run_ecephys_module(module: str, args_path: Path, output_json: Path, logger: StageLogger) -> None:
    command = [
        sys.executable,
        "-W",
        "ignore",
        "-m",
        f"ecephys_spike_sorting.modules.{module}",
        "--input_json",
        str(args_path),
        "--output_json",
        str(output_json),
    ]
    logger.output(" ".join(command))
    subprocess.check_call(command)


def copy_existing_custom_ks4_to_preprocessed(cfg: PipelineConfig, logger: StageLogger) -> list[Path]:
    copied_dirs: list[Path] = []
    for probe_id in cfg.normalized_probe_ids():
        output_dir = cfg.custom_ks4_probe_sort_dir(probe_id)
        if not output_dir.exists():
            raise FileNotFoundError(f"Custom KS4 output does not exist for probe {probe_id}: {output_dir}")
        copied_dirs.append(_copy_custom_results_to_preprocessed(cfg, probe_id, output_dir, logger))
    return copied_dirs


def run_custom_ks4(cfg: PipelineConfig, logger: StageLogger) -> None:
    custom_repo = cfg.custom_kilosort_repo_dir
    if not (custom_repo / "kilosort").is_dir():
        raise FileNotFoundError(f"Custom KS4 repository does not contain a kilosort package: {custom_repo}")

    _prepend_path(custom_repo)
    _prepend_path(cfg.ecephys_repo_dir)
    _clear_kilosort_imports()
    helper = importlib.import_module("ecephys_spike_sorting.modules.ks4_helper.__main__")
    create_input = importlib.import_module("ecephys_spike_sorting.scripts.create_input_json")

    manifest: list[dict[str, Any]] = []
    for probe_index, probe_id in enumerate(cfg.normalized_probe_ids()):
        input_file = _catgt_ap_file(cfg, probe_id)
        output_dir = cfg.custom_ks4_probe_sort_dir(probe_id)
        args_path = cfg.json_dir / f"{cfg.run_label}_imec{probe_id}_custom_ks4-input.json"
        args_path.parent.mkdir(parents=True, exist_ok=True)
        args = _ks4_args(
            cfg, probe_id, probe_index, input_file, output_dir, args_path,
            create_input.createInputJson,
        )

        sidecar_name = args.pop("_custom_ks4_sidecar_name", "custom KS4 sidecar")
        logger.log(f"Running {sidecar_name} for probe {probe_id}")
        logger.output(f"Kilosort repository: {custom_repo}")
        logger.output(f"Kilosort input: {input_file}")
        logger.output(f"Kilosort output: {output_dir}")
        helper.run_ks4(args)

        if cfg.custom_ks4_run_quality_metrics:
            prefix = f"{cfg.run_label}_imec{probe_id}_custom_ks4"
            _run_ecephys_module("mean_waveforms", args_path, cfg.json_dir / f"{prefix}-mean_waveforms-output.json", logger)
            _run_ecephys_module("quality_metrics", args_path, cfg.json_dir / f"{prefix}-quality_metrics-output.json", logger)

        curated_dir = _copy_custom_results_to_preprocessed(cfg, probe_id, output_dir, logger)
        manifest.append({
            "probe_id": int(probe_id),
            "input_file": str(input_file),
            "output_dir": str(output_dir),
            "curated_output_dir": str(curated_dir),
            "input_json": str(args_path),
            "sidecar": sidecar_name,
            "quality_metrics": bool(cfg.custom_ks4_run_quality_metrics),
        })

    manifest_path = cfg.logs_dir / "custom_ks4_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.log(f"Kilosort4S/custom KS4 sidecar stage finished. Manifest: {manifest_path}")
