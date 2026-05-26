from __future__ import annotations

import importlib
import json
import os
import shutil
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


def _somatic_ops_overrides(cfg: PipelineConfig) -> dict[str, Any]:
    return {
        "somatic_output_subdir": "somatic_identity",
        "somatic_fragment_merge_max_depth_um": float(cfg.somatic_fragment_merge_max_depth_um),
        "somatic_fragment_merge_same_shank_only": bool(cfg.somatic_fragment_merge_same_shank_only),
        "somatic_fragment_merge_min_soma_similarity": float(cfg.somatic_fragment_merge_min_soma_similarity),
        "somatic_fragment_merge_max_isi_violation_fraction": float(cfg.somatic_fragment_merge_max_isi_violation_fraction),
        "somatic_fragment_merge_max_duplicate_fraction": float(cfg.somatic_fragment_merge_max_duplicate_fraction),
        "somatic_state_group_full_template_similarity": float(cfg.somatic_state_group_full_template_similarity),
        "somatic_refractory_ms": float(cfg.somatic_refractory_ms),
        "somatic_duplicate_ms": float(cfg.somatic_duplicate_ms),
        "somatic_conflict_ratio_threshold": float(cfg.somatic_conflict_ratio_threshold),
        "somatic_max_spikes_per_unit_for_conflict_metrics": int(cfg.somatic_max_spikes_per_unit_for_conflict_metrics),
        "somatic_state_channel_radius": int(cfg.somatic_state_channel_radius),
        "tmin": float(cfg.ks_tmin),
    }


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
    "metrics.csv",
    "waveform_metrics.csv",
    "mean_waveforms.npy",
    "cluster_group.tsv",
    "cluster_KSLabel.tsv",
    "cluster_ContamPct.tsv",
    "cluster_Amplitude.tsv",
)


SOMATIC_CURATED_FILES = (
    "spike_times.npy",
    "spike_unit_ids.npy",
    "spike_state_ids.npy",
    "spike_ks4_fragment_ids.npy",
    "unit_soma_templates.npy",
    "unit_full_mean_templates.npy",
    "unit_num_states.tsv",
    "unit_quality_metrics.tsv",
    "unit_identity_conflict_fraction.tsv",
    "state_parent_unit.tsv",
    "state_ks4_fragments.tsv",
    "state_templates_local.npy",
    "state_template_channels.npy",
    "state_spike_counts.tsv",
    "state_fraction.tsv",
    "ks4_fragment_to_unit.tsv",
    "ks4_fragment_to_state.tsv",
    "somatic_merge_evidence.tsv",
    "somatic_identity_metadata.json",
    "somatic_input_manifest.json",
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
    curated_dir = cfg.preprocessed_dir / "custom_ks4_somatic" / f"imec{probe_id}"
    if curated_dir.exists():
        shutil.rmtree(curated_dir)
    somatic_src = output_dir / "somatic_identity"
    somatic_dst = curated_dir / "somatic_identity"
    copied: list[dict[str, str]] = []

    for name in TOP_LEVEL_CURATED_FILES:
        _copy_if_exists(output_dir / name, curated_dir / name, copied)
    for name in SOMATIC_CURATED_FILES:
        _copy_if_exists(somatic_src / name, somatic_dst / name, copied)

    manifest_path = curated_dir / "custom_ks4_somatic_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(copied, indent=2), encoding="utf-8")
    logger.output(f"Custom somatic curated copy: {curated_dir}")
    logger.output(f"Custom somatic curated files copied: {len(copied)}")
    return curated_dir


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
    _clear_kilosort_imports()
    merger = importlib.import_module("kilosort.somatic_fragment_merge")

    manifest: list[dict[str, Any]] = []
    ops_overrides = _somatic_ops_overrides(cfg)
    subset_duration = float(cfg.custom_ks4_reference_duration_s or 0.0)

    for probe_id in cfg.normalized_probe_ids():
        reference_dir = cfg.probe_sort_dir(probe_id)
        output_dir = cfg.custom_ks4_probe_sort_dir(probe_id)
        if not reference_dir.exists():
            raise FileNotFoundError(
                "Regular KS4 output is required before custom somatic fragment merging. "
                f"Missing: {reference_dir}"
            )

        logger.log(f"Merging regular KS4 fragments into somatic units for probe {probe_id}")
        logger.output(f"Reference regular KS4: {reference_dir}")
        logger.output(f"Custom somatic output: {output_dir}")
        if subset_duration > 0:
            logger.output(f"Fragment merge subset duration: {subset_duration:g} seconds")

        result = merger.run_fragment_template_merge(
            reference_dir,
            output_dir,
            ops_overrides=ops_overrides,
            subset_duration_s=subset_duration,
            progress=logger.output,
        )
        curated_dir = _copy_custom_results_to_preprocessed(cfg, probe_id, output_dir, logger)
        metadata = result["metadata"]
        logger.output(
            "Custom somatic fragment merge: "
            f"{metadata['n_ks4_fragments']} fragments -> "
            f"{metadata['n_custom_units']} units -> {metadata['n_states']} states"
        )
        manifest.append({
            "probe_id": int(probe_id),
            "reference_ks4_dir": str(reference_dir),
            "output_dir": str(output_dir),
            "curated_output_dir": str(curated_dir),
            "somatic_identity_dir": str(result["somatic_output_dir"]),
            "method": metadata["method"],
            "n_ks4_fragments": int(metadata["n_ks4_fragments"]),
            "n_custom_units": int(metadata["n_custom_units"]),
            "n_states": int(metadata["n_states"]),
            "custom_ks4_reference_duration_s": subset_duration,
        })

    manifest_path = cfg.logs_dir / "custom_ks4_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.log(f"Custom somatic fragment merge finished. Manifest: {manifest_path}")
