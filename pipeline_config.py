from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any


class PipelineConfigError(ValueError):
    pass


def _norm_path(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path.resolve(strict=False))))


def _same_or_nested(path: Path, parent: Path) -> bool:
    path_norm = _norm_path(path)
    parent_norm = _norm_path(parent)
    try:
        return os.path.commonpath([path_norm, parent_norm]) == parent_norm
    except ValueError:
        return False


def _overlaps(a: Path, b: Path) -> bool:
    return _same_or_nested(a, b) or _same_or_nested(b, a)


def _require_absolute(label: str, path: Path, errors: list[str]) -> None:
    if not path.is_absolute():
        errors.append(f"{label} must be an absolute path: {path}")


def _require_existing(label: str, path: Path, errors: list[str], *, directory: bool | None = None) -> None:
    if not path.exists():
        errors.append(f"{label} does not exist: {path}")
        return
    if directory is True and not path.is_dir():
        errors.append(f"{label} must be a directory: {path}")
    if directory is False and not path.is_file():
        errors.append(f"{label} must be a file: {path}")


def _parent_exists(label: str, path: Path, errors: list[str]) -> None:
    parent = path if path.exists() and path.is_dir() else path.parent
    if not parent.exists():
        errors.append(f"{label} parent directory does not exist: {parent}")


def _has_contents(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return True
    try:
        next(path.iterdir())
    except StopIteration:
        return False
    except OSError:
        return True
    return True


def _tokens(text: str) -> list[str]:
    return [part.strip() for part in text.split() if part.strip()]


FORBIDDEN_CATGT_PREFIXES = (
    "-dir=",
    "-dest=",
    "-run=",
    "-g=",
    "-t=",
    "-prb=",
    "-obx=",
    "-out=",
)
FORBIDDEN_CATGT_FLAGS = {"-ap", "-lf", "-ni", "-ob"}
SHELL_META_CHARS = {"&", "|", ">", "<", ";"}


@dataclass
class PipelineConfig:
    processing_python: str = r"E:\SZYMON\Documents\Miniconda\envs\data_proc_gui\python.exe"
    spikeglx_run: str = r"G:\SpikeGLX_data_saving_dir_Marcin_Szymon\SpikeGLX_savefiles\myRun_TEST_g0"
    stim_cam_run: str = r"G:\SpikeGLX_data_saving_dir_Marcin_Szymon\Stim_and_cam_savefiles\Run_TEST_stim_and_cam"
    catgt_exe: str = r"E:\SZYMON\Documents\CatGT-win\CatGT.exe"
    tprime_exe: str = r"E:\SZYMON\Documents\TPrime-win\TPrime.exe"
    cwaves_path: str = r"E:\SZYMON\Documents\C_Waves-win"
    preprocessed_root: str = ""
    catgt_dest: str = ""
    json_directory: str = ""
    kilosort_output_tmp: str = ""
    ecephys_directory: str = ""
    npy_matlab_repository: str = ""
    kilosort_repository: str = ""
    kilosort20_repository: str = ""
    kilosort25_repository: str = ""
    kilosort30_repository: str = ""
    custom_kilosort_repository: str = ""
    custom_ks4_reference_duration_s: float = 0.0
    custom_ks4_run_quality_metrics: bool = True
    somatic_fragment_merge_max_depth_um: float = 50.0
    somatic_fragment_merge_same_shank_only: bool = True
    somatic_fragment_merge_min_soma_similarity: float = 0.9
    somatic_fragment_merge_max_isi_violation_fraction: float = 0.1
    somatic_fragment_merge_max_duplicate_fraction: float = 0.1
    somatic_state_group_full_template_similarity: float = 0.98
    somatic_refractory_ms: float = 1.5
    somatic_duplicate_ms: float = 0.25
    somatic_conflict_ratio_threshold: float = 0.9
    somatic_max_spikes_per_unit_for_conflict_metrics: int = 1000
    somatic_state_channel_radius: int = 30

    gate_index: int = 0
    trial_start: int = 0
    trial_end: int = 0
    n_probes: int = 1
    probe_ids: list[int] = field(default_factory=lambda: [0])
    brain_regions: list[str] = field(default_factory=lambda: ["cortex"])
    onebox_streams: str = ""

    ni_word: int = 0
    sync_bit: int = 0
    rotation_bit: int = 1
    switching_bit: int = 2
    camera_bit: int = 3
    tprime_syncperiod_s: float = 1.0
    tprime_reference_stream: str = "imec0"
    ap_sync_word: int = 384
    ap_sync_bit: int = 6
    sync_threshold: int = 500
    event_threshold: int = 0
    sync_crop_enabled: bool = False
    sync_crop_start_index: int = 0
    sync_crop_end_index: int = 0

    drop_first_switching_intervals: int = 2
    drop_first_rotation_intervals: int = 0

    catgt_ap_filter: str = "butter,12,300,9000"
    catgt_loccar_um: str = "40,140"
    catgt_gfix: str = "0.40,0.10,0.02"
    log_name: str = "pipeline_log.csv"
    run_catgt: bool = True
    run_tprime: bool = True
    ni_present: bool = True
    obx_present: bool = False
    car_mode: str = "gblcar"
    loccar_min: int = 40
    loccar_max: int = 160
    process_lf: bool = True
    catgt_cmd_string: str = "-prb_fld -out_prb_fld -apfilter=butter,12,300,10000 -lffilter=butter,12,1,500 -gfix=0.4,0.10,0.02"
    ni_obx_extract_string: str = (
        "-xd=0,0,-1,1,0 -xid=0,0,-1,1,0 "
        "-xd=0,0,-1,2,0 -xid=0,0,-1,2,0 "
        "-xd=0,0,-1,3,0 -xid=0,0,-1,3,0"
    )
    create_aux_timepoints: bool = False
    event_ex_param_str: str = ""
    probe_geometry_mode: str = "metadata"
    custom_probe_geometry: str = ""

    ks_ver: str = "4"
    modules: list[str] = field(default_factory=lambda: ["ks4_helper", "kilosort_postprocessing", "mean_waveforms", "quality_metrics"])
    ref_per_ms_by_region: dict[str, float] = field(default_factory=lambda: {"cortex": 1.5})
    ks_th2_by_region: dict[str, str] = field(default_factory=lambda: {"default": "[10,4]", "cortex": "[10,4]", "medulla": "[10,4]", "thalamus": "[10,4]"})
    ks_th3_by_region: dict[str, str] = field(default_factory=lambda: {"default": "[9,9]", "cortex": "[9,9]", "medulla": "[9,9]", "thalamus": "[9,9]"})
    ks_th4_by_region: dict[str, str] = field(default_factory=lambda: {"default": "[8,9]", "cortex": "[8,9]", "medulla": "[8,9]", "thalamus": "[8,9]"})
    ks_remDup: int = 0
    ks_saveRez: int = 1
    ks_copy_fproc: int = 0
    ks_templateRadius_um: int = 163
    ks_whiteningRadius_um: int = 163
    ks_minfr_goodchannels: float = 0.1
    ks_CAR: int = 0
    ks_nblocks: int = 6
    ks_doFilter: int = 0
    ks4_duplicate_spike_ms: float = 0.25
    ks4_min_template_size_um: int = 10
    ks4_det: bool = False
    ks_tmin: float = 0.0
    ks_tmax: float = -1.0
    ks_CSBseed: int = 1
    ks_LTseed: int = 1
    ks_helper_noise_threshold: int = 20
    c_waves_snr_um: int = 160
    c_waves_calc_half: bool = False
    include_pc_metrics: bool = True
    noise_template_use_rf: bool = False

    @property
    def spikeglx_path(self) -> Path:
        return Path(self.spikeglx_run)

    @property
    def stim_cam_path(self) -> Path:
        return Path(self.stim_cam_run)

    @property
    def run_and_gate(self) -> tuple[str, int]:
        run, gate = self.spikeglx_path.name.rsplit("_g", 1)
        return run, int(gate)

    @property
    def run_label(self) -> str:
        return self.spikeglx_path.name

    @property
    def spikeglx_data_dir(self) -> Path:
        return self.spikeglx_path.parent

    @property
    def catgt_output_dir(self) -> Path:
        return self.processing_work_dir / "CatGT_runs"

    @property
    def catgt_root(self) -> Path:
        run, gate = self.run_and_gate
        return self.catgt_output_dir / f"catgt_{run}_g{gate}"

    @property
    def json_dir(self) -> Path:
        return self.processing_work_dir / "ecephys_json"

    @property
    def kilosort_tmp_dir(self) -> Path:
        return self.processing_work_dir / "kilosort_tmp"

    @property
    def ecephys_package_dir(self) -> Path:
        if self.ecephys_directory.strip():
            return Path(self.ecephys_directory)
        raise PipelineConfigError("ecephys package directory is required.")

    @property
    def ecephys_repo_dir(self) -> Path:
        package_dir = self.ecephys_package_dir
        return package_dir.parent if package_dir.name == "ecephys_spike_sorting" else package_dir

    @property
    def ks_output_tag(self) -> str:
        return {"2.0": "ks2", "2.5": "ks25", "3.0": "ks3", "4": "ks4"}.get(str(self.ks_ver), "ks4")

    def probe_sort_dir(self, probe_id: int) -> Path:
        run, gate = self.run_and_gate
        return self.catgt_root / f"{run}_g{gate}_imec{probe_id}" / f"imec{probe_id}_{self.ks_output_tag}"

    @property
    def custom_kilosort_repo_dir(self) -> Path:
        if self.custom_kilosort_repository.strip():
            return Path(self.custom_kilosort_repository)
        return Path(__file__).resolve().parent / "Kilosort_state_enhanced"

    def custom_ks4_probe_sort_dir(self, probe_id: int) -> Path:
        run, gate = self.run_and_gate
        return self.catgt_root / f"{run}_g{gate}_imec{probe_id}" / f"imec{probe_id}_ks4_somatic"

    @property
    def output_root(self) -> Path:
        if not self.preprocessed_root.strip():
            raise PipelineConfigError("Analysis output root is required. Choose a directory outside the SpikeGLX raw data folder.")
        return Path(self.preprocessed_root)

    @property
    def run_output_dir(self) -> Path:
        return self.output_root / self.run_label

    @property
    def preprocessed_dir(self) -> Path:
        return self.run_output_dir / "preprocessed_data"

    @property
    def processing_work_dir(self) -> Path:
        return self.run_output_dir / "processing_work"

    @property
    def logs_dir(self) -> Path:
        return self.processing_work_dir / "logs"

    def normalized_probe_ids(self) -> list[int]:
        if self.probe_ids:
            return [int(p) for p in self.probe_ids]
        return list(range(int(self.n_probes)))

    def normalized_brain_regions(self) -> list[str]:
        regions = [str(region).strip() for region in self.brain_regions if str(region).strip()]
        probes = self.normalized_probe_ids()
        if not regions:
            return ["cortex"] * len(probes)
        if len(regions) == 1 and len(probes) > 1:
            return regions * len(probes)
        if len(regions) != len(probes):
            raise ValueError("Brain regions must be one value or match the number of selected probes.")
        return regions

    def _required_text_fields(self, selected_stage_keys: list[str] | None = None) -> dict[str, str]:
        selected = set(selected_stage_keys or [])
        require_ecephys = not selected or "ecephys_pipeline" in selected
        fields = {
            "processing_python": "Processing Python",
            "spikeglx_run": "SpikeGLX raw run",
            "stim_cam_run": "Stim/cam run",
            "preprocessed_root": "Analysis output root",
        }
        if require_ecephys:
            fields["ecephys_directory"] = "ecephys package"
            if self.run_catgt:
                fields["catgt_exe"] = "CatGT"
            if self.run_tprime:
                fields["tprime_exe"] = "TPrime"
            if "mean_waveforms" in self.modules:
                fields["cwaves_path"] = "C_Waves"
            if str(self.ks_ver) in {"2.0", "2.5", "3.0"} or "kilosort_helper" in self.modules:
                fields["npy_matlab_repository"] = "npy-matlab repo"
                fields["kilosort_repository"] = "Kilosort repo"
        return fields

    def _validate_required_paths(self, selected_stage_keys: list[str] | None, errors: list[str]) -> None:
        for attr, label in self._required_text_fields(selected_stage_keys).items():
            if not str(getattr(self, attr)).strip():
                errors.append(f"{label} path is required.")

    def _validate_existing_paths(self, selected_stage_keys: list[str] | None, errors: list[str]) -> None:
        selected = set(selected_stage_keys or [])
        require_ecephys = not selected or "ecephys_pipeline" in selected

        path_specs: list[tuple[str, Path, bool | None]] = [
            ("Processing Python", Path(self.processing_python), False),
            ("SpikeGLX raw run", self.spikeglx_path, True),
            ("Stim/cam run", self.stim_cam_path, True),
        ]
        if require_ecephys:
            path_specs.append(("ecephys package", self.ecephys_package_dir, True))
            if self.run_catgt:
                path_specs.append(("CatGT", Path(self.catgt_exe), None))
            if self.run_tprime:
                path_specs.append(("TPrime", Path(self.tprime_exe), None))
            if "mean_waveforms" in self.modules:
                path_specs.append(("C_Waves", Path(self.cwaves_path), None))
            if str(self.ks_ver) in {"2.0", "2.5", "3.0"} or "kilosort_helper" in self.modules:
                path_specs.append(("npy-matlab repo", Path(self.npy_matlab_repository), True))
                path_specs.append(("Kilosort repo", Path(self.kilosort_repository), True))
        if "custom_ks4" in selected:
            path_specs.append(("Custom KS4 repo", self.custom_kilosort_repo_dir, True))

        for label, path, directory in path_specs:
            _require_absolute(label, path, errors)
            _require_existing(label, path, errors, directory=directory)

        output_specs = [("Analysis output root", self.output_root)]
        for label, path in output_specs:
            _require_absolute(label, path, errors)
            _parent_exists(label, path, errors)
            if path.exists() and not path.is_dir():
                errors.append(f"{label} must be a directory: {path}")
            if path.anchor and _norm_path(path) == _norm_path(Path(path.anchor)):
                errors.append(f"{label} cannot be a drive root: {path}")

    def _validate_raw_files(self, errors: list[str]) -> None:
        try:
            run, gate = self.run_and_gate
        except Exception as exc:
            errors.append(f"SpikeGLX raw run folder must be named like <run>_g<gate>: {exc}")
            return

        trial = int(self.trial_start)
        expected: list[Path] = []
        for probe_id in self.normalized_probe_ids():
            probe_dir = self.spikeglx_path / f"{run}_g{gate}_imec{probe_id}"
            expected.extend(
                [
                    probe_dir / f"{run}_g{gate}_t{trial}.imec{probe_id}.ap.bin",
                    probe_dir / f"{run}_g{gate}_t{trial}.imec{probe_id}.ap.meta",
                ]
            )
            if self.process_lf:
                expected.extend(
                    [
                        probe_dir / f"{run}_g{gate}_t{trial}.imec{probe_id}.lf.bin",
                        probe_dir / f"{run}_g{gate}_t{trial}.imec{probe_id}.lf.meta",
                    ]
                )
        if self.ni_present:
            expected.extend(
                [
                    self.spikeglx_path / f"{run}_g{gate}_t{trial}.nidq.bin",
                    self.spikeglx_path / f"{run}_g{gate}_t{trial}.nidq.meta",
                ]
            )

        missing = [path for path in expected if not path.exists()]
        if missing:
            errors.append("Required raw SpikeGLX files are missing; refusing to run:\n" + "\n".join(str(path) for path in missing))

    def _validate_path_separation(self, errors: list[str]) -> None:
        protected_inputs = [
            ("SpikeGLX raw data root", self.spikeglx_data_dir),
            ("SpikeGLX raw run", self.spikeglx_path),
            ("Stim/cam run", self.stim_cam_path),
        ]
        writable_dirs = [
            ("Analysis output root", self.output_root),
            ("Run output folder", self.run_output_dir),
            ("Analysis run output", self.preprocessed_dir),
            ("Processing work folder", self.processing_work_dir),
            ("CatGT output", self.catgt_output_dir),
            ("Pipeline JSON", self.json_dir),
            ("Kilosort temp", self.kilosort_tmp_dir),
        ]

        for out_label, out_path in writable_dirs:
            for in_label, in_path in protected_inputs:
                if _overlaps(out_path, in_path):
                    errors.append(
                        f"{out_label} overlaps protected input {in_label}.\n"
                        f"  {out_label}: {out_path}\n"
                        f"  {in_label}: {in_path}"
                    )

    def _validate_output_targets(self, errors: list[str]) -> None:
        if _has_contents(self.run_output_dir):
            errors.append(
                "Run output folder already exists and is not empty; refusing to mix new output with previous output.\n"
                "Delete the existing run output folder or choose a different dedicated output root.\n"
                f"  Run output folder: {self.run_output_dir}"
            )
        if self.output_root.exists() and self.output_root.is_dir():
            direct_files = [path for path in self.output_root.iterdir() if path.is_file()]
            if direct_files:
                errors.append(
                    "Analysis output root may contain only per-run folders; refusing to mix run files at the root.\n"
                    + "\n".join(str(path) for path in direct_files[:10])
                )

    def _validate_catgt_arguments(self, errors: list[str]) -> None:
        for field, label in [
            (self.catgt_cmd_string, "CatGT command string"),
            (self.ni_obx_extract_string, "NI/OBX extract string"),
        ]:
            for token in _tokens(field):
                lower = token.lower()
                if lower in FORBIDDEN_CATGT_FLAGS or any(lower.startswith(prefix) for prefix in FORBIDDEN_CATGT_PREFIXES):
                    errors.append(f"{label} cannot override raw/input/output routing option: {token}")
                if any(char in token for char in SHELL_META_CHARS):
                    errors.append(f"{label} cannot contain shell metacharacters: {token}")
                if ":\\" in token or ":/" in token:
                    errors.append(f"{label} cannot contain filesystem paths: {token}")

    def _validate_sync_crop(self, errors: list[str]) -> None:
        if not self.sync_crop_enabled:
            return
        if self.sync_crop_start_index < 0:
            errors.append("Sync crop start edge index must be >= 0.")
        if self.sync_crop_end_index <= self.sync_crop_start_index:
            errors.append("Sync crop end edge index must be greater than the start edge index.")
        if not self.ni_present:
            errors.append("Sync crop requires NI/NIDQ sync extraction to crop stimulus TTL outputs.")

    def _validate_custom_ks4_options(self, selected_stage_keys: list[str] | None, errors: list[str]) -> None:
        selected = set(selected_stage_keys or [])
        if "custom_ks4" not in selected:
            return
        if float(self.custom_ks4_reference_duration_s) < 0:
            errors.append("Custom KS4 reference duration must be >= 0 seconds. Use 0 to reuse the full regular KS4 crop.")
        if float(self.somatic_fragment_merge_max_depth_um) < 0:
            errors.append("Somatic fragment merge max depth must be >= 0.")
        if not (0 <= float(self.somatic_fragment_merge_min_soma_similarity) <= 1):
            errors.append("Somatic fragment merge minimum soma similarity must be between 0 and 1.")
        if not (0 <= float(self.somatic_fragment_merge_max_isi_violation_fraction) <= 1):
            errors.append("Somatic fragment merge max ISI violation fraction must be between 0 and 1.")
        if not (0 <= float(self.somatic_fragment_merge_max_duplicate_fraction) <= 1):
            errors.append("Somatic fragment merge max duplicate fraction must be between 0 and 1.")
        if not (0 <= float(self.somatic_state_group_full_template_similarity) <= 1):
            errors.append("Somatic state full-template grouping similarity must be between 0 and 1.")
        if float(self.somatic_refractory_ms) < 0:
            errors.append("Somatic refractory ms must be >= 0.")
        if not (0 <= float(self.somatic_duplicate_ms) <= 0.5):
            errors.append("Somatic duplicate ms must be between 0 and 0.5.")
        if not (0 <= float(self.somatic_conflict_ratio_threshold) <= 1):
            errors.append("Somatic conflict ratio threshold must be between 0 and 1.")
        if int(self.somatic_max_spikes_per_unit_for_conflict_metrics) < 1:
            errors.append("Somatic max spikes per unit for conflict metrics must be at least 1.")
        if int(self.somatic_state_channel_radius) < 0:
            errors.append("Somatic state channel radius must be >= 0.")

    def validate_for_run(self, selected_stage_keys: list[str] | None = None, *, allow_existing_output: bool = False) -> None:
        errors: list[str] = []
        self._validate_required_paths(selected_stage_keys, errors)
        if errors:
            raise PipelineConfigError("\n".join(errors))

        self.normalized_brain_regions()
        self._validate_existing_paths(selected_stage_keys, errors)
        self._validate_raw_files(errors)
        self._validate_path_separation(errors)
        if not allow_existing_output:
            self._validate_output_targets(errors)
        self._validate_catgt_arguments(errors)
        self._validate_sync_crop(errors)
        self._validate_custom_ks4_options(selected_stage_keys, errors)
        if errors:
            raise PipelineConfigError("\n\n".join(errors))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        valid_fields = {item.name for item in dataclass_fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in valid_fields})
