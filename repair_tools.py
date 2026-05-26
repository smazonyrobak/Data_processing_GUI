from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MOVE_START_RE = re.compile(r"MOVE_START\s+(\d+)(?:/(\d+))?\s+(.+)\s*$", re.IGNORECASE)
MOVE_DONE_RE = re.compile(r"MOVE_DONE\s+(\d+)\s+(.+)\s*$", re.IGNORECASE)
REPAIR_SCHEMA_VERSION = 2
KEYS = ("global_step", "source_move_index", "repair_trial_index")


class RepairError(ValueError):
    pass


@dataclass(frozen=True)
class SegmentSpec:
    folder: Path
    label: str = ""
    include_key: str = "global_step"
    include_ranges: tuple[tuple[int, int], ...] | None = None


@dataclass
class RepairPlan:
    output_folder: Path
    segments: list[dict[str, Any]]
    included_moves: list[dict[str, Any]]
    excluded_moves: list[dict[str, Any]]
    crop: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ready_to_write(self) -> bool:
        return bool(self.included_moves) and not self.errors


def normalise_key(value: str | None) -> str:
    key = str(value or "global_step").strip().lower().replace(" ", "_")
    aliases = {
        "global": "global_step",
        "global_step": "global_step",
        "source": "source_move_index",
        "source_move": "source_move_index",
        "source_move_index": "source_move_index",
        "repair": "repair_trial_index",
        "repair_trial": "repair_trial_index",
        "repair_trial_index": "repair_trial_index",
    }
    if key not in aliases:
        raise RepairError(f"Unsupported repair key {value!r}. Use one of: {', '.join(KEYS)}.")
    return aliases[key]


def parse_ranges(text: str) -> tuple[tuple[int, int], ...] | None:
    ranges: list[tuple[int, int]] = []
    for token in re.split(r"[,;\s]+", text.strip()):
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = _int(left)
            end = _int(right)
        else:
            start = end = _int(token)
        if start is None or end is None:
            raise RepairError(f"Cannot parse range token: {token!r}")
        if end < start:
            start, end = end, start
        ranges.append((start, end))
    return tuple(ranges) if ranges else None


def parse_values(text: str) -> set[int]:
    values: set[int] = set()
    for start, end in parse_ranges(text) or ():
        values.update(range(start, end + 1))
    return values


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _load_trials(folder: Path) -> tuple[list[dict[str, str]], list[str], Path]:
    candidates = [
        folder / "stimulus_trials.csv",
        folder / "somatosensory_stimulation" / "run_001_sequence.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise RepairError(f"No stimulus trial table found in {folder}")
    rows, fields = _read_csv(path)
    if not rows:
        raise RepairError(f"Stimulus trial table is empty: {path}")
    if "global_step" not in fields:
        fields = ["global_step", *fields]
    for index, row in enumerate(rows, start=1):
        row.setdefault("global_step", str(index))
        if not str(row["global_step"]).strip():
            row["global_step"] = str(index)
    return rows, fields, path


def _load_events(folder: Path) -> tuple[list[dict[str, str]], list[str], Path]:
    path = folder / "events.csv"
    if not path.exists():
        raise RepairError(f"No events.csv found in {folder}")
    rows, fields = _read_csv(path)
    for index, row in enumerate(rows):
        row["_source_event_row_index"] = str(index)
    return rows, fields, path


def _event_message(row: dict[str, str]) -> str:
    text = str(row.get("message", "")).strip()
    if text.upper().startswith("ARDUINO:"):
        return text.split(":", 1)[1].strip()
    return text


def _completed_moves(events: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    starts: dict[int, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []

    for row in events:
        event_type = str(row.get("event_type", "")).strip().lower()
        message = _event_message(row)
        if event_type == "move_start":
            match = MOVE_START_RE.search(message)
            if match is None:
                continue
            source_move_index = int(match.group(1))
            starts[source_move_index] = {
                "source_move_index": source_move_index,
                "source_total_moves": _int(match.group(2)),
                "move_label": match.group(3).strip(),
                "source_start_timestamp": row.get("timestamp", ""),
                "source_start_relative_to_first_camera_frame_s": _float(row.get("relative_to_first_camera_frame_s")),
                "start_event_row": int(row.get("_source_event_row_index", "0")),
            }
        elif event_type == "move_done":
            match = MOVE_DONE_RE.search(message)
            if match is None:
                continue
            source_move_index = int(match.group(1))
            move = starts.pop(source_move_index, None)
            if move is None:
                incomplete.append(
                    {
                        "source_move_index": source_move_index,
                        "move_label": match.group(2).strip(),
                        "exclude_reason": "move_done_without_move_start",
                        "source_end_timestamp": row.get("timestamp", ""),
                    }
                )
                continue
            start_s = move.get("source_start_relative_to_first_camera_frame_s")
            end_s = _float(row.get("relative_to_first_camera_frame_s"))
            move.update(
                {
                    "source_end_timestamp": row.get("timestamp", ""),
                    "source_end_relative_to_first_camera_frame_s": end_s,
                    "source_duration_s": None if start_s is None or end_s is None else end_s - start_s,
                    "end_event_row": int(row.get("_source_event_row_index", "0")),
                }
            )
            completed.append(move)

    for move in starts.values():
        incomplete.append({**move, "exclude_reason": "move_start_without_move_done"})
    return completed, incomplete


def _value_for(record: dict[str, Any], key: str) -> int | None:
    key = normalise_key(key)
    if key == "repair_trial_index":
        return _int(record.get("repair_trial_index"))
    return _int(record.get(key))


def _in_ranges(value: int | None, ranges: tuple[tuple[int, int], ...] | None) -> bool:
    if value is None:
        return False
    return not ranges or any(start <= value <= end for start, end in ranges)


def analyse_segment(spec: SegmentSpec, source_segment_index: int) -> dict[str, Any]:
    folder = Path(spec.folder)
    if not folder.is_dir():
        raise RepairError(f"Segment folder does not exist: {folder}")

    include_key = normalise_key(spec.include_key)
    trials, trial_fields, trial_path = _load_trials(folder)
    events, event_fields, event_path = _load_events(folder)
    metadata = _read_json(folder / "recording_metadata.json")
    complete, incomplete = _completed_moves(events)
    trial_by_source_move = {index: row for index, row in enumerate(trials, start=1)}

    moves: list[dict[str, Any]] = []
    completed_source_moves: set[int] = set()
    for move in complete:
        source_move_index = int(move["source_move_index"])
        completed_source_moves.add(source_move_index)
        trial = trial_by_source_move.get(source_move_index)
        record = _base_record(spec, source_segment_index, folder, metadata, trial_path, event_path, move)
        if trial is None:
            record.update(include=False, exclude_reason="no_matching_stimulus_trial")
        else:
            record.update(_trial_fields(trial))
            key_value = _value_for(record, include_key)
            record["include_key_value"] = key_value
            if not _in_ranges(key_value, spec.include_ranges):
                record.update(include=False, exclude_reason="outside_include_ranges")
            else:
                record.update(include=True, exclude_reason="")
        moves.append(record)

    for source_move_index, trial in trial_by_source_move.items():
        if source_move_index in completed_source_moves:
            continue
        record = _base_record(
            spec,
            source_segment_index,
            folder,
            metadata,
            trial_path,
            event_path,
            {"source_move_index": source_move_index, "move_label": trial.get("move_label", "")},
        )
        record.update(_trial_fields(trial))
        key_value = _value_for(record, include_key)
        if _in_ranges(key_value, spec.include_ranges):
            record.update(include=False, exclude_reason="trial_has_no_completed_move", include_key_value=key_value)
            moves.append(record)

    for move in incomplete:
        trial = trial_by_source_move.get(int(move.get("source_move_index") or -1))
        record = _base_record(spec, source_segment_index, folder, metadata, trial_path, event_path, move)
        if trial:
            record.update(_trial_fields(trial))
        record.update(include=False, exclude_reason=move.get("exclude_reason", "incomplete_move"))
        moves.append(record)

    return {
        "source_segment_index": source_segment_index,
        "label": spec.label or folder.name,
        "folder": str(folder),
        "include_key": include_key,
        "include_ranges": spec.include_ranges,
        "stimulus_trials_csv": str(trial_path),
        "events_csv": str(event_path),
        "recording_metadata_json": str(folder / "recording_metadata.json"),
        "camera_first_frame_at": metadata.get("camera_first_frame_at") or metadata.get("first_frame_at"),
        "trial_count": len(trials),
        "completed_move_count": len(complete),
        "incomplete_move_count": len(incomplete),
        "trial_fields": trial_fields,
        "event_fields": event_fields,
        "event_rows": events,
        "moves": moves,
    }


def _base_record(
    spec: SegmentSpec,
    source_segment_index: int,
    folder: Path,
    metadata: dict[str, Any],
    trial_path: Path,
    event_path: Path,
    move: dict[str, Any],
) -> dict[str, Any]:
    return {
        **move,
        "source_segment_index": source_segment_index,
        "source_segment_label": spec.label or folder.name,
        "source_folder": str(folder),
        "source_stimulus_trials_csv": str(trial_path),
        "source_events_csv": str(event_path),
        "source_camera_first_frame_at": metadata.get("camera_first_frame_at") or metadata.get("first_frame_at"),
    }


def _trial_fields(trial: dict[str, str]) -> dict[str, Any]:
    return {
        "global_step": _int(trial.get("global_step")),
        "trial_move_label": trial.get("move_label", ""),
        "trial_row": dict(trial),
    }


def build_repair_plan(
    specs: list[SegmentSpec],
    output_folder: Path,
    *,
    exclude_key: str = "global_step",
    exclude_values: set[int] | None = None,
    crop_key: str = "repair_trial_index",
    crop_start: int | None = None,
    crop_end: int | None = None,
) -> RepairPlan:
    if not specs:
        raise RepairError("Add at least one source segment.")

    exclude_key = normalise_key(exclude_key)
    crop_key = normalise_key(crop_key)
    excluded_values = set(exclude_values or set())
    segments = [analyse_segment(spec, index) for index, spec in enumerate(specs, start=1)]

    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for segment in segments:
        for move in segment["moves"]:
            if move.get("include"):
                included.append(move)
            else:
                excluded.append(move)

    included.sort(key=lambda row: (int(row["source_segment_index"]), int(row.get("source_move_index") or 0)))
    _assign_repair_indices(included)

    if excluded_values:
        kept: list[dict[str, Any]] = []
        for move in included:
            if _value_for(move, exclude_key) in excluded_values:
                excluded.append({**move, "include": False, "exclude_reason": f"manual_{exclude_key}_exclusion"})
            else:
                kept.append(move)
        included = kept
        _assign_repair_indices(included)

    warnings = _plan_warnings(included)
    errors = [] if included else ["No completed moves are included."]
    crop = _crop_info(included, crop_key, crop_start, crop_end)
    return RepairPlan(Path(output_folder), segments, included, excluded, crop, warnings, errors)


def _assign_repair_indices(moves: list[dict[str, Any]]) -> None:
    for index, move in enumerate(moves):
        move["repair_trial_index"] = index
        trial = dict(move.get("trial_row") or {})
        trial["repair_trial_index"] = str(index)
        trial["source_segment_index"] = str(move["source_segment_index"])
        trial["source_move_index"] = str(move.get("source_move_index", ""))
        move["trial_row"] = trial


def _plan_warnings(included: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    global_steps = [_int(move.get("global_step")) for move in included]
    global_steps = [step for step in global_steps if step is not None]
    if len(global_steps) != len(set(global_steps)):
        duplicates = sorted({step for step in global_steps if global_steps.count(step) > 1})
        warnings.append("Duplicate global_step values are included: " + ", ".join(map(str, duplicates[:20])))
    if global_steps and global_steps != sorted(global_steps):
        warnings.append("Included global_step values are not monotonic in repaired order.")
    return warnings


def _crop_info(
    included: list[dict[str, Any]],
    crop_key: str,
    crop_start: int | None,
    crop_end: int | None,
) -> dict[str, Any]:
    if not included:
        return {"crop_key": crop_key, "crop_start": crop_start, "crop_end": crop_end}
    if crop_start is None:
        crop_start = _value_for(included[0], crop_key)
    if crop_end is None:
        crop_end = _value_for(included[-1], crop_key)

    start_move = next((move for move in included if _value_for(move, crop_key) == crop_start), None)
    end_move = next((move for move in reversed(included) if _value_for(move, crop_key) == crop_end), None)
    if start_move:
        start_move["is_crop_start_anchor"] = True
    if end_move:
        end_move["is_crop_end_anchor"] = True
    return {
        "crop_key": crop_key,
        "crop_start": crop_start,
        "crop_end": crop_end,
        "start_anchor": _anchor(start_move),
        "end_anchor": _anchor(end_move),
    }


def _anchor(move: dict[str, Any] | None) -> dict[str, Any] | None:
    if move is None:
        return None
    return {
        "repair_trial_index": move.get("repair_trial_index"),
        "global_step": move.get("global_step"),
        "source_segment_index": move.get("source_segment_index"),
        "source_move_index": move.get("source_move_index"),
        "timestamp": move.get("source_start_timestamp") or move.get("source_end_timestamp"),
        "relative_to_first_camera_frame_s": move.get("source_start_relative_to_first_camera_frame_s"),
        "source_camera_first_frame_at": move.get("source_camera_first_frame_at"),
    }


def plan_summary(plan: RepairPlan) -> str:
    lines = [f"Output folder: {plan.output_folder}", "", "Segments:"]
    for segment in plan.segments:
        lines.append(
            f"  {segment['source_segment_index']}. {segment['label']} | "
            f"{segment['trial_count']} trials, {segment['completed_move_count']} completed, "
            f"include by {segment['include_key']} {segment['include_ranges'] or 'all'}"
        )
    lines.extend(["", f"Included completed moves: {len(plan.included_moves)}"])
    if plan.included_moves:
        first = plan.included_moves[0]
        last = plan.included_moves[-1]
        lines.append(
            f"Repair trials: {first['repair_trial_index']} ({first.get('move_label')}) "
            f"to {last['repair_trial_index']} ({last.get('move_label')})"
        )
        lines.append(f"Original global_step span: {first.get('global_step')} to {last.get('global_step')}")
    lines.append(f"Excluded/incomplete moves or trial rows: {len(plan.excluded_moves)}")
    lines.extend(["", "Crop anchors:"])
    lines.append(f"  key: {plan.crop.get('crop_key')}")
    lines.append(f"  start: {plan.crop.get('crop_start')}")
    lines.append(f"  end: {plan.crop.get('crop_end')}")
    if plan.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"  - {warning}" for warning in plan.warnings)
    if plan.errors:
        lines.extend(["", "Errors:"])
        lines.extend(f"  - {error}" for error in plan.errors)
    return "\n".join(lines)


def write_repaired_run(plan: RepairPlan, *, allow_existing_output: bool = False) -> None:
    if not plan.ready_to_write:
        raise RepairError("Repair plan is not ready to write:\n" + "\n".join(plan.errors))
    output = plan.output_folder.resolve(strict=False)
    _check_output_path(output, [Path(segment["folder"]) for segment in plan.segments], allow_existing_output)
    output.mkdir(parents=True, exist_ok=True)

    trial_fields = _trial_fieldnames(plan)
    trial_rows = [move["trial_row"] for move in plan.included_moves]
    _write_csv(output / "stimulus_trials.csv", trial_rows, trial_fields)

    _write_csv(output / "completed_moves_repaired.csv", _completed_rows(plan), _completed_fields())
    _write_csv(output / "excluded_moves_repaired.csv", _excluded_rows(plan), _excluded_fields())
    event_rows, event_fields = _event_rows(plan)
    _write_csv(output / "events.csv", event_rows, event_fields)

    manifest = _manifest(plan, output)
    manifest_text = json.dumps(manifest, indent=2)
    (output / "repair_manifest.json").write_text(manifest_text, encoding="utf-8")
    (output / "recording_metadata.json").write_text(json.dumps(_recording_metadata(plan, output), indent=2), encoding="utf-8")


def _check_output_path(output: Path, sources: list[Path], allow_existing_output: bool) -> None:
    if not output.is_absolute():
        raise RepairError(f"Repair output folder must be absolute: {output}")
    for source in sources:
        source = source.resolve(strict=False)
        if output == source or output in source.parents or source in output.parents:
            raise RepairError(f"Repair output must be separate from source data:\n  output: {output}\n  source: {source}")
    if output.exists() and any(output.iterdir()) and not allow_existing_output:
        raise RepairError(f"Repair output folder already exists and is not empty: {output}")


def _trial_fieldnames(plan: RepairPlan) -> list[str]:
    fields = ["repair_trial_index", "source_segment_index", "source_move_index"]
    for segment in plan.segments:
        for field in segment["trial_fields"]:
            if field not in fields:
                fields.append(field)
    return fields


def _completed_fields() -> list[str]:
    return [
        "repair_trial_index",
        "global_step",
        "move_label",
        "source_segment_index",
        "source_segment_label",
        "source_folder",
        "source_move_index",
        "source_total_moves",
        "source_start_timestamp",
        "source_end_timestamp",
        "source_start_relative_to_first_camera_frame_s",
        "source_end_relative_to_first_camera_frame_s",
        "source_duration_s",
        "source_camera_first_frame_at",
        "is_crop_start_anchor",
        "is_crop_end_anchor",
    ]


def _completed_rows(plan: RepairPlan) -> list[dict[str, Any]]:
    return [{field: move.get(field) for field in _completed_fields()} for move in plan.included_moves]


def _excluded_fields() -> list[str]:
    return [
        "repair_trial_index",
        "global_step",
        "move_label",
        "source_segment_index",
        "source_segment_label",
        "source_folder",
        "source_move_index",
        "exclude_reason",
        "source_start_timestamp",
        "source_end_timestamp",
    ]


def _excluded_rows(plan: RepairPlan) -> list[dict[str, Any]]:
    return [{field: move.get(field) for field in _excluded_fields()} for move in plan.excluded_moves]


def _event_rows(plan: RepairPlan) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    fields: list[str] = []
    segments = {int(segment["source_segment_index"]): segment for segment in plan.segments}
    extras = [
        "repair_trial_index",
        "global_step",
        "source_segment_index",
        "source_segment_label",
        "source_folder",
        "source_event_row_index",
        "repair_action",
    ]

    for move in plan.included_moves:
        segment = segments[int(move["source_segment_index"])]
        for event_row_index in [move.get("start_event_row"), move.get("end_event_row")]:
            if event_row_index is None:
                continue
            row = dict(segment["event_rows"][int(event_row_index)])
            source_event_row_index = row.pop("_source_event_row_index", event_row_index)
            row.update(
                {
                    "repair_trial_index": move.get("repair_trial_index"),
                    "global_step": move.get("global_step"),
                    "source_segment_index": move.get("source_segment_index"),
                    "source_segment_label": move.get("source_segment_label"),
                    "source_folder": move.get("source_folder"),
                    "source_event_row_index": source_event_row_index,
                    "repair_action": "included_completed_move",
                }
            )
            rows.append(row)
            for field in row:
                if field not in fields:
                    fields.append(field)
    for field in extras:
        if field not in fields:
            fields.append(field)
    return rows, fields


def _manifest(plan: RepairPlan, output: Path) -> dict[str, Any]:
    return {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "repair_type": "stim_cam_repair",
        "output_folder": str(output),
        "source_segments": [_public_segment(segment) for segment in plan.segments],
        "included_move_count": len(plan.included_moves),
        "excluded_move_count": len(plan.excluded_moves),
        "crop": plan.crop,
        "warnings": plan.warnings,
        "notes": [
            "Original source folders were not modified.",
            "Use this folder as the Stim/cam run in the main processing tab.",
        ],
    }


def _public_segment(segment: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "source_segment_index",
        "label",
        "folder",
        "include_key",
        "include_ranges",
        "stimulus_trials_csv",
        "events_csv",
        "recording_metadata_json",
        "camera_first_frame_at",
        "trial_count",
        "completed_move_count",
        "incomplete_move_count",
    ]
    return {key: segment.get(key) for key in keys}


def _recording_metadata(plan: RepairPlan, output: Path) -> dict[str, Any]:
    return {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "app": "Data_processing_GUI repair tab",
        "status": "repaired",
        "repair_manifest": str(output / "repair_manifest.json"),
        "stimulus_trials_csv": str(output / "stimulus_trials.csv"),
        "events_csv": str(output / "events.csv"),
        "source_segments": [_public_segment(segment) for segment in plan.segments],
    }

