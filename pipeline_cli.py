from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from pipeline_config import PipelineConfig
from pipeline_runner import FileLogger, run_selected_stages


LOG_PREFIX = "__PIPELINE_LOG__ "
OUT_PREFIX = "__PIPELINE_OUT__ "
ORIGINAL_STDOUT = sys.stdout


def emit_prefixed(prefix: str, message: str) -> None:
    lines = str(message).splitlines() or [""]
    for line in lines:
        print(prefix + line, file=ORIGINAL_STDOUT, flush=True)


def emit_log(message: str) -> None:
    emit_prefixed(LOG_PREFIX, message)


def emit_output(message: str) -> None:
    emit_prefixed(OUT_PREFIX, message)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stages", required=True)
    parser.add_argument("--validated-output", action="store_true")
    args = parser.parse_args(argv)

    try:
        cfg = PipelineConfig.load(Path(args.config))
        stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
        cfg.validate_for_run(stages, allow_existing_output=args.validated_output)
        logger = FileLogger(cfg, emit_log=emit_log, emit_output=emit_output)
        logger.log("Pipeline run started")
        run_selected_stages(cfg, stages, logger, allow_existing_output=args.validated_output)
        logger.log("Pipeline run finished")
        return 0
    except Exception:
        emit_output(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
