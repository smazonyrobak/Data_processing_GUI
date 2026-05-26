from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_config import PipelineConfig
from stages.custom_ks4 import copy_existing_custom_ks4_to_preprocessed


class PrintLogger:
    def log(self, message: str) -> None:
        print(message, flush=True)

    def output(self, message: str) -> None:
        print(message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy custom somatic KS4 outputs into preprocessed_data.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = PipelineConfig.load(Path(args.config))
    copied_dirs = copy_existing_custom_ks4_to_preprocessed(cfg, PrintLogger())
    for path in copied_dirs:
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
