from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Protocol

from process_utils import close_job, guarded_popen


class StageLogger(Protocol):
    def log(self, message: str) -> None: ...
    def output(self, message: str) -> None: ...


def read_meta(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in Path(path).read_text().splitlines() if "=" in line)


def run_command(cmd: list[object], cwd: Path | None, logger: StageLogger) -> None:
    printable = " ".join(str(part) for part in cmd)
    logger.output(printable)
    proc = guarded_popen(
        [str(part) for part in cmd],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            logger.output(line.rstrip())
        code = proc.wait()
    finally:
        close_job(proc)
    if code != 0:
        raise subprocess.CalledProcessError(code, [str(part) for part in cmd])


def require_files(paths: Iterable[Path], message: str) -> None:
    missing = [Path(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(message + "\n" + "\n".join(str(path) for path in missing))


def line_count(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)
