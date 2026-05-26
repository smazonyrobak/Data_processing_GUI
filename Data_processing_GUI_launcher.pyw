from __future__ import annotations

import ctypes
import os
import runpy
import sys
import traceback
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT = APP_DIR / "preprocessing_gui.py"
ERROR_LOG = APP_DIR / "Data_processing_GUI_launcher_error.log"
CONDA_ENV_DIR = Path(sys.executable).resolve().parent.parent
APP_USER_MODEL_ID = "MarcinSzymon.Neuropixels.DataProcessingGUI"


def set_windows_app_id() -> None:
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        pass


def activate_current_conda_env_for_child_processes() -> None:
    env_dirs = [
        CONDA_ENV_DIR,
        CONDA_ENV_DIR / "Library" / "mingw-w64" / "bin",
        CONDA_ENV_DIR / "Library" / "usr" / "bin",
        CONDA_ENV_DIR / "Library" / "bin",
        CONDA_ENV_DIR / "Scripts",
    ]
    existing = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(str(path) for path in env_dirs if path.exists()) + os.pathsep + existing
    os.environ.setdefault("CONDA_PREFIX", str(CONDA_ENV_DIR))


def show_message_box(title: str, message: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
    except Exception:
        pass


def main() -> int:
    try:
        if not MAIN_SCRIPT.exists():
            raise FileNotFoundError(f"Could not find main GUI script:\n{MAIN_SCRIPT}")
        set_windows_app_id()
        activate_current_conda_env_for_child_processes()
        sys.path.insert(0, str(APP_DIR))
        os.chdir(str(APP_DIR))
        runpy.run_path(str(MAIN_SCRIPT), run_name="__main__")
        return 0
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 0
    except Exception:
        details = traceback.format_exc()
        try:
            ERROR_LOG.write_text(details, encoding="utf-8")
        except Exception:
            pass
        show_message_box(
            "Data processing GUI launcher error",
            "The application could not start.\n\n"
            f"Details were written to:\n{ERROR_LOG}",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
