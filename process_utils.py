from __future__ import annotations

import ctypes
import subprocess
import sys
from typing import Any


JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JobObjectExtendedLimitInformation = 9
JobObjectBasicAccountingInformation = 1


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", ctypes.c_uint32),
        ("TotalProcesses", ctypes.c_uint32),
        ("ActiveProcesses", ctypes.c_uint32),
        ("TotalTerminatedProcesses", ctypes.c_uint32),
    ]


def _raise_last_error(action: str) -> None:
    err = ctypes.get_last_error()
    raise OSError(err, f"{action} failed with Windows error {err}")


def _create_kill_on_close_job() -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        _raise_last_error("CreateJobObjectW")

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = kernel32.SetInformationJobObject(
        job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        kernel32.CloseHandle(job)
        _raise_last_error("SetInformationJobObject")
    return job


def close_job(proc: subprocess.Popen[Any]) -> None:
    job = getattr(proc, "_kill_on_close_job", None)
    if job:
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(job)
        proc._kill_on_close_job = None


def job_active_processes(proc: subprocess.Popen[Any]) -> int | None:
    job = getattr(proc, "_kill_on_close_job", None)
    if not job or sys.platform != "win32":
        return None
    info = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
    ok = ctypes.WinDLL("kernel32", use_last_error=True).QueryInformationJobObject(
        job,
        JobObjectBasicAccountingInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
        None,
    )
    if not ok:
        return None
    return int(info.ActiveProcesses)


def guarded_popen(*popen_args: Any, **popen_kwargs: Any) -> subprocess.Popen[Any]:
    """Start a process that dies with this Python process on Windows.

    Child processes inherit the job, so closing/crashing the parent Python
    process also closes the job handle and terminates the tool tree.
    """
    if sys.platform != "win32":
        return subprocess.Popen(*popen_args, **popen_kwargs)

    flags = int(popen_kwargs.pop("creationflags", 0))
    job = _create_kill_on_close_job()
    proc: subprocess.Popen[Any] | None = None
    try:
        proc = subprocess.Popen(*popen_args, creationflags=flags, **popen_kwargs)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.AssignProcessToJobObject(job, int(proc._handle)):
            err = ctypes.get_last_error()
            # If the parent is already inside a restrictive job, Windows may
            # refuse adding a nested one. In that case the child has inherited
            # the parent's job, which is still the important cleanup path.
            if err != 5:
                _raise_last_error("AssignProcessToJobObject")
            kernel32.CloseHandle(job)
            job = 0
        proc._kill_on_close_job = job
        return proc
    except Exception:
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        if job:
            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(job)
        raise


def guarded_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    proc = guarded_popen(cmd, **kwargs)
    try:
        return subprocess.CompletedProcess(cmd, proc.wait())
    finally:
        close_job(proc)
