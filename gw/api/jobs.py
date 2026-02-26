from __future__ import annotations

import time
import uuid
import threading
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Callable

# Maximum time (seconds) a job subprocess is allowed to run before being killed.
_DEFAULT_TIMEOUT_S = 300  # 5 minutes

# Maximum number of completed jobs to keep in memory.  Older finished jobs
# are pruned whenever a new job is created.
_MAX_FINISHED_JOBS = 50


@dataclass
class Job:
    id: str
    state: str = "queued"
    exit_code: Optional[int] = None
    last_lines: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)


_JOBS: Dict[str, Job] = {}
_JOB_LOCK = threading.Lock()


def _prune_finished_jobs() -> None:
    """Remove oldest finished jobs when the pool exceeds the limit.

    Must be called while *not* holding ``_JOB_LOCK`` — it acquires it internally.
    """
    with _JOB_LOCK:
        finished = [
            (j.created_at, jid)
            for jid, j in _JOBS.items()
            if j.state in ("done", "error")
        ]
        if len(finished) <= _MAX_FINISHED_JOBS:
            return
        finished.sort()  # oldest first
        to_remove = finished[: len(finished) - _MAX_FINISHED_JOBS]
        for _ts, jid in to_remove:
            _JOBS.pop(jid, None)


def create_job() -> Job:
    jid = str(uuid.uuid4())
    job = Job(id=jid)
    with _JOB_LOCK:
        _JOBS[jid] = job
    # Prune old finished jobs to prevent unbounded memory growth
    _prune_finished_jobs()
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _JOB_LOCK:
        return _JOBS.get(job_id)


def _append(job: Job, line: str, max_lines: int = 200) -> None:
    """Append a line to the job's output buffer (thread-safe via list append)."""
    job.last_lines.append(line)
    if len(job.last_lines) > max_lines:
        job.last_lines = job.last_lines[-max_lines:]


def run_subprocess_job(
    job: Job,
    cmd: List[str],
    cwd: Optional[str] = None,
    on_done: Optional[Callable[[Job], None]] = None,
    timeout: int = _DEFAULT_TIMEOUT_S,
) -> None:
    """Run *cmd* in a background thread and stream output to *job*.

    Parameters
    ----------
    timeout : int
        Maximum seconds to wait for the subprocess.  If exceeded the process
        is killed and the job is marked as an error.
    """
    def _runner():
        job.state = "running"
        proc: Optional[subprocess.Popen] = None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            assert proc.stdout is not None

            # Use a deadline so the process cannot run forever
            deadline = time.monotonic() + timeout
            for line in proc.stdout:
                _append(job, line.rstrip("\n"))
                if time.monotonic() > deadline:
                    _append(job, f"[timeout] Job exceeded {timeout}s — killing process.")
                    proc.kill()
                    break

            job.exit_code = proc.wait(timeout=30)
            job.state = "done" if job.exit_code == 0 else "error"
        except subprocess.TimeoutExpired:
            if proc is not None:
                proc.kill()
            job.state = "error"
            job.error = f"Job timed out after {timeout}s"
            job.exit_code = 1
        except Exception as e:
            job.state = "error"
            job.error = str(e)
            job.exit_code = 1
            # Ensure zombie process is cleaned up
            if proc is not None:
                try:
                    proc.kill()
                except OSError:
                    pass
        finally:
            if on_done:
                try:
                    on_done(job)
                except Exception:
                    pass  # Don't let callback errors crash the runner

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
