"""Long work submitted now and collected later.

Building an ontology from a document takes minutes. That is the model's speed, not something
this code can shorten: one call on a 4.8k-character document measured 180 seconds, and a staged
extraction is several calls. What *was* ours to fix is that the caller had to hold an HTTP
connection open for the whole of it.

That arrangement fails in every direction. Clients time out and cannot tell a slow extraction
from a dead server. Proxies close idle connections. A Ctrl-C, a dropped wifi or a redeploy
destroys work that was nearly finished and has already been paid for. And there is no way to ask
"is it still going?" — the only signal is a socket that has not closed yet.

So a job is submitted, an id comes back at once, and the result is collected when it is ready.
The work is unchanged and takes exactly as long; what changes is that nothing has to sit on a
socket while it happens, and an interrupted client loses nothing.

Jobs are held in memory, which is the right size for this: a lost job on restart costs a re-run,
and durable storage would be a database dependency bought for a demo. It is stated rather than
assumed, because "the job vanished" is otherwise a mystery.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"


@dataclass
class Job:
    """One unit of submitted work and whatever is known about it so far."""

    job_id: str
    kind: str
    state: str = PENDING
    submitted_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    result: Any = None
    error: str | None = None
    detail: Any = None  # a structured error body, when the failure has one
    label: str = ""  # what was submitted, for a listing that means something

    def as_dict(self, *, include_result: bool = True) -> dict[str, Any]:
        now = time.monotonic()
        started = self.started_at
        elapsed = (
            (self.finished_at or now) - started if started is not None else None
        )
        out: dict[str, Any] = {
            "job_id": self.job_id,
            "kind": self.kind,
            "label": self.label,
            "state": self.state,
            "seconds": round(elapsed, 1) if elapsed is not None else None,
            # A caller polling needs to know whether to poll again, and inferring that from a
            # state name is a guess they should not have to make.
            "finished": self.state in (DONE, FAILED),
        }
        if self.state == FAILED:
            out["error"] = self.error
            if self.detail is not None:
                out["detail"] = self.detail
        if include_result and self.state == DONE:
            out["result"] = self.result
        return out


class JobStore:
    """Runs submitted work on background threads and keeps what came of it.

    A thread per job rather than a pool: these are minutes-long HTTP waits, a handful at a time,
    and a pool bound would make one caller's five-paradigm comparison queue behind another's.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, work: Callable[[], Any], *, label: str = "") -> Job:
        job = Job(
            job_id=uuid.uuid4().hex[:12],
            kind=kind,
            label=label,
            submitted_at=time.monotonic(),
        )
        with self._lock:
            self._jobs[job.job_id] = job

        def run() -> None:
            job.state, job.started_at = RUNNING, time.monotonic()
            try:
                job.result = work()
                job.state = DONE
            except Exception as exc:  # noqa: BLE001 - the point is to record any failure
                job.state = FAILED
                job.error = f"{type(exc).__name__}: {exc}"
                # Endpoints raise HTTPException with a structured body; keeping it means a
                # polled failure says the same thing the synchronous call would have.
                job.detail = getattr(exc, "detail", None)
                job.traceback = traceback.format_exc()  # type: ignore[attr-defined]
            finally:
                job.finished_at = time.monotonic()

        threading.Thread(target=run, name=f"job-{job.job_id}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        """Every job, newest first, without the results — a listing is for finding one."""
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: -j.submitted_at)
        return [j.as_dict(include_result=False) for j in jobs]
