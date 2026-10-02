"""Where background work executes. Callers only `submit`, so the threads used today can be
swapped for a task queue (e.g. Cloud Tasks) without changing them."""

import threading
from collections.abc import Callable
from typing import Protocol


class TaskRunner(Protocol):
    def submit(self, task: Callable[[], None]) -> None: ...


class ThreadRunner:
    """A daemon thread per task, in this process (fine for one user)."""

    def submit(self, task: Callable[[], None]) -> None:
        threading.Thread(target=task, daemon=True).start()


class InlineRunner:
    """Runs the task before `submit` returns (tests, scripts)."""

    def submit(self, task: Callable[[], None]) -> None:
        task()
