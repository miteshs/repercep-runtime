"""repercep_scheduler — Rust-backed request scheduler.

The public Python API for the scheduler lives at ``repercep.runtime.scheduler``;
this package exposes the raw ``_native`` extension module that wrapper imports
from. Application code should not import from this package directly.
"""

from repercep_scheduler._native import Scheduler, SchedulerError

__all__ = ["Scheduler", "SchedulerError"]
