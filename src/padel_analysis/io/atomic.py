"""Writing a file so that a crash never leaves it half-written.

Every hand-made truth of this project is rewritten after each answer. Writing in place
truncates the file before filling it: a power cut at that instant once left a file of
zeros, erasing forty-five thousand frames of assignments and every arbitration already
made. Here the new content is published only once it is complete and on the disk.
"""

import json
import os
import tempfile
from pathlib import Path


def write_json_atomically(path: str | Path, payload: object, indent: int | None = None) -> None:
    """Write `payload` as JSON to `path`, replacing the previous file in one step.

    A failure at any point leaves the previous file untouched and no temporary file
    behind.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f"{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
