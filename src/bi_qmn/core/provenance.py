"""Import-closure source snapshots.

P1 D1-v2 and D1-v3 were withdrawn before execution because a whole-tree source hash drifted when
unrelated modules (a P2 audit patch, the Model C namespace) changed. Hashing the modules that are
actually loaded by a run, plus its declared configuration and protocol files, records every byte
that can influence the computation while not invalidating a frozen run when an unrelated module
is added. Drift in any loaded module is still detected and blocks execution.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_closure_snapshot(
    root: str | Path,
    *,
    package: str = "bi_qmn",
    extra_files: Iterable[str] = (),
) -> dict[str, str]:
    root_path = Path(root).resolve()
    snapshot: dict[str, str] = {}
    for name, module in sorted(sys.modules.items()):
        if not (name == package or name.startswith(package + ".")):
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(root_path).as_posix()
        except ValueError:
            relative = path.as_posix()
        snapshot[relative] = file_sha256(path)
    for relative in extra_files:
        snapshot[Path(relative).as_posix()] = file_sha256(root_path / relative)
    return dict(sorted(snapshot.items()))


def snapshot_hash(snapshot: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(snapshot), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def snapshot_drift(declared: Mapping[str, str], current: Mapping[str, str]) -> dict[str, list[str]]:
    return {
        "changed": sorted(path for path in declared if path in current and current[path] != declared[path]),
        "missing": sorted(path for path in declared if path not in current),
        "added": sorted(path for path in current if path not in declared),
    }
