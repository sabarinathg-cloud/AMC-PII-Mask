from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import List
import fnmatch
import logging

from .audio_io import ffprobe_audio
from .utils import write_csv, write_json

logger = logging.getLogger(__name__)


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _excluded(path: Path, root: Path, config) -> bool:
    if config is None:
        return False
    if getattr(config.paths, "exclude_output_root_from_discovery", True) and _is_under(path, Path(config.paths.output_root)):
        return True
    if getattr(config.paths, "exclude_work_dir_from_discovery", True) and _is_under(path, Path(config.paths.work_dir)):
        return True
    rel = str(path.relative_to(root)) if _is_under(path, root) else str(path)
    for pat in getattr(config.paths, "extra_exclude_globs", []) or []:
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(str(path), pat):
            return True
    return False


def discover_audio_files(
    input_root: str | Path,
    input_glob: str,
    config=None,
    output_root: str | Path | None = None,
    work_dir: str | Path | None = None,
    auto_exclude_output_and_work_dirs: bool = True,
) -> List[Path]:
    root = Path(input_root)
    files = sorted(p for p in root.glob(input_glob) if p.is_file())
    if config is not None:
        return [p for p in files if not _excluded(p, root, config)]
    if not auto_exclude_output_and_work_dirs:
        return files
    out_root = Path(output_root) if output_root is not None else None
    work = Path(work_dir) if work_dir is not None else None
    result = []
    for p in files:
        if out_root is not None and _is_under(p, out_root):
            continue
        if work is not None and _is_under(p, work):
            continue
        result.append(p)
    return result


def build_manifest(config) -> list[dict]:
    files = discover_audio_files(config.paths.input_root, config.paths.input_glob, config=config)
    if config.runtime.shard_count > 1:
        files = [p for idx, p in enumerate(files) if idx % config.runtime.shard_count == config.runtime.shard_index]
    if config.runtime.limit is not None:
        files = files[: int(config.runtime.limit)]

    logger.info("Discovered %d audio files", len(files))
    rows: list[dict] = []

    def probe(p: Path) -> dict:
        try:
            meta = ffprobe_audio(p, ffprobe_path=config.runtime.ffprobe_path)
            meta["status"] = "ok"
            return meta
        except Exception as e:
            return {"path": str(p), "status": "probe_failed", "error": repr(e)}

    workers = max(1, int(config.runtime.num_manifest_workers))
    max_pending = max(workers * 8, 64)
    it = iter(files)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pending = set()
        for _ in range(min(max_pending, len(files))):
            try:
                pending.add(ex.submit(probe, next(it)))
            except StopIteration:
                break
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                rows.append(fut.result())
                completed += 1
                if completed % max(1, int(config.runtime.log_every)) == 0:
                    logger.info("Probed %d/%d files", completed, len(files))
                try:
                    pending.add(ex.submit(probe, next(it)))
                except StopIteration:
                    pass

    out_dir = Path(config.paths.work_dir) / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "audio_manifest.json", rows)
    if config.runtime.write_csv_reports:
        write_csv(out_dir / "audio_manifest.csv", rows)
    return rows
