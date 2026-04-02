from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass
from typing import List

from nodmod import S3MSong


@dataclass
class LoadFailure:
    path: str
    exc_type: str
    message: str
    traceback: str


def find_s3m_files(root: str) -> List[str]:
    matches: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith('.s3m'):
                matches.append(os.path.join(dirpath, name))
    return matches


def load_s3m(path: str) -> None:
    song = S3MSong()
    song.load(path, verbose=False)
    if song.pattern_seq:
        _ = song.get_note(0, 0, 0)


def main() -> int:
    root = r"G:\My Drive\Moduli"

    if not os.path.isdir(root):
        print(f"Root folder not found: {root}")
        return 2

    s3m_files = find_s3m_files(root)
    s3m_files.sort()

    print(f"Found {len(s3m_files)} .s3m/.S3M files under: {root}")

    failures: List[LoadFailure] = []
    skipped = 0
    loaded = 0
    start = time.time()

    for idx, path in enumerate(s3m_files, start=1):
        try:
            load_s3m(path)
            loaded += 1
        except NotImplementedError as exc:
            message = str(exc)
            if "Not an S3M module" in message or "Adlib S3M instruments" in message:
                skipped += 1
            else:
                failures.append(
                    LoadFailure(
                        path=path,
                        exc_type=type(exc).__name__,
                        message=message,
                        traceback="".join(traceback.format_exc()).strip(),
                    )
                )
        except Exception as exc:  # noqa: BLE001 - intentional broad catch for reporting
            failures.append(
                LoadFailure(
                    path=path,
                    exc_type=type(exc).__name__,
                    message=str(exc),
                    traceback="".join(traceback.format_exc()).strip(),
                )
            )
        if idx % 25 == 0:
            print(f"Processed {idx}/{len(s3m_files)} files...")

    elapsed = time.time() - start
    print("\nSummary")
    print(f"Total: {len(s3m_files)}")
    print(f"Loaded: {loaded}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {len(failures)}")
    print(f"Elapsed: {elapsed:.2f}s")

    if failures:
        print("\nFailures")
        for fail in failures:
            print("-")
            print(f"Path: {fail.path}")
            print(f"Error: {fail.exc_type}: {fail.message}")
            print("Traceback:")
            print(fail.traceback)

    return 1 if failures or loaded == 0 else 0


if __name__ == "__main__":
    rc = main()
    if rc == 0:
        print("OK: test_s3m_load_all.py")
    raise SystemExit(rc)