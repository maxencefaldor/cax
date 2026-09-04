"""Execute the example notebooks and package them for the documentation.

The notebooks in `examples/` are stored stripped: outputs are not committed, because a
notebook's outputs are base64 blobs that git cannot delta, so every refresh would add
their full size to the repository's history forever. But a library about self-organizing
systems is worth *seeing*, so the documentation shows the notebooks executed.

This script is how the outputs are made. It runs the notebooks where the compute is —
your machine, with its accelerators — and writes the executed copies to a build
directory that is never committed. The result is packaged as a release asset that the
docs workflow downloads at build time, so the published site shows every output while
the repository stays the size of its source.

Execution is cached: a notebook is re-run only when its own source changes or when the
library it imports does. Editing one notebook re-runs one notebook; changing `cax`
re-runs the notebooks, but only once, and in the background.

Usage:
    python docs/scripts/execute.py                 # execute what is stale
    python docs/scripts/execute.py --all           # execute everything
    python docs/scripts/execute.py 20_lenia 31_boids
    python docs/scripts/execute.py --package       # write the release tarball
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
BUILD = ROOT / "docs" / "build" / "notebooks"
CACHE = ROOT / "docs" / "build" / "cache.json"
TARBALL = ROOT / "docs" / "build" / "notebooks.tar.gz"
TIMEOUT = 2 * 60 * 60  # a training notebook may legitimately run for hours


def library_digest() -> str:
    """Hash the library source, so a change to `cax` invalidates every notebook."""
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def notebook_digest(path: Path, library: str) -> str:
    """Hash a notebook's own source together with the library it imports."""
    cells = json.loads(path.read_text())["cells"]
    source = "".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
    )
    return hashlib.sha256(f"{library}{source}".encode()).hexdigest()


def execute(path: Path, device: str | None) -> tuple[bool, float, str]:
    """Execute one notebook into the build directory.

    Returns a `(ok, seconds, message)` tuple.
    """
    BUILD.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    if device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = device
    start = time.perf_counter()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            f"--ExecutePreprocessor.timeout={TIMEOUT}",
            "--output-dir",
            str(BUILD),
            "--output",
            path.name,
            str(path),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=environment,
    )
    elapsed = time.perf_counter() - start
    if result.returncode == 0:
        return True, elapsed, ""
    error = [line for line in result.stderr.splitlines() if "Error" in line]
    return False, elapsed, (error[-1] if error else "failed")[:120]


def package() -> None:
    """Write the tarball the docs workflow downloads."""
    executed = sorted(BUILD.glob("*.ipynb"))
    if not executed:
        raise SystemExit("nothing executed yet")
    with tarfile.open(TARBALL, "w:gz") as tar:
        for path in executed:
            tar.add(path, arcname=f"examples/{path.name}")
    size = TARBALL.stat().st_size / 1e6
    print(f"\n{TARBALL.relative_to(ROOT)}: {len(executed)} notebooks, {size:.1f} MB")
    print("upload with:")
    print(f"  gh release upload docs-notebooks {TARBALL.relative_to(ROOT)} --clobber")


def main() -> None:
    """Execute the stale notebooks and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names", nargs="*", help="notebook stems; default is all stale ones"
    )
    parser.add_argument("--all", action="store_true", help="ignore the cache")
    parser.add_argument("--package", action="store_true", help="only write the tarball")
    parser.add_argument(
        "--device", default=None, help="CUDA_VISIBLE_DEVICES for the run"
    )
    args = parser.parse_args()

    if args.package:
        package()
        return

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    library = library_digest()

    notebooks = sorted(EXAMPLES.glob("*.ipynb"))
    if args.names:
        notebooks = [p for p in notebooks if p.stem in args.names]

    stale = [
        p
        for p in notebooks
        if args.all or args.names or cache.get(p.name) != notebook_digest(p, library)
    ]
    fresh = len(notebooks) - len(stale)
    print(
        f"{len(notebooks)} notebooks: {len(stale)} to execute, "
        f"{fresh} already current\n"
    )

    failures = []
    for index, path in enumerate(stale, 1):
        print(f"[{index}/{len(stale)}] {path.stem} ... ", end="", flush=True)
        ok, elapsed, message = execute(path, args.device)
        if ok:
            # Re-read before writing so shards on other devices are not clobbered.
            cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
            cache[path.name] = notebook_digest(path, library)
            CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
            print(f"ok ({elapsed:.0f}s)")
        else:
            failures.append((path.stem, message))
            print(f"FAILED ({elapsed:.0f}s) {message}")

    print(f"\nexecuted {len(stale) - len(failures)}/{len(stale)}")
    for name, message in failures:
        print(f"  FAILED {name}: {message}")
    if BUILD.exists() and any(BUILD.glob("*.ipynb")):
        package()


if __name__ == "__main__":
    main()
