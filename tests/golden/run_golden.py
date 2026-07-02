"""Capture golden-master artifacts from the LEGACY chase pipeline.

Run this ONCE before refactoring (and re-run intentionally when a behavior
change is accepted). It executes the legacy pipeline with the fixed golden
config, then snapshots the selected artifacts into the golden artifact dir.

Usage:
    HF_HUB_OFFLINE=1 python tests/golden/run_golden.py            # capture baseline
    HF_HUB_OFFLINE=1 python tests/golden/run_golden.py --check    # run + compare

The pipeline entry point is resolved indirectly so this harness keeps working
after the legacy runner is replaced by ``python -m loris run`` in Phase 6.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE))

from golden_config import (  # noqa: E402
    GOLDEN_ARGV,
    GOLDEN_ARTIFACT_DIR,
    GOLDEN_ARTIFACTS,
)

# Pipeline entry: (module, is_cli_module). Swapped to loris in Phase 6.
_PIPELINE_ENTRY = os.environ.get("LORIS_PIPELINE_ENTRY", "run_loris_chase_pipeline")


def _run_pipeline(exp_base: Path) -> Path:
    """Run the pipeline into exp_base and return the created experiment dir."""
    exp_base.mkdir(parents=True, exist_ok=True)
    argv = list(GOLDEN_ARGV) + ["--exp_dir", str(exp_base)]
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Pin to single-threaded BLAS/OMP for reproducible golden artifacts. Force
    # (not setdefault): the shell exports OMP_NUM_THREADS=0, which libgomp
    # rejects; "1" also removes thread-scheduling nondeterminism.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    # Force CPU for the whole process tree. predicates._get_embedding_model and
    # the cluster encoder build SentenceTransformer with no device arg (-> CUDA
    # by default), and they run inside joblib loky WORKER processes that don't
    # inherit in-process monkeypatches. Many workers each loading MiniLM on the
    # GPU exhausts it (CUDA OOM). Hiding the GPU keeps everything on CPU, which
    # is also what makes the golden artifacts reproducible bit-for-bit.
    env["CUDA_VISIBLE_DEVICES"] = ""
    # Ensure the torch.backends.mps compat shim loads in EVERY interpreter,
    # including loky worker subprocesses (which don't inherit parent
    # monkeypatches). sitecustomize.py on PYTHONPATH is auto-imported by site.
    _sc_dir = str(_HERE / "_sitecustomize")
    env["PYTHONPATH"] = _sc_dir + os.pathsep + env.get("PYTHONPATH", "")

    before = set(p.name for p in exp_base.iterdir())
    cmd = [sys.executable, "-u", "-m", _PIPELINE_ENTRY] + argv
    print(f"[golden] running: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(_REPO), env=env)
    print(f"[golden] pipeline exited {proc.returncode} in {time.time()-t0:.1f}s", flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"pipeline failed with code {proc.returncode}")

    after = [p for p in exp_base.iterdir() if p.is_dir() and p.name not in before]
    if not after:
        # maybe reused an existing dir; take newest
        after = [p for p in exp_base.iterdir() if p.is_dir()]
    if not after:
        raise SystemExit("no experiment directory was created")
    exp_dir = max(after, key=lambda p: p.stat().st_mtime)
    print(f"[golden] experiment dir: {exp_dir}", flush=True)
    return exp_dir


def capture() -> None:
    exp_base = _REPO / "tests" / "golden" / "_run"
    if exp_base.exists():
        shutil.rmtree(exp_base)
    exp_dir = _run_pipeline(exp_base)

    dest = _REPO / GOLDEN_ARTIFACT_DIR
    dest.mkdir(parents=True, exist_ok=True)
    captured = []
    for name in GOLDEN_ARTIFACTS:
        src = exp_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)
            captured.append(name)
        else:
            print(f"[golden] WARNING: artifact missing: {name}", flush=True)
    print(f"[golden] captured {len(captured)} artifacts -> {dest}", flush=True)
    print("[golden] " + ", ".join(captured), flush=True)


def check() -> int:
    exp_base = _REPO / "tests" / "golden" / "_run_check"
    if exp_base.exists():
        shutil.rmtree(exp_base)
    exp_dir = _run_pipeline(exp_base)
    from compare import compare_dir  # noqa: E402
    ok = compare_dir(exp_dir, _REPO / GOLDEN_ARTIFACT_DIR)
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="run pipeline and compare against stored golden artifacts")
    args = ap.parse_args()
    if args.check:
        raise SystemExit(check())
    capture()


if __name__ == "__main__":
    main()
