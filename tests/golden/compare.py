"""Compare a fresh experiment dir against stored golden artifacts.

Comparison modes (see golden_config.GOLDEN_ARTIFACTS):
  json_canonical -> json.dumps(sort_keys=True) byte-equality (true bit-for-bit)
  npy_exact      -> np.array_equal
  json_metrics   -> numeric compare of GOLDEN_METRIC_KEYS with atol=1e-3
                    (rule layer is bit-exact; only embedding-layer F1 floats
                    get tolerance — see _cmp_json_metrics)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from golden_config import GOLDEN_ARTIFACTS, GOLDEN_METRIC_KEYS  # noqa: E402


def _canonical_bytes(path: Path) -> bytes:
    with open(path, "r") as f:
        obj = json.load(f)
    return json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _cmp_json_canonical(fresh: Path, golden: Path) -> tuple[bool, str]:
    if _canonical_bytes(fresh) == _canonical_bytes(golden):
        return True, "byte-identical (canonical JSON)"
    return False, "canonical JSON differs"


def _cmp_json_metrics(fresh: Path, golden: Path) -> tuple[bool, str]:
    """Compare final metric keys with an absolute tolerance.

    The rule layer (rules.json, PatternStore, candidate predicates, fire_masks)
    is verified bit-for-bit elsewhere. The final F1 numbers, however, flow
    through the *test-time* sentence-transformer embeddings (sim_graph /
    SimPredicate), and sentence-transformers' encode() is not bit-reproducible
    on CPU (sub-1e-4 jitter in the embedding floats propagates into a handful
    of borderline sim edges → a few flipped predictions → ~1e-4 F1 wobble).
    So model/embedding-layer metrics get atol=1e-3; the rule layer stays exact.
    """
    METRIC_ATOL = 1e-3
    with open(fresh) as f:
        a = json.load(f)
    with open(golden) as f:
        b = json.load(f)
    diffs = []
    for k in GOLDEN_METRIC_KEYS:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if abs(va - vb) > METRIC_ATOL:
                diffs.append(f"{k}: fresh={va!r} golden={vb!r} (|Δ|>{METRIC_ATOL})")
        elif va != vb:
            diffs.append(f"{k}: fresh={va!r} golden={vb!r}")
    if diffs:
        return False, "; ".join(diffs)
    return True, f"{len(GOLDEN_METRIC_KEYS)} metric keys within atol={METRIC_ATOL}"


def _cmp_npy_exact(fresh: Path, golden: Path) -> tuple[bool, str]:
    import numpy as np
    a = np.load(fresh, allow_pickle=False)
    b = np.load(golden, allow_pickle=False)
    if a.shape != b.shape:
        return False, f"shape {a.shape} != {b.shape}"
    if np.array_equal(a, b):
        return True, f"array_equal shape={a.shape}"
    n = int((a != b).sum()) if a.shape == b.shape else -1
    return False, f"{n} differing elements"


_DISPATCH = {
    "json_canonical": _cmp_json_canonical,
    "json_metrics": _cmp_json_metrics,
    "npy_exact": _cmp_npy_exact,
}


def compare_dir(fresh_dir: Path, golden_dir: Path) -> bool:
    fresh_dir = Path(fresh_dir)
    golden_dir = Path(golden_dir)
    all_ok = True
    print(f"[compare] fresh={fresh_dir}")
    print(f"[compare] golden={golden_dir}")
    for name, mode in GOLDEN_ARTIFACTS.items():
        gpath = golden_dir / name
        fpath = fresh_dir / name
        if not gpath.exists():
            print(f"  SKIP {name}: no golden baseline")
            continue
        if not fpath.exists():
            print(f"  FAIL {name}: missing in fresh run")
            all_ok = False
            continue
        ok, msg = _DISPATCH[mode](fpath, gpath)
        print(f"  {'PASS' if ok else 'FAIL'} {name} [{mode}]: {msg}")
        all_ok = all_ok and ok
    print(f"[compare] {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
    return all_ok


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python compare.py <fresh_dir> <golden_dir>")
        raise SystemExit(2)
    raise SystemExit(0 if compare_dir(Path(sys.argv[1]), Path(sys.argv[2])) else 1)
