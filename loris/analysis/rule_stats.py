"""Predicate / rule statistics over LORIS ``rules.json`` artifacts.

Aggregates the rules discovered across one or more experiment runs and reports
the "necessary statistics" a LORIS paper writeup needs:

  * number of predicates by **type**, split into **ML** predicates
    (``MLPredicate``, ``MLThresholdPredicate``) vs **logic/textual** predicates
    (``Match``/``Freq``/``Cooccur``/``Before`` + ``Label`` + graph ``Sim``/``Group``);
  * predicate **length** — rule-body arity distribution and predicate-string length;
  * predicate **roles** — feature/text · ml · label · propagation/graph, plus the
    consequence op (``add``/``remove``/``replace``) and the discovery stage;
  * **representative rules** — the top-N rules by validation F1-gain, per dataset
    and per label, rendered human-readable.

The taxonomy mirrors ``loris.predicates._core`` and the textual-type set used by
the pipeline's own per-cluster logging in ``loris.pipeline.steps``. Parsing is on
the raw ``rules.json`` dicts (not ``RDLSet.load``) so it is robust to old runs and
never needs the model pool.

CLI
---
    python -m loris.analysis.rule_stats \\
        --runs-glob 'experiments/baselines_loris/_runs/*' \\
        --out experiments/analysis/rule_stats

Pass ``--runs-glob`` multiple times to union the old runs and a later rework:

    python -m loris.analysis.rule_stats \\
        --runs-glob 'experiments/baselines_loris/_runs/*' \\
        --runs-glob 'experiments/baselines_loris_rework/_runs/*' \\
        --out experiments/analysis/rule_stats
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Predicate taxonomy (mirrors loris/predicates/_core.py) ────────────────────
ML_TYPES = {"MLPredicate", "MLThresholdPredicate"}
# textual set as used by loris/pipeline/steps.py per-cluster logging
TEXTUAL_TYPES = {"MatchPredicate", "FreqPredicate", "CooccurPredicate", "BeforePredicate"}
LABEL_TYPES = {"LabelPredicate"}
GRAPH_TYPES = {"SimPredicate", "GroupPredicate"}

# Known dataset tokens, longest-first so 'reuters21578' wins over any prefix.
KNOWN_DATASETS = [
    "reuters21578", "aapd", "arxiv", "bgc", "rcv1", "eurlex",
    "hupd", "pubmed", "goodreads", "micol",
]


def role_of(ptype: str) -> str:
    """Coarse role for a predicate type (feature/text · ml · label · propagation)."""
    if ptype in ML_TYPES:
        return "ml"
    if ptype in TEXTUAL_TYPES:
        return "logic_text"
    if ptype in LABEL_TYPES:
        return "label"
    if ptype in GRAPH_TYPES:
        return "propagation_graph"
    return "other"


def is_ml(ptype: str) -> bool:
    return ptype in ML_TYPES


# ── Human-readable rendering (dict-port of the predicate reprs) ───────────────
def _raw(pat: Any) -> str:
    """Pattern dict -> its raw string."""
    if isinstance(pat, dict):
        return str(pat.get("raw", pat))
    return str(pat)


def pred_readable(p: Dict[str, Any]) -> str:
    """Compact readable form of one predicate dict, matching the pipeline style."""
    t = p.get("type", "?")
    if t == "MLThresholdPredicate":
        return f"ml_thresh('{p.get('model_name')}', label='{p.get('label')}', thresh={p.get('threshold')})"
    if t == "MLPredicate":
        return f"ml('{p.get('model_name')}', label='{p.get('label')}')"
    if t == "MatchPredicate":
        neg = "¬" if p.get("negate") else ""
        return f"{neg}match({p.get('attr')}, '{_raw(p.get('r'))}')"
    if t == "FreqPredicate":
        return f"freq({p.get('attr')}, '{_raw(p.get('r'))}' {p.get('op')} {p.get('eta')})"
    if t == "CooccurPredicate":
        return f"cooccur({p.get('attr')}, '{_raw(p.get('r1'))}', '{_raw(p.get('r2'))}')"
    if t == "BeforePredicate":
        return f"before({p.get('attr')}, '{_raw(p.get('r1'))}', '{_raw(p.get('r2'))}')"
    if t == "LabelPredicate":
        return f"label('{p.get('label')}', {p.get('op')})"
    if t == "SimPredicate":
        return f"sim(thresh={p.get('threshold')})"
    if t == "GroupPredicate":
        return f"group({p.get('attr_name')}, n={p.get('group_count')})"
    # generic fallback
    fields = ", ".join(f"{k}={v!r}" for k, v in p.items() if k != "type")
    return f"{t}({fields})"


def rule_readable(r: Dict[str, Any]) -> str:
    body = r.get("body", []) or []
    body_str = " ∧ ".join(pred_readable(p) for p in body) if body else "⊤"
    op = {"add": "+", "remove": "-", "replace": "=", "equal": "=="}.get(
        r.get("consequence_op", "add"), "+")
    return f"{body_str} → {op}{r.get('consequence')}"


# ── Loading ───────────────────────────────────────────────────────────────────
def dataset_of(run_dir: Path) -> str:
    """Infer the dataset name from a run directory name."""
    name = run_dir.name.lower()
    for ds in KNOWN_DATASETS:
        if ds in name:
            return ds
    # fallback: middle token of <group>_<method>_<dataset>_s<seed>
    parts = run_dir.name.split("_")
    return parts[-2] if len(parts) >= 2 else run_dir.name


def find_rules_json(run_dir: Path) -> Optional[Path]:
    """Locate the single rules.json under a run directory (shallowest wins)."""
    cands = sorted(run_dir.rglob("rules.json"), key=lambda p: (len(p.parts), str(p)))
    return cands[0] if cands else None


def load_rules(run_dir: Path) -> Tuple[List[dict], List[str], Optional[Path]]:
    """Return (rules, label_names, path) for a run dir, or ([],[],None)."""
    path = find_rules_json(run_dir)
    if path is None:
        return [], [], None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return [], [], path
    return data.get("rules", []) or [], data.get("label_names", []) or [], path


# ── Per-run statistics ────────────────────────────────────────────────────────
def rule_stats_for_run(run_dir: Path, top_n: int = 5) -> Optional[Dict[str, Any]]:
    rules, label_names, path = load_rules(run_dir)
    if path is None:
        return None

    type_counts: Counter = Counter()
    role_counts: Counter = Counter()
    op_counts: Counter = Counter()
    stage_counts: Counter = Counter()
    body_len_hist: Counter = Counter()
    pred_str_lens: List[int] = []
    n_ml = 0
    n_logic = 0  # everything that is not ML (textual + label + graph)

    for r in rules:
        body = r.get("body", []) or []
        body_len_hist[len(body)] += 1
        op_counts[r.get("consequence_op", "add")] += 1
        stage = (r.get("val_stats") or {}).get("stage", "unknown")
        stage_counts[stage] += 1
        for p in body:
            t = p.get("type", "?")
            type_counts[t] += 1
            role_counts[role_of(t)] += 1
            pred_str_lens.append(len(pred_readable(p)))
            if is_ml(t):
                n_ml += 1
            else:
                n_logic += 1

    body_lens = [len(r.get("body", []) or []) for r in rules]

    def _len_stats(xs: List[int]) -> Dict[str, float]:
        if not xs:
            return {"mean": 0.0, "median": 0.0, "max": 0, "min": 0}
        return {
            "mean": round(statistics.fmean(xs), 3),
            "median": statistics.median(xs),
            "max": max(xs),
            "min": min(xs),
        }

    return {
        "run": run_dir.name,
        "dataset": dataset_of(run_dir),
        "rules_json": str(path),
        "n_rules": len(rules),
        "n_predicates": sum(type_counts.values()),
        "predicate_counts_by_type": dict(type_counts.most_common()),
        "ml_vs_logic": {
            "ml": n_ml,
            "logic": n_logic,
            "ml_ratio": round(n_ml / max(1, n_ml + n_logic), 4),
        },
        "roles": dict(role_counts.most_common()),
        "consequence_ops": dict(op_counts.most_common()),
        "by_stage": dict(stage_counts.most_common()),
        "body_len_distribution": {str(k): body_len_hist[k] for k in sorted(body_len_hist)},
        "body_len_stats": _len_stats(body_lens),
        "predicate_string_len": _len_stats(pred_str_lens),
        "representative_rules": representative_rules(rules, top_n=top_n),
    }


def representative_rules(rules: List[dict], top_n: int = 5,
                         per_label: bool = False) -> List[Dict[str, Any]]:
    """Top-N rules by score (globally, or one per label if per_label)."""
    def _entry(r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "consequence": r.get("consequence"),
            "consequence_op": r.get("consequence_op", "add"),
            "score": round(float(r.get("score", 0.0)), 4),
            "body_len": len(r.get("body", []) or []),
            "stage": (r.get("val_stats") or {}).get("stage", "unknown"),
            "readable": rule_readable(r),
        }

    if per_label:
        best: Dict[str, dict] = {}
        for r in rules:
            lbl = r.get("consequence")
            if lbl not in best or float(r.get("score", 0)) > float(best[lbl].get("score", 0)):
                best[lbl] = r
        chosen = sorted(best.values(), key=lambda r: -float(r.get("score", 0)))[:top_n]
    else:
        chosen = sorted(rules, key=lambda r: -float(r.get("score", 0)))[:top_n]
    return [_entry(r) for r in chosen]


# ── Aggregation across runs ────────────────────────────────────────────────────
def aggregate_runs(run_dirs: List[Path], top_n: int = 5) -> Dict[str, Any]:
    per_run: List[Dict[str, Any]] = []
    for d in sorted(run_dirs):
        s = rule_stats_for_run(d, top_n=top_n)
        if s is not None:
            per_run.append(s)

    agg_types: Counter = Counter()
    agg_roles: Counter = Counter()
    agg_ops: Counter = Counter()
    agg_stage: Counter = Counter()
    agg_body_hist: Counter = Counter()
    total_ml = 0
    total_logic = 0
    total_rules = 0
    # dataset -> pooled rules for representative selection
    ds_rules: Dict[str, List[dict]] = defaultdict(list)
    per_dataset_rulecount: Counter = Counter()

    for s in per_run:
        for t, c in s["predicate_counts_by_type"].items():
            agg_types[t] += c
        for role, c in s["roles"].items():
            agg_roles[role] += c
        for op, c in s["consequence_ops"].items():
            agg_ops[op] += c
        for st, c in s["by_stage"].items():
            agg_stage[st] += c
        for bl, c in s["body_len_distribution"].items():
            agg_body_hist[int(bl)] += c
        total_ml += s["ml_vs_logic"]["ml"]
        total_logic += s["ml_vs_logic"]["logic"]
        total_rules += s["n_rules"]
        per_dataset_rulecount[s["dataset"]] += s["n_rules"]

    # pool raw rules per dataset for representative rules (re-read; cheap)
    for d in sorted(run_dirs):
        rules, _, path = load_rules(d)
        if path is not None:
            ds_rules[dataset_of(d)].extend(rules)

    rep_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for ds, rules in sorted(ds_rules.items()):
        # dedup identical readable bodies, keep highest score
        best: Dict[str, dict] = {}
        for r in rules:
            key = rule_readable(r)
            if key not in best or float(r.get("score", 0)) > float(best[key].get("score", 0)):
                best[key] = r
        rep_by_dataset[ds] = representative_rules(list(best.values()), top_n=top_n)

    return {
        "n_runs": len(per_run),
        "total_rules": total_rules,
        "total_predicates": sum(agg_types.values()),
        "predicate_counts_by_type": dict(agg_types.most_common()),
        "ml_vs_logic": {
            "ml": total_ml,
            "logic": total_logic,
            "ml_ratio": round(total_ml / max(1, total_ml + total_logic), 4),
        },
        "roles": dict(agg_roles.most_common()),
        "consequence_ops": dict(agg_ops.most_common()),
        "by_stage": dict(agg_stage.most_common()),
        "body_len_distribution": {str(k): agg_body_hist[k] for k in sorted(agg_body_hist)},
        "rules_per_dataset": dict(per_dataset_rulecount.most_common()),
        "representative_rules_by_dataset": rep_by_dataset,
        "per_run": per_run,
    }


# ── Markdown rendering ─────────────────────────────────────────────────────────
def render_markdown(agg: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("# LORIS predicate / rule statistics\n")
    L.append(f"Runs aggregated: **{agg['n_runs']}**  ·  "
             f"total rules: **{agg['total_rules']}**  ·  "
             f"total predicates: **{agg['total_predicates']}**\n")

    # ── Per-experiment breakdown (one row per run/cell) ──────────────────────
    L.append("## Per-experiment breakdown\n")
    L.append("Each experiment cell (method × dataset) separately: rules, predicate "
             "counts, ML-vs-logic split, consequence ops, body length, discovery stages.\n")
    L.append("| experiment | dataset | rules | preds | ML | logic | add | remove | "
             "avg body | stages |")
    L.append("| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | :-- |")
    for s in sorted(agg["per_run"], key=lambda x: (x["dataset"], x["run"])):
        ml = s["ml_vs_logic"]
        ops = s["consequence_ops"]
        stages = ",".join(f"{k}:{v}" for k, v in s["by_stage"].items())
        L.append(
            f"| {s['run']} | {s['dataset']} | {s['n_rules']} | {s['n_predicates']} | "
            f"{ml['ml']} | {ml['logic']} | {ops.get('add', 0)} | {ops.get('remove', 0)} | "
            f"{s['body_len_stats']['mean']} | {stages} |")
    L.append("")
    L.append("Per-experiment predicate-type counts and representative rules are in "
             "`rule_stats.json` under `per_run[].predicate_counts_by_type` and "
             "`per_run[].representative_rules`.\n")

    L.append("## Aggregate (all experiments)\n")

    ml = agg["ml_vs_logic"]
    L.append("## ML vs logic predicates\n")
    L.append("| class | count | share |")
    L.append("| :-- | --: | --: |")
    tot = max(1, ml["ml"] + ml["logic"])
    L.append(f"| ML (MLPredicate, MLThreshold) | {ml['ml']} | {ml['ml']/tot:.1%} |")
    L.append(f"| logic (text/label/graph) | {ml['logic']} | {ml['logic']/tot:.1%} |")
    L.append("")

    L.append("## Predicate count by type\n")
    L.append("| type | role | count |")
    L.append("| :-- | :-- | --: |")
    for t, c in agg["predicate_counts_by_type"].items():
        L.append(f"| {t} | {role_of(t)} | {c} |")
    L.append("")

    L.append("## Roles\n")
    L.append("| role | count |")
    L.append("| :-- | --: |")
    for r, c in agg["roles"].items():
        L.append(f"| {r} | {c} |")
    L.append("")

    L.append("## Consequence ops · discovery stage\n")
    L.append("| op | count |  | stage | count |")
    L.append("| :-- | --: | -- | :-- | --: |")
    ops = list(agg["consequence_ops"].items())
    stages = list(agg["by_stage"].items())
    for i in range(max(len(ops), len(stages))):
        o = f"{ops[i][0]} | {ops[i][1]}" if i < len(ops) else " | "
        s = f"{stages[i][0]} | {stages[i][1]}" if i < len(stages) else " | "
        L.append(f"| {o} |  | {s} |")
    L.append("")

    L.append("## Rule body length (arity) distribution\n")
    L.append("| body_len | #rules |")
    L.append("| --: | --: |")
    for k, v in agg["body_len_distribution"].items():
        L.append(f"| {k} | {v} |")
    L.append("")

    L.append("## Rules per dataset\n")
    L.append("| dataset | #rules |")
    L.append("| :-- | --: |")
    for ds, c in agg["rules_per_dataset"].items():
        L.append(f"| {ds} | {c} |")
    L.append("")

    L.append("## Representative rules (top by F1-gain, per dataset)\n")
    for ds, reps in agg["representative_rules_by_dataset"].items():
        L.append(f"### {ds}\n")
        for r in reps:
            L.append(f"- `{r['readable']}`  — score={r['score']}, "
                     f"len={r['body_len']}, stage={r['stage']}")
        L.append("")

    return "\n".join(L) + "\n"


# ── CLI ────────────────────────────────────────────────────────────────────────
def _expand_globs(globs: List[str]) -> List[Path]:
    dirs: List[Path] = []
    seen = set()
    for g in globs:
        for m in glob.glob(g):
            p = Path(m)
            if p.is_dir() and str(p) not in seen:
                seen.add(str(p))
                dirs.append(p)
    return dirs


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate LORIS predicate/rule statistics.")
    ap.add_argument("--runs-glob", action="append", required=True, dest="runs_glob",
                    help="glob for run directories (repeatable to union sets)")
    ap.add_argument("--out", required=True,
                    help="output path prefix; writes <out>.json and <out>.md")
    ap.add_argument("--top-n", type=int, default=5, help="representative rules per group")
    args = ap.parse_args(argv)

    run_dirs = _expand_globs(args.runs_glob)
    if not run_dirs:
        print(f"No run directories matched: {args.runs_glob}")
        return 1

    agg = aggregate_runs(run_dirs, top_n=args.top_n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    md_path = out.with_suffix(".md")
    json_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False))
    md_path.write_text(render_markdown(agg))

    print(f"Aggregated {agg['n_runs']} runs → {agg['total_rules']} rules, "
          f"{agg['total_predicates']} predicates")
    print(f"  ML/logic: {agg['ml_vs_logic']['ml']}/{agg['ml_vs_logic']['logic']} "
          f"(ml_ratio={agg['ml_vs_logic']['ml_ratio']})")
    print(f"  types: {agg['predicate_counts_by_type']}")
    print(f"  ops: {agg['consequence_ops']}")
    print(f"  body_len: {agg['body_len_distribution']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
