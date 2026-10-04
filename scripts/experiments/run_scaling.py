#!/usr/bin/env python
"""Execute only predeclared, genuinely prepared nested D/Sigma/Lambda points."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

from loris.discovery.manifest import bound
from loris.experiments import workflow as wf


def digest_rule(record):
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def is_subsequence(small, large):
    it = iter(large)
    return all(any(v == x for v in it) for x in small)


def main(sweep_path, output, dry_run=False):
    source = Path(sweep_path).resolve(strict=True)
    spec = wf.read(source)
    if spec.get("schema") != "loris_prepared_scaling_sweep_v1" or spec["dimension"] not in (
        "document_ids",
        "rule_ids",
        "labels",
    ):
        raise ValueError("declare document_ids, rule_ids or labels scaling axis")
    points = spec["points"]
    if len(points) < 2 or len({p["id"] for p in points}) != len(points):
        raise ValueError("at least two uniquely named prepared scaling points required")
    for i, point in enumerate(points):
        if not point["id"].replace("_", "").replace("-", "").isalnum():
            raise ValueError("unsafe point name")
        for axis in ("document_ids", "rule_ids", "labels"):
            values = point[axis]
            if not values or len(set(values)) != len(values):
                raise ValueError("explicit complete unique scaling axes required")
            if i:
                previous = points[i - 1][axis]
                if axis == spec["dimension"]:
                    if len(values) <= len(previous) or not is_subsequence(previous, values):
                        raise ValueError("scaling axis must be strictly nested in preserved order")
                elif values != previous:
                    raise ValueError("only the declared scaling dimension may change")
        bound(point["assets"], source.parent)
    if dry_run:
        return dict(
            status="declared_plan_only_not_asset_certified",
            dimension=spec["dimension"],
            points=[
                dict(
                    id=p["id"],
                    counts={a: len(p[a]) for a in ("document_ids", "rule_ids", "labels")},
                )
                for p in points
            ],
        )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    workspaces = []
    # Validate every prepared point before any model training/refinement starts.
    for point in points:
        work = out / point["id"]
        wf.bind_assets(bound(point["assets"], source.parent), work)
        config = wf.read(work / "ASSETS.json")
        if config["rules_mode"] != "reuse":
            raise ValueError("scaling requires preprepared complete program assets")
        wf.main(["audit", "--workspace", str(work)])
        wf.main(["rules", "--workspace", str(work)])
        app = wf.manifest(work, "labeling_manifest")
        with bound(app["context"], work).open("rb") as stream:
            context = pickle.load(stream)
        actual_rules = wf.read(wf.program(work))["rules"]
        if context["row_ids"] != point["document_ids"] or context["labels"] != point["labels"]:
            raise ValueError(
                "prepared application context does not match declared full scaling axes"
            )
        if [digest_rule(r) for r in actual_rules] != point["rule_ids"]:
            raise ValueError("prepared ordered program differs from declared rule hashes")
        for name in ("reliability_protocol", "evaluation"):
            wf.asset(work, name)
        workspaces.append(work)
    for work in workspaces:
        for stage in ("calibrate", "label", "evaluate", "collect"):
            wf.main([stage, "--workspace", str(work)])
    result = dict(
        status="executed",
        dimension=spec["dimension"],
        workspaces=[str(p) for p in workspaces],
        synthetic_or_truncated_assets_generated=False,
        scaling_points_are_not_independent_seed_repetitions=True,
    )
    wf.write(out / "SWEEP.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    print(json.dumps(main(a.sweep, a.output, a.dry_run), indent=2))
