"""Copy small, exact ordered programs; reference large immutable BO assets."""

import json
from pathlib import Path
import re
import shutil

from loris.discovery.manifest import file_record, bound
from loris._engine.rlearner import atomic_json


def export_rules(source_index, output):
    source = Path(source_index).resolve(strict=True)
    index = json.loads(source.read_text())
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("rule registry exists; choose an independent destination")
    out.mkdir(parents=True)
    entries = []
    for item in index["completed"]:
        original = bound(item["canonical_ordered_program"], source.parent)
        raw = json.loads(original.read_text())
        if len(raw["rules"]) != item["rules"]:
            raise ValueError("rule count differs from authenticated index")
        slug = re.sub(r"[^a-z0-9_]+", "_", item["dataset"].lower())
        target = out / slug / "SELECTED.json"
        target.parent.mkdir()
        shutil.copyfile(original, target)
        copied = file_record(target)
        if copied["sha256"] != item["canonical_ordered_program"]["sha256"]:
            raise RuntimeError("ordered program copy differs")
        copied["path"] = str(target.relative_to(out))
        entries.append(
            dict(
                dataset=item["dataset"],
                rules=item["rules"],
                local_program=copied,
                original=item,
                producer_unchanged=True,
            )
        )
    result = dict(
        schema="loris_ordered_rule_registry_v1",
        source_index=file_record(source),
        entries=entries,
        copied_assets="ordered programs only",
        large_assets="original immutable references; no checkpoint/probability/DAG copies",
        scope="Registry is not a standalone dataset/model/BO archive.",
    )
    atomic_json(out / "INDEX.json", result)
    return result


def check_complete_program(program_path, completion_path=None):
    """Reject a candidate pool, partial BO result or mismatched completion."""
    from loris._engine.rdl_reference import Rule
    from loris.rule_forms import audit_program_rule_forms

    path = Path(program_path).resolve(strict=True)
    program = json.loads(path.read_text())
    if not isinstance(program, dict) or not isinstance(program.get("rules"), list):
        raise ValueError("complete learned program must contain an ordered rules array")
    if not program.get("all_candidates_sequentially_evaluated"):
        raise ValueError("all candidate admissions must have completed")
    trials = program.get("complete_BO_trials")
    clusters = program.get("clusters")
    if type(trials) is not int or trials < 1 or not isinstance(clusters, list) or not clusters:
        raise ValueError("complete BO trial and cluster records required")
    if sum(c.get("complete_BO_trials", -1) for c in clusters) != trials:
        raise ValueError("cluster BO counts differ from the completed program")
    if len({c["cluster"] for c in clusters}) != len(clusters):
        raise ValueError("duplicate cluster completion")
    if program.get("evaluation", {}).get("status") != "fixpoint":
        raise ValueError("selected program did not reach a successful closure")
    for record in program["rules"]:
        Rule.from_dict(record)
    if completion_path is None:
        raise ValueError("a completion receipt is required; a program filename is insufficient")
    receipt_path = Path(completion_path).resolve(strict=True)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "complete" or receipt.get("selected_rules") != len(
        program["rules"]
    ):
        raise ValueError("completion status or selected rule count differs")
    if receipt.get("complete_BO_trials") != trials:
        raise ValueError("completion trial count differs")
    actual = file_record(path)
    selection = receipt.get("selection", {})
    if selection.get("sha256") != actual["sha256"]:
        raise ValueError("completion receipt is bound to another ordered program")
    if receipt.get("proof") is not None:
        from loris._storage.implicit_seed_proof import validate_proof_roots
        proof_path = bound(receipt["proof"], receipt_path.parent)
        validate_proof_roots(proof_path)
    return dict(
        status="verified_complete_program",
        rules=len(program["rules"]),
        complete_BO_trials=trials,
        clusters=len(clusters),
        program=actual,
        completion=file_record(receipt_path),
        all_rules_decoded=True,
        historical_receipt_not_new_protocol_result=True,
        current_rule_form_audit=audit_program_rule_forms(program["rules"]),
    )


def check_rules(index_path):
    from loris.rule_forms import audit_program_rule_forms

    path = Path(index_path).resolve(strict=True)
    value = json.loads(path.read_text())
    rows = []
    for item in value["entries"]:
        program = bound(item["local_program"], path.parent)
        if len(json.loads(program.read_text())["rules"]) != item["rules"]:
            raise ValueError("rule count mismatch")
        completion = item.get("completion") or item.get("original", {}).get("completion")
        if completion:
            complete = bound(completion, path.parent)
            check_complete_program(program, complete)
            scope = "program_and_completion_verified"
        else:
            # v1 external registries without completion are visibly weaker.
            from loris._engine.rdl_reference import Rule

            for record in json.loads(program.read_text())["rules"]:
                Rule.from_dict(record)
            scope = "program_only_not_complete_discovery_certified"
        rows.append(
            dict(
                dataset=item["dataset"],
                rules=item["rules"],
                status="verified",
                scope=scope,
                current_rule_form_audit=audit_program_rule_forms(
                    json.loads(program.read_text())["rules"]
                ),
            )
        )
    return dict(
        status="verified",
        entries=rows,
        large_assets_replayed=False,
        verified_scope="historical integrity and completion; current execution additionally requires a valid rule form audit",
    )


def check_execution_program(program_path, completion_path):
    """Accept a complete learned program or an explicitly certified subset ablation."""
    from loris.rule_forms import validate_program_rule_forms

    receipt_path = Path(completion_path).resolve(strict=True)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "loris_nested_program_subset_v1":
        result = check_complete_program(program_path, completion_path)
        validate_program_rule_forms(json.loads(Path(program_path).read_text())["rules"])
        return result
    required = {
        "schema",
        "status",
        "selection",
        "parent_program",
        "parent_completion",
        "parent_indices",
        "scope",
    }
    if (
        set(receipt) != required
        or receipt["status"] != "complete"
        or receipt["scope"] != "fixed_program_subset_ablation"
    ):
        raise ValueError("explicit complete-parent subset ablation receipt required")
    parent = bound(receipt["parent_program"], receipt_path.parent)
    completion = bound(receipt["parent_completion"], receipt_path.parent)
    proof = check_complete_program(parent, completion)
    program = Path(program_path).resolve(strict=True)
    actual = file_record(program)
    if receipt["selection"].get("sha256") != actual["sha256"]:
        raise ValueError("subset receipt binds another ordered program")
    indices = receipt["parent_indices"]
    rules = json.loads(parent.read_text())["rules"]
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(i) is not int or not 0 <= i < len(rules) for i in indices)
        or indices != sorted(set(indices))
    ):
        raise ValueError(
            "subset indices must be unique increasing positions in the complete parent"
        )
    selected = json.loads(program.read_text())
    if selected.get("rules") != [rules[i] for i in indices]:
        raise ValueError("subset rules differ from certified complete parent positions")
    rule_form_audit = validate_program_rule_forms(selected["rules"])
    return dict(
        status="verified_complete_parent_subset_ablation",
        rules=len(indices),
        program=actual,
        completion=file_record(receipt_path),
        parent=proof,
        new_discovery_result_claimed=False,
        current_rule_form_audit=rule_form_audit,
    )
