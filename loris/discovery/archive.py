"""Read-only discovery status and lossless BO inventory with SQLite snapshots."""

import datetime
import json
from pathlib import Path
import sqlite3

from .manifest import bound, file_record, read_json

FIT_PRECISION_REJECTIONS = ("fit_precision_below_floor", "blocked_Gamma_contradiction")


def status(directory):
    directory = Path(directory).resolve(strict=True)
    completed = directory / "COMPLETE.json"
    clusters = []
    for folder in sorted((directory / "learning").glob("cluster_*")):
        trials = sorted(folder.glob("trial_*.json"))
        progress = folder / "worker_status.json"
        clusters.append(
            {
                "directory": folder.name,
                "stored_trials": len(trials),
                "pending": (folder / "pending.json").exists(),
                "worker": read_json(progress) if progress.exists() else None,
            }
        )
    if completed.exists():
        receipt = read_json(completed)
        if receipt.get("schema") == "loris_completed_BO_admission_continuation_v1":
            selected = read_json(bound(receipt["selection"], directory))
            proof_path = bound(receipt["proof"], directory)
            proof = read_json(proof_path)
            from loris._storage.implicit_seed_proof import validate_proof_roots
            validate_proof_roots(proof, proof_path.parent)
            facts = bound(proof["facts"], directory)
            bound(proof["proof_records"], directory)
            if (selected["evaluation"]["status"] != "fixpoint"
                    or len(selected["rules"]) != receipt["selected_rules"]
                    or proof["complete_facts_conflicts_rounds_scores_equal"] is not True
                    or proof["evaluation"]["positive_metrics"] != selected["evaluation"]["positive_metrics"]):
                raise ValueError("continuation final program/proof differs")
            return dict(status="complete_verified_admission_continuation",run=str(directory),
                rules=receipt["selected_rules"],positive_metrics=selected["evaluation"]["positive_metrics"],
                complete_receipt=file_record(completed),new_BO=0,new_training=0,new_Chase=0,
                approximate=True,source_pools=selected["clusters"],facts=file_record(facts))
        selected_path = bound(receipt["selection"], directory)
        selected = read_json(selected_path)
        if receipt.get("status") != "complete" or selected["evaluation"]["status"] != "fixpoint":
            raise ValueError("completion must bind a successful full program")
        if len(selected["rules"]) != receipt["selected_rules"]:
            raise ValueError("rule count differs from completed ordered program")
        proof_path = bound(receipt["proof"], directory)
        proof = read_json(proof_path)
        from loris._storage.implicit_seed_proof import validate_proof_roots
        validate_proof_roots(proof, proof_path.parent)
        facts = bound(proof["facts"], directory)
        if "proof_records" in proof:
            bound(proof["proof_records"], directory)
        if (
            proof.get("complete_facts_conflicts_rounds_scores_equal") is not True
            or proof["evaluation"]["status"] != "fixpoint"
            or proof["evaluation"]["positive_metrics"] != selected["evaluation"]["positive_metrics"]
        ):
            raise ValueError("proof replay does not certify the selected successful program")
        expected = receipt["complete_BO_trials"]
        if sum(c["stored_trials"] for c in clusters) != expected or any(
            c["pending"] for c in clusters
        ):
            raise ValueError("completed BO has missing/extra trials or a pending proposal")
        # New descending-order runs bind their exact admission permutation.
        # Do not report completion after losing that required replay identity.
        if receipt.get("archive") is not None:
            archive = read_json(bound(receipt["archive"], directory))
            if archive.get("implicit_seed_roots", []) != proof.get("implicit_seed_roots", []):
                raise ValueError("Archive compact proof root references differ")
            for entry in archive.get("clusters", []):
                order_record = entry.get("records", {}).get("admission_order")
                if order_record is not None:
                    bound(order_record, directory)
        return {
            "status": "complete_verified_identity",
            "run": str(directory),
            "rules": receipt["selected_rules"],
            "BO_trials": expected,
            "positive_metrics": selected["evaluation"]["positive_metrics"],
            "complete_receipt": file_record(completed),
            "clusters": clusters,
            "facts": file_record(facts),
            "new_training": 0,
            "new_BO": 0,
            "new_Chase": 0,
        }
    state = directory / "STATUS.json"
    return {
        "status": "incomplete",
        "run": str(directory),
        "clusters": clusters,
        "last_status": read_json(state) if state.exists() else None,
        "process_liveness_not_inferred_from_status_files": True,
    }


def snapshot_sqlite(source, destination):
    """SQLite backup includes committed WAL pages; no raw main-file copying."""
    source, destination = Path(source).resolve(strict=True), Path(destination)
    if destination.exists():
        raise FileExistsError(str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
    dst = sqlite3.connect(str(destination))
    try:
        src.backup(dst)
        integrity = dst.execute("PRAGMA integrity_check").fetchall()
        if integrity != [("ok",)]:
            raise ValueError("SQLite snapshot failed integrity check: " + str(integrity))
        tables = [
            r[0]
            for r in dst.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ]
        counts = {
            name: dst.execute('SELECT count(*) FROM "' + name.replace('"', '""') + '"').fetchone()[
                0
            ]
            for name in tables
        }
    finally:
        dst.close()
        src.close()
    return {
        "source": str(source),
        "snapshot": file_record(destination),
        "method": "sqlite_connection_backup_including_committed_WAL",
        "integrity": "ok",
        "table_rows": counts,
    }


def _semantic_files(run):
    for name in ("learning", "initialization"):
        directory = run / name
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if (
                    path.is_file()
                    and "__pycache__" not in path.parts
                    and not path.name.endswith(("-wal", "-shm", "-journal"))
                ):
                    yield path
    for path in sorted(run.iterdir()):
        if path.is_file() and path.suffix in {
            ".json",
            ".jsonl",
            ".npz",
            ".sqlite",
            ".sqlite3",
            ".gz",
            ".bin",
        }:
            yield path
    workers = run / "worker_scores"
    if workers.exists():
        yield from sorted(
            p for p in workers.iterdir() if p.is_file() and p.suffix in {".sqlite", ".sqlite3"}
        )


def _verify_cluster(folder, prefix):
    """Validate both complete layers and ordered prefix transitions; no Chase."""
    from loris._engine.rlearner import digest

    protocol = read_json(folder / "protocol.json")
    complete = read_json(folder / "BO_COMPLETE.json")
    protocol_id = digest(protocol)
    records = [read_json(p) for p in sorted(folder.glob("trial_*.json"))]
    if (
        complete["protocol_id"] != protocol_id
        or complete["complete_trials"] != len(records)
        or complete["all_trial_records_sha256"] != digest(records)
    ):
        raise ValueError("complete BO digest/count/protocol mismatch: " + str(folder))
    binding_id = digest(protocol["evaluator"])

    def key(rules):
        return digest({"binding_id": binding_id, "ordered_rules": rules})

    admissions = folder / "admission"
    before = read_json(admissions / "before.json")
    if before["rules"] != prefix or before["evaluation"]["program_key"] != key(prefix):
        raise ValueError("admission starts from a different ordered prefix")
    if len(list(admissions.glob("candidate_*.json"))) != len(records):
        raise ValueError("missing or extra sequential admission records")
    categories = {}
    prefix = list(prefix)
    from .search.admission import order_trials

    run_protocol_path = folder.parent / "protocol.json"
    order = (
        read_json(run_protocol_path).get("admission_order", "proposal_order")
        if run_protocol_path.exists()
        else "proposal_order"
    )
    numbers = order_trials(records, order)
    epsilon = 1e-12  # Historical archives used this explicit implementation tolerance.
    batch_mode = False
    if order != "proposal_order" or (admissions / "ORDER.json").exists():
        order_receipt = read_json(admissions / "ORDER.json")
        batch_mode = order_receipt.get("policy") == "batch_prefilter"
        if batch_mode and (type(order_receipt.get("batch_size")) is not int or order_receipt["batch_size"] < 1):
            raise ValueError("invalid approximation batch size")
        if (
            order_receipt["order"] != order
            or order_receipt["trial_numbers"] != numbers
            or order_receipt["raw_records_sha256"] != digest(records)
        ):
            raise ValueError("declared score-order admission differs from its raw proposals")
        epsilon = order_receipt.get("strict_gain_epsilon", 1e-12)
        if type(epsilon) not in (int, float) or epsilon not in (0.0, 1e-12):
            raise ValueError("unsupported stored admission gain tolerance")
    for position, number in enumerate(numbers):
        if batch_mode and position % order_receipt["batch_size"] == 0:
            batch_prefix = list(prefix)
        proposal = records[number]
        if (
            proposal.get("number") != number
            or proposal.get("protocol_id") != protocol_id
            or not {"params", "configuration", "standalone", "rule"} <= set(proposal)
        ):
            raise ValueError("raw proposal identity/axis/order differs")
        if not (folder / ("trial_%05d.json" % number)).is_file():
            raise ValueError("trial sequence has gaps or wrong filenames")
        standalone = proposal["standalone"]
        evaluation = standalone.get("evaluation")
        kind = evaluation["status"] if evaluation is not None else "invalid_or_no_rule"
        categories[kind] = categories.get(kind, 0) + 1
        if standalone.get("selected_prefix_used") is not False:
            raise ValueError("standalone BO must not score an admitted prefix")
        raw = proposal["rule"]
        if raw is not None:
            if evaluation is None or evaluation["program_key"] != key([raw]):
                raise ValueError("standalone score belongs to another rule")
            if evaluation["status"] != "fixpoint" and (
                evaluation.get("positive_metrics") is not None or standalone["score"] != -1
            ):
                raise ValueError("failed candidate has a normal F1 score")
        receipt = read_json(admissions / ("candidate_%05d.json" % number))
        if order != "proposal_order" and receipt.get("admission_position") != position:
            raise ValueError("admission position differs from declared decreasing-score order")
        if (
            receipt["trial_number"] != number
            or receipt["candidate"] != raw
            or receipt["standalone"] != standalone
            or receipt["selected_before"] != len(prefix)
        ):
            raise ValueError("admission does not match its original proposal and prefix")
        decision = receipt["decision"]
        if type(decision["accepted"]) is not bool:
            raise ValueError("explicit accepted/rejected decision required")
        if raw is None:
            if decision["accepted"]:
                raise ValueError("invalid rule was admitted")
        else:
            baseline, proposed = decision["baseline"], decision["evaluation"]
            evaluation_prefix = prefix
            if batch_mode:
                if receipt.get("batch_prefix_key") != key(batch_prefix):
                    raise ValueError("parallel batch prefix differs")
                if not receipt.get("rechecked_after_prefix_change"):
                    if decision != receipt.get("prefilter_decision"):
                        raise ValueError("prefilter decision changed without a recheck")
                    if decision["accepted"] and prefix != batch_prefix:
                        raise ValueError("stale accepted prefilter decision was not rechecked")
                    evaluation_prefix = batch_prefix
            gated = decision.get("reason") in FIT_PRECISION_REJECTIONS
            if gated:
                # Fit-precision gate: rejected on the rule's own evaluation.
                if (decision["accepted"] or baseline["status"] != "fixpoint"
                        or baseline["program_key"] != key(evaluation_prefix)
                        or proposed["program_key"] != key([raw])
                        or decision["delta_macro_f1"] is not None):
                    raise ValueError("fit-precision rejection has an inconsistent receipt")
            elif (
                baseline["status"] != "fixpoint"
                or baseline["program_key"] != key(evaluation_prefix)
                or proposed["program_key"] != key(evaluation_prefix + [raw])
            ):
                raise ValueError("whole-program score belongs to a different ordered program")
            delta = (
                proposed["positive_metrics"]["macro_f1"] - baseline["positive_metrics"]["macro_f1"]
                if proposed["status"] == "fixpoint"
                else None
            )
            if gated:
                pass
            elif order == "random_consistent":
                expected = proposed["status"] == "fixpoint" and raw not in evaluation_prefix
                if (
                    decision.get("criterion") != "consistent_distinct_rule"
                    or decision.get("score_used_for_selection") is not False
                    or decision["delta_macro_f1"] != delta
                    or decision["accepted"] != expected
                ):
                    raise ValueError(
                        "random admission differs from score-independent consistency and deduplication"
                    )
            elif decision["delta_macro_f1"] != delta or decision["accepted"] != (
                delta is not None and delta > epsilon
            ):
                raise ValueError("admission disagrees with strict complete-program F1 gain")
        if decision["accepted"]:
            prefix.append(raw)
        if receipt["selected_after"] != len(prefix) or receipt["selected_program_key"] != key(
            prefix
        ):
            raise ValueError("admission resulting prefix differs")
    after = read_json(admissions / "after.json")
    if (
        after["rules"] != prefix
        or after["all_candidates_processed"] != len(records)
        or after["evaluation"]["program_key"] != key(prefix)
    ):
        raise ValueError("cluster final prefix is incomplete")
    return {
        "directory": folder.name,
        "trials": len(records),
        "standalone_statuses": categories,
        "BO_records_digest_verified": True,
        "all_sequential_admissions_verified": not batch_mode,
        "approximate_batch_admissions_verified": batch_mode,
    }, prefix


def archive_index(directory, output, *, snapshot_studies=True):
    """Index complete proposals and dependencies without copying model assets.

    This is a bound inventory, not a self-contained copy of the whole project.
    Study snapshots are independently readable backups. All indexed original
    files and external model/context dependencies must be retained for replay.
    Rejected candidate proof DAGs are not claimed to have been materialized.
    """
    run = Path(directory).resolve(strict=True)
    checked = status(run)
    if checked["status"] != "complete_verified_identity":
        raise ValueError("final archival inventory requires a completed quiescent discovery")
    out = Path(output).resolve()
    if out == run or run in out.parents:
        raise ValueError("archive output must be outside the immutable source run")
    out.mkdir(parents=True, exist_ok=False)
    clusters, prefix = [], []
    for folder in sorted((run / "learning").glob("cluster_*")):
        summary, prefix = _verify_cluster(folder, prefix)
        clusters.append(summary)
    selected = read_json(bound(read_json(run / "COMPLETE.json")["selection"], run))
    if prefix != selected["rules"]:
        raise ValueError("final selected program differs from all original admission decisions")
    files = [
        dict(file_record(p), relative_path=str(p.relative_to(run)), bytes=p.stat().st_size)
        for p in _semantic_files(run)
    ]
    snapshots = []
    if snapshot_studies:
        for path in sorted((run / "learning").glob("cluster_*/optuna.sqlite")):
            snapshots.append(snapshot_sqlite(path, out / "studies" / path.parent.name / path.name))
    # Completion identity must not change during the inventory operation.
    if checked["complete_receipt"] != file_record(run / "COMPLETE.json"):
        raise RuntimeError("run changed while being archived")
    value = {
        "schema": "loris_complete_BO_archive_index_v1",
        "source_run": str(run),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "indexed_complete_BO",
        "completion": checked,
        "clusters": clusters,
        "files": files,
        "study_snapshots": snapshots,
        "indexed_bytes": sum(x["bytes"] for x in files),
        "models_probabilities_and_contexts": "references in original INPUTS/protocol; retained, not duplicated",
        "cache_reuse": "exact data/labels/roots/models/thresholds/text/semantics/program identities only",
        "prefix_change": "rescore affected whole-program admissions; stored proposals remain reusable",
        "all_rejected_candidate_DAGs_materialized": False,
        "archive_is_self_contained": False,
        "initialization": (
            "the initializer protocol declares ALL_FIT_CANDIDATES or "
            "EVALUATED_FIT_CANDIDATES plus ANALYTICALLY_SKIPPED; "
            "startup seeds consume original BO slots"
        ),
        "source_files": [file_record(__file__)],
    }
    with (out / "BO_ARCHIVE_INDEX.json").open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return value
