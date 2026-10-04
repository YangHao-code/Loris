"""Author-approved approximate admission with a deterministic batch boundary.

Rejects against the batch-start prefix are not reconsidered within this pass.
Survivors are committed in the original order and rechecked whenever that
prefix changed. Thus accepted programs keep the original criterion; candidates
that need an earlier member of this batch may be missed. This is not Algorithm
1 equivalence. Workers and batch size are deliberately independent controls.
"""
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from loris._engine.rlearner import digest, immutable_json, rule_record
from loris.rule_forms import parse_author_rule

POLICIES = ("strict", "batch_prefilter")


def decide(evaluator, selected, rule, order):
    if order != "random_consistent":
        return evaluator.admit(selected, rule, epsilon=0.0)
    baseline = evaluator.evaluate(selected)
    if baseline["status"] != "fixpoint":
        raise ValueError("random admission requires a consistent prefix")
    duplicate = rule_record(rule) in [rule_record(r) for r in selected]
    proposed = evaluator.evaluate(tuple(selected) + (rule,))
    accepted = proposed["status"] == "fixpoint" and not duplicate
    return dict(line=12, accepted=accepted, criterion="consistent_distinct_rule",
                delta_macro_f1=(proposed["positive_metrics"]["macro_f1"] -
                                baseline["positive_metrics"]["macro_f1"]
                                if proposed["status"] == "fixpoint" else None),
                baseline=baseline, evaluation=proposed, score_used_for_selection=False,
                reason="duplicate" if duplicate else "consistent_random_rule" if accepted else "conflict")


_worker = None
_source_scope = None


def _initialize(context, output, lexical_cache_dir, gamma_protection=False, fit_precision_floor=None):
    # Linux fork shares the already loaded, fixed input arrays. Never reuse a
    # parent's SQLite connection. Worker score caches are intentionally memory
    # only; the parent commits batch decisions and raw BO remains unchanged.
    global _worker, _source_scope
    from loris.discovery.runner import evaluator_pair
    from .source_cache import cached_source_extraction
    _source_scope = cached_source_extraction()
    _source_scope.__enter__()
    _worker = evaluator_pair(context, Path(output), "prefilter_%d" % os.getpid(),
                             lexical_cache_dir, persistent_scores=False,
                             gamma_protection=gamma_protection, fit_precision_floor=fit_precision_floor)


def _score(task):
    prefix, raw, order = task
    runner, evaluator, admission = _worker
    before = (evaluator.actual_calls, evaluator.actual_compute_seconds, evaluator.cache_hits)
    started = time.monotonic()
    decision = decide(admission, tuple(parse_author_rule(r) for r in prefix),
                      parse_author_rule(raw), order)
    return decision, dict(calls=evaluator.actual_calls - before[0],
                          compute_seconds=evaluator.actual_compute_seconds - before[1],
                          cache_hits=evaluator.cache_hits - before[2],
                          seconds=time.monotonic() - started, pid=os.getpid())


class ParallelScorer:
    def __init__(self, context, output, workers=8, lexical_cache_dir=None,
                 gamma_protection=False, fit_precision_floor=None):
        if type(workers) is not int or workers < 1:
            raise ValueError("admission workers must be positive")
        self.workers = workers
        self.calls = 0
        self.compute_seconds = 0.0
        # fork is required here to avoid duplicating million-document contexts
        # through pickle. Unsupported hosts fail explicitly before launching.
        self.pool = ProcessPoolExecutor(max_workers=workers,
                                       mp_context=multiprocessing.get_context("fork"),
                                       initializer=_initialize,
                                       initargs=(context, str(output), lexical_cache_dir,
                                                 gamma_protection, fit_precision_floor))

    def __call__(self, selected, rules, order):
        prefix = [rule_record(r) for r in selected]
        results = list(self.pool.map(_score, [(prefix, rule_record(r), order) for r in rules], chunksize=1))
        self.calls += sum(r[1]["calls"] for r in results)
        self.compute_seconds += sum(r[1]["compute_seconds"] for r in results)
        return [r[0] for r in results], [r[1] for r in results]

    def close(self):
        self.pool.shutdown(wait=True)


def admit_batches(*, order, directory, records, numbers, selected, evaluator,
                  rule_parser, score_batch, batch_size=8, on_progress=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("admission batch size must be positive")
    directory = Path(directory)
    selected = list(selected)
    immutable_json(directory / "ORDER.json", dict(
        order=order, trial_numbers=numbers, raw_records_sha256=digest(records),
        policy="batch_prefilter", batch_size=batch_size,
        tie_break="original trial number within unchanged cluster order",
        initial_rejections_revisited=False, changed_prefix_survivors_rechecked=True,
        approximate=True, strict_gain_epsilon=0.0, score_guided=order != "random_consistent"))
    immutable_json(directory / "before.json", dict(
        rules=[rule_record(r) for r in selected], evaluation=evaluator.evaluate(selected)))
    for start in range(0, len(numbers), batch_size):
        block = numbers[start:start + batch_size]
        commit = directory / ("batch_%05d.json" % (start // batch_size))
        prefix_key = evaluator.program_key(selected)
        if commit.exists():
            saved = json.loads(commit.read_text())
            if saved["prefix_key"] != prefix_key or saved["trial_numbers"] != block:
                raise ValueError("committed admission batch has different inputs/prefix")
            receipts = []
            for number in block:
                receipt = json.loads((directory / ("candidate_%05d.json" % number)).read_text())
                if receipt["candidate"] != records[number]["rule"] or receipt["selected_before"] != len(selected):
                    raise ValueError("committed candidate differs from raw trial or prefix")
                if receipt["decision"]["accepted"]:
                    selected.append(rule_parser(receipt["candidate"]))
                if receipt["selected_program_key"] != evaluator.program_key(selected):
                    raise ValueError("committed candidate program key differs")
                receipts.append(receipt)
            if saved["receipts_sha256"] != digest(receipts):
                raise ValueError("committed candidate receipts changed")
        else:
            started = time.monotonic()
            valid_numbers = [n for n in block if records[n]["rule"] is not None]
            raw_decisions, costs = score_batch(selected,
                [rule_parser(records[n]["rule"]) for n in valid_numbers], order) if valid_numbers else ([], [])
            if len(raw_decisions) != len(valid_numbers):
                raise ValueError("parallel scorer returned incomplete decisions")
            initial = dict(zip(valid_numbers, raw_decisions))
            prefilter_seconds = time.monotonic() - started
            prefix_count = len(selected)
            receipts, rechecks = [], 0
            for offset, number in enumerate(block):
                record = records[number]
                raw = record["rule"]
                preliminary = initial.get(number)
                rechecked = False
                if raw is None:
                    decision = dict(line=12, accepted=False, reason=record["no_body_reason"])
                elif not preliminary["accepted"] or len(selected) == prefix_count:
                    decision = preliminary
                else:
                    rechecked = True
                    rechecks += 1
                    decision = decide(evaluator, selected, rule_parser(raw), order)
                receipt = dict(trial_number=number, admission_position=start + offset,
                               candidate=raw, standalone=record["standalone"], decision=decision,
                               selected_before=len(selected), policy="batch_prefilter",
                               batch_prefix_key=prefix_key, prefilter_decision=preliminary,
                               rechecked_after_prefix_change=rechecked)
                if decision["accepted"]:
                    selected.append(rule_parser(raw))
                receipt.update(selected_after=len(selected), selected_program_key=evaluator.program_key(selected))
                immutable_json(directory / ("candidate_%05d.json" % number), receipt)
                receipts.append(receipt)
            immutable_json(commit, dict(prefix_key=prefix_key, trial_numbers=block,
                receipts_sha256=digest(receipts), worker_costs=costs,
                prefilter_seconds=prefilter_seconds, total_seconds=time.monotonic() - started,
                survivor_rechecks=rechecks, selected_after=len(selected)))
        if on_progress:
            on_progress(dict(phase="line12_parallel_batch_admission", processed=start + len(block),
                budget=len(records), selected=len(selected), admission_order=order,
                admission_policy="batch_prefilter", admission_batch_size=batch_size,
                actual_parent_chase_calls=evaluator.actual_calls,
                actual_prefilter_chase_calls=getattr(score_batch, "calls", None),
                cache_hits=evaluator.cache_hits))
    immutable_json(directory / "after.json", dict(rules=[rule_record(r) for r in selected],
        evaluation=evaluator.evaluate(selected), all_candidates_processed=len(records),
        admission_policy="batch_prefilter", approximate=True))
    return selected
