"""Real ordinary-Chase residual, rollback, polarity and portable CLI checks."""
import copy
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from loris.document import Document
from loris._engine.rdl_reference import Doc, Label, ML, RefGamma, Rule
from loris.discovery.manifest import file_record
from loris.labeling import cli
from loris.labeling.residual_aux import ResidualAuxWorkflow, dual_gate, masked_teacher_bce


def fixture(tmp_path, *, alias="new_aux", application_gamma=True):
    n, k = 8, 2
    truth = np.zeros((n, k), bool); truth[[0, 7]] = True
    old = np.full((n, k), .1, np.float32); old[:6] = .9
    arb = np.where(truth, .9, .1).astype(np.float32)
    contexts = {}
    for name in ("calibration", "application"):
        gamma = RefGamma.empty(n, k)
        if name == "application" and application_gamma:
            neg = np.zeros((n, k), bool); neg[0, 0] = True
            gamma = RefGamma(np.zeros_like(neg), neg)
        contexts[name] = dict(docs=[Document(cnt="text%d" % i) for i in range(n)],
            row_ids=[name + str(i) for i in range(n)], labels=["A", "B"], gamma=gamma,
            model_probabilities={"old": old.copy(), "arb": arb.copy()}, model_decisions={})
    rules = (Rule((Doc("x"), ML("x", "old", "A", .7)), Label("x", "A")),
             Rule((Doc("x"), Label("x", "A")), Label("x", "B")))
    policy = dict(conflict_policy="unknown_arbitrate", gamma_wins=True,
                  arbitration=dict(alias="arb", default_threshold=.5, thresholds={}), fallback=None)
    args = dict(contexts=contexts, rules=rules, alias=alias,
        receiver_masks={name: np.ones(n, bool) for name in contexts},
        calibration_rows=np.ones(n, bool), calibration_truth=truth, policy=policy,
        cache_dir=tmp_path / "cache")
    return args, truth


def predictions(truth, good=True):
    p = np.where(truth, .995, .01).astype(np.float32) if good else np.full(truth.shape, .5, np.float32)
    return {name: p.copy() for name in ("calibration", "application")}


def checkpoint(tmp_path, name):
    path = tmp_path / (name + ".cp"); path.write_bytes(name.encode())
    return file_record(path)


def test_true_signed_rules_complete_closure_gamma_and_old_tables(tmp_path):
    args, truth = fixture(tmp_path)
    original = {name: (ctx["model_probabilities"]["old"].copy(), ctx["gamma"].neg.copy())
                for name, ctx in args["contexts"].items()}
    workflow = ResidualAuxWorkflow(**args)
    prefix = [r.to_dict() for r in args["rules"]]
    record = workflow.consider("trained1", checkpoint(tmp_path, "one"), predictions(truth), training_receipt={"updates": 64})
    assert record["accepted"] and record["actual_complete_calibration_quality"]["macro_f1"] == 1
    assert record["positive_thresholds"] and record["negative_thresholds"]
    assert [r.to_dict() for r in workflow.accepted_rules[:2]] == prefix
    assert np.array_equal(workflow.outputs["calibration"], truth)
    assert not workflow.outputs["application"][0, 0]  # Immutable original negative Gamma.
    assert workflow.old_closure_calls == 2 and record["training_receipt"]["updates"] == 64
    assert all(not isinstance(a, Label) for r in workflow.accepted_rules[2:] for a in r.body)
    for name, ctx in args["contexts"].items():
        assert np.array_equal(ctx["model_probabilities"]["old"], original[name][0])
        assert np.array_equal(ctx["gamma"].neg, original[name][1])


def test_rejected_candidate_has_current_endpoint_and_keeps_last_accepted(tmp_path):
    args, truth = fixture(tmp_path); workflow = ResidualAuxWorkflow(**args)
    workflow.consider("good", checkpoint(tmp_path, "good"), predictions(truth))
    retained = {name: value.copy() for name, value in workflow.outputs.items()}
    cp = copy.deepcopy(workflow.accepted_checkpoint)
    record = workflow.consider("bad", checkpoint(tmp_path, "bad"), predictions(truth, False))
    assert not record["accepted"] and record["retained_candidate"] == "good"
    assert workflow.accepted_checkpoint == cp and len(workflow.trajectory) == 2
    assert not np.array_equal(workflow.current_outputs["application"], retained["application"])
    assert all(np.array_equal(workflow.outputs[name], value) for name, value in retained.items())
    assert set(record["actual_complete_outputs"]) == {"calibration", "application"}


def test_complement_refreshes_with_each_probability_update(tmp_path):
    args, truth = fixture(tmp_path); workflow = ResidualAuxWorkflow(**args)
    a = workflow.candidate_context("calibration", np.full(truth.shape, .1, np.float32))
    b = workflow.candidate_context("calibration", np.full(truth.shape, .9, np.float32))
    name = workflow.names["complement"]
    assert np.allclose(a["model_probabilities"][name], .9)
    assert np.allclose(b["model_probabilities"][name], .1)
    assert name not in b["static_model_probabilities"]
    assert a["model_probabilities"]["old"] is b["model_probabilities"]["old"]


def test_actual_chase_gate_rejects_matrix_only_negative_improvement(tmp_path):
    args, truth = fixture(tmp_path)
    # The original arbitration disagrees with a good new teacher. It remains
    # immutable, so new negative rules cannot remove these unknown-pair outputs.
    for ctx in args["contexts"].values():
        ctx["model_probabilities"]["arb"][:] = .99
    args["rules"] = args["rules"][:1]
    workflow = ResidualAuxWorkflow(**args)
    p = predictions(truth)
    # Remove recall gain; only screened negative corrections are proposed.
    for values in p.values():
        values[7] = .01
        values[:, 1] = .01
    record = workflow.consider("proxy_only", checkpoint(tmp_path, "proxy"), p)
    assert record["negative_thresholds"] and not record["accepted"]
    assert workflow.accepted_candidate == "frozen_old_program"


@pytest.mark.parametrize("change", ["docs", "attributes", "decisions", "policy", "Gamma", "old_Pi"])
def test_internal_semantic_mutation_rejected_before_candidate_chase(tmp_path, change):
    args, truth = fixture(tmp_path); workflow = ResidualAuxWorkflow(**args)
    ctx = workflow.contexts["calibration"]
    if change == "docs": ctx["docs"][0].cnt = "changed"
    if change == "attributes": ctx["attribute_values"] = {"mtd.group": ["changed"] * 8}
    if change == "decisions": ctx["model_decisions"]["old"] = np.ones((8, 2), bool)
    if change == "policy": workflow.policy["arbitration"]["default_threshold"] = .8
    if change == "Gamma": ctx["gamma"] = RefGamma(np.ones((8, 2), bool), np.zeros((8, 2), bool))
    if change == "old_Pi": ctx["model_probabilities"]["old"] = np.zeros((8, 2), np.float32)
    with pytest.raises(ValueError, match="Frozen"):
        workflow.consider("changed", checkpoint(tmp_path, "changed"), predictions(truth))
    assert workflow.trajectory[-1]["status"] == "failed"


def test_external_source_mutation_does_not_change_private_frozen_program(tmp_path):
    args, truth = fixture(tmp_path); workflow = ResidualAuxWorkflow(**args)
    args["contexts"]["calibration"]["docs"][0].cnt = "caller changed"
    args["contexts"]["calibration"]["model_probabilities"]["old"][:] = 0
    assert workflow.consider("new", checkpoint(tmp_path, "new"), predictions(truth))["accepted"]


def test_dual_gate_requires_both_metrics_and_strict_improvement():
    incumbent = dict(macro_f1=.7, micro_f1=.8)
    assert not dual_gate(dict(macro_f1=.69, micro_f1=.9), incumbent)
    assert not dual_gate(dict(macro_f1=.8, micro_f1=.79), incumbent)
    assert not dual_gate(incumbent, incumbent)
    assert dual_gate(dict(macro_f1=.7, micro_f1=.81), incumbent)


def test_calibration_answers_cannot_be_gamma_roots_and_alias_collision(tmp_path):
    args, _ = fixture(tmp_path)
    args["contexts"]["calibration"]["gamma"] = RefGamma(np.ones((8, 2), bool), np.zeros((8, 2), bool))
    with pytest.raises(ValueError, match="cannot seed"):
        ResidualAuxWorkflow(**args)
    args, _ = fixture(tmp_path, alias="old")
    with pytest.raises(ValueError, match="collides"):
        ResidualAuxWorkflow(**args)


def test_unknown_teacher_zero_gradient_detach_and_full_cell_denominator():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[.3, float("nan")], [-.4, .8]], requires_grad=True)
    targets = torch.tensor([[1., 0.], [0., 0.]], requires_grad=True)
    observed = torch.tensor([[True, False], [True, False]])
    weights = torch.tensor([[.7, 0.], [.4, 0.]], requires_grad=True)
    loss = masked_teacher_bce(logits, targets, observed, weights)
    expected = (torch.nn.functional.softplus(-logits[0, 0]) * .7
                + torch.nn.functional.softplus(logits[1, 0]) * .4) / 4
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad[:, 1].eq(0).all() and targets.grad is weights.grad is None


def test_halt_policy_honors_original_gamma_protection(tmp_path):
    args, _ = fixture(tmp_path)
    args["policy"] = dict(conflict_policy="halt", gamma_wins=True, arbitration=None, fallback=None)
    workflow = ResidualAuxWorkflow(**args)
    assert not workflow.old_outputs["application"][0, 0]


@pytest.mark.parametrize("kind", ["shape", "dtype", "overlap"])
def test_invalid_original_gamma_rejected(kind, tmp_path):
    args, _ = fixture(tmp_path)
    gp, gn = np.zeros((8, 2), bool), np.zeros((8, 2), bool)
    if kind == "shape": gp = np.zeros((7, 2), bool)
    if kind == "dtype": gp = gp.astype(np.float32)
    if kind == "overlap": gp[0, 0] = gn[0, 0] = True
    args["contexts"]["application"]["gamma"] = SimpleNamespace(pos=gp, neg=gn)
    with pytest.raises(ValueError, match="Gamma"):
        ResidualAuxWorkflow(**args)


def test_strict_failed_candidate_is_recorded_and_cannot_replace_incumbent(tmp_path):
    args, truth = fixture(tmp_path)
    args["policy"] = dict(conflict_policy="halt", gamma_wins=True, arbitration=None, fallback=None)
    workflow = ResidualAuxWorkflow(**args)
    retained = workflow.outputs["calibration"].copy()
    with pytest.raises(ValueError, match="Chase failed"):
        workflow.consider("fails", checkpoint(tmp_path, "fails"), predictions(truth))
    assert workflow.trajectory[-1]["status"] == "failed" and not workflow.trajectory[-1]["accepted"]
    assert workflow.accepted_checkpoint is None and np.array_equal(workflow.outputs["calibration"], retained)


def test_frozen_fallback_remains_fixed_while_negative_rules_block_its_fill(tmp_path):
    args, truth = fixture(tmp_path); args["rules"] = ()
    args["policy"]["fallback"] = dict(alias="old", default_threshold=.7, thresholds={})
    workflow = ResidualAuxWorkflow(**args)
    record = workflow.consider("fallback_corrections", checkpoint(tmp_path, "fallback"), predictions(truth))
    assert record["accepted"] and np.array_equal(workflow.outputs["calibration"], truth)
    assert workflow.policy["fallback"] == args["policy"]["fallback"]
    assert np.array_equal(workflow.contexts["calibration"]["model_probabilities"]["old"],
                          args["contexts"]["calibration"]["model_probabilities"]["old"])


def test_old_endpoint_and_checkpoint_identity_corruptions_fail_closed(tmp_path):
    args, truth = fixture(tmp_path); workflow = ResidualAuxWorkflow(**args)
    cp = checkpoint(tmp_path, "bad_hex"); cp["sha256"] = "z" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        workflow.consider("bad_hex", cp, predictions(truth))
    workflow.old_outputs["calibration"] = np.zeros(truth.shape, bool)
    with pytest.raises(ValueError, match="old endpoint"):
        workflow.consider("bad_old", checkpoint(tmp_path, "old"), predictions(truth))


@pytest.mark.parametrize("proposal_policy", [None, "quantile41_rare_allowance_one"])
def test_portable_cli_uses_native_axes_and_exports_every_current_checkpoint(tmp_path, proposal_policy):
    args, truth = fixture(tmp_path)
    protocol = dict(schema="loris_residual_aux_protocol_v1", contexts={}, program=None,
                    alias="new_aux", policy=args["policy"], calibration_answers=None, candidates=[])
    if proposal_policy is not None:
        protocol["proposal_policy"] = proposal_policy
    for name, context in args["contexts"].items():
        path = tmp_path / (name + ".pkl")
        with path.open("wb") as stream: pickle.dump(context, stream)
        protocol["contexts"][name] = dict(context=file_record(path), receiver_row_ids=context["row_ids"])
    program = tmp_path / "program.json"; program.write_text(json.dumps([r.to_dict() for r in args["rules"]]))
    protocol["program"] = file_record(program)
    answers = tmp_path / "calibration.npz"
    np.savez(answers, row_ids=args["contexts"]["calibration"]["row_ids"], labels=["A", "B"],
             targets=truth, observed=np.ones_like(truth))
    protocol["calibration_answers"] = file_record(answers)
    for name, good in (("good", True), ("bad", False)):
        candidate = dict(candidate_id=name, checkpoint=checkpoint(tmp_path, name), probabilities={})
        for context_name, values in predictions(truth, good).items():
            packet = tmp_path / (name + context_name + ".npz")
            np.savez(packet, probabilities=values, row_ids=args["contexts"][context_name]["row_ids"], labels=["A", "B"])
            candidate["probabilities"][context_name] = file_record(packet)
        protocol["candidates"].append(candidate)
    path = tmp_path / "protocol.json"; path.write_text(json.dumps(protocol))
    assert cli.main(["residual-aux", "--protocol", str(path), "--output", str(tmp_path / "out")]) == 0
    complete = json.loads((tmp_path / "out/COMPLETE.json").read_text())
    assert complete["accepted_candidate"] == "good" and not complete["B_or_test_used_for_selection"]
    assert complete["threshold_proposal_policy"]["name"] == (proposal_policy or "coarse")
    expected = complete["threshold_proposal_policy"]
    assert all(record["threshold_proposal_policy"] == {key: expected[key] for key in record["threshold_proposal_policy"]}
               for record in complete["trajectory"])
    assert (tmp_path / "out/initial_application_output.npz").is_file()
    assert (tmp_path / "out/candidate_00001_PROGRAM.json").is_file()
    with np.load(tmp_path / "out/candidate_00001_application_output.npz", allow_pickle=False) as current:
        with np.load(tmp_path / "out/application_output.npz", allow_pickle=False) as retained:
            assert not np.array_equal(current["prediction"], retained["prediction"])
            with np.load(tmp_path / "out/retained_00001_application_output.npz", allow_pickle=False) as step:
                assert np.array_equal(step["prediction"], retained["prediction"])
    with pytest.raises(FileExistsError):
        cli.main(["residual-aux", "--protocol", str(path), "--output", str(tmp_path / "out")])


def _scalar_threshold_oracle(old, probabilities, truth, *, quantile=False,
                             minimum_removals=5, maximum_true_loss=.10, rare_loss=0):
    """Independent finite-grid enumeration; no vector count/proposal helper."""
    from loris.labeling.residual_aux import QUANTILE_ANCHORS, POSITIVE_GRID, NEGATIVE_GRID
    p = probabilities.astype(np.float64)
    def counts(values, y):
        return np.asarray([sum(bool(a) and bool(b) for a, b in zip(values, y)),
                           sum(bool(a) and not bool(b) for a, b in zip(values, y)),
                           sum(not bool(a) and bool(b) for a, b in zip(values, y))], np.int64)
    def f1(c):
        d = 2 * c[0] + c[1] + c[2]
        return float(2 * c[0] / d) if d else 0.
    def micro(c):
        return f1(c.sum(1))
    def grid(values):
        values = np.unique(values)
        edges = (values[:-1] + values[1:]) * .5 if len(values) > 1 else values
        if len(edges):
            try: q = np.quantile(edges, np.linspace(0, 1, 41), method="nearest")
            except TypeError: q = np.quantile(edges, np.linspace(0, 1, 41), interpolation="nearest")
        else: q = []
        return np.unique(np.r_[q, QUANTILE_ANCHORS])
    current = np.stack([counts(old[:, j], truth[:, j]) for j in range(old.shape[1])], axis=1)
    choices = []
    for j in range(old.shape[1]):
        positives = grid(p[~old[:, j], j]) if quantile else POSITIVE_GRID
        negatives = grid(p[old[:, j], j]) if quantile else np.asarray(NEGATIVE_GRID[1:])
        negatives = [None] + [float(v) for v in negatives if not quantile or v < 1.]
        options = []; before = f1(current[:, j]); ntrue = int(truth[:, j].sum())
        limit = max(rare_loss, int(np.floor(ntrue * maximum_true_loss + 1e-12)))
        for pos in positives:
            for neg in negatives:
                if neg is not None and neg >= pos: continue
                remove = old[:, j] & (p[:, j] <= neg) if neg is not None else np.zeros(len(old), bool)
                loss = int((remove & truth[:, j]).sum())
                if neg is not None and (remove.sum() < minimum_removals or not ntrue
                        or (loss > limit if quantile else loss / ntrue > maximum_true_loss)): continue
                pred = (old[:, j] | (~old[:, j] & (p[:, j] >= pos))) & ~remove
                c = counts(pred, truth[:, j]); score = f1(c)
                if score > before + 1e-12:
                    options.append((score, float(pos), -1. if neg is None else neg, c))
        options.sort(key=lambda v: (v[0], v[1], -v[2]), reverse=True)
        if options: choices.append((options[0][0] - before, j, options))
    choices.sort(key=lambda v: (-v[0], v[1])); positive, negative = {}, {}
    for _, j, options in choices:
        for _, pc, nc, c in options:
            proposal = current.copy(); proposal[:, j] = c
            if micro(proposal) + 1e-12 < micro(current): continue
            current = proposal; positive[j] = pc
            if nc >= 0: negative[j] = nc
            break
    return positive, negative


@pytest.mark.parametrize("policy", ["coarse", "quantile41_rare_allowance_one"])
def test_vectorized_thresholds_equal_independent_scalar_oracle(policy):
    from loris.labeling.residual_aux import fit_thresholds
    rng = np.random.RandomState(619)
    for _ in range(14):
        old = rng.rand(19, 4) < .45; truth = rng.rand(19, 4) < .25
        # Deliberate equal scores, zero/one endpoints and float32 bin boundaries.
        p = rng.choice(np.asarray([0., .005, .03, .049999, .05, .1, .17, .5, .8, .99, 1.], np.float32), (19, 4))
        quantile = policy != "coarse"
        expected = _scalar_threshold_oracle(old, p, truth, quantile=quantile,
            minimum_removals=3 if quantile else 5, rare_loss=1 if quantile else 0)
        assert fit_thresholds(old, p, truth, proposal_policy=policy) == expected


def test_quantile_midpoint_is_not_float32_rounded_and_rare_allowance_is_explicit():
    from loris.labeling.residual_aux import _quantile_grid, fit_thresholds
    left = np.float32(.2); right = np.nextafter(left, np.float32(1.))
    midpoint = (float(left) + float(right)) * .5
    assert midpoint in _quantile_grid(np.asarray([left, right], np.float32))
    assert midpoint != float(np.float32(midpoint))
    # Sacrificing one rare TP can improve F1 if it removes enough false positives.
    old = np.ones((8, 1), bool); truth = np.zeros_like(old); truth[[0, 7]] = True
    p = np.asarray([[.01]] * 7 + [[.99]], np.float32)
    coarse = fit_thresholds(old, p, truth)
    latest = fit_thresholds(old, p, truth, proposal_policy="quantile41_rare_allowance_one")
    assert not coarse[1] and latest[1]
    assert latest == _scalar_threshold_oracle(old, p, truth, quantile=True, minimum_removals=3, rare_loss=1)


def test_named_quantile_policy_is_frozen_and_actual_gate_keeps_old_program(tmp_path):
    args, truth = fixture(tmp_path)
    args["proposal_policy"] = "quantile41_rare_allowance_one"
    workflow = ResidualAuxWorkflow(**args)
    assert workflow.proposal_policy == dict(name="quantile41_rare_allowance_one", minimum_removals=3,
        maximum_true_loss=.1, rare_loss_allowance=1, quantile_points=41)
    record = workflow.consider("quantile", checkpoint(tmp_path, "quantile"), predictions(truth))
    assert record["accepted"] and record["threshold_proposal_policy"] == workflow.proposal_policy
    assert [r.to_dict() for r in workflow.accepted_rules[:len(args["rules"])]] == [r.to_dict() for r in args["rules"]]
    workflow.proposal_policy["minimum_removals"] = 4
    with pytest.raises(ValueError, match="Frozen"):
        workflow.consider("mutated_quantile", checkpoint(tmp_path, "mutated_quantile"), predictions(truth))


def test_negative_midpoint_native_rule_exactly_matches_raw_probability_cut(tmp_path):
    from loris.labeling.residual_aux import residual_rules
    from loris._engine.typed_runtime import IndexedChaseRunner
    args, _ = fixture(tmp_path)
    args["rules"] = ()
    workflow = ResidualAuxWorkflow(**args)
    left = np.float32(.05); right = np.nextafter(left, np.float32(1.))
    cut = (float(left) + float(right)) * .5
    p = np.full((8, 2), right, np.float32); p[:4, 0] = left
    # Isolate the threshold predicate; guard/scope are explicit frozen tables.
    context = workflow.candidate_context("calibration", p)
    names = workflow.names
    context["static_model_probabilities"][names["old_present"]] = np.ones((8, 2), np.float64)
    rules = residual_rules((), workflow.labels, names, {}, {"A": cut})
    runner = IndexedChaseRunner(context, tmp_path / "native_midpoint")
    try:
        runner.precompute(rules); closure = runner(rules, context["gamma"])
        assert closure.status == "fixpoint"
        assert np.array_equal(closure.negative_facts[:, 0], p[:, 0].astype(np.float64) <= cut)
        assert not closure.negative_facts[:, 1].any()
        assert context["model_probabilities"][names["complement"]].dtype == np.float64
    finally:
        runner.close()


def test_unrepresentable_float64_complement_boundary_fails_closed(tmp_path):
    args, _ = fixture(tmp_path)
    truth = np.zeros((8, 2), bool); truth[[4, 5, 6]] = True
    args["calibration_truth"] = truth
    args["proposal_policy"] = "quantile41_rare_allowance_one"
    for ctx in args["contexts"].values():
        ctx["model_probabilities"]["old"][:] = .9
        ctx["model_probabilities"]["arb"] = np.where(truth, .9, .1)
    workflow = ResidualAuxWorkflow(**args)
    p = np.repeat(np.asarray([0., 1e-17, 2e-17, 3e-17, 4e-17, .0001, .99, 1.])[:, None], 2, axis=1)
    # 1-P coalesces tiny but distinct values at1. The fitting midpoint must
    # never be silently compiled to a broader native negative predicate.
    retained = workflow.outputs["calibration"].copy()
    with pytest.raises(ValueError, match="exact probability cut"):
        workflow.consider("lossy_boundary", checkpoint(tmp_path, "lossy_boundary"),
                          {"calibration": p, "application": p})
    assert not workflow.trajectory[-1]["accepted"]
    assert workflow.accepted_checkpoint is None and np.array_equal(workflow.outputs["calibration"], retained)


@pytest.mark.parametrize("bad", [{"name": "unknown"}, {"name": "quantile41_rare_allowance_one", "quantile_points": 40},
    {"name": "quantile41_rare_allowance_one", "rare_loss_allowance": 0}, {"name": "coarse", "minimum_removals": True},
    {"name": "coarse", "maximum_true_loss": float("nan")}])
def test_invalid_proposal_configuration_fails_before_closure(tmp_path, bad):
    args, _ = fixture(tmp_path); args["proposal_policy"] = bad
    with pytest.raises(ValueError): ResidualAuxWorkflow(**args)
