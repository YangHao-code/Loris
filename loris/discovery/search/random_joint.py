"""Score-independent joint proposals for the paper's LORISrandRD control.

The PDF does not specify the random distribution. We retain the main
sampler's full-support, family-balanced sparse Bernoulli prior for every
body indicator, and draw heads/active parameters with Optuna RandomSampler.
No fit witness, initialized candidate, validation score, or selected prefix
can affect these draws. Invalid configurations still consume the budget.
"""

from pathlib import Path

from optuna.samplers import RandomSampler

from loris._engine.rlearner import digest, immutable_json
from loris._engine.sparse_bo import JointBinaryTPESampler
from .sampler import source_record

SCHEMA = "loris_random_joint_full_support_v1"


class RandomJointSampler(JointBinaryTPESampler):
    def __init__(self, include_keys, *, seed, feature_families=None):
        super().__init__(include_keys, seed=seed, feature_families=feature_families)
        prior = dict(self.protocol)
        self.delegate = RandomSampler(seed=(seed + 100003) % 2**32)
        self.protocol = dict(
            schema=SCHEMA,
            seed=seed,
            source=source_record(__file__),
            include_keys=list(self.include_keys),
            dimension=self.dimension,
            body_distribution="independent family-balanced Bernoulli; same sparse prior as main TPE",
            feature_on_probabilities=self.feature_probabilities,
            family_prior=prior["feature_family_prior"],
            head_and_active_parameters="Optuna RandomSampler over unchanged declared domains",
            fit_witness_used=False,
            initial_configurations=[],
            score_guided=False,
            observations="recorded for exact recovery and audit only; never read by proposal policy",
            hard_body_size_cap=None,
            invalid_candidates_consume_budget=True,
            validation_score_used_for_admission=False,
            admission="proposal order; distinct legal rules whose cumulative closure is consistent",
            final_rule_count_matched_to_main=False,
        )

    def sample_relative(self, study, trial, search_space):
        if set(search_space) != set(self.include_keys):
            raise ValueError("random joint proposals require the unchanged complete feature axis")
        chosen = self.rng.random_sample(self.dimension) < self.prior_probabilities
        self.last_diagnostic = dict(
            schema=SCHEMA,
            trial=trial.number,
            branch="random_joint",
            score_guided=False,
            fit_witness_used=False,
        )
        return {key: int(value) for key, value in zip(self.include_keys, chosen)}

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self.delegate.sample_independent(study, trial, param_name, param_distribution)


class RandomJointSpace:
    """Same grammar/axes, explicitly disabling all fit and repaired startup seeds."""

    def __init__(self, base, directory):
        self._base = base
        self._directory = Path(directory)
        receipt = dict(
            schema=SCHEMA,
            source=source_record(__file__),
            space_manifest_id=digest(base.manifest()),
            seed_count=0,
            fit_witness_used=False,
            initial_configurations=[],
            reason="random joint proposals start at trial zero; no fit seed ranking",
        )
        immutable_json(self._directory / "COMPLETE.json", receipt)
        self._receipt = source_record(self._directory / "COMPLETE.json")

    def __getattr__(self, name):
        return getattr(self._base, name)

    def manifest(self):
        if source_record(self._directory / "COMPLETE.json") != self._receipt:
            raise ValueError("random initialization receipt changed")
        return dict(
            self._base.manifest(),
            initialization_override=dict(
                schema=SCHEMA,
                receipt=self._receipt,
                seed_count=0,
                same_candidate_space=True,
                fit_witness_used=False,
            ),
        )

    def iter_initial_configurations(self, count, *, seed, **kwargs):
        return iter(())

    def initial_configurations(self, count, *, seed, **kwargs):
        return []
