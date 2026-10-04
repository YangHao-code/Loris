"""Full-support, joint binary TPE proposals for large paper configuration spaces.

This is an explicitly chosen BO surrogate/prior, NOT an equivalence speedup of
Optuna's default per-bit TPE. Independent Beta-Bernoulli densities model good
and remaining observations. Acquisition maximizes log l(z)-log g(z) over full
sampled vectors, rather than assembling each bit's independently best draw.
Every declared feature is present, with positive ON and OFF probability; there
is no feature dropout, body-length cap, or candidate rule bank. Heads and active
continuous/categorical parameters use Optuna TPE. This finite acquisition
search does not claim exhaustive optimization or guaranteed effective rules.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import operator
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution
from optuna.samplers import BaseSampler, TPESampler
from optuna.trial import TrialState


SCHEMA = 'paper36_joint_binary_beta_bernoulli_tpe_v1'

# Execution caches are deliberately outside sampler vars. Dense pickles and
# compact checkpoints retain exactly the original algorithmic state. Weak keys
# also let finished cluster samplers and their full axes be released normally.
_RELATIVE_SEARCH_CACHE = WeakKeyDictionary()
_PARAMETER_CODEC_CACHE = WeakKeyDictionary()
_PACKED_ON_INDEX_CACHE = WeakKeyDictionary()
_HISTORY_TOTAL_CACHE = WeakKeyDictionary()


def _immutable_packed(packed):
    from loris._engine.hashing import immutable_buffer
    return (type(packed) is np.ndarray and packed.dtype == np.uint8 and packed.ndim == 1
            and immutable_buffer(packed))


class JointBinaryTPESampler(BaseSampler):
    def __init__(self, include_keys, *, seed, expected_body_size=3., prior_strength=1.,
                 startup_trials=30, acquisition_candidates=64, good_fraction=.1,
                 feature_families=None):
        self.include_keys = tuple(include_keys)
        if not self.include_keys or len(set(self.include_keys)) != len(self.include_keys):
            raise ValueError('all unique binary feature keys must be declared')
        self.dimension = len(self.include_keys)
        if not np.isfinite(expected_body_size) or expected_body_size <= 0:
            raise ValueError('positive expected body size required')
        if not np.isfinite(prior_strength) or prior_strength <= 0:
            raise ValueError('strictly positive prior strength required')
        if (type(seed) is not int or type(startup_trials) is not int or startup_trials < 1
                or type(acquisition_candidates) is not int or acquisition_candidates < 1
                or not 0 < good_fraction < 1):
            raise ValueError('invalid BO seed, startup, acquisition or quantile budget')
        # A small space cannot have expected size 3 while preserving OFF support.
        # The explicit maximum .5 is a probability prior, never a length cap.
        self.prior_probability = min(.5, expected_body_size / self.dimension)
        if feature_families is None:
            self.prior_probabilities = np.full(self.dimension, self.prior_probability)
            family_descriptor = None
        else:
            if (set(feature_families) != set(self.include_keys)
                    or any(not isinstance(f, str) or not f for f in feature_families.values())):
                raise ValueError('each declared feature needs one nonempty family')
            families = sorted(set(feature_families.values()))
            counts = {f: sum(value == f for value in feature_families.values()) for f in families}
            self.prior_probabilities = np.array([min(.5, expected_body_size / len(families) /
                counts[feature_families[key]]) for key in self.include_keys])
            family_descriptor = {'allocation': 'equal expected length per represented family',
                'feature_families': dict(feature_families), 'counts': counts}
        self.prior_strength = float(prior_strength)
        self.startup_trials, self.acquisition_candidates = startup_trials, acquisition_candidates
        self.good_fraction = float(good_fraction)
        self.rng = np.random.RandomState(seed)
        self.delegate = TPESampler(seed=seed + 100003, n_startup_trials=startup_trials)
        self.observations = {}
        self.last_diagnostic = None
        self.protocol = {'schema': SCHEMA, 'optuna_version': optuna.__version__, 'seed': seed,
            'include_keys': list(self.include_keys), 'dimension': self.dimension,
            'prior_expected_size_requested': float(expected_body_size),
            'uniform_prior_on_probability': self.prior_probability if feature_families is None else None,
            'prior_strength': self.prior_strength,
            'feature_family_prior': family_descriptor,
            'actual_prior_expected_size': float(self.prior_probabilities.sum()),
            'initialization': 'same Bernoulli probabilities; config initializer adds an available guard for an unguarded ML; no selected feature removed',
            'startup_trials': startup_trials, 'acquisition_candidates': acquisition_candidates,
            'good_fraction': self.good_fraction, 'good_count': 'ceil(fraction * n), at least one, at most n-1',
            'objective_direction': 'maximize', 'joint_acquisition': 'factorized log l(z) - log g(z)',
            'all_features_present': True, 'hard_body_size_cap': None, 'feature_dropout': False,
            'historical_rule_inputs': False, 'equivalent_to_default_optuna_TPE': False,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    @property
    def feature_probabilities(self):
        return {key: float(p) for key, p in zip(self.include_keys, self.prior_probabilities)}

    def infer_relative_search_space(self, study, trial):
        cached = _RELATIVE_SEARCH_CACHE.get(self)
        if cached is None or cached[0] != self.include_keys:
            distributions = {key: CategoricalDistribution([0, 1]) for key in self.include_keys}
            objects = tuple(distributions.values())
            _RELATIVE_SEARCH_CACHE[self] = (self.include_keys, distributions, objects,
                                            tuple(d.choices for d in objects))
        else:
            distributions, objects, choices = cached[1], cached[2], cached[3]
            # Returned distributions are ordinary Optuna objects. Detect an
            # external mutation instead of silently reusing a changed domain.
            # Fast exact test: same objects, same class and the very same
            # (immutable, validated at creation) choices tuples; otherwise the
            # complete original field-by-field check decides.
            unchanged = (all(map(operator.is_, distributions.values(), objects))
                         and all(map(operator.is_, map(type, objects), itertools.repeat(CategoricalDistribution)))
                         and all(map(operator.is_, map(operator.attrgetter('choices'), objects), choices)))
            if not unchanged and any(type(d) is not CategoricalDistribution or type(d.choices) is not tuple
                   or len(d.choices) != 2 or type(d.choices[0]) is not int
                   or type(d.choices[1]) is not int or d.choices != (0, 1)
                   for d in distributions.values()):
                raise ValueError('cached full binary distribution was mutated')
        # Callers may alter their mapping without deleting a cached feature.
        # The distribution objects are shared, never constructed twice here.
        return dict(distributions)

    def _record(self, trial, values):
        if len(values) != 1 or not np.isfinite(values[0]):
            raise ValueError('single finite standalone score required')
        from loris._engine.compact_storage import PARAM_ATTR, ParameterCodec
        compact = trial.user_attrs.get(PARAM_ATTR)
        if compact is not None:
            cached = _PARAMETER_CODEC_CACHE.get(self)
            if (cached is None or cached[0] != self.include_keys
                    or cached[1] != compact['space_id']):
                codec = ParameterCodec(self.include_keys, compact['space_id'])
                _PARAMETER_CODEC_CACHE[self] = (self.include_keys, compact['space_id'], codec)
            else:
                codec = cached[2]
            codec.frozen_parameters(trial)
            packed = codec.packed_mask(compact)
        else:
            if any(key not in trial.params or trial.params[key] not in (0, 1) for key in self.include_keys):
                raise ValueError('every binary indicator must be recorded for every completed trial')
            mask = np.fromiter((trial.params[key] for key in self.include_keys), dtype=np.uint8,
                               count=self.dimension)
            packed = np.packbits(mask)
        # Permanently read-only bytes-backed row: equal values, and later
        # history sums can reuse it without re-authenticating its bytes.
        packed = np.frombuffer(np.ascontiguousarray(packed).tobytes(), np.uint8)
        value = (float(values[0]), packed)
        if trial.number in self.observations:
            old = self.observations[trial.number]
            if old[0] != value[0] or not np.array_equal(old[1], packed):
                raise ValueError('sampler observation differs from completed BO trial')
        else:
            self.observations[trial.number] = value

    def _history_axis(self, study):
        for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
            if trial.number not in self.observations:
                self._record(trial, trial.values)
        ordered = sorted(self.observations)
        return ordered, np.array([self.observations[i][0] for i in ordered]), np.array(ordered)

    def _history(self, study):
        # Public compatibility/debugging API retains the complete dense state.
        ordered, scores, numbers = self._history_axis(study)
        if not ordered:
            return np.empty((0, self.dimension), np.uint8), np.array([]), np.array([], int)
        masks = np.array([np.unpackbits(self.observations[i][1])[:self.dimension] for i in ordered])
        return masks, scores, numbers

    def _packed_history_representation(self, number):
        packed = self.observations[number][1]
        if (type(packed) is not np.ndarray or packed.dtype != np.uint8 or packed.ndim != 1
                or packed.shape != ((self.dimension + 7) // 8,)):
            raise ValueError('packed sampler history differs from the complete binary axis')
        # The observations remain ordinary mutable arrays for compatibility.
        # Authenticate every byte on every use; in-place changes cannot reuse
        # stale ON indices merely because an ndarray keeps the same identity.
        cache = _PACKED_ON_INDEX_CACHE.setdefault(self, {})
        saved = cache.get(number)
        if (saved is not None and len(saved) > 3 and saved[3] is packed
                and saved[0] == self.dimension and _immutable_packed(packed)):
            # The very same permanently read-only bytes-backed row object.
            return saved[2], None if saved[2] is not None else np.unpackbits(packed)[:self.dimension]
        identity = hashlib.sha256(packed.tobytes()).digest()
        if saved is None or saved[0] != self.dimension or saved[1] != identity:
            mask = np.unpackbits(packed)[:self.dimension]
            # A fully ON history is legal. Do not turn its packed storage into
            # a permanent 8-byte index for every bit. The choice below affects
            # only exact storage/counting, never any candidate or probability.
            sparse_enough = int(mask.sum(dtype=np.uint64)) * np.dtype(np.intp).itemsize <= packed.nbytes
            indices = np.flatnonzero(mask) if sparse_enough else None
            if indices is not None:
                indices = np.frombuffer(indices.tobytes(), dtype=indices.dtype)
            cache[number] = (self.dimension, identity, indices, packed)
            return indices, None if indices is not None else mask
        else:
            indices = saved[2]
        return indices, None if indices is not None else np.unpackbits(packed)[:self.dimension]

    def _packed_on_indices(self, number):
        indices, mask = self._packed_history_representation(number)
        if indices is None:
            indices = np.flatnonzero(mask)
            indices = np.frombuffer(indices.tobytes(), dtype=indices.dtype)
        return indices

    def _history_counts(self, ordered, good_numbers=()):
        """Exactly dense uint8 sums/any, without materializing history x D.

        uint8.sum(axis=0) uses uint64. Integer additions have the same exact
        result in any row order, and all following floating-point operations
        stay in sample_relative unchanged. Every packed row is authenticated.

        remaining = total - good and seen = total > 0 are exact identities of
        these unsigned integer sums. The total over all rows is maintained
        incrementally; it is rebuilt from scratch unless every row is the very
        same immutable, previously authenticated observation object.
        """
        total = self._history_total(ordered)
        good = np.zeros(self.dimension, np.uint64)
        for number in good_numbers:
            indices, mask = self._packed_history_representation(number)
            if indices is not None:
                np.add.at(good, indices, np.uint64(1))
            else:
                good += mask
        remaining = total - good
        seen = total > 0
        return good, remaining, seen

    def _history_total(self, ordered):
        """Exact sum of all ordered rows.

        Rows whose packed bytes are permanently immutable are accumulated once
        (keyed by the identity of their observation tuple); any mutable row is
        authenticated and added afresh on every call, as before.
        """
        cache = _HISTORY_TOTAL_CACHE.get(self)
        wanted = set(ordered)
        if (cache is None or cache['dimension'] != self.dimension
                or not cache['members'].keys() <= wanted
                or not all(self.observations.get(n) is obj for n, obj in cache['members'].items())):
            cache = {'dimension': self.dimension, 'members': {},
                     'total': np.zeros(self.dimension, np.uint64)}
            _HISTORY_TOTAL_CACHE[self] = cache
        members, cached_total = cache['members'], cache['total']
        extra = np.zeros(self.dimension, np.uint64)
        for number in ordered:
            if number in members:
                continue
            observation = self.observations[number]
            immutable = _immutable_packed(observation[1])
            target = cached_total if immutable else extra
            indices, mask = self._packed_history_representation(number)
            if indices is not None:
                np.add.at(target, indices, np.uint64(1))
            else:
                target += mask
            if immutable:
                members[number] = observation
        return cached_total + extra

    def sample_relative(self, study, trial, search_space):
        if set(search_space) != set(self.include_keys):
            raise ValueError('binary acquisition may not silently drop a declared feature')
        if study.direction != optuna.study.StudyDirection.MAXIMIZE:
            raise ValueError('standalone BO must maximize accuracy')
        ordered, scores, numbers = self._history_axis(study)
        count = len(scores)
        if count < self.startup_trials:
            chosen = self.rng.random_sample(self.dimension) < self.prior_probabilities
            diagnostic = {'phase': 'sparse_prior_startup', 'history': count}
            _, _, seen = self._history_counts(ordered)
        else:
            ranking = np.lexsort((numbers, -scores))
            n_good = min(count - 1, max(1, int(math.ceil(self.good_fraction * count))))
            good_sum, remaining_sum, seen = self._history_counts(ordered, numbers[ranking[:n_good]])
            alpha = self.prior_strength * self.prior_probabilities
            p_good = (good_sum + alpha) / (n_good + self.prior_strength)
            p_rest = (remaining_sum + alpha) / (count - n_good + self.prior_strength)
            if not ((p_good > 0).all() and (p_good < 1).all() and (p_rest > 0).all() and (p_rest < 1).all()):
                raise ValueError('finite positive probability support was lost')
            draws = self.rng.random_sample((self.acquisition_candidates, self.dimension)) < p_good
            on_ratio = np.log(p_good) - np.log(p_rest)
            off_ratio = np.log1p(-p_good) - np.log1p(-p_rest)
            acquisition = draws @ (on_ratio - off_ratio) + off_ratio.sum()
            selected = int(np.argmax(acquisition))
            chosen = draws[selected]
            diagnostic = {'phase': 'joint_binary_TPE', 'history': count, 'good_count': n_good,
                'acquisition_candidates': self.acquisition_candidates,
                'selected_acquisition': float(acquisition[selected]),
                'min_on_probability': float(p_good.min()), 'min_off_probability': float((1-p_good).min()),
                'mean_draw_length': float(draws.sum(axis=1).mean())}
        diagnostic.update({'trial': trial.number, 'selected_body_indicators': int(chosen.sum()),
                           'previously_unseen_selected': int(np.sum(chosen & ~seen))})
        self.last_diagnostic = diagnostic
        # Same mapping as {key: int(value) ...}: int64 tolist yields Python ints.
        return dict(zip(self.include_keys, np.asarray(chosen, bool).astype(np.int64).tolist()))

    def sample_independent(self, study, trial, param_name, param_distribution):
        if param_name in self.include_keys:
            raise ValueError('binary indicator escaped the complete joint proposal')
        return self.delegate.sample_independent(study, trial, param_name, param_distribution)

    def before_trial(self, study, trial):
        self.delegate.before_trial(study, trial)

    def after_trial(self, study, trial, state, values):
        self.delegate.after_trial(study, trial, state, values)
        if state == TrialState.COMPLETE:
            self._record(trial, values)

    def reseed_rng(self):
        # Concurrent reseeding would change the explicitly single-seed protocol.
        raise RuntimeError('joint binary discovery is a deterministic sequential study')
