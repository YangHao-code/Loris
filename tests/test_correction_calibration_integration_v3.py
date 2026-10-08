"""Independent calibration uses each candidate's refreshed complement table."""
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from loris._engine.typed_session import array_identity
from loris.discovery.manifest import bound, file_record
from loris.labeling import cli
from loris.labeling.aux_extension import prepare, probability_views
from loris.labeling.reliability import calibration_texts, main_calibrate, refresh_current
from tests.test_current_model_reliability import prediction_record
from tests.test_fit_reliability import calibration_fixture


def test_two_candidate_calibrations_refresh_complement_but_freeze_old_models(tmp_path):
    args, old_calibration, _ = calibration_fixture(tmp_path)
    prepared = prepare(args.manifest, tmp_path / "v3", protocol_path=args.protocol)
    mp = Path(prepared["labeling_manifest"]["path"])
    m = json.loads(mp.read_text())
    alias = m["dynamic"]["alias"]
    complement = "v3_calibration_complement"
    m["aux_extension"].update(default_threshold=.9, correction=dict(
        complement_alias=complement, positive_active=[], negative_rule_indices=[1],
        negative_default_threshold=.2, negative_thresholds={}, min_gap=.05))
    m["loop"].update(conflict_policy="unknown_arbitrate",
                      unknown_semantics="retain_consequences",
                      gamma_conflict_policy="validated_gamma_wins",
                      conflict_update_policy="fresh_from_gamma")
    m["arbitration"] = dict(alias="aux", default_threshold=.5, thresholds={})
    mp.write_text(json.dumps(m))
    initial_output = tmp_path / "initial_calibration"
    assert main_calibrate(SimpleNamespace(manifest=str(mp),
        protocol=prepared["reliability_protocol"]["path"], output=str(initial_output))) == 0
    m, target, rules, artifact, _ = cli.load(mp, weighting="provenance",
        source_reliability=initial_output / "RELIABILITY.json")
    original_source = array_identity(old_calibration["model_probabilities"]["aux"])
    initial_complement = array_identity(
        np.subtract(np.float64(1), old_calibration["model_probabilities"]["aux"], dtype=np.float64))
    for i, value in enumerate((.99, .01)):
        checkpoint = tmp_path / ("candidate%d.pt" % i)
        checkpoint.write_text("candidate-%d" % i)
        m["dynamic"]["checkpoint"] = file_record(checkpoint)
        application = np.full((len(target["row_ids"]), len(target["labels"])), .1, np.float32)
        application[:, 0] = value
        target["model_probabilities"].update(probability_views(m, application))
        prediction_folder = tmp_path / ("prediction%d" % i)
        prediction_folder.mkdir()
        prediction = prediction_record(prediction_folder, calibration_texts(artifact),
            target["labels"], m["dynamic"]["checkpoint"], value=value)
        artifact, _ = refresh_current(artifact, target, rules, m, prediction,
            tmp_path / ("refresh%d" % i))
        # The original Aux-dependent rule remains on its initial .1 table.
        assert artifact["rule_scores"][0]["support"] == 0
        # Corrective negative fires in the 18 red contexts only when A is low.
        assert artifact["rule_scores"][-1]["support"] == (18 if value < .2 else 0)
        protocol_path = bound(artifact["protocol"], Path("."))
        protocol = json.loads(protocol_path.read_text())
        snapshot_path = bound(protocol["context"], protocol_path.parent)
        with snapshot_path.open("rb") as stream:
            snapshot = pickle.load(stream)
        current = snapshot["model_probabilities"][alias]
        np.testing.assert_array_equal(snapshot["model_probabilities"][complement],
            np.subtract(np.float64(1), current, dtype=np.float64))
        assert artifact["calibration_binding"]["model_tables"][complement] != initial_complement
        assert artifact["calibration_binding"]["model_tables"]["aux"] == original_source
        assert artifact["calibration_binding"]["static_model_tables"]["aux"] == original_source
        assert artifact["calibration_binding"]["static_model_tables"][complement] == initial_complement
        assert artifact["binding"]["conflict_update_policy"] == "fresh_from_gamma"
