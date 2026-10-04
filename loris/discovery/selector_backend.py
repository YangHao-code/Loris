"""Selector-only input boundary; no online labeling or feedback imports.

The numerical Top-K selector is retained in selection.py. This maintained
wrapper binds its own bytes separately so a changed launcher cannot silently
resume a run created by a different execution protocol.
"""

import json
from pathlib import Path

import numpy as np

from .manifest import bound, file_record


FIELDS = {
    "arrays",
    "row_ids",
    "aliases",
    "labels",
    "input_binding",
    "protocol",
    "model_interpretation",
}
EXTENSION_FIELDS = {
    "experimental_extension",
    "additional_proposal",
    "additional_admission_extension",
    "additional_execution_extension",
}


def validate_selector_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("selector JSON object manifest required")
    extensions = sorted(set(manifest) & EXTENSION_FIELDS)
    if extensions:
        raise ValueError(
            "select-models cannot execute experimental extension fields: " + ", ".join(extensions)
        )
    unknown = set(manifest) - FIELDS - {"experiment_declaration"}
    if unknown:
        raise ValueError("unknown select-models manifest fields: " + ", ".join(sorted(unknown)))
    missing = FIELDS - {"model_interpretation"} - set(manifest)
    if missing:
        raise ValueError("missing select-models manifest fields: " + ", ".join(sorted(missing)))


def run_selector_manifest(manifest_path, output, *, resume=False):
    output = Path(output).resolve()
    if resume:
        if not output.is_dir():
            raise ValueError("resume requires an existing model-selector run")
        protocol = json.loads((output / "PROTOCOL.json").read_text())
        if protocol.get("schema") != "paper36_aligned_joint_vote_selection_v1":
            raise ValueError("resume refuses historical or unrelated experiment directories")
    elif output.exists():
        raise FileExistsError("selector requires a new output directory: " + str(output))
    path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(path.read_text())
    validate_selector_manifest(manifest)
    from loris._learning.selection import train_model_selection, ModelSelectionProtocol

    with np.load(bound(manifest["arrays"], path.parent), allow_pickle=False) as arrays:
        interpretation = manifest.get("model_interpretation")
        if interpretation is not None:
            if (
                not isinstance(interpretation, dict)
                or set(interpretation) - {"row_ids", "probabilities", "decisions"}
                or "row_ids" not in interpretation
            ):
                raise ValueError(
                    "model_interpretation requires full row_ids and alias-to-NPZ tables"
                )
            interpretation = dict(interpretation)
            for kind in ("probabilities", "decisions"):
                fields = interpretation.get(kind, {})
                if not isinstance(fields, dict):
                    raise ValueError(
                        "model interpretation tables must map aliases to NPZ array names"
                    )
                if any(not isinstance(key, str) or key not in arrays for key in fields.values()):
                    raise ValueError("model interpretation references a missing NPZ array")
                interpretation[kind] = {alias: arrays[key] for alias, key in fields.items()}
        return train_model_selection(
            features=arrays["features"],
            decisions=arrays["decisions"],
            truth=arrays["truth"],
            cluster_assignments=arrays["cluster_assignments"],
            row_ids=manifest["row_ids"],
            aliases=manifest["aliases"],
            labels=manifest["labels"],
            input_binding=dict(
                manifest["input_binding"],
                manifest=file_record(path),
                selector_wrapper=file_record(__file__),
            ),
            protocol=ModelSelectionProtocol(**manifest["protocol"]),
            output_dir=output,
            model_interpretation=interpretation,
        )
