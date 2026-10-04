"""Optional diagnostic exports; in-memory inference/provenance is never removed."""
import gzip
import json
import numpy as np

from loris._engine.typed_session import plain


def export_facts(folder, result):
    from .conflict_protocol import unknown_mask
    history_arrays = {}
    if hasattr(result, "history_nodes"):
        history_arrays = dict(execution_gamma_positive=result.persistent_gamma.pos,
                              execution_gamma_negative=result.persistent_gamma.neg,
                              temporary_seed_positive=result.temporary_seed[0],
                              temporary_seed_negative=result.temporary_seed[1])
        roots = getattr(getattr(result.history_nodes, "mapping", None), "base", None)
        if roots is not None:
            history_arrays.update(initial_root_gamma_positive=roots.roots.gamma.pos,
                                  initial_root_gamma_negative=roots.roots.gamma.neg)
    np.savez_compressed(folder / "FACTS.npz", positive=result.positive_facts,
        negative=result.negative_facts, conflict_unknown=unknown_mask(result),
        arbitration_positive=getattr(result,"arbitration_positive",np.zeros_like(result.positive_facts)),
        arbitration_negative=getattr(result,"arbitration_negative",np.zeros_like(result.negative_facts)),
        **history_arrays)


def export_provenance(folder, result):
    with gzip.open(str(folder / "PROOF.jsonl.gz"), "wt") as f:
        compact = getattr(result.proof_dag, "compact", None)
        if compact is not None:
            f.write(json.dumps(plain(compact()), allow_nan=False) + "\n")
        else:
            for key in result.proof_dag:
                f.write(json.dumps(plain(dict(key=key, node=result.proof_dag[key])), allow_nan=False) + "\n")
        f.write(json.dumps(dict(subset=result.subset_facts.to_dict(),
                               strict_subset=result.strict_subset_facts.to_dict())) + "\n")
    if hasattr(result, "history_nodes"):
        from .unknown_incremental import export_history
        with gzip.open(str(folder / "HISTORY.json.gz"), "wt") as f:
            json.dump(export_history(result), f, allow_nan=False)


def should_export(loop, call, phase):
    mode = loop.get("intermediate_exports", "periodic")
    if mode == "all":
        return True
    if mode == "final" or phase in ("feedback_check", "LLM_proposal_validation", "LLM_check_base_refresh"):
        return False
    return call == 1 or call % loop.get("history_export_interval", 10) == 0
