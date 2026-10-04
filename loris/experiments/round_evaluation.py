"""Score saved decisions after inference, using the separate evaluation truth."""

from pathlib import Path

import numpy as np

from .evaluation import metrics


def evaluate_rounds(run, completion, context, evaluation_ids, truth):
    run = Path(run).resolve()
    expected = len(evaluation_ids)
    label_count = len(context["labels"])
    truth_rows = {row_id: i for i, row_id in enumerate(evaluation_ids)}
    records = []
    for item in completion.get("round_history", []):
        path = (run / item["file"]).resolve()
        if run not in path.parents:
            raise ValueError("round snapshot must belong to its labeling run")
        record = dict(item)
        record.update(output_metrics=None, logical_metrics=None,
                      aux_model_metrics_at_0_5=None, dynamic_predicate_F1={})
        if not path.is_file():
            record.update(scoring_status="missing_snapshot", scored_documents=0,
                          expected_documents=expected)
            records.append(record)
            continue
        with np.load(path, allow_pickle=False) as values:
            rows = values["rows"]
            if (rows.ndim != 1 or rows.dtype.kind not in "iu" or len(set(rows.tolist())) != len(rows)
                    or np.any(rows < 0) or np.any(rows >= len(context["row_ids"]))):
                raise ValueError("invalid saved round document row axis")
            ids = [context["row_ids"][int(row)] for row in rows]
            if ("row_ids" in values.files and values["row_ids"].tolist() != ids
                    or "labels" in values.files and values["labels"].tolist() != context["labels"]):
                raise ValueError("saved round row/label identities differ from application context")
            saved = [i for i, row_id in enumerate(ids) if row_id in truth_rows]
            actual_truth = [truth_rows[ids[i]] for i in saved]
            count = len(saved)
            record.update(
                scoring_status="complete" if count == expected and expected else "incomplete_saved_rows",
                scored_documents=count, expected_documents=expected,
                row_identity="explicit" if "row_ids" in values.files else "legacy_global_row_indices",
            )
            for field, name in (("logical", "logical_metrics"), ("output_positive", "output_metrics"),
                                ("aux_raw", "aux_model_metrics_at_0_5")):
                if field not in values.files or values[field].shape != (len(rows), label_count) or values[field].dtype != np.bool_:
                    raise ValueError("saved round prediction axes or dtype differ: " + field)
                record[name] = metrics(truth[actual_truth], values[field][saved]) if count else None
            if np.any(values["logical"] & ~values["output_positive"]):
                raise ValueError("saved round output omits retained logical positives")
            for occurrence, label in item["occurrence_labels"].items():
                if (type(label) is not int or not 0 <= label < label_count or occurrence not in values.files
                        or values[occurrence].shape != (len(rows),) or values[occurrence].dtype != np.bool_):
                    raise ValueError("saved dynamic predicate decision/label axis differs")
                record["dynamic_predicate_F1"][occurrence] = (
                    metrics(truth[actual_truth, label:label+1], values[occurrence][saved, None]) if count else None
                )
        records.append(record)
    return dict(
        status=("no_saved_rounds" if not records else "complete" if all(
            r["scoring_status"] == "complete" for r in records
        ) else "incomplete_saved_rows"),
        source="independent_post_run_evaluation_truth",
        expected_documents=expected,
        missing_rows_are_not_scored_as_negative=True,
        rounds=records,
    )
