"""Aggregate actual receipts only; planned entries never acquire invented scores."""

import csv
import json
from pathlib import Path


def collect(matrix_path, output):
    matrix = json.loads(Path(matrix_path).read_text())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    costs = []
    per_label = []
    for experiment in matrix["experiments"]:
        folder = Path(experiment["output"])
        results = folder / "RESULTS.json"
        complete = folder / "COMPLETE.json"
        records = []
        if experiment.get("kind") == "fit_source_reliability":
            value = {}
            if experiment["status"] == "complete_reused":
                from loris.discovery.manifest import bound

                cp = bound(experiment["receipt"], Path(matrix_path).resolve().parent)
                c = json.loads(cp.read_text())
                value = json.loads(bound(c["results"], cp.parent).read_text())
            elif results.exists():
                value = json.loads(results.read_text())
            label_axis = []
            records = [
                dict(
                    name=experiment["id"],
                    prediction_object="fit_source_reliability_not_final_P",
                    status=value.get("status", experiment["status"]),
                    metrics=None,
                )
            ]
            if value:
                costs.append(
                    dict(
                        id=experiment["id"],
                        seconds=value.get("seconds"),
                        updates=0,
                        queries=0,
                        source_reviews=0,
                        Chase_calls=len(value.get("costs", [])),
                    )
                )
        elif results.exists():
            value = json.loads(results.read_text())
            records = value.get("records", [value["record"]] if "record" in value else [])
            label_axis = value.get("labels", [])
        elif complete.exists():
            value = json.loads(complete.read_text())
            label_axis = []
            records = [
                dict(
                    name=experiment["id"],
                    prediction_object="successful_Chase_P",
                    status=value.get("status"),
                    metrics=value.get("normal_metrics"),
                )
            ]
            costs.append(
                dict(
                    id=experiment["id"],
                    seconds=value.get("elapsed_seconds"),
                    updates=value.get("actual_optimizer_updates"),
                    queries=value.get("actual_label_requests"),
                    source_reviews=value.get("actual_source_reviews"),
                    Chase_calls=value.get("actual_Chase_calls"),
                )
            )
        elif experiment["status"] == "complete_reused":
            from loris.discovery.manifest import bound

            receipt = bound(experiment["receipt"], Path(matrix_path).resolve().parent)
            value = json.loads(receipt.read_text())
            label_axis = []
            original_input = receipt.parent / "INPUTS.json"
            if original_input.exists():
                label_axis = json.loads(original_input.read_text()).get("labels", [])
            if not label_axis:
                protocol = receipt.parent / "learning/cluster_00/protocol.json"
                if protocol.exists():
                    label_axis = json.loads(protocol.read_text())["space"]["labels"]
            m = value.get("final_positive_metrics")
            if m is not None:
                m = dict(
                    m,
                    TP=sum(m["tp"]),
                    FP=sum(m["fp"]),
                    FN=sum(m["fn"]),
                    coverage=value.get("dev_coverage"),
                )
            records = [
                dict(
                    name="frozen_discovery",
                    prediction_object="successful_Chase_P",
                    status="complete_reused",
                    metrics=m,
                )
            ]
        else:
            records = [dict(name=experiment["id"], status=experiment["status"], metrics=None)]
        for record in records:
            m = record.get("metrics") or {}
            rows.append(
                dict(
                    id=experiment["id"],
                    dataset=experiment["dataset"],
                    name=record["name"],
                    status=record.get("status"),
                    object=record.get("prediction_object"),
                    macro_f1=m.get("macro_f1"),
                    micro_f1=m.get("micro_f1"),
                    TP=m.get("TP"),
                    FP=m.get("FP"),
                    FN=m.get("FN"),
                    precision=m.get("precision"),
                    recall=m.get("recall"),
                    coverage=m.get("coverage"),
                    completeness=m.get("multi_label_completeness"),
                    evaluation=experiment.get("evaluation", "development_or_unexecuted"),
                )
            )
            if "per_label_f1" in m:
                for i, f1 in enumerate(m["per_label_f1"]):
                    per_label.append(
                        dict(
                            id=experiment["id"],
                            name=record["name"],
                            label=label_axis[i] if label_axis else i,
                            F1=f1,
                            TP=m["tp"][i],
                            FP=m["fp"][i],
                            FN=m["fn"][i],
                        )
                    )

    def write(name, values, fields):
        with (out / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(values)

    write(
        "MAIN_RESULTS.csv",
        rows,
        [
            "id",
            "dataset",
            "name",
            "status",
            "object",
            "macro_f1",
            "micro_f1",
            "TP",
            "FP",
            "FN",
            "precision",
            "recall",
            "coverage",
            "completeness",
            "evaluation",
        ],
    )
    write(
        "COSTS.csv", costs, ["id", "seconds", "updates", "queries", "source_reviews", "Chase_calls"]
    )
    write("PER_LABEL.csv", per_label, ["id", "name", "label", "F1", "TP", "FP", "FN"])
    (out / "REPORTING_SCOPE.json").write_text(
        json.dumps(
            dict(
                actual_rows=len(rows),
                planned_scores_are_null=True,
                no_cross_object_component_subtraction=True,
                development_selection_bias_not_removed=True,
            ),
            indent=2,
        )
        + "\n"
    )
    return dict(status="complete", rows=len(rows), output=str(out))
