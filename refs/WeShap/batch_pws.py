import os
datasets = [
    # "youtube",
    # "imdb",
    # "yelp",
    # "trec",
    # "medical_abstract",
    # "mushroom",
    # "census",
    # "KSDD-1",
    "indoor-outdoor",
    # "voc07-animal",
]

numeric_datasets = ("census", "mushroom", "KSDD-1", "indoor-outdoor", "voc07-animal")
multiclass_datasets = ("trec", "medical_abstract")
image_datasets = ("indoor-outdoor", "voc07-animal")
text_datasets = ("youtube", "imdb", "yelp", "trec", "medical_abstract")

label_models = ["Snorkel"]

mode = "revise"  # Mode of the experiment. One of "revise", "rank"
device = "cuda:0"
eval_metrics = ["weshap", "if", "acc", "cov", "acc-cov"]  # Metrics used to rank or revise label functions
# eval_metrics = ["weshap"]
use_soft_labels = True  # Use soft labels or hard labels
tune_label_model = True  # Tune label model or not
tune_end_model = True  # Tune end model or not
tune_weshap_params = True  # Tune WEShap parameters or not
tune_revision_threshold = True  # Tune revision threshold or not

abstain_strategy = "discard"  # Strategy for non-covered instances. One of "discard", "keep", "random"
runs = 3  # Number of runs for each experiment
interval = 10  # Interval of runs to save results
metric = "acc"  # Metric used to tune label model and end model

for dataset in datasets:
    if dataset in text_datasets:
        end_models = ["bert"]
    elif dataset in image_datasets:
        end_models = ["resnet-50"]
    else:
        end_models = ["mlp"]

    for label_model in label_models:
        for end_model in end_models:
            if dataset in numeric_datasets:
                if dataset == "mushroom":
                    lf_tag = "feature_exhaustive"
                else:
                    lf_tag = None
                feature_extractor = "None"
            else:
                lf_tag = "ngram_random_100"
                if dataset in ["imdb", "yelp"]:
                    feature_extractor = "bertweet-sentiment"
                else:
                    feature_extractor = "sentence-bert"

            for eval_metric in eval_metrics:
                if eval_metric in ["weshap", "if"] and mode == "revise":
                    revision_modes = ["filter", "mute"]
                else:
                    revision_modes = ["filter"]

                for revision_mode in revision_modes:
                    cmd = f"python run_pws.py --dataset-name {dataset} --feature-extractor {feature_extractor} " \
                          f"--mode {mode} --revision-type {revision_mode} --abstain-strategy {abstain_strategy} " \
                          f"--tune-metric {metric} --label-model {label_model} --end-model {end_model} " \
                           f"--eval-metric {eval_metric} --save-wandb --runs {runs} --interval {interval} --device {device}"

                    if use_soft_labels:
                        cmd += " --use-soft-labels"

                    if lf_tag is not None:
                        cmd += f" --lf-tag {lf_tag}"

                    if tune_label_model:
                        cmd += " --tune-label-model"

                    if tune_end_model:
                        cmd += " --tune-end-model"

                    if tune_weshap_params:
                        cmd += " --tune-weshap-params"

                    if eval_metric == "if":
                        cmd += " --load-if --save-if"

                    if tune_revision_threshold:
                        cmd += " --tune-revision-threshold"

                    print(cmd)
                    os.system(cmd)

            # # run golden baseline
            # cmd = (f"python run_pws.py --dataset-name {dataset} --feature-extractor {feature_extractor} --mode golden"
            #        f" --end-model {end_model} --lf-tag {lf_tag} --save-wandb --runs {runs}")
            # print(cmd)
            # os.system(cmd)