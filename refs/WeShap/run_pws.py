"""
Run the PWS pipeline
"""
import argparse
from data_utils import load_wrench_data
from utils import print_dataset_stats
import numpy as np
import os
from utils import run_pws_pipeline, get_lf_orders, load_lf_names, filter_unhelpful_lfs, \
    mute_unhelpful_weak_labels, compute_if_score, run_if_revision, tune_proxy_model_params
import wandb
from pathlib import Path
from eval import evaluate_golden_baseline
from snorkel.labeling.analysis import LFAnalysis
from weshap import WeShapAnalysis
import warnings

warnings.filterwarnings("ignore")

def main(args):
    train_dataset, valid_dataset, test_dataset = load_wrench_data(data_root=args.dataset_path,
                                                                  dataset_name=args.dataset_name,
                                                                  feature=args.feature_extractor,
                                                                  image_root=args.image_root)

    # sample validation set for evaluating WeShap and other metrics
    if args.valid_sample_frac < 1.0:
        sampled_valid_ids = np.random.choice(len(valid_dataset.ids), int(len(valid_dataset.ids) * args.valid_sample_frac),
                                            replace=False)
        sampled_valid_dataset = valid_dataset.create_subset(sampled_valid_ids)
    elif args.valid_sample_frac > 1:
        sampled_valid_ids = np.random.choice(len(valid_dataset.ids), int(args.valid_sample_frac), replace=False)
        sampled_valid_dataset = valid_dataset.create_subset(sampled_valid_ids)
    else:
        sampled_valid_dataset = valid_dataset
        sampled_valid_ids = np.arange(len(valid_dataset)).astype(int)

    print(f"Dataset path: {args.dataset_path}, name: {args.dataset_name}")
    print_dataset_stats(train_dataset, split="train")
    print_dataset_stats(valid_dataset, split="valid")
    print_dataset_stats(test_dataset, split="test")

    if args.save_wandb:
        group_id = wandb.util.generate_id()
        config_dict = vars(args)
        config_dict["group_id"] = group_id

    rng = np.random.default_rng(args.seed)
    if args.lf_tag is not None:
        # load LF from file
        lf_path = f"{args.dataset_path}/{args.dataset_name}/lf_{args.lf_tag}.npz"
        if not os.path.exists(lf_path):
            print(f"LF file {lf_path} does not exist")
            exit(1)
        wl = np.load(lf_path)
        train_dataset.weak_labels = wl["L_train"].tolist()
        valid_dataset.weak_labels = wl["L_valid"].tolist()

        sampled_valid_dataset.weak_labels = wl["L_valid"][sampled_valid_ids].tolist()
        sampled_valid_dataset.n_lf = len(sampled_valid_dataset.weak_labels[0])

        train_dataset.n_lf = len(train_dataset.weak_labels[0])
        valid_dataset.n_lf = len(valid_dataset.weak_labels[0])

    L_train = np.array(train_dataset.weak_labels)
    L_valid = np.array(valid_dataset.weak_labels)
    summary = LFAnalysis(L_train).lf_summary(Y=np.array(train_dataset.labels))
    lf_names = load_lf_names(args)
    if lf_names is not None:
        summary["lf_names"] = lf_names

    pws_configs = {
        "label_model": args.label_model,
        "end_model": args.end_model,
        "tune_label_model": args.tune_label_model,
        "tune_end_model": args.tune_end_model,
        "tune_metric": args.tune_metric,
        "abstain_strategy": args.abstain_strategy,
        "distance_metric": args.distance_metric,
        "use_soft_labels": args.use_soft_labels,
    }
    if args.tune_weshap_params:
        weshap_params = tune_proxy_model_params(sampled_valid_dataset, tune_metric="acc")
        if args.save_wandb:
            config_dict["weshap_distance_metric"] = weshap_params["distance_metric"]
            config_dict["weshap_k"] = weshap_params["n_neighbors"]
            config_dict["weshap_weights"] = weshap_params["weights"]
    else:
        weshap_params = {
            'n_neighbors': args.weshap_k,
            'distance_metric': args.weshap_distance_metric,
            "weights": args.weshap_weights
        }

    if args.mode == "rank":
        for run in range(args.runs):
            if args.save_wandb:
                wandb.init(
                    project="LLMDP-eval",
                    config=config_dict
                )
                wandb.define_metric("test_acc", summary="mean")
                wandb.define_metric("test_f1", summary="mean")
                wandb.define_metric("test_auc", summary="mean")

            train_dataset.weak_labels = L_train.tolist()
            valid_dataset.weak_labels = L_valid.tolist()
            n_lf = L_train.shape[1]
            lf_nums = list(range(args.interval, n_lf, args.interval))
            if len(lf_nums) == 0 or lf_nums[-1] != n_lf:
                lf_nums.append(n_lf)

            # rank LFs
            print(f"Rank LFs based on {args.eval_metric}")
            lf_order, lf_values = get_lf_orders(args, train_dataset, valid_dataset, rng, weshap_params)
            for lf_num in lf_nums:
                train_dataset.weak_labels = L_train[:, lf_order[:lf_num]].tolist()
                valid_dataset.weak_labels = L_valid[:, lf_order[:lf_num]].tolist()
                lf_stats, label_stats, test_perf = run_pws_pipeline(train_dataset, valid_dataset, test_dataset,
                                                                    **pws_configs, device=args.device)
                print(f"LF num: {lf_num}, Test perf: {test_perf}")

                if args.save_wandb:
                    wandb.log({
                        "lf_num": lf_num,
                        "lf_acc_avg": lf_stats["lf_acc_avg"],
                        "lf_cov_avg": lf_stats["lf_cov_avg"],
                        "lf_overlap_avg": lf_stats["lf_overlap_avg"],
                        "lf_conflict_avg": lf_stats["lf_conflict_avg"],
                        "train_precision": label_stats["accuracy"],
                        "train_coverage": label_stats["coverage"],
                        "test_acc": test_perf["acc"],
                        "test_f1": test_perf["f1"],
                        "test_auc": test_perf["auc"],
                    })

            if args.save_wandb:
                wandb.finish()

    elif args.mode == "revise":
        for run in range(args.runs):
            if args.save_wandb:
                wandb.init(
                    project="LLMDP-eval",
                    config=config_dict
                )

            # run the original pipeline
            train_dataset.weak_labels = L_train.tolist()
            valid_dataset.weak_labels = L_valid.tolist()
            train_dataset.n_lf = len(train_dataset.weak_labels[0])
            valid_dataset.n_lf = len(valid_dataset.weak_labels[0])
            origin_lf_stats, origin_label_stats, origin_test_perf = run_pws_pipeline(
                train_dataset, valid_dataset, test_dataset, **pws_configs, device=args.device)
            print(f"Original Test perf: {origin_test_perf}")
            if args.revision_type in ["filter", "filter+mute"]:
                print(f"Filter LFs based on {args.eval_metric}")
                # filter unhelpful LFs
                lf_order, lf_values = get_lf_orders(args, train_dataset, sampled_valid_dataset, rng, weshap_params)
                selected_lfs, threshold = filter_unhelpful_lfs(train_dataset, sampled_valid_dataset, lf_values,
                                                               pws_configs, tune_threshold=args.tune_revision_threshold)
                train_dataset.weak_labels = L_train[:, selected_lfs].tolist()
                valid_dataset.weak_labels = L_valid[:, selected_lfs].tolist()
                train_dataset.n_lf = len(train_dataset.weak_labels[0])
                valid_dataset.n_lf = len(valid_dataset.weak_labels[0])

            if args.revision_type in ["mute", "filter+mute"]:
                if args.eval_metric == "weshap":
                    analysis = WeShapAnalysis(train_dataset, sampled_valid_dataset,
                                              n_neighbors=weshap_params["n_neighbors"],
                                              weights=weshap_params["weights"],
                                              metric=weshap_params["distance_metric"])
                    weshap_contributions = analysis.calculate_contribution()
                    L_train_mute, threshold = mute_unhelpful_weak_labels(train_dataset, sampled_valid_dataset,
                                                                         weshap_contributions, pws_configs,
                                                                         tune_threshold=args.tune_revision_threshold)
                    train_dataset.weak_labels = L_train_mute.tolist()

                elif args.eval_metric == "if":
                    if args.revision_type == "mute":
                        # reuse the IF score
                        if_path = Path(
                            args.dataset_path) / args.dataset_name / f"if_{args.if_type}_{args.if_mode}_{args.label_model}.npy"
                        if_score = compute_if_score(train_dataset, valid_dataset,
                                                    if_type=args.if_type, mode=args.if_mode,label_model=args.label_model,
                                                    load_if=args.load_if, save_if=args.save_if, if_path=if_path)
                    else:
                        if_score = None

                    revised_lf_stats, revised_label_stats, revised_test_perf, threshold = run_if_revision(
                        train_dataset, valid_dataset, test_dataset, args.if_type, args.if_mode, pws_configs,
                        tune_threshold=args.tune_revision_threshold, if_score=if_score, device=args.device)

                else:
                    raise ValueError(f"Invalid eval_metric: {args.eval_metric} for weak label revision")

            # run the revised pipeline
            if not (args.eval_metric == "if" and args.revision_type in ["mute","filter+mute"]):
                revised_lf_stats, revised_label_stats, revised_test_perf = run_pws_pipeline(
                    train_dataset, valid_dataset, test_dataset, **pws_configs, device=args.device)

            print(f"Revised Test perf: {revised_test_perf}")
            if args.save_wandb:
                wandb.log({
                    "origin_lf_num": L_train.shape[1],
                    "origin_lf_acc_avg": origin_lf_stats["lf_acc_avg"],
                    "origin_lf_cov_avg": origin_lf_stats["lf_cov_avg"],
                    "origin_lf_overlap_avg": origin_lf_stats["lf_overlap_avg"],
                    "origin_lf_conflict_avg": origin_lf_stats["lf_conflict_avg"],
                    "origin_train_precision": origin_label_stats["accuracy"],
                    "origin_train_coverage": origin_label_stats["coverage"],
                    "origin_test_acc": origin_test_perf["acc"],
                    "origin_test_f1": origin_test_perf["f1"],
                    "origin_test_auc": origin_test_perf["auc"],
                    "revision_threshold": threshold,
                    "revised_lf_num": len(train_dataset.weak_labels[0]),
                    "revised_lf_acc_avg": revised_lf_stats["lf_acc_avg"],
                    "revised_lf_cov_avg": revised_lf_stats["lf_cov_avg"],
                    "revised_lf_overlap_avg": revised_lf_stats["lf_overlap_avg"],
                    "revised_lf_conflict_avg": revised_lf_stats["lf_conflict_avg"],
                    "revised_train_precision": revised_label_stats["accuracy"],
                    "revised_train_coverage": revised_label_stats["coverage"],
                    "revised_test_acc": revised_test_perf["acc"],
                    "revised_test_f1": revised_test_perf["f1"],
                    "revised_test_auc": revised_test_perf["auc"],
                })
                wandb.finish()

    else:
        for run in range(args.runs):
            if args.save_wandb:
                wandb.init(
                    project="LLMDP-eval",
                    config=config_dict
                )
            # run the original pipeline
            test_perf = evaluate_golden_baseline(train_dataset, valid_dataset, test_dataset, args)
            if args.save_wandb:
                wandb.log({
                    "test_acc": test_perf["acc"],
                    "test_f1": test_perf["f1"],
                    "test_auc": test_perf["auc"],
                })
                wandb.finish()

                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Usage of WeShap
    parser.add_argument("--mode", type=str, default="rank", choices=["rank", "revise", "golden"], help="Use Mode")
    parser.add_argument("--revision-type", type=str, default="mute", choices=["filter", "mute", "filter+mute"])
    # dataset
    parser.add_argument("--dataset-path", type=str, default="./data", help="dataset path")
    parser.add_argument("--dataset-name", type=str, default="youtube", help="dataset name")
    parser.add_argument("--image-root", type=str, default="../companion/ch03_creating_datasets/cv/data/")
    parser.add_argument("--feature-extractor", type=str, default="bert", help="feature for training end model")
    # label functions
    parser.add_argument("--lf-tag", type=str, default=None, help="tag of LF to use. If None, use original LFs")
    # label function selection
    parser.add_argument("--eval-metric", type=str, default="weshap",
                        choices=["random", "if", "weshap", "acc", "cov", "acc-cov"], help="metric used to evaluate LFs")
    # weshap settings
    parser.add_argument("--feature-transform", type=str, default="none", choices=["none", "pca", "lda"], help="feature transform method")
    parser.add_argument("--target-dimension", type=int, default=10, help="target dimension for feature transform")
    parser.add_argument("--weshap-distance-metric", type=str, default="euclidean", choices=["euclidean","manhattan", "cosine"], help="distance metric for WEShap")
    parser.add_argument("--weshap-k", type=int, default=10, help="k for WEShap")
    parser.add_argument("--weshap-weights", type=str, default="uniform", choices=["uniform", "distance"], help="weight for WEShap")
    parser.add_argument("--tune-weshap-params", action="store_true", help="set to true if tune WEShap hyperparameters")
    parser.add_argument("--tune-revision-threshold", action="store_true", help="set to true if tune revision threshold")
    # influence function settings
    parser.add_argument("--if-type", type=str, default="if", choices=["if", "sif", "relatif"], help="type of IF")
    parser.add_argument("--if-mode", type=str, default="RW", choices=["RW", "WM", "normal"], help="mode of IF")
    parser.add_argument("--load-if", action="store_true", help="set to true if load precomputed IF")
    parser.add_argument("--save-if", action="store_true", help="set to true if save IF")
    # data programming
    parser.add_argument("--label-model", type=str, default="Snorkel", choices=["Snorkel", "MV"], help="label model used in DP paradigm")
    parser.add_argument("--use-soft-labels", action="store_true", help="set to true if use soft labels when training end model")
    parser.add_argument("--end-model", type=str, default="logistic", choices=["logistic", "mlp", "knn", "bert", "resnet-50"], help="end model in DP paradigm")
    parser.add_argument("--distance-metric", type=str, default="euclidean", choices=["cosine", "euclidean", "manhatten"], help="distance metric for KNN")
    parser.add_argument("--abstain-strategy", type=str, default="discard", choices=["discard", "keep", "random"], help="strategy for abstained data")
    parser.add_argument("--tune-label-model", action="store_true", help="tune label model hyperparameters")
    parser.add_argument("--tune-end-model", action="store_true", help="tune end model hyperparameters")
    parser.add_argument("--tune-metric", type=str, default="acc", choices=["acc", "f1"], help="metric used to tune model hyperparameters")
    # experiment
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--valid-sample-frac", type=float, default=1.0, help="validation size")
    parser.add_argument("--interval", type=int, default=10, help="interval of LF number for evaluating LF ranking")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0", help="device for training")
    parser.add_argument("--save-wandb", action="store_true", help="save results to wandb")
    args = parser.parse_args()
    main(args)

