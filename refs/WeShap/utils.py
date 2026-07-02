import numpy as np
from label_model import get_wrench_label_model
from end_model import train_disc_model, random_argmax, train_resnet
from wrench.search import grid_search
from wrench.search_space import SEARCH_SPACE
from wrench.explainer import Explainer, modify_training_labels
from wrench.endmodel import EndClassifierModel
from snorkel.labeling.analysis import LFAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim
import torch.nn as nn
import copy
from eval import evaluate_lfs,evaluate_labels,evaluate_disc_model, evaluate_resnet
from weshap import WeShapAnalysis
import optuna
from pathlib import Path
import os
import json
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from data_utils import FeaturesDataset
import wandb
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image

def plot_correction(weshap_correction, if_correction, n_error, title, save_path):
    plt.figure()
    plt.rcParams.update({'font.size': 16})
    x = np.arange(len(weshap_correction)) / len(weshap_correction)
    plot_range = x < 0.5
    weshap_auc = np.mean(weshap_correction[plot_range]) / n_error
    if_auc = np.mean(if_correction[plot_range]) / n_error
    plt.plot(x[plot_range], weshap_correction[plot_range]/n_error, "-", label="WeShap (AUC={:.3f})".format(weshap_auc))
    plt.plot(x[plot_range], if_correction[plot_range]/n_error, ":", label="SIF (AUC={:.3f})".format(if_auc))
    plt.plot(x[plot_range], x[plot_range], "--", label="Random (AUC=0.250)")
    plt.title(title)
    plt.legend()
    plt.xlabel("Fraction of inspected labels")
    plt.ylabel("Fraction of corrected labels")
    plt.savefig(save_path)
    plt.close()


def filter_unhelpful_lfs(train_dataset, valid_dataset, lf_values, pws_configs, tune_threshold=True):
    lf_order = np.argsort(lf_values)[::-1]
    if tune_threshold:
        L_train = np.array(train_dataset.weak_labels)
        L_valid = np.array(valid_dataset.weak_labels)
        train_dataset = copy.copy(train_dataset)
        valid_dataset = copy.copy(valid_dataset)
        pws_configs = copy.copy(pws_configs)
        pws_configs["tune_label_model"] = False
        pws_configs["tune_end_model"] = False
        if pws_configs["end_model"] in ["bert", "resnet-50"]:
            # Use fixed feature expression instead of tuning the whole end model
            pws_configs["end_model"] = "logistic"

        n_lf = L_train.shape[1]
        def objective(trial):
            params = {"selected_lf_num": trial.suggest_int("selected_lf_num", 3, n_lf)}
            selected_lf = lf_order[:params["selected_lf_num"]]
            active_indices = np.max(L_train, axis=1) != -1
            train_dataset.weak_labels = L_train[:, selected_lf].tolist()
            valid_dataset.weak_labels = L_valid[:, selected_lf].tolist()
            revised_active_indices = np.max(L_train[:, selected_lf], axis=1) != -1
            if np.sum(revised_active_indices) < 0.1 * np.sum(active_indices):
                # the number of active instances is too small
                return -1

            try:
                lf_stats, label_stats, test_perf = run_pws_pipeline(train_dataset, valid_dataset, valid_dataset, **pws_configs)
            except ValueError:
                return -1

            if np.isnan(test_perf["acc"]):
                return -1

            return test_perf["acc"]

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        last_lf = lf_order[study.best_params["selected_lf_num"]-1]
        threshold = lf_values[last_lf]
        return lf_order[:study.best_params["selected_lf_num"]], threshold
    else:
        threshold = 0.0
        lf_size = np.sum(lf_values >= threshold)
        return lf_order[:lf_size], threshold


def mute_unhelpful_weak_labels(train_dataset, valid_dataset, contribution_matrix, pws_configs, tune_threshold=True):
    L_train = np.array(train_dataset.weak_labels)
    if tune_threshold:
        train_dataset = copy.copy(train_dataset)
        pws_configs = copy.copy(pws_configs)
        pws_configs["tune_label_model"] = False
        pws_configs["tune_end_model"] = False
        if pws_configs["end_model"] in ["bert", "resnet-50"]:
            # Use fixed feature expression instead of tuning the whole end model
            pws_configs["end_model"] = "logistic"

        min_contribution = np.min(contribution_matrix.flatten())
        max_contribution = np.max(contribution_matrix.flatten())
        def objective(trial):
            params = {"threshold": trial.suggest_float("threshold", min_contribution, max_contribution)}
            active_indices = np.max(L_train, axis=1) != -1
            muted_L_train = copy.copy(L_train)
            muted_L_train[contribution_matrix < params["threshold"]] = -1
            revised_active_indices = np.max(muted_L_train, axis=1) != -1
            if np.sum(revised_active_indices) < 0.1 * np.sum(active_indices):
                # the number of active instances is too small
                return -1

            covered_classes = np.unique(muted_L_train)
            for c in range(train_dataset.n_class):
                if c not in covered_classes:
                    # the class is not covered
                    return -1

            train_dataset.weak_labels = muted_L_train.tolist()
            try:
                lf_stats, label_stats, test_perf = run_pws_pipeline(train_dataset, valid_dataset, valid_dataset, **pws_configs)
            except ValueError:
                return -1

            if np.isnan(test_perf["acc"]):
                return -1

            return test_perf["acc"]


        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        muted_L_train = copy.copy(L_train)
        muted_L_train[contribution_matrix < study.best_params["threshold"]] = -1
        return muted_L_train, study.best_params["threshold"]
    else:
        threshold = 0.0
        muted_L_train = copy.copy(L_train)
        muted_L_train[contribution_matrix < threshold] = -1
        return muted_L_train, threshold


def tune_proxy_model_params(valid_dataset, tune_metric='acc'):
    """
    Tune the parameters of proxy model.
    Tune metric: "acc" for accuracy on validation set.
    """
    def objective(trial):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 5, 80),
            'distance_metric': trial.suggest_categorical('distance_metric', ['euclidean', 'manhattan', 'cosine']),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance'])
        }
        knn = KNeighborsClassifier(n_neighbors=params['n_neighbors'], metric=params["distance_metric"],
                                    weights=params['weights'])
        knn.fit(valid_dataset.features, valid_dataset.labels)
        valid_preds = knn.predict(valid_dataset.features)
        if tune_metric == 'acc':
            return accuracy_score(valid_dataset.labels, valid_preds)
        else:
            raise NotImplementedError

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    return study.best_params


def load_lf_names(args):
    if args.lf_tag is not None:
        lf_name_path = f"{args.dataset_path}/{args.dataset_name}/lf_{args.lf_tag}_names.txt"
    else:
        lf_name_path = f"{args.dataset_path}/{args.dataset_name}/lf_names.txt"

    if not os.path.exists(lf_name_path):
        print(f"LF names file {lf_name_path} does not exist.")
        return None

    with open(lf_name_path, "r") as f:
        lf_names = f.readlines()
        lf_names = [lf_name.strip() for lf_name in lf_names]

    return lf_names


def convert_lf_names_to_txt(dataset_path, dataset_name):
    lf_name_path = f"{dataset_path}/{dataset_name}/label_functions.json"
    if os.path.exists(lf_name_path):
        dict = json.load(open(lf_name_path))
        lf_names = []
        for key in dict:
            lf_names.append(dict[key]["tag"] + "->" + str(dict[key]["label"]) + "\n")

        with open(f"{dataset_path}/{dataset_name}/lf_names.txt", "w") as f:
            f.writelines(lf_names)


def get_lf_orders(args, train_dataset, valid_dataset, rng, weshap_params):
    """
    Get the order of LFs according to different metrics
    """

    L_train = np.array(train_dataset.weak_labels)
    L_valid = np.array(valid_dataset.weak_labels)

    lf_train_stats = LFAnalysis(L_train).lf_summary(Y=np.array(train_dataset.labels))
    lf_valid_stats = LFAnalysis(L_valid).lf_summary(Y=np.array(valid_dataset.labels))
    n_lf = L_train.shape[1]

    if args.eval_metric == "if":
        if_path = Path(args.dataset_path) / args.dataset_name / f"if_{args.if_type}_{args.if_mode}_{args.label_model}.npy"
        if_score = compute_if_score(train_dataset, valid_dataset,
                                    if_type=args.if_type, mode=args.if_mode, label_model=args.label_model,
                                    save_if=args.save_if, load_if=args.load_if, if_path=if_path)
        # lf_values = np.sum(if_score, axis=(0,2))
        lf_values = np.abs(np.sum(if_score, axis=(0, 2)))   # Use absolute IF values to rank LFs
        lf_order = np.argsort(lf_values)[::-1]

    elif args.eval_metric == "acc":
        lf_values = lf_valid_stats["Emp. Acc."].values
        lf_order = np.argsort(lf_values)[::-1]

    elif args.eval_metric == "cov":
        lf_values = lf_train_stats["Coverage"].values
        lf_order = np.argsort(lf_values)[::-1]

    elif args.eval_metric == "acc-cov":
        lf_values = (2 * lf_valid_stats["Emp. Acc."].values - 1) * lf_train_stats["Coverage"].values
        lf_order = np.argsort(lf_values)[::-1]

    elif args.eval_metric == "weshap":
        if args.abstain_strategy == "discard":
            # remove uncovered data as they will not be used in the PWS pipeline
            covered_train_dataset = train_dataset.get_covered_subset()
            weshap_train_dataset = covered_train_dataset
        else:
            weshap_train_dataset = train_dataset

        if args.feature_transform == "none":
            analysis = WeShapAnalysis(weshap_train_dataset, valid_dataset,
                                      n_neighbors=weshap_params["n_neighbors"],
                                      weights=weshap_params["weights"],
                                      metric=weshap_params["distance_metric"])
            lf_values = analysis.calculate_weshap_score()
            lf_order = np.argsort(lf_values)[::-1]

        elif args.feature_transform == "pca":
            pca = PCA(n_components=args.target_dimension)
            train_dataset_transformed = copy.copy(weshap_train_dataset)
            train_dataset_transformed.features = pca.fit_transform(weshap_train_dataset.features)
            valid_dataset_transformed = copy.copy(valid_dataset)
            valid_dataset_transformed.features = pca.transform(valid_dataset.features)
            analysis = WeShapAnalysis(train_dataset_transformed,
                                      valid_dataset_transformed,
                                      n_neighbors=weshap_params["n_neighbors"],
                                      weights=weshap_params["weights"],
                                      metric=weshap_params["distance_metric"])
            lf_values = analysis.calculate_weshap_score()
            lf_order = np.argsort(lf_values)[::-1]

        elif args.feature_transform == "lda":
            LDA = LinearDiscriminantAnalysis(n_components=min(args.target_dimension, train_dataset.n_class-1))
            LDA.fit(valid_dataset.features, valid_dataset.labels)
            train_dataset_transformed = copy.copy(weshap_train_dataset)
            train_dataset_transformed.features = LDA.transform(weshap_train_dataset.features)
            valid_dataset_transformed = copy.copy(valid_dataset)
            valid_dataset_transformed.features = LDA.transform(valid_dataset.features)
            analysis = WeShapAnalysis(train_dataset_transformed,
                                      valid_dataset_transformed,
                                      n_neighbors=weshap_params["n_neighbors"],
                                      weights=weshap_params["weights"],
                                      metric=weshap_params["distance_metric"])
            lf_values = analysis.calculate_weshap_score()
            lf_order = np.argsort(lf_values)[::-1]

        else:
            raise NotImplementedError

    elif args.eval_metric == "random":
        lf_values = np.ones(n_lf)
        lf_order = rng.permutation(n_lf)

    else:
        raise ValueError(f"Unknown eval metric {args.eval_metric}")

    return lf_order, lf_values


def approximate_label_model(explainer, L, y, w0=None):
    L_aug = explainer.augment_label_matrix(L)

    N, M, C = L_aug.shape
    C = C - 1

    L_aug_ = csr_matrix(L_aug.reshape(N, -1))

    if w0 is not None:
        x0 = w0 - w0.min(axis=-1, keepdims=True)
    else:
        x0 = np.zeros(shape=(M, C + 1, C))
        for i in range(C):
            x0[:, i+1, i] = 1.0

    def func(x):
        x = x.reshape(M * (C + 1), C)
        P = L_aug_.dot(x)
        Z = np.sum(P, axis=1, keepdims=True)
        y_hat = P / Z
        diff = (y_hat - y).flatten()
        # P = L_aug_ @ x
        # Z = np.sum(P, axis=1, keepdims=True)
        # y_hat = P / Z
        return diff

    res = least_squares(func, x0.flatten(), bounds=(0, np.inf), ftol=1e-3, xtol=1e-3, gtol=1e-3)

    approx_w = res.x.reshape(x0.shape)
    explainer.register_label_model(approx_w, 'identity')
    return approx_w


def compute_if_score(train_data, valid_data,
                     if_type="if", mode="RW",
                     label_model="Snorkel",
                     save_if=False, load_if=False, if_path=None):
    """
    Compute the IF score using WS Explainer
    """
    if load_if and if_path is not None and os.path.exists(if_path):
        if_score = np.load(if_path)
        return if_score

    label_model = get_wrench_label_model(label_model)
    label_model.fit(dataset_train=train_data,
                    dataset_valid=valid_data)
    covered_train_data = train_data.get_covered_subset()
    L = np.array(covered_train_data.weak_labels)
    aggregated_soft_labels = label_model.predict_proba(covered_train_data)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    explainer = Explainer(L.shape[1], train_data.n_class)
    approximate_label_model(explainer, L, aggregated_soft_labels)
    lr, weight_decay, epochs, batch_size = 0.01, 0.0, 1000, 128
    IF_score = explainer.compute_IF_score(
        L, np.array(covered_train_data.features), np.array(valid_data.features),
        np.array(valid_data.labels), if_type=if_type, mode=mode,
        lr=lr, weight_decay=weight_decay, epochs=epochs, batch_size=batch_size,
        device=device
    )  # IF score (N * M * C). The influence of LF j on instance i when it fires for class c
    if save_if and if_path is not None:
        np.save(if_path, IF_score)

    return IF_score


def run_if_revision(train_dataset, valid_dataset, test_dataset, if_type, if_mode, pws_configs, tune_threshold=True, if_score=None,
                    device="cuda"):
    """
    Run the pipeline of IF revision
    """
    # train label model
    L_train = np.array(train_dataset.weak_labels)
    lf_polarities = np.unique(L_train.reshape(-1))
    if L_train.shape[1] > 3 and len(lf_polarities) >= 3:
        label_model_ = get_wrench_label_model(pws_configs["label_model"], verbose=False)
        search_space = SEARCH_SPACE.get(pws_configs["label_model"], None)
    else:
        label_model_ = get_wrench_label_model("MV", verbose=False)
        search_space = None

    if search_space is not None and pws_configs["tune_label_model"]:
        searched_paras = grid_search(label_model_, dataset_train=train_dataset, dataset_valid=valid_dataset,
                                     metric="acc", direction='auto',
                                     search_space=search_space,
                                     n_repeats=1, n_trials=100, parallel=False)
        label_model_ = get_wrench_label_model(pws_configs["label_model"], **searched_paras, verbose=False)

    try:
        label_model_.fit(dataset_train=train_dataset,
                         dataset_valid=valid_dataset,
                         )
    except:
        label_model_ = get_wrench_label_model("MV", verbose=False)
        label_model_.fit(dataset_train=train_dataset,
                         dataset_valid=valid_dataset,
                         )
    # run explainer
    explainer = Explainer(train_dataset.n_lf, train_dataset.n_class)
    covered_train_data = train_dataset.get_covered_subset()
    covered_train_indices = L_train.max(axis=1) >= 0
    L = np.array(covered_train_data.weak_labels)
    xs_tr = np.array(covered_train_data.features)
    aggregated_soft_labels = label_model_.predict_proba(covered_train_data)
    approx_w = approximate_label_model(explainer, L, aggregated_soft_labels)
    if if_score is None:
        if_score = compute_if_score(train_dataset, valid_dataset, if_type=if_type,
                                    mode=if_mode, label_model=pws_configs["label_model"],
                                    load_if=False, save_if=False)

    if pws_configs["end_model"] in ["bert", "resnet-50"]:
        # Use fixed feature expression instead of tuning the whole end model
        end_model_tuning = "logistic"
    else:
        end_model_tuning = pws_configs["end_model"]

    def objective(trial):
        params = {"alpha": trial.suggest_float("alpha", 0.7, 1.0)}
        modified_soft_labels = modify_training_labels(aggregated_soft_labels, L, approx_w, if_score, params["alpha"],
                                                      sample_method='weight', normal_if=False, act_func='identity', normalize=True)
        disc_model = train_disc_model(model_type=end_model_tuning,
                                      distance_metric=pws_configs["distance_metric"],
                                      xs_tr=xs_tr,
                                      ys_tr_soft=modified_soft_labels,
                                      valid_dataset=valid_dataset,
                                      use_soft_labels=pws_configs["use_soft_labels"],
                                      tune_end_model=False,
                                      tune_metric="acc")

        valid_perf = evaluate_disc_model(disc_model, valid_dataset)
        return valid_perf["acc"]

    if tune_threshold:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)
        alpha = study.best_params["alpha"]
        print("Best alpha: {}".format(alpha))
        modified_soft_labels = modify_training_labels(aggregated_soft_labels, L, approx_w, if_score, alpha,
                                                      sample_method='weight', normal_if=False, act_func='identity', normalize=True)
    else:
        alpha = 0.9
        modified_soft_labels = modify_training_labels(aggregated_soft_labels, L, approx_w, if_score, alpha,
                                                        sample_method='weight', normal_if=False, act_func='identity', normalize=True)

    if pws_configs["end_model"] == "bert":
        # Finetune end model
        device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        torch.cuda.empty_cache()
        disc_model = EndClassifierModel(
            batch_size=32,
            real_batch_size=16,  # for accumulative gradient update
            test_batch_size=128,
            n_steps=1000,
            backbone='BERT',
            backbone_model_name='bert-base-cased',
            backbone_max_tokens=128,
            backbone_fine_tune_layers=-1,  # fine  tune all
            optimizer='AdamW',
            optimizer_lr=5e-5,
            optimizer_weight_decay=0.0,
        )
        disc_model.fit(
            dataset_train=covered_train_data,
            y_train=modified_soft_labels,
            dataset_valid=valid_dataset,
            evaluation_step=10,
            metric='acc',
            patience=10,
            device=device
        )
        test_perf = evaluate_disc_model(disc_model, test_dataset)

    elif pws_configs["end_model"] == "resnet-50":
        # Load the pre-trained ResNet-50 model
        disc_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Modify the final layer to match the number of classes in your dataset
        num_classes = train_dataset.n_class
        disc_model.fc = nn.Linear(disc_model.fc.in_features, num_classes)
        modified_hard_labels = random_argmax(modified_soft_labels)

        train_dataset_ = FeaturesDataset(train_dataset.images[covered_train_indices], modified_hard_labels)
        valid_dataset_ = FeaturesDataset(valid_dataset.images, valid_dataset.labels)
        test_dataset_ = FeaturesDataset(test_dataset.images, test_dataset.labels)
        train_loader = DataLoader(train_dataset_, batch_size=64, shuffle=True)
        valid_loader = DataLoader(valid_dataset_, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset_, batch_size=64, shuffle=False)
        # Train the resnet-50 model
        device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        torch.cuda.empty_cache()
        disc_model = train_resnet(disc_model, train_loader, valid_loader,
                                  n_epochs=50, lr=1e-4, weight_decay=1e-5, device=device)
        test_perf = evaluate_resnet(disc_model, test_loader, device)

    else:
        disc_model = train_disc_model(model_type=pws_configs["end_model"],
                                      distance_metric=pws_configs["distance_metric"],
                                      xs_tr=xs_tr,
                                      ys_tr_soft=modified_soft_labels,
                                      valid_dataset=valid_dataset,
                                      use_soft_labels=pws_configs["use_soft_labels"],
                                      tune_end_model=pws_configs["tune_end_model"],
                                      tune_metric="acc")
        test_perf = evaluate_disc_model(disc_model, test_dataset)

    cur_L_train = np.array(train_dataset.weak_labels)
    train_labels = np.array(train_dataset.labels)
    revised_lf_stats = evaluate_lfs(train_labels, cur_L_train, n_class=train_dataset.n_class)
    ys_tr = np.repeat(-1, len(train_labels))
    covered_train_indices = np.max(cur_L_train, axis=1) >= 0
    ys_tr[covered_train_indices] = np.argmax(modified_soft_labels, axis=1)
    revised_label_stats = evaluate_labels(train_labels, ys_tr, n_class=train_dataset.n_class)

    return revised_lf_stats, revised_label_stats, test_perf, alpha


def print_dataset_stats(dataset, split="train"):
    print("{} size: {}".format(split, len(dataset)))
    values, counts = np.unique(dataset.labels, return_counts=True)
    freq = counts / len(dataset)
    print("Label distribution: ", freq)


def run_pws_pipeline(train_dataset, valid_dataset, test_dataset, label_model, end_model,
                     tune_label_model=False, tune_end_model=False, tune_metric="acc", abstain_strategy="discard",
                     distance_metric="euclidean", use_soft_labels=True, device="cuda"):
    """
    Run the pipeline of PWS
    """
    L_train = np.array(train_dataset.weak_labels)
    train_labels = np.array(train_dataset.labels)
    lf_train_stats = evaluate_lfs(train_labels, L_train, n_class=train_dataset.n_class)
    lf_polarities = np.unique(L_train.reshape(-1))
    if L_train.shape[1] > 3 and len(lf_polarities) >= 3:
        label_model_ = get_wrench_label_model(label_model, verbose=False)
        search_space = SEARCH_SPACE.get(label_model, None)
    else:
        label_model_ = get_wrench_label_model("MV", verbose=False)
        search_space = None

    if search_space is not None and tune_label_model:
        if tune_metric == "f1":
            if train_dataset.n_class == 2:
                metric = "f1_binary"
            else:
                metric = "f1_micro"
        else:
            metric = tune_metric

        searched_paras = grid_search(label_model_, dataset_train=train_dataset, dataset_valid=valid_dataset,
                                     metric=metric, direction='auto',
                                     search_space=search_space,
                                     n_repeats=1, n_trials=100, parallel=False)
        label_model_ = get_wrench_label_model(label_model, **searched_paras, verbose=False)

    try:
        label_model_.fit(dataset_train=train_dataset,
                        dataset_valid=valid_dataset,
                        )
    except:
        label_model_ = get_wrench_label_model("MV", verbose=False)
        label_model_.fit(dataset_train=train_dataset,
                        dataset_valid=valid_dataset,
                        )

    ys_tr = label_model_.predict(train_dataset)
    ys_tr_soft = label_model_.predict_proba(train_dataset)

    train_covered_indices = (np.max(L_train, axis=1) != -1) & (ys_tr != -1)  # non-abstain indices
    ys_tr[~train_covered_indices] = -1
    train_label_stats = evaluate_labels(train_labels, ys_tr, n_class=train_dataset.n_class)

    if abstain_strategy == "discard":
        # discard non-abstain indices
        xs_tr = train_dataset.features[train_covered_indices, :]
        ys_tr_soft = ys_tr_soft[train_covered_indices, :]
    elif abstain_strategy == "keep":
        # keep the predictions for abstain indices
        xs_tr = train_dataset.features
    elif abstain_strategy == "random":
        # randomly assign labels to abstain indices
        xs_tr = train_dataset.features
        ys_tr_soft[~train_covered_indices, :] = 1 / train_dataset.n_class
    else:
        raise ValueError("Invalid abstain strategy: {}".format(abstain_strategy))

    if end_model  in ["knn", "logistic", "mlp"]:
        # Use fixed feature expression, no need to tune end model
        disc_model = train_disc_model(model_type=end_model,
                                      distance_metric=distance_metric,
                                      xs_tr=xs_tr,
                                      ys_tr_soft=ys_tr_soft,
                                      valid_dataset=valid_dataset,
                                      use_soft_labels=use_soft_labels,
                                      tune_end_model=tune_end_model,
                                      tune_metric=tune_metric)
        test_perf = evaluate_disc_model(disc_model, test_dataset)

    elif end_model == "bert":
        # Finetune end model
        if abstain_strategy == "discard":
            indices = [i for i in range(len(train_covered_indices)) if train_covered_indices[i]]
            downstream_train_dataset = train_dataset.create_subset(indices)
        else:
            downstream_train_dataset = train_dataset

        device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        torch.cuda.empty_cache()
        disc_model = EndClassifierModel(
            batch_size=32,
            real_batch_size=16,  # for accumulative gradient update
            test_batch_size=128,
            n_steps=1000,
            backbone='BERT',
            backbone_model_name='bert-base-cased',
            backbone_max_tokens=128,
            backbone_fine_tune_layers=-1,  # fine  tune all
            optimizer='AdamW',
            optimizer_lr=5e-5,
            optimizer_weight_decay=0.0,
        )
        disc_model.fit(
            dataset_train=downstream_train_dataset,
            y_train=ys_tr_soft,
            dataset_valid=valid_dataset,
            evaluation_step=10,
            metric='acc',
            patience=10,
            device=device
        )
        test_perf = evaluate_disc_model(disc_model, test_dataset)

    elif end_model == "resnet-50":
        # Load the pre-trained ResNet-50 model
        disc_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Modify the final layer to match the number of classes in your dataset
        num_classes = train_dataset.n_class  # Change this to the number of classes in your dataset
        disc_model.fc = nn.Linear(disc_model.fc.in_features, num_classes)

        # Create datasets in the format compatible of resnet-50 model
        if abstain_strategy == "discard":
            ys_tr = random_argmax(ys_tr_soft)
            train_dataset_ = FeaturesDataset(train_dataset.images[train_covered_indices], ys_tr)
        else:
            train_dataset_ = FeaturesDataset(train_dataset.images, ys_tr)

        valid_dataset_ = FeaturesDataset(valid_dataset.images, valid_dataset.labels)
        test_dataset_ = FeaturesDataset(test_dataset.images, test_dataset.labels)
        train_loader = DataLoader(train_dataset_, batch_size=64, shuffle=True)
        valid_loader = DataLoader(valid_dataset_, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset_, batch_size=64, shuffle=False)

        # Train the resnet-50 model
        device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        torch.cuda.empty_cache()
        disc_model = train_resnet(disc_model, train_loader, valid_loader,
                                  n_epochs=50, lr=1e-4, weight_decay=1e-5, device=device)
        test_perf = evaluate_resnet(disc_model, test_loader, device)

    else:
        raise ValueError(f"Unsupported end model: {end_model}")

    return lf_train_stats, train_label_stats, test_perf



