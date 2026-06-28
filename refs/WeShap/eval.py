"""
Evaluate the quality of the LFs, label model, and the final labels.
"""
import numpy as np
import wrench.basemodel
from wrench.endmodel import EndClassifierModel
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from snorkel.labeling.analysis import LFAnalysis
from data_utils import FeaturesDataset
from torch.utils.data import DataLoader
from end_model import train_resnet, train_disc_model


def evaluate_lfs(labels, L_train, lf_classes=None, n_class=2):
    # evaluate the coverage, accuracy of label functions
    n_lf = L_train.shape[1]
    n_active = np.sum(L_train != -1, axis=0)
    n_correct = np.sum(L_train == labels.reshape(-1, 1), axis=0)
    lf_covs = n_active / len(labels)
    lf_accs = np.divide(n_correct, n_active, out=np.repeat(np.nan, len(n_active)), where=n_active != 0)
    lf_cov_avg = np.mean(lf_covs)
    lf_acc_avg = np.mean(lf_accs[lf_covs != 0])
    # evaluate conflicts, overlaps
    lf_stats = LFAnalysis(L_train).lf_summary(Y=labels)
    lf_overlap_avg = lf_stats["Overlaps"].mean()
    lf_conflict_avg = lf_stats["Conflicts"].mean()
    results = {
        "n_lf": n_lf,
        "lf_cov_avg": lf_cov_avg,
        "lf_acc_avg": lf_acc_avg,
        "lf_overlap_avg": lf_overlap_avg,
        "lf_conflict_avg": lf_conflict_avg,
    }

    # evaluate the LF quality per class
    if lf_classes is not None:
        lf_num_pc = []
        lf_acc_avg_pc = []
        lf_cov_avg_pc = []
        lf_cov_total_pc = []
        for c in range(n_class):
            active_lfs = lf_classes == c
            lf_num_pc.append(np.sum(active_lfs))
            if np.sum(active_lfs) != 0:
                lf_acc_avg_pc.append(np.nanmean(lf_accs[active_lfs]))
                lf_cov_avg_pc.append(np.mean(lf_covs[active_lfs]))
                L_train_pc = L_train[:, active_lfs]
                cov_pc = np.mean(np.max(L_train_pc, axis=1) == c)
                lf_cov_total_pc.append(cov_pc)
            else:
                # no LF emits label for class c
                lf_acc_avg_pc.append(np.nan)
                lf_cov_avg_pc.append(np.nan)
                lf_cov_total_pc.append(np.nan)

        results["n_lf_per_class"] = lf_num_pc
        results["lf_acc_per_class"] = lf_acc_avg_pc
        results["lf_cov_per_class"] = lf_cov_avg_pc
        results["lf_cov_total_per_class"] = lf_cov_total_pc

    return results


def evaluate_labels(labels, preds, n_class=2):
    # Evaluate the prediction results. -1 in preds mean the label model rejects making prediction.
    covered_indices = preds != -1
    covered_labels = labels[covered_indices]
    covered_preds = preds[covered_indices]
    if -1 in covered_labels:
        # ground truth labels are missing
        results = {
            "coverage": np.sum(covered_indices) / len(preds),
            "accuracy": np.nan,
        }
        return results

    coverage = np.sum(covered_indices) / len(preds)
    accuracy = accuracy_score(covered_labels, covered_preds)

    average = "binary" if n_class == 2 else "macro"
    precision = precision_score(covered_labels, covered_preds, average=average)
    recall = recall_score(covered_labels, covered_preds, average=average)
    f1 = f1_score(covered_labels, covered_preds, average=average)

    # label distribution and predicted label distribution
    label_dist = np.zeros(n_class, dtype=float)
    pred_label_dist = np.zeros(n_class, dtype=float)
    for c in range(n_class):
        label_dist[c] = np.sum(labels == c) / len(labels)
        pred_label_dist[c] = np.sum(covered_preds == c) / len(covered_preds)
    # class-specific label quality
    precision_perclass = precision_score(covered_labels, covered_preds, average=None)
    recall_perclass = recall_score(covered_labels, covered_preds, average=None)
    f1_perclass = f1_score(covered_labels, covered_preds, average=None)
    # confusion matrix
    cm = confusion_matrix(covered_labels, covered_preds)
    results = {
        "coverage": coverage,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "label_distribution": label_dist.tolist(),
        "pred_distribution": pred_label_dist.tolist(),
        "per_class_precision": precision_perclass.tolist(),
        "per_class_recall": recall_perclass.tolist(),
        "per_class_f1": f1_perclass.tolist(),
        "confusion_matrix": cm
    }
    return results


def evaluate_disc_model(disc_model, test_dataset):
    if disc_model is None:
        return {
            "acc": np.nan,
            "auc": np.nan,
            "f1": np.nan
        }
    if isinstance(disc_model, wrench.basemodel.BaseClassModel):
        y_pred = disc_model.predict(test_dataset)
        y_probs = disc_model.predict_proba(test_dataset)
    else:
        y_pred = disc_model.predict(test_dataset.features)
        y_probs = disc_model.predict_proba(test_dataset.features)

    test_acc = accuracy_score(test_dataset.labels, y_pred)
    if test_dataset.n_class == 2:
        test_auc = roc_auc_score(test_dataset.labels, y_probs[:, 1])
        test_f1 = f1_score(test_dataset.labels, y_pred)
    else:
        test_auc = roc_auc_score(test_dataset.labels, y_probs, average="macro", multi_class="ovo")
        test_f1 = f1_score(test_dataset.labels, y_pred, average="macro")

    results = {
        "acc": test_acc,
        "auc": test_auc,
        "f1": test_f1
    }
    return results

def evaluate_resnet(disc_model, test_loader, device):
    disc_model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = disc_model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_predictions)
    test_auc = roc_auc_score(all_labels, all_probs)
    test_f1 = f1_score(all_labels, all_predictions)
    results = {
        "acc": test_acc,
        "auc": test_auc,
        "f1": test_f1
    }
    return results


def evaluate_golden_baseline(train_dataset, valid_dataset, test_dataset, args):
    # train an end model use ground truth labels
    ys = np.array(train_dataset.labels)
    ys_onehot = np.zeros((len(ys), train_dataset.n_class))
    ys_onehot[range(len(ys_onehot)), ys] = 1
    if args.end_model == "bert":
        device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
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
            dataset_train=train_dataset,
            y_train=ys_onehot,
            dataset_valid=valid_dataset,
            evaluation_step=10,
            metric='acc',
            patience=10,
            device=device
        )
        test_perf = evaluate_disc_model(disc_model, test_dataset)

    elif args.end_model == "resnet-50":
        # Load the pre-trained ResNet-50 model
        disc_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Modify the final layer to match the number of classes in your dataset
        num_classes = train_dataset.n_class  # Change this to the number of classes in your dataset
        disc_model.fc = nn.Linear(disc_model.fc.in_features, num_classes)

        train_dataset_ = FeaturesDataset(train_dataset.images, train_dataset.labels)
        valid_dataset_ = FeaturesDataset(valid_dataset.images, valid_dataset.labels)
        test_dataset_ = FeaturesDataset(test_dataset.images, test_dataset.labels)
        train_loader = DataLoader(train_dataset_, batch_size=64, shuffle=True)
        valid_loader = DataLoader(valid_dataset_, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset_, batch_size=64, shuffle=False)

        # Train the resnet-50 model
        device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
        torch.cuda.empty_cache()
        disc_model = train_resnet(disc_model, train_loader, valid_loader,
                                  n_epochs=50, lr=1e-4, weight_decay=1e-5, device=device)
        test_perf = evaluate_resnet(disc_model, test_loader, device)

    else:
        disc_model = train_disc_model(model_type=args.end_model,
                                      distance_metric=args.distance_metric,
                                      xs_tr=train_dataset.features,
                                      ys_tr_soft=ys_onehot,
                                      valid_dataset=valid_dataset,
                                      use_soft_labels=args.use_soft_labels,
                                      tune_end_model=args.tune_end_model,
                                      tune_metric=args.tune_metric)
        test_perf = evaluate_disc_model(disc_model, test_dataset)

    return test_perf




