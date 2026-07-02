# !/usr/bin/env python
# coding:utf-8
"""
Tencent is pleased to support the open source community by making NeuralClassifier available.
Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the MIT License (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at
http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for thespecific language governing permissions and limitations under
the License.
"""

import sys
import time

import torch
from torch.utils.data import DataLoader

import util
from config import Config
from dataset.classification_dataset import ClassificationDataset
from dataset.collator import ClassificationCollator
from dataset.collator import ClassificationType
from dataset.collator import FastTextCollator
from evaluate.classification_evaluate import \
    ClassificationEvaluator as cEvaluator
from model.classification.drnn import DRNN
from model.classification.fasttext import FastText
from model.classification.textcnn import TextCNN
from model.classification.textvdcnn import TextVDCNN
from model.classification.textrnn import TextRNN
from model.classification.textrcnn import TextRCNN
from model.classification.transformer import Transformer
from model.classification.dpcnn import DPCNN
from model.classification.attentive_convolution import AttentiveConvNet
from model.classification.region_embedding import RegionEmbedding
from model.classification.hmcn import HMCN
from model.model_util import get_optimizer, get_hierar_relations
from util import ModeType
import itertools
from evaluate.classification_evaluate import ClassificationEvaluator as cEvaluator

import numpy as np
from sklearn.metrics import roc_auc_score

ClassificationDataset, ClassificationCollator, FastTextCollator, cEvaluator,
FastText, TextCNN, TextRNN, TextRCNN, DRNN, TextVDCNN, Transformer, DPCNN,
AttentiveConvNet, RegionEmbedding


def get_classification_model(model_name, dataset, conf):
    model = globals()[model_name](dataset, conf)
    model = model.cuda(conf.device) if conf.device.startswith("cuda") else model
    return model


def load_checkpoint(file_name, conf, model, optimizer):
    checkpoint = torch.load(file_name)
    conf.train.start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def eval(conf):
    logger = util.Logger(conf)
    model_name = conf.model_name
    dataset_name = "ClassificationDataset"
    collate_name = "FastTextCollator" if model_name == "FastText" \
        else "ClassificationCollator"

    test_dataset = globals()[dataset_name](conf, conf.data.test_json_files)
    collate_fn = globals()[collate_name](conf, len(test_dataset.label_map))
    test_data_loader = DataLoader(
        test_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    empty_dataset = globals()[dataset_name](conf, [])
    model = get_classification_model(model_name, empty_dataset, conf)
    optimizer = get_optimizer(conf, model)
    load_checkpoint(conf.eval.model_dir, conf, model, optimizer)
    model.eval()
    is_multi = False
    if conf.task_info.label_type == ClassificationType.MULTI_LABEL:
        is_multi = True
    predict_probs = []
    standard_labels = []
    evaluator = cEvaluator(conf.eval.dir)
    for batch in test_data_loader:
        if model_name == "HMCN":
            (global_logits, local_logits, logits) = model(batch)
        else:
            logits = model(batch)
        if not is_multi:
            result = torch.nn.functional.softmax(logits, dim=1).cpu().tolist()
        else:
            result = torch.sigmoid(logits).cpu().tolist()
        predict_probs.extend(result)
        standard_labels.extend(batch[ClassificationDataset.DOC_LABEL_LIST])
    (_, precision_list, recall_list, fscore_list, right_list,
     predict_list, standard_list) = \
        evaluator.evaluate(
            predict_probs, standard_label_ids=standard_labels, label_map=empty_dataset.label_map,
            threshold=conf.eval.threshold, top_k=conf.eval.top_k,
            is_flat=conf.eval.is_flat, is_multi=is_multi)
    logger.warn(
        "Performance is precision: %f, "
        "recall: %f, fscore: %f, right: %d, predict: %d, standard: %d." % (
            precision_list[0][cEvaluator.MICRO_AVERAGE],
            recall_list[0][cEvaluator.MICRO_AVERAGE],
            fscore_list[0][cEvaluator.MICRO_AVERAGE],
            right_list[0][cEvaluator.MICRO_AVERAGE],
            predict_list[0][cEvaluator.MICRO_AVERAGE],
            standard_list[0][cEvaluator.MICRO_AVERAGE]))
    evaluator.save()


def generate_eval_results(sublogit_E_X_Y, evaluator, standard_labels, conf, label_map):
    proba = torch.sigmoid(sublogit_E_X_Y).mean(0).cpu().tolist()

    (_, precision_list, recall_list, fscore_list, right_list,
    predict_list, standard_list) = \
        evaluator.evaluate(
            proba, standard_label_ids=standard_labels, label_map=label_map,
            threshold=conf.eval.threshold, top_k=conf.eval.top_k,
            is_flat=conf.eval.is_flat, is_multi=True)
    results = { "micro_fscore": fscore_list[0][cEvaluator.MICRO_AVERAGE],
                    "precision": precision_list[0][cEvaluator.MICRO_AVERAGE],
                    "recall": recall_list[0][cEvaluator.MICRO_AVERAGE],
                    "right": right_list[0][cEvaluator.MICRO_AVERAGE],
                    "predict": predict_list[0][cEvaluator.MICRO_AVERAGE],
                    "standard": standard_list[0][cEvaluator.MICRO_AVERAGE],
                    "macro_fscore": fscore_list[0][cEvaluator.MACRO_AVERAGE],
                        "precision_list": precision_list, 
                    "recall_list": recall_list, 
                    "fscore_list": fscore_list, 
                    "right_list": right_list,
                    "predict_list":predict_list, 
                    "standard_list":standard_list}
    return proba, results, fscore_list[0][cEvaluator.MICRO_AVERAGE]


def generate_topk_eval_results(proba, evaluator, standard_labels, conf, label_map, results):
    for topk in [1,3,5,10]:
        (_, precision_list_k, recall_list_k, fscore_list_k, right_list_k,
            predict_list_k, standard_list_k) = \
        evaluator.evaluate(
            proba, standard_label_ids=standard_labels, label_map=label_map,
            threshold=conf.eval.threshold, top_k=topk,
            is_flat=conf.eval.is_flat, is_multi=True)
        results.update({"more_info_top%d" % topk : {"precision@%d" % topk: precision_list_k[0][cEvaluator.MICRO_AVERAGE], 
                                "recall@%d" % topk: recall_list_k[0][cEvaluator.MICRO_AVERAGE], 
                                "micro_fscore@%d" % topk: fscore_list_k[0][cEvaluator.MICRO_AVERAGE],
                                "macro_fscore@%d" % topk: fscore_list_k[0][cEvaluator.MACRO_AVERAGE],
                                "precision_list_top%d" % topk: fscore_list_k, 
                                "recall_list_top%d" % topk: right_list_k,
                                "fscore_list_top%d" % topk: fscore_list_k, 
                                "right_list_top%d" % topk: right_list_k,
                                "predict_list_top%d" % topk: predict_list_k, 
                                "standard_list_top%d" % topk: standard_list_k}})
    return results


def evalresults(logits_E_X_Y, standard_labels, conf, label_map, cur_acqIdx=None):

    evaluator = cEvaluator(conf.eval.dir)
    proba, results, _ = generate_eval_results(logits_E_X_Y, evaluator, standard_labels, conf, label_map)
    results = generate_topk_eval_results(proba, evaluator, standard_labels, conf, label_map, results)
       
    return results



def ndcg(rel_true, rel_pred, p=None, form="linear"):
    """ Returns normalized Discounted Cumulative Gain
    Args:
        rel_true (1-D Array): relevance lists for particular user, (n_songs,)
        rel_pred (1-D Array): predicted relevance lists, (n_pred,)
        p (int): particular rank position
        form (string): two types of nDCG formula, 'linear' or 'exponential'
    Returns:
        ndcg (float): normalized discounted cumulative gain score [0, 1]
    """
    rel_true = np.sort(rel_true)[::-1]
    p = min(len(rel_true), min(len(rel_pred), p))
    discount = 1 / (np.log2(np.arange(p) + 2))

    if form == "linear":
        idcg = np.sum(rel_true[:p] * discount)
        dcg = np.sum(rel_pred[:p] * discount)
    elif form == "exponential" or form == "exp":
        idcg = np.sum([2**x - 1 for x in rel_true[:p]] * discount)
        dcg = np.sum([2**x - 1 for x in rel_pred[:p]] * discount)
    else:
        raise ValueError("Only supported for two formula, 'linear' or 'exp'")

    return dcg / idcg




if __name__ == '__main__':
    config = Config(config_file=sys.argv[1])
    eval(config)
