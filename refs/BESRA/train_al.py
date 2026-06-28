#!/usr/bin/env python
#coding:utf-8


import os, gc
import shutil
import sys
import time

import torch
from torch.utils.data import DataLoader

import util
from config import Config
from dataset.classification_dataset import ClassificationDataset
from dataset.collator import ClassificationCollator
from dataset.collator import FastTextCollator
from dataset.collator import ClassificationType
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
from model.classification.bert import BERT
from model.classification.attentive_convolution import AttentiveConvNet
from model.classification.region_embedding import RegionEmbedding
from model.classification.hmcn import HMCN
from model.loss import ClassificationLoss
from model.model_util import get_optimizer, get_hierar_relations
from util import ModeType
from pytorchtools import EarlyStopping


from eval import evalresults

import random
import pdb
import numpy as np
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from acquisition import (
    random_queries_batch,
    reshape_3d_logits,
    least_confidence_scores,
    bald_scores,
    max_entropy_acquisition_function,
    besra,
    MaximumLossReductionMaximalConfidence,
    AdaptiveActiveLearning,
    MultilabelWithAuxiliaryLearner,
    CostSensitiveReferencePairEncoding,
    audi,
    CVIRS,
    GPB2M
 
)

ClassificationDataset, ClassificationCollator, FastTextCollator, ClassificationLoss, cEvaluator
FastText, TextCNN, TextRNN, TextRCNN, DRNN, TextVDCNN, Transformer, DPCNN, AttentiveConvNet, RegionEmbedding
BERT

def get_data_loader(benchmarks, collate_name, conf, label_length):
    """Get data loader: Train, Validate, Test
    """

    collate_fn = globals()[collate_name](conf, label_length)

    train_data_loader = DataLoader(
        benchmarks["train"], batch_size=conf.train.batch_size, shuffle=True,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    validate_data_loader = DataLoader(
        benchmarks["dev"], batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    # test_dataset = globals()[dataset_name](conf, conf.data.test_json_files)
    test_data_loader = DataLoader(
        benchmarks["test"], batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    unlabel_data_loader = DataLoader(
        benchmarks["unlabel"], batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    return train_data_loader, validate_data_loader, test_data_loader, unlabel_data_loader


def get_classification_model(model_name, dataset, conf):
    """Get classification model from configuration
    """
    model = globals()[model_name](dataset, conf)
    model = model.cuda(conf.device) if conf.device.startswith("cuda") else model
    return model



class ClassificationTrainer(object):
    def __init__(self, label_map, logger, evaluator, conf, loss_fn):
        self.label_map = label_map
        self.logger = logger
        self.evaluator = evaluator
        self.conf = conf
        self.loss_fn = loss_fn
        if self.conf.task_info.hierarchical:
            self.hierar_relations = get_hierar_relations(
                    self.conf.task_info.hierar_taxonomy, label_map)

    def train(self, data_loader, model, optimizer, stage, epoch, scheduler=None):
        if self.conf.model_name == "BERT" and epoch != 0:
            optimizer = get_optimizer(self.conf, model)
        model.update_lr(optimizer, epoch)
        model.train()
        return self.run(data_loader, model, optimizer, stage, epoch,
                        ModeType.TRAIN, scheduler)

    def eval(self, data_loader, model, optimizer, stage, epoch, mcflag=False):
        model.eval()
        if mcflag:
            enable_dropout(model)
            model.dropout.p = self.conf.al_info.emsemble_info.emsemble_dropout
        with torch.no_grad():
            return self.run(data_loader, model, optimizer, stage, epoch)

    def run(self, data_loader, model, optimizer, stage,
            epoch, mode=ModeType.EVAL, scheduler=None):
        is_multi = False
        # multi-label classifcation
        if self.conf.task_info.label_type == ClassificationType.MULTI_LABEL:
            is_multi = True
        predict_probs = []
        standard_labels = []
        logits_li = []
        num_batch = data_loader.__len__()
        total_loss = 0.
        

        for step, batch in enumerate(tqdm(data_loader)):
        # for batch in data_loader:
            # hierarchical classification using hierarchy penalty loss
            if self.conf.task_info.hierarchical:
                logits = model(batch)
                linear_paras = model.linear.weight
                is_hierar = True
                used_argvs = (self.conf.task_info.hierar_penalty, linear_paras, self.hierar_relations)
                loss = self.loss_fn(
                    logits,
                    batch[ClassificationDataset.DOC_LABEL].to(self.conf.device),
                    is_hierar,
                    is_multi,
                    *used_argvs)
            # hierarchical classification with HMCN
            elif self.conf.model_name == "HMCN":
                (global_logits, local_logits, logits) = model(batch)
                loss = self.loss_fn(
                    global_logits,
                    batch[ClassificationDataset.DOC_LABEL].to(self.conf.device),
                    False,
                    is_multi)
                loss += self.loss_fn(
                    local_logits,
                    batch[ClassificationDataset.DOC_LABEL].to(self.conf.device),
                    False,
                    is_multi)
            # flat classificaiton
            else:
                logits = model(batch) 
                loss = self.loss_fn(
                    logits,
                    batch[ClassificationDataset.DOC_LABEL].to(self.conf.device),
                    False,
                    is_multi)
            if mode == ModeType.TRAIN:
                if scheduler:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    optimizer.step()
                    scheduler.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                continue
            total_loss += loss.item()
            if not is_multi:
                result = torch.nn.functional.softmax(logits, dim=1).cpu().tolist()
            else:
                result = torch.sigmoid(logits).cpu().tolist()
        
            logits_li.append(logits)
     
            predict_probs.extend(result)
            standard_labels.extend(batch[ClassificationDataset.DOC_LABEL_LIST])

        if mode == ModeType.EVAL:
            total_loss = total_loss / num_batch
            ## predict_list: list of dicts, each dict is the predict label of each sample
            ## standard_list: list of dicts, each dict is the original label of each sample
            ## right_list: list of dicts, each dict is the correct label of each sample
            (_, precision_list, recall_list, fscore_list, right_list,
             predict_list, standard_list) = \
                self.evaluator.evaluate(
                    predict_probs, standard_label_ids=standard_labels, label_map=self.label_map,
                    threshold=self.conf.eval.threshold, top_k=self.conf.eval.top_k,
                    is_flat=self.conf.eval.is_flat, is_multi=is_multi)
            # pdb.set_trace()
            # precision_list[0] save metrics of flat classification
            # precision_list[1:] save metrices of hierarchical classification
            self.logger.warn(
                "%s performance at epoch %d is precision: %f, "
                "recall: %f, fscore: %f, macro-fscore: %f, right: %d, predict: %d, standard: %d.\n"
                "Loss is: %f." % (
                    stage, epoch, precision_list[0][cEvaluator.MICRO_AVERAGE],
                    recall_list[0][cEvaluator.MICRO_AVERAGE],
                    fscore_list[0][cEvaluator.MICRO_AVERAGE],
                    fscore_list[0][cEvaluator.MACRO_AVERAGE],
                    right_list[0][cEvaluator.MICRO_AVERAGE],
                    predict_list[0][cEvaluator.MICRO_AVERAGE],
                        standard_list[0][cEvaluator.MICRO_AVERAGE], total_loss))
            ## record txt file to save the best performance

            results = {"stage": stage, 
                         "epoch": epoch,
                         "micro_fscore": fscore_list[0][cEvaluator.MICRO_AVERAGE],
                         "precision": precision_list[0][cEvaluator.MICRO_AVERAGE],
                         "recall": recall_list[0][cEvaluator.MICRO_AVERAGE],
                         "right": right_list[0][cEvaluator.MICRO_AVERAGE],
                         "predict": predict_list[0][cEvaluator.MICRO_AVERAGE],
                         "standard": standard_list[0][cEvaluator.MICRO_AVERAGE],
                         "macro_fscore": fscore_list[0][cEvaluator.MACRO_AVERAGE],
                        "loss": total_loss,
                        "precision_list": precision_list, 
                        "recall_list": recall_list, 
                        "fscore_list": fscore_list, 
                        "right_list": right_list,
                        "predict_list":predict_list, 
                        "standard_list":standard_list}

            for topk in [1,3,5,10]:
                (_, precision_list_k, recall_list_k, fscore_list_k, right_list_k,
                predict_list_k, standard_list_k) = \
                    self.evaluator.evaluate(
                        predict_probs, standard_label_ids=standard_labels, label_map=self.label_map,
                        threshold=self.conf.eval.threshold, top_k=topk,
                        is_flat=self.conf.eval.is_flat, is_multi=is_multi)
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
      
            return fscore_list[0][cEvaluator.MICRO_AVERAGE], results, [item for sublist in logits_li for item in sublist], standard_labels, total_loss


def write_result_to_txt(save_path, results):
    fileName = save_path + "/result.txt"
    file = open(fileName, "a+")
    results = str(results) + "\n"
    file.writelines([results])
    file.close()

def load_checkpoint(file_name, model, optimizer):
    checkpoint = torch.load(file_name)
    best_performance = checkpoint["best_performance"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return best_performance


def save_checkpoint(state, file_prefix):
    file_name = file_prefix + "_best"
    torch.save(state, file_name)


def train(logger, conf, benchmarks, label_length, acqIdx=0, train_logit_save=False):

    if not os.path.exists(conf.checkpoint_dir):
        os.makedirs(conf.checkpoint_dir)

    model_name = conf.model_name
    dataset_name = "ClassificationDataset"
    collate_name = "FastTextCollator" if model_name == "FastText" \
        else "ClassificationCollator"
    train_data_loader, validate_data_loader, test_data_loader, unlabel_data_loader = \
        get_data_loader(benchmarks, collate_name, conf, label_length)
    empty_dataset = globals()[dataset_name](conf, [], mode="train")
    model = get_classification_model(model_name, empty_dataset, conf)
    loss_fn = globals()["ClassificationLoss"](
        label_size=label_length, loss_type=conf.train.loss_type)
    optimizer = get_optimizer(conf, model)
    evaluator = cEvaluator(conf.eval.dir)
    trainer = globals()["ClassificationTrainer"](
        empty_dataset.label_map, logger, evaluator, conf, loss_fn)

    if conf.model_name == "BERT":
        t_total = int(len(train_data_loader) / conf.train.batch_size / conf.bert.gradient_accumulation_steps * conf.train.num_epochs)
        warmup_steps = 0.05 * t_total
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)


    best_epoch = 1
    best_performance = -0.1
    model_file_prefix = conf.checkpoint_dir + "/" + model_name
    if conf.early_stopping:
        early_stopping = EarlyStopping(patience=conf.early_stopping_patience, verbose=conf.early_stopping_verbose)
    for epoch in range(conf.train.start_epoch,
                       conf.train.start_epoch + conf.train.num_epochs):
        start_time = time.time()
        trainer.train(train_data_loader, model, optimizer, "Train", epoch, scheduler if conf.model_name == 'BERT' else None)
        # trainer.eval(train_data_loader, model, optimizer, "Train", epoch)
        performance, _, _, _, total_loss = trainer.eval(
            validate_data_loader, model, optimizer, "Validate", epoch)
        # trainer.eval(test_data_loader, model, optimizer, "test", epoch)
        if conf.early_stopping:
            early_stopping.path = model_file_prefix
            early_stopping.save_info = {
                    'epoch': epoch,
                    'model_name': model_name,
                    'state_dict': model.state_dict(),
                    'best_performance': best_performance,
                    'optimizer': optimizer.state_dict(),
                }
            early_stopping(performance, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        else:
            if performance > best_performance:  # record the best model
                best_epoch = epoch
                best_performance = performance
                save_checkpoint({
                    'epoch': epoch,
                    'model_name': model_name,
                    'state_dict': model.state_dict(),
                    'best_performance': best_performance,
                    'optimizer': optimizer.state_dict(),
                }, model_file_prefix)
        time_used = time.time() - start_time
        logger.info("Epoch %d cost time: %d second" % (epoch, time_used))

    load_checkpoint(model_file_prefix + "_best" , model, optimizer)
    _, results, test_logits, standard_labels, _ = trainer.eval(test_data_loader, model, optimizer, "Best test", best_epoch)

    if train_logit_save:
        _, _, train_logits, train_standard_labels, _ = trainer.eval(train_data_loader, model, optimizer, "Best train", best_epoch)
        conf.al_info.train_logit_save.train_logits = train_logits
        conf.al_info.train_logit_save.train_standard_labels = train_standard_labels

    ensemble_logitsInfo = []

    if conf.full_train:
        return performance, results, ensemble_logitsInfo, standard_labels
    
    _, _, unlabel_logits, _, _ = trainer.eval(unlabel_data_loader, model, optimizer, "Unlabel", best_epoch)
    test_logits_li = []

    if conf.al_info.emsemble_info.emopt:
        ensemble_logitsInfo.append(unlabel_logits)
        test_logits_li.append(test_logits)
        # ensemble_logitsInfo.append({"unlabel": unlabel_logits, "test":test_logits})
        if conf.al_info.emsemble_info.emsemble_method == "mc_dropout":
            for eachMCM in range(conf.al_info.emsemble_info.emsemble_num - 1):
                sub_output_dir = os.path.join(conf.al_output_dir, str(eachMCM + 2))
                if not os.path.exists(sub_output_dir):
                    os.makedirs(sub_output_dir)
                _, test_results, test_logits, _, _ = trainer.eval(test_data_loader, model, optimizer, "Best test", best_epoch, mcflag=conf.al_info.emsemble_info.emopt)
                _, _, unlabel_logits, _, _ = trainer.eval(unlabel_data_loader, model, optimizer, "Unlabel", best_epoch, mcflag=conf.al_info.emsemble_info.emopt)
                write_result_to_txt(sub_output_dir, {"Acquired sample": acqIdx, "Best test": test_results})
                ensemble_logitsInfo.append(unlabel_logits)
                test_logits_li.append(test_logits)
                # ensemble_logitsInfo.append({"unlabel": unlabel_logits, "test":test_logits})
            ensemble_logitsInfo = {"unlabel": ensemble_logitsInfo, "test":test_logits_li}
        elif conf.al_info.emsemble_info.emsemble_method == "deep_en":
            ensemble_logitsInfo = {"unlabel": unlabel_logits, "test":test_logits}
    else:
        ensemble_logitsInfo.append(unlabel_logits)
    
    return performance, results, ensemble_logitsInfo, standard_labels



def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train(True)


def train_al(conf):

    conf.al_output_dir = os.path.join(conf.al_output_dir, conf.task_info.label_type, conf.taskname, conf.model_name)
    conf.al_output_dir = os.path.join(conf.al_output_dir, str(conf.al_info.exp_seed))

    if conf.full_train:
        prefixname = "%s" % (
                    'Fulltrain_ep%s' % (str(conf.train.num_epochs))
                )
    else:
        prefixname = "%s%s%s%s%s%s%s" % (
            "%s" % (conf.al_info.query_strategy),
            "Dyopt" if conf.al_info.dynamicsvalopt else "",
            "Reopt" if conf.al_info.retrainopt else "",
            "%s_e%s" % (conf.al_info.emsemble_info.emsemble_method, conf.al_info.emsemble_info.emsemble_num) if conf.al_info.emsemble_info.emopt else "",
            "Xp%sSp%sAl%sBe%sCl%s" % (conf.al_info.elr_info.xprime, conf.al_info.elr_info.xpsplit, conf.al_info.elr_info.alpha, conf.al_info.elr_info.beta, conf.al_info.cl_method) if conf.al_info.query_strategy == "besra" else "",
            '_b%s' % (str(conf.al_info.acq_batchsize))
        )  

    conf.al_output_dir = os.path.join(conf.al_output_dir, prefixname)
    conf.checkpoint_dir = os.path.join(conf.al_output_dir, conf.checkpoint_dir)

    conf.log.logger_file =  os.path.join(conf.al_output_dir, conf.log.logger_file)

    if not os.path.exists(conf.al_output_dir):
        os.makedirs(conf.al_output_dir)

    logger = util.Logger(conf)

    dataset_name = "ClassificationDataset"

    traindev_dataset = globals()[dataset_name](
        conf, conf.data.train_json_files, generate_dict=True)

    label_length = len(traindev_dataset.label_map) 

    test_dataset = globals()[dataset_name](conf, conf.data.test_json_files)

    if conf.al_info.val_info.reducevalopt:
        valid_indices, train_indices = generate_random_indices(list(traindev_dataset), conf.al_info.val_info.val_seed, 
                        conf.al_info.val_info.valsize)
        train_dataset = subsetByIndices(list(traindev_dataset), train_indices)
        validate_dataset = subsetByIndices(list(traindev_dataset), valid_indices)
    else:
        train_dataset = list(traindev_dataset)
        validate_dataset = list(globals()[dataset_name](conf, conf.data.validate_json_files))

    if conf.full_train:
        conf.al_info.emsemble_info.emopt = False
        benchmarks = {
                "train": train_dataset,
                "dev": validate_dataset,
                "test": test_dataset,
                "unlabel": []
            }
        cur_acqIdx = len(train_dataset)
        start_time = time.time()
        _, results, ensemble_logitsInfo, _ = train(logger, conf, benchmarks, label_length, acqIdx=cur_acqIdx)
        end_time = time.time()
        resutls_txt = {"Full train samples": cur_acqIdx, "valid samples": str(len(validate_dataset))}
        resutls_txt.update({"Test res": results, "train time": end_time - start_time})
        write_result_to_txt(conf.al_output_dir, resutls_txt)
        exit(0)
    else:
        label_indices, unlabeled_indices = generate_random_indices(train_dataset, conf.al_info.exp_seed,
                conf.al_info.initial_label_size)
        labeled_dataset = subsetByIndices(train_dataset, label_indices)
        unlabel_dataset = subsetByIndices(train_dataset, unlabeled_indices)
    write_result_to_txt(conf.al_output_dir, {"Initial label pool": label_indices, "Initial unlabeled pool": unlabeled_indices,
                                              "configures": conf})
    benchmarks = {
                    "train": labeled_dataset,
                    "dev": validate_dataset,
                    "test": test_dataset,
                    "unlabel": unlabel_dataset
                }

    if conf.al_info.cont_train.contopt:
        acqIdx = conf.al_info.cont_train.acqIdx
    else:
        acqIdx = 0

    for eachAcq in range(acqIdx, conf.al_info.max_acq_label_size, conf.al_info.acq_batchsize) :
        cur_acqIdx = str(eachAcq)
        # pdb.set_trace()
        start_time = time.time()
        if conf.al_info.emsemble_info.emopt:
            if conf.al_info.emsemble_info.emsemble_method == "deep_en":
                ensemble_logitsInfo = []
                test_logitsLi = []
                for eachDM in range(conf.al_info.emsemble_info.emsemble_num):
                    sub_output_dir = os.path.join(conf.al_output_dir, str(eachDM + 1))
                    if not os.path.exists(sub_output_dir):
                        os.makedirs(sub_output_dir)
                    _, test_results, logitsInfo, standard_labels = train(logger, conf, benchmarks, label_length, acqIdx=cur_acqIdx)
                    write_result_to_txt(sub_output_dir, {"Acquired sample": cur_acqIdx, "Best test": test_results})
                    ensemble_logitsInfo.append(logitsInfo['unlabel'])
                    test_logitsLi.append(logitsInfo['test'])
                testlogits_E_X_Y = reshape_3d_logits(test_logitsLi)
            elif conf.al_info.emsemble_info.emsemble_method == "mc_dropout":
                _, results, ensemble_logitsInfo, standard_labels = train(logger, conf, benchmarks, label_length, acqIdx=cur_acqIdx)
                testlogits_E_X_Y = reshape_3d_logits(ensemble_logitsInfo["test"])
                ensemble_logitsInfo = ensemble_logitsInfo['unlabel']
            empty_dataset = globals()[dataset_name](conf, [], mode="train")
            results = evalresults(testlogits_E_X_Y, standard_labels, conf, empty_dataset.label_map, cur_acqIdx)
     
        else:
            _, results, ensemble_logitsInfo, _ = train(logger, conf, benchmarks, label_length, acqIdx=cur_acqIdx, train_logit_save=conf.al_info.train_logit_save.opt)
        end_time = time.time()
        resutls_txt = {"AcqIdx": cur_acqIdx}
        resutls_txt.update({"Test res": results, "train time": end_time - start_time})

  
        logits_E_X_Y = reshape_3d_logits(ensemble_logitsInfo)
        # pdb.set_trace()
        start_time = time.time()
        sampled_index = query_method(conf, logits_E_X_Y, cur_acqIdx, conf.al_info.acq_batchsize, unlabeled_indices)
        end_time = time.time()

        label_indices, unlabeled_indices = update_labeled_and_unlabeled_pool(sampled_index, label_indices,
                                                                                    unlabeled_indices)
        resutls_txt.update({"sampled_index": sampled_index, "len tr":len(labeled_dataset),"len va":len(validate_dataset),
                            "query time": end_time - start_time})
        # pdb.set_trace()
        labeled_dataset = subsetByIndices(train_dataset, label_indices)
        unlabel_dataset = subsetByIndices(train_dataset, unlabeled_indices)
        benchmarks = {
                    "train": labeled_dataset,
                    "dev": validate_dataset,
                    "test": test_dataset,
                    "unlabel": unlabel_dataset
                }
        write_result_to_txt(conf.al_output_dir, resutls_txt)



def update_labeled_and_unlabeled_pool(sampled_index, label_index, unlabeled_index):
    # print(sampled_index)
    label_index_new = label_index + sampled_index
    for x in sampled_index:
        unlabeled_index.remove(x)

    return label_index_new, unlabeled_index


def query_method(conf, ensemble_logitsInfo, cur_acqIdx, acq_bs, unlabeled_indices, sigmoid=False):
    if conf.al_info.query_strategy == 'rand':
        return random_queries_batch(conf, ensemble_logitsInfo, cur_acqIdx, acq_bs, unlabeled_indices)
    elif conf.al_info.query_strategy == 'lc':
        return least_confidence_scores(conf, ensemble_logitsInfo, acq_bs, unlabeled_indices)
    elif conf.al_info.query_strategy == 'bald':
        return bald_scores(conf, ensemble_logitsInfo, acq_bs, unlabeled_indices)
    elif conf.al_info.query_strategy == 'maxent':
        return max_entropy_acquisition_function(conf, ensemble_logitsInfo, acq_bs, unlabeled_indices)
    elif conf.al_info.query_strategy == 'besra':
        return besra(conf, ensemble_logitsInfo, acq_bs, unlabeled_indices)
    elif conf.al_info.query_strategy == 'mmc':
        return MaximumLossReductionMaximalConfidence(conf, ensemble_logitsInfo, acq_bs, unlabeled_indices)
    elif conf.al_info.query_strategy == 'adaptive':
        return AdaptiveActiveLearning(conf, ensemble_logitsInfo, acq_bs, unlabeled_indices)
    elif conf.al_info.query_strategy == 'shlr':
        return MultilabelWithAuxiliaryLearner(conf, ensemble_logitsInfo, acq_bs, unlabeled_indices)
    elif conf.al_info.query_strategy == 'csmlal':
        return CostSensitiveReferencePairEncoding(conf, ensemble_logitsInfo, acq_bs, unlabeled_indices)
    elif conf.al_info.query_strategy == 'audi':
        return audi(conf, ensemble_logitsInfo, acq_bs, unlabeled_indices, cur_acqIdx)
    elif conf.al_info.query_strategy == 'cvirs':
        return CVIRS(conf, ensemble_logitsInfo, acq_bs, unlabeled_indices, cur_acqIdx)
    elif conf.al_info.query_strategy == 'gpb2m':
        return GPB2M(conf, ensemble_logitsInfo, acq_bs, unlabeled_indices, cur_acqIdx)
    



def subsetByIndices(features, indices):
    return [features[i] for i in indices]

def generate_random_indices(train_dataset, seed, size):
    max_len = len(train_dataset)
    indices = range(max_len)
    random.seed(seed)
    samples_indices = random.sample(indices, size)
    samples_indices_set = set(samples_indices)
    available_indices = [i for i in indices if i not in samples_indices_set]
    return samples_indices, available_indices


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    config = Config(config_file=sys.argv[1])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.train.visible_device_list)
    config.al_info.exp_seed = int(config.al_info.exp_seed)
    config.al_info.val_info.val_seed = int(config.al_info.val_info.val_seed)
    set_seed(config.al_info.exp_seed)
    train_al(config)
    gc.collect() 
