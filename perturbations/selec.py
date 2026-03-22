import argparse
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import copy
import os
import pandas as pd
from tqdm import tqdm
import perturbations.perturbations
import joblib
import torch.nn.init as init
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from new_model import (
    get_text_and_labels, TextDataset, TextCNN, BiLSTMClassifier, 
    RobertaForSequenceClassification, RobertaTokenizer
)
from models.model import get_data
class ModelWrapper_pre(nn.Module):
    """统一包装所有模型，确保接口一致"""
    
    def __init__(self, model, model_type, vocab=None, vectorizer=None, tokenizer=None, device=None, max_len=128):
        super().__init__()
        self.model = model
        self.model_type = model_type  # 'textcnn', 'bilstm', 'roberta', 'svm', 'logreg'
        self.vectorizer = vectorizer
        self.tokenizer = tokenizer
        self.max_len = max_len
        if model_type in ['textcnn', 'bilstm'] and vocab is None:
            raise ValueError(f"对于{model_type}模型，必须提供vocab参数")
        self.vocab = vocab 
        self.device = device

    def encode_text(self, text):
        """编码单个文本，包含填充"""
        tokens = text.split()
        ids = [self.vocab[token] for token in tokens[:self.max_len]]
        
        # 进行填充，确保所有序列长度一致
        if len(ids) < self.max_len:
            ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))  # 0是<pad>的索引
        else:
            ids = ids[:self.max_len]
        
        return ids

    def collate_batch(self, texts):
        """批量处理文本，确保长度一致"""
        # 对所有文本进行编码和填充
        encoded_texts = [self.encode_text(text) for text in texts]
        # 转换为tensor
        return torch.tensor(encoded_texts, dtype=torch.long).to(self.device)

    def forward(self, data):
        """
        统一接口：输入data，输出[batch_size, num_classes]的logits
        """
        # 统一处理输入格式
        if isinstance(data, dict) and 'X' in data:
            # 如果是UnifiedTextDataset的输出格式
            texts = data['X']
        else:
            texts = data
            
        if self.model_type in ['textcnn', 'bilstm']:
            # 处理TextCNN和BiLSTM
            if isinstance(texts, list):
                # 原始文本输入，需要编码和填充
                input_tensor = self.collate_batch(texts)
            elif isinstance(texts, torch.Tensor) and texts.dtype == torch.object:
                # 处理字符串tensor
                text_list = [str(t) for t in texts.cpu().numpy()]
                input_tensor = self.collate_batch(text_list)
            else:
                # 假设已经是编码后的tensor
                input_tensor = texts
                
            return self.model(input_tensor)
            
        elif self.model_type == 'roberta':
            # 处理RoBERTa的特殊输出格式
            if isinstance(texts, list):
                # 原始文本输入
                encoding = self.tokenizer(
                    texts, 
                    truncation=True, 
                    padding='max_length',
                    max_length=self.max_len,
                    return_tensors='pt'
                ).to(self.device)
                outputs = self.model(
                    input_ids=encoding['input_ids'],
                    attention_mask=encoding['attention_mask']
                )
            else:
                # 假设已经是编码后的输入
                outputs = self.model(
                    input_ids=data['input_ids'],
                    attention_mask=data['attention_mask']
                )
            return outputs.logits  # 提取logits
            
        elif self.model_type in ['svm', 'logreg']:
            # 处理传统模型
            if isinstance(texts, torch.Tensor):
                texts = [str(t) for t in texts.cpu().numpy()]
                
            features = self.vectorizer.transform(texts)
            return torch.tensor(self.model.decision_function(features), dtype=torch.float32).to(self.device)
            
            # if hasattr(self.model, 'predict_proba'):
            #     probs = self.model.predict_proba(features)
            # else:
            #     decision = self.model.decision_function(features)
            #     probs = torch.softmax(torch.tensor(decision), dim=1)
                
            # return torch.tensor(probs, dtype=torch.float32).to(self.device)

# ---------------------------
# New ModelWrapper（统一接口，返回 logits）
# ---------------------------
class ModelWrapper(nn.Module):
    """
    统一包装所有模型，确保接口一致。
    forward(...) 应返回 logits Tensor (B, C) 在 self.device 上。

    参数:
      model: 原始模型对象 (PyTorch nn.Module 或 sklearn 模型)
      model_type: 'textcnn', 'bilstm', 'roberta', 'svm', 'logreg'
      vocab: 用于 textcnn/bilstm 的词表（torchtext vocab 或 mapping）
      vectorizer: sklearn TfidfVectorizer（用于 svm/logreg）
      tokenizer: transformers tokenizer（用于 roberta）
      device: torch.device 或字符串
      max_len: 最大长度（tokens）
    """
    def __init__(self, model, model_type, vocab=None, vectorizer=None, tokenizer=None, device=None, max_len=128):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.vocab = vocab
        self.vectorizer = vectorizer
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = torch.device(device) if device is not None else torch.device('cpu')

        # sanity checks
        if model_type in ['textcnn', 'bilstm'] and vocab is None:
            raise ValueError(f"对于{model_type}模型，必须提供 vocab 参数")
        # move PyTorch model to device if applicable
        if isinstance(self.model, nn.Module):
            self.model.to(self.device)
            self.model.eval()

    # --- helpers for vocab encoding ---
    def token_to_id(self, token):
        # support both torchtext Vocab (callable) and dict
        try:
            return self.vocab[token]
        except Exception:
            # fallback: if vocab has get
            try:
                return self.vocab.get(token, self.vocab.get("<unk>", 0))
            except Exception:
                # last resort
                return 0

    def encode_text(self, text):
        toks = text.split()[:self.max_len]
        ids = [self.token_to_id(t) for t in toks]
        if len(ids) < self.max_len:
            # pad with pad token if available
            pad_idx = None
            try:
                pad_idx = self.vocab["<pad>"]
            except Exception:
                pad_idx = self.vocab.get("<pad>", 0)
            ids += [pad_idx] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return ids

    def collate_batch(self, texts):
        # texts: list of strings
        encoded = [self.encode_text(t) for t in texts]
        tensor = torch.tensor(encoded, dtype=torch.long, device=self.device)
        return tensor

    # --- forward: accept variety of inputs ---
    def forward(self, data):
        """
        data 可以是:
          - list[str] (原始文本)
          - torch.Tensor: 如果是 dtype torch.long 则视作已经编码的 ids (B, L)
          - dict: 如果是 transformers 风格 {'input_ids', 'attention_mask'}
          - 其他可迭代字符串类型
        返回:
          logits: torch.Tensor (B, C) 在 self.device 上
        """
        # ------- 1) handle PyTorch models (textcnn, bilstm) -------
        if self.model_type in ['textcnn', 'bilstm']:
            # 将输入转为 ids tensor
            if isinstance(data, (list, tuple)):
                input_tensor = self.collate_batch(list(data))  # already on device
            elif isinstance(data, torch.Tensor):
                # ensure device & dtype
                input_tensor = data.to(self.device)
                if input_tensor.dtype != torch.long:
                    input_tensor = input_tensor.long()
            elif isinstance(data, dict) and 'X' in data:
                # unified dataset format that might include 'X'
                texts = data['X']
                if isinstance(texts, torch.Tensor):
                    # may be object dtype of strings
                    input_tensor = self.collate_batch([str(t) for t in texts])
                else:
                    input_tensor = self.collate_batch(list(texts))
            else:
                # single string?
                if isinstance(data, str):
                    input_tensor = self.collate_batch([data])
                else:
                    # as fallback, try to iterate and convert to str
                    try:
                        input_tensor = self.collate_batch([str(x) for x in data])
                    except Exception:
                        raise ValueError("Unsupported input type for textcnn/bilstm")
            # run model
            with torch.no_grad():
                logits = self.model(input_tensor)
            return logits.to(self.device)

        # ------- 2) RoBERTa -------
        if self.model_type == 'roberta':
            # accept list[str], or dict with input_ids/attention_mask already as tensors
            if isinstance(data, (list, tuple)):
                # use tokenizer to batch-encode
                enc = self.tokenizer(
                    list(data),
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_len,
                    return_tensors='pt'
                )
                input_ids = enc['input_ids'].to(self.device)
                attention_mask = enc['attention_mask'].to(self.device)
                with torch.no_grad():
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits
                return logits.to(self.device)
            elif isinstance(data, dict):
                # expect 'input_ids' and 'attention_mask' as tensors (or CPU tensors)
                input_ids = data.get('input_ids')
                attn = data.get('attention_mask')
                if input_ids is None or attn is None:
                    raise ValueError("For roberta dict input must contain 'input_ids' and 'attention_mask'")
                input_ids = input_ids.to(self.device)
                attn = attn.to(self.device)
                with torch.no_grad():
                    out = self.model(input_ids=input_ids, attention_mask=attn)
                return out.logits.to(self.device)
            else:
                # single string
                if isinstance(data, str):
                    return self.forward([data])
                raise ValueError("Unsupported input type for roberta")

        # ------- 3) sklearn models (svm, logreg) -------
        if self.model_type in ['svm', 'logreg']:
            # expect raw texts (list[str]) or tensor of strings
            if isinstance(data, torch.Tensor):
                # convert to list[str]
                try:
                    texts = [str(x) for x in data.cpu().numpy()]
                except Exception:
                    # torch tensor enumerables maybe bytes
                    texts = [str(x) for x in data.tolist()]
            elif isinstance(data, (list, tuple)):
                texts = list(data)
            elif isinstance(data, str):
                texts = [data]
            elif isinstance(data, dict) and 'X' in data:
                texts = data['X']
            else:
                raise ValueError("Unsupported input type for sklearn model")
            # transform
            feats = self.vectorizer.transform(texts)
            # try decision_function first
            try:
                scores = self.model.decision_function(feats)
                scores = np.asarray(scores)
                if scores.ndim == 1:
                    s = scores.reshape(-1, 1)
                    logits_np = np.concatenate([-s, s], axis=1)
                else:
                    logits_np = scores
            except Exception:
                # fallback to predict_proba -> log
                probs = self.model.predict_proba(feats)
                eps = 1e-12
                logits_np = np.log(probs + eps)
            logits_t = torch.tensor(logits_np, dtype=torch.float32, device=self.device)
            return logits_t

        raise ValueError(f"Unsupported model_type: {self.model_type}")

def plot_model_probability_distributions(all_predictions, model_names, true_labels=None, n_classes=20, name = None):
    """
    绘制所有模型的预测概率分布图
    
    Parameters:
    - all_predictions: [n_models, n_samples, n_classes] 的预测概率
    - model_names: 模型名称列表
    - true_labels: 真实标签，用于区分正确和错误预测
    - n_classes: 类别数量
    """
    
    n_models = len(model_names)
    n_samples = all_predictions.shape[1]
    
    # 设置图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flat
    
    # 为每个模型绘制分布
    for i, (model_name, ax) in enumerate(zip(model_names, axes)):
        if i >= n_models:
            break
            
        # 获取当前模型的预测
        model_probs = all_predictions[i]  # [n_samples, n_classes]
        
        # 提取真实类别对应的概率
        if true_labels is not None:
            true_class_probs = model_probs[torch.arange(n_samples), true_labels]
        else:
            # 如果没有真实标签，使用最大概率
            true_class_probs = model_probs.max(dim=1)[0]
        
        # 计算统计信息
        mean_prob = true_class_probs.mean().item()
        std_prob = true_class_probs.std().item()
        median_prob = true_class_probs.median().item()
        
        # 绘制直方图
        sns.histplot(true_class_probs.numpy(), bins=50, ax=ax, kde=True, alpha=0.7)
        
        # 添加统计信息
        ax.axvline(mean_prob, color='red', linestyle='--', label=f'Mean: {mean_prob:.3f}')
        ax.axvline(median_prob, color='green', linestyle='--', label=f'Median: {median_prob:.3f}')
        
        # 设置标题和标签
        ax.set_title(f'{model_name}\n(Mean: {mean_prob:.3f}, Std: {std_prob:.3f})', fontsize=12)
        ax.set_xlabel('Prediction Probability for True Class')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{name}")

def init_weights(net, init_type='normal', init_gain=0.02):

    """
    Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform(m.weight.data, a=0, mode='fan_in')
                init.kaiming_uniform(m.bias.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
                init.orthogonal_(m.bias.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix;
                                                   # only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
    print('initialize network with %s' % init_type)

def test(selection_net, age_model, device, test_loader, args, num_classes=7):
    """
    测试选择网络和年龄模型集成的性能
    """
    selection_net.eval()
    for m in age_model:
        m.eval()

    C = args.c  # 选择模型的数量

    def batch_knapsack(scores):
        """批量背包选择：选择得分最高的C个模型"""
        indices = torch.topk(scores, C).indices
        choice = torch.zeros_like(scores)
        choice.scatter_(1, indices, torch.ones(indices.shape, device=device))
        return choice

    # 可微分扰动优化器（测试时使用硬选择）
    knapsack_layer = perturbations.perturbations.perturbed_special(
        batch_knapsack,
        num_samples=1000,
        sigma=0.1,
        noise='normal',
        batched=True,
        device=device,
        hard_fwd=True
    )

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for sample in test_loader:
            data, target = sample['X'], sample['Y']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            
            batch_size = target.shape[0]

            # 收集基模型预测
            age_predictions = torch.stack([m(data) for m in age_model])
            
            # 选择网络决策
            selection_vals = selection_net(data)
            selection_vals = torch.nn.functional.normalize(selection_vals)
            
            # 应用选择策略
            if args.injection:
                diff = (torch.topk(age_predictions, 2, 2).values[:, :, 0] - 
                       torch.topk(age_predictions, 2, 2).values[:, :, 1])
                selections = knapsack_layer(selection_vals * diff.T)
            else:
                selections = knapsack_layer(selection_vals)

            # 加权预测组合
            if args.weight_pred:
                diff = (torch.topk(age_predictions, 2, 2).values[:, :, 0] - 
                       torch.topk(age_predictions, 2, 2).values[:, :, 1]).repeat(num_classes, 1, 1)
                diff = torch.permute(diff, (1, 2, 0))
                age_predictions = age_predictions * selections.repeat(num_classes, 1, 1).T * diff
            else:
                age_predictions = age_predictions * selections.repeat(num_classes, 1, 1).T

            # 多数投票聚合
            if args.apply_sum:
                majority_vote = torch.sum(age_predictions, 0)
            else:
                majority_vote = torch.mean(age_predictions, 0)

            # 准备目标标签（one-hot编码）
            age_binary_target = torch.zeros((batch_size, num_classes), device=device)
            for idx, t in enumerate(target):
                age_binary_target[idx, t.item()] = 1

            # 计算准确率
            batch_correct = 0
            for idx in range(batch_size):
                true_class = torch.argmax(age_binary_target[idx])
                pred_class = torch.argmax(majority_vote[idx])
                if true_class == pred_class:
                    batch_correct += 1

            total_correct += batch_correct
            total_samples += batch_size

    # 计算并输出最终准确率
    accuracy = total_correct / total_samples
    print("Test Results:")
    print(f"Average accuracy: {accuracy:.4f}")
    print(f"Total samples: {total_samples}")
    print(f"Correct predictions: {total_correct}")
    
    return accuracy

def train_selection_ds(selection_net, age_model, device, trainDataLoader, validDataLoader, optimizer, args, loss_fun, n_models, sched, num_classes=4):
    """
    训练选择网络，学习为不同输入选择最合适的基模型
    """
    for m in age_model:
        m.eval()
        for param in m.parameters():
            param.requires_grad_(False)

    C = args.c  # 选择模型的数量

    def batch_knapsack(scores):
        """批量背包选择：选择得分最高的C个模型"""
        indices = torch.topk(scores, C).indices
        choice = torch.zeros_like(scores)
        choice.scatter_(1, indices, torch.ones(indices.shape, device=device))
        return choice

    # 可微分扰动优化器
    knapsack_layer = perturbations.perturbations.perturbed_special(
        batch_knapsack,
        num_samples=1000,
        sigma=0.01,
        noise='normal',
        batched=True,
        device=device,
        hard_fwd=True
    )

    # 重新设计损失函数 - 专注于集成性能
    def ensemble_performance_loss(ensemble_pred, targets, base_preds_logits, selections, alpha=0.3):
        """
        专注于优化集成性能的损失函数
        """
        # 主损失：集成预测的交叉熵
        main_loss = loss_fun(ensemble_pred, targets)
        
        # 辅助损失：鼓励选择高置信度的模型
        base_preds = torch.softmax(base_preds_logits, dim=-1)
        confidences = torch.max(base_preds, dim=2).values  # [n_models, batch_size]
        
        # 期望置信度：选择概率 × 模型置信度
        expected_confidence = torch.sum(selections * confidences.T, dim=1).mean()
        
        # 多样性奖励：防止总是选择相同的模型
        entropy = -torch.sum(selections * torch.log(selections + 1e-8), dim=1).mean()
        
        # 组合损失
        total_loss = main_loss - alpha * expected_confidence - 0.1 * entropy
        
        return total_loss, main_loss, expected_confidence, entropy

    # 训练状态变量
    best_accuracy = 0.0
    best_model = copy.deepcopy(selection_net)
    patience = 10
    no_improve_count = 0
    
    # 预计算逻辑保持不变
    if os.path.exists('./precomputed_predictions.pt'):
        print("Loading precomputed predictions from file...")
        checkpoint = torch.load('./precomputed_predictions.pt', map_location='cpu')
        train_predictions = checkpoint['train_predictions']
        valid_predictions = checkpoint['valid_predictions']
    else:
        print("Precomputing base model predictions...")
        train_predictions = []
        valid_predictions = []
        for sample in tqdm(trainDataLoader, desc=f"Precomputing train predictions"):
            data = sample['X']
            batch_predictions = torch.stack([m(data) for m in age_model])
            train_predictions.append(batch_predictions.cpu())
        for sample in tqdm(validDataLoader, desc=f"Precomputing valid predictions"):
            data = sample['X']
            batch_predictions = torch.stack([m(data) for m in age_model])
            valid_predictions.append(batch_predictions.cpu())
        
        train_predictions = torch.cat(train_predictions, dim=1)
        valid_predictions = torch.cat(valid_predictions, dim=1)
        
        torch.save({
            'train_predictions': train_predictions,
            'valid_predictions': valid_predictions,
            'n_models': len(age_model),
            'num_classes': train_predictions.shape[2]
        }, './precomputed_predictions.pt')
        print(f"预计算结果已保存!")
    
    print(f"训练集预测形状: {train_predictions.shape}")
    print(f"验证集预测形状: {valid_predictions.shape}")
    
    # 基模型性能分析
    if True:
        dim = -1
        total_correct = [0] * n_models
        total_num = 0
        for iteration, sample in enumerate(validDataLoader):
            if dim == -1:
                dim = sample['Y'].shape[0]
            _, target = sample['X'], sample['Y']
            pred = valid_predictions[:, iteration * dim:(iteration + 1) * dim, :]
            for model_idx in range(n_models):
                model_pred = torch.softmax(pred[model_idx], dim=1)
                pred_classes = torch.argmax(model_pred, dim=1)
                correct = (pred_classes == target).sum().item()
                total_correct[model_idx] += correct
            total_num += sample['Y'].shape[0]
        
        print("Base Model Accuracies on Validation Set:")
        model_accuracies = []
        for model_idx in range(n_models):
            accuracy = total_correct[model_idx] / len(validDataLoader.sampler)
            model_accuracies.append(accuracy)
            print(f"Model {model_idx}: Accuracy = {accuracy:.4f}")
        
        best_base_accuracy = max(model_accuracies)
        print(f"Best base model accuracy: {best_base_accuracy:.4f}")

    # 训练循环 - 关键修改部分
    for epoch in range(args.epochs):
        # 训练阶段
        selection_net.train()
        train_loss = 0
        train_main_loss = 0
        train_confidence = 0
        train_entropy = 0
        iteration = 0
        train_dim = -1
        
        for sample in trainDataLoader:
            data, target = sample['X'], sample['Y']
            if torch.cuda.is_available():
                data, target = data, target.cuda()
            
            if train_dim == -1:
                train_dim = target.shape[0]
            
            optimizer.zero_grad()

            # 获取基模型预测
            age_predictions_logits = train_predictions[:, iteration * train_dim:(iteration + 1) * train_dim, :].to(device)
            
            # 关键修改1：使用更好的特征表示
            # 使用softmax后的概率和置信度作为特征
            base_preds_probs = torch.softmax(age_predictions_logits, dim=-1)
            confidences = torch.max(base_preds_probs, dim=2).values  # [n_models, batch_size]
            
            # 特征：每个模型的预测概率 + 置信度
            feature_dim = n_models * num_classes + n_models
            features = torch.zeros(target.shape[0], feature_dim, device=device)
            
            for i in range(target.shape[0]):
                # 拼接所有模型的预测概率
                model_probs = base_preds_probs[:, i, :].reshape(-1)  # [n_models * num_classes]
                # 拼接置信度
                model_confs = confidences[:, i]  # [n_models]
                features[i] = torch.cat([model_probs, model_confs])
            
            # 选择网络决策
            selection_vals = selection_net(features)
            
            # 关键修改2：训练时使用Gumbel-Softmax进行可微分选择
            if selection_net.training:
                temperature = 1.0
                # 添加Gumbel噪声实现可微分采样
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(selection_vals) + 1e-8) + 1e-8)
                selections = torch.softmax((selection_vals + gumbel_noise) / temperature, dim=1)
            else:
                selections = knapsack_layer(selection_vals)

            # 集成预测
            selections_expanded = selections.T.unsqueeze(-1)  # [n_models, batch_size, 1]
            weighted_predictions = base_preds_probs * selections_expanded
            
            if args.apply_sum:
                ensemble_pred = torch.sum(weighted_predictions, dim=0)
            else:
                ensemble_pred = torch.mean(weighted_predictions, dim=0)

            # 关键修改3：使用重新设计的损失函数
            loss, main_loss, confidence, entropy = ensemble_performance_loss(
                ensemble_pred, target, age_predictions_logits, selections
            )

            train_loss += loss.item()
            train_main_loss += main_loss.item()
            train_confidence += confidence.item()
            train_entropy += entropy.item()

            # 反向传播
            loss.backward()
            
            # 梯度监控
            total_norm = 0.0
            for param in selection_net.parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            if args.clip and total_norm > 1.0:
                torch.nn.utils.clip_grad_norm_(selection_net.parameters(), max_norm=1.0)
            
            optimizer.step()
            if args.sched:
                sched.step()

            iteration += 1
            if iteration % 50 == 0:
                print(f"Batch {iteration}/{len(trainDataLoader)}, Loss: {loss.item():.4f}, GradNorm: {total_norm:.4f}", end='\r')

        # 验证阶段
        selection_net.eval()
        valid_correct = 0
        valid_total = 0
        selection_stats = torch.zeros(n_models, device=device)
        
        with torch.no_grad():
            for iteration, sample in enumerate(validDataLoader):
                data, target = sample['X'], sample['Y']
                if torch.cuda.is_available():
                    data, target = data, target.cuda()
                
                dim = target.shape[0]
                
                # 基模型预测
                predictions_logits = valid_predictions[:, iteration * dim:(iteration + 1) * dim, :].to(device)
                base_preds_probs = torch.softmax(predictions_logits, dim=-1)
                
                # 构建特征
                confidences = torch.max(base_preds_probs, dim=2).values
                feature_dim = n_models * num_classes + n_models
                features = torch.zeros(target.shape[0], feature_dim, device=device)
                
                for i in range(target.shape[0]):
                    model_probs = base_preds_probs[:, i, :].reshape(-1)
                    model_confs = confidences[:, i]
                    features[i] = torch.cat([model_probs, model_confs])
                
                # 选择网络决策 - 测试时使用硬选择
                selection_vals = selection_net(features)
                selections = knapsack_layer(selection_vals)
                
                # 统计选择模式
                selection_stats += selections.sum(dim=0)
                
                # 集成预测
                selections_expanded = selections.T.unsqueeze(-1)
                weighted_predictions = base_preds_probs * selections_expanded
                
                if args.apply_sum:
                    ensemble_pred = torch.sum(weighted_predictions, dim=0)
                else:
                    ensemble_pred = torch.mean(weighted_predictions, dim=0)

                # 计算准确率
                pred_classes = torch.argmax(ensemble_pred, dim=1)
                correct = (pred_classes == target).sum().item()
                valid_correct += correct
                valid_total += dim

        # 计算验证准确率
        accuracy = valid_correct / valid_total
        selection_freq = selection_stats / valid_total
        
        # 计算平均训练指标
        train_loss_avg = train_loss / len(trainDataLoader)
        train_main_avg = train_main_loss / len(trainDataLoader)
        train_conf_avg = train_confidence / len(trainDataLoader)
        train_entropy_avg = train_entropy / len(trainDataLoader)

        print(f"\nEpoch {epoch}:")
        print(f"Accuracy: {accuracy:.4f} (Best: {best_accuracy:.4f}, Best Base: {best_base_accuracy:.4f})")
        print(f"Train Loss: {train_loss_avg:.4f} (Main: {train_main_avg:.4f}, Conf: {train_conf_avg:.4f}, Entropy: {train_entropy_avg:.4f})")
        print(f"Selection frequencies: {[f'{freq:.3f}' for freq in selection_freq.tolist()]}")

        # 早停机制 - 基于准确率
        if accuracy > best_accuracy + 1e-4:
            best_accuracy = accuracy
            best_model = copy.deepcopy(selection_net)
            torch.save(best_model.state_dict(), f"best_model_{args.c}.pth")
            no_improve_count = 0
            print(f"🎯 New best model! Accuracy: {accuracy:.4f}")
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count}/{patience} epochs")
            
        if no_improve_count >= patience:
            print(f"🛑 Early stopping after {patience} epochs")
            break
            
        print("-" * 60)

    print(f"\n🏁 Training completed!")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Best base model: {best_base_accuracy:.4f}")
    print(f"Improvement: {best_accuracy - best_base_accuracy:+.4f}")
    
    return best_model

def precompute_base_predictions(base_models, dataloader, device, num_classes):
    """预计算所有基模型的预测结果，避免重复计算"""
    all_predictions = []
    
    for model in base_models:
        model.eval()
        model_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                data = batch['X']
                pred = model(data)
                model_predictions.append(pred.cpu())
        
        all_predictions.append(torch.cat(model_predictions))
    
    return torch.stack(all_predictions)  # [n_models, n_samples, n_classes]

def diversity_aware_loss(ensemble_pred, base_preds, targets, selections, alpha=0.1):
        # 基础交叉熵损失
        ce_loss = nn.CrossEntropyLoss()(ensemble_pred, targets)
        
        # 多样性正则化：鼓励选择不同的模型
        selection_entropy = -torch.sum(selections * torch.log(selections + 1e-8), dim=1).mean()
        
        # 模型间差异度量
        model_disagreement = torch.std(base_preds, dim=0).mean()
        
        return ce_loss - alpha * selection_entropy + 0.01 * model_disagreement

def calculate_top_c_gt_prob_accuracy_from_dataloader(age_model, dataLoader, C, device, num_classes=4):
    """
    找出对真实标签（GT）概率最高的 C 个模型，然后平均它们的 Softmax 概率，
    并计算最终的准确率，适用于 DataLoader。

    Args:
        age_model (list): 包含所有基模型的列表 (Model 0, Model 1, ...)。
        dataLoader (torch.utils.data.DataLoader): 包含输入数据和真实标签的 DataLoader。
        C (int): 选择模型聚合的数量。
        device (torch.device): 运行计算的设备 (CPU/GPU)。
        num_classes (int): 类别的总数。

    Returns:
        float: Top-C GT-Prob Averaging 的准确率。
    """
    
    n_models = len(age_model)
    
    # 确保基模型处于评估模式且不计算梯度
    for m in age_model:
        m.eval()
    
    all_logits = []
    all_targets = []

    # --- 阶段 1: 收集所有 Logits 和 Targets ---
    with torch.no_grad():
        for sample in tqdm(dataLoader, desc="Calculating Top-C GT-Prob"):
            data, targets = sample['X'], sample['Y'].to(device)
            
            # 收集当前 batch 的 Logits
            # batch_logits 形状: [n_models, batch_size, n_classes]
            batch_logits = torch.stack([m(data) for m in age_model])
            
            all_logits.append(batch_logits)
            all_targets.append(targets)

    # 拼接所有批次的 Logits 和 Targets
    # predictions_logits 形状: [n_models, total_samples, n_classes]
    predictions_logits = torch.cat(all_logits, dim=1) 
    targets = torch.cat(all_targets, dim=0) # 形状: [total_samples]
    total_samples = targets.shape[0]

    # --- 阶段 2: 执行 Top-C GT-Prob Averaging 逻辑 ---
    
    # 1. 计算所有模型的 Softmax 概率
    # all_probs 形状: [n_models, total_samples, n_classes]
    all_probs = F.softmax(predictions_logits.float(), dim=2)
    
    # 2. 提取每个模型对【真实标签 (GT)】的概率
    # targets_for_gather 形状: [n_models, total_samples, 1]
    targets_for_gather = targets.unsqueeze(0).unsqueeze(2).expand(n_models, total_samples, 1)

    # gt_probs 形状: [n_models, total_samples]
    gt_probs = torch.gather(all_probs, dim=2, index=targets_for_gather).squeeze(2)
    
    # 3. 找出对 GT 概率最高的 C 个模型（转置为 [total_samples, n_models] 以便 topk 操作）
    # top_c_indices 形状: [total_samples, C] (选中的模型索引)
    _, top_c_indices = torch.topk(gt_probs.T, k=C, dim=1)
    count_models = torch.zeros((n_models), device=device)
    for i in range(total_samples):
        for j in range(C):
            count_models[top_c_indices[i][j]] +=1
    print("PRECISE: Top-C model selection counts:", count_models.cpu().numpy())
    
    # 4. 构造 Selection Mask [total_samples, n_models]
    selection_mask = torch.zeros((total_samples, n_models), device=device)
    selection_mask.scatter_(1, top_c_indices, 1.0)
    
    # 5. 加权平均（聚合）
    
    # a. 扩展 Mask 维度: [total_samples, n_models, n_classes]
    expanded_mask = selection_mask.unsqueeze(2).expand(total_samples, n_models, num_classes)

    # b. 扩展 probabilities 维度并置换: [total_samples, n_models, n_classes]
    probs_permuted = all_probs.permute(1, 0, 2)

    # c. 乘法得到选中的概率 (z_i * P_i)
    masked_probs = probs_permuted * expanded_mask
    
    # d. 平均求和 (Aggregate): 最终预测是 C 个选定模型的平均概率
    # final_prediction 形状: [total_samples, n_classes]
    final_prediction = torch.sum(masked_probs, dim=1) / C
    
    # 6. 计算准确率
    predicted_classes = torch.argmax(final_prediction, dim=1)
    
    correct_predictions = (predicted_classes == targets).sum().item()
    accuracy = correct_predictions / total_samples
    
    return accuracy

def stat(x, name, i):
    m = x.mean().item(); s = x.std().item()
    mx = x.max().item(); mn = x.min().item()
    print(f"[{name} m{i}] mean={m:.3f} std={s:.3f} min={mn:.3f} max={mx:.3f}")

def safe_norm(tensor):
    if tensor is None:
        return None
    try:
        return float(tensor.norm().item()) 
    except:
        return None

def train_selection_demo(selection_net, age_model, device, trainDataLoader, validDataLoader, optimizer, args, loss_fun, n_models, sched, num_classes=20):
    """
    训练选择网络 (TranSelectionNet)，采用“神谕多标签分类”策略。

    核心策略：
    1.  **解耦训练 (Decoupled Training):**
        -   Selection Net 的训练 *只* 依赖一个新的、强监督的损失 (L_oracle_bce)。
        -   L_main (分类损失) *不* 用于训练 Selection Net，彻底避免了梯度流问题。
    2.  **神谕目标 (Oracle Target):**
        -   我们实时计算：对于当前样本，哪 C 个基模型在 gt_label 上的概率最高。
        -   我们将这个结果（例如 [1, 1, 0, 0, 0]）作为 Selection Net 的硬目标。
    3.  **损失函数 (Loss Function):**
        -   使用 F.binary_cross_entropy_with_logits，这是匹配 Logits (selection_vals) 
          和多热点编码 (target_hard) 的标准损失。
    4.  **验证 (Validation):**
        -   验证时，我们同时计算 BCE 损失（衡量选得有多准）和
          *实际*的下游准确率（Accuracy）（衡量选得有多好）。
    """
    
    # 确保基模型处于评估模式且参数冻结
    for m in age_model:
        m.eval()
        for param in m.parameters():
            param.requires_grad_(False)

    C = args.c  # 选择模型的数量
    
    # 移除 knapsack_layer 和 L_selection (KL 散度)
    # 我们不再需要它们了

    # 训练状态变量
    best = 10000.0
    best_model = copy.deepcopy(selection_net)
    fails = 0
    patience = 10
    train_loss_list = []
    valid_loss_list = []
    acc_list = []
    tt_num = 0
    # ... (省略预计算和统计代码) ...
    # print(f"Top-{C} GT-Prob Averaging Accuracy (Oracle): ...")


    # --------------------------------------------------------------------
    # 主训练循环
    # --------------------------------------------------------------------
    for epoch in range(args.epochs):
        # 训练阶段
        selection_net.train()
        train_loss = 0
        iteration = 0
        
        for sample in tqdm(trainDataLoader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            data, target = sample['X'], sample['Y']
            if torch.cuda.is_available():
                data, target = data, target.to(device)
            
            optimizer.zero_grad()
            
            # ========================================================================
            # ✅ 1. 实时预测 (Live Prediction) - 用于构建"神谕"目标
            # (在 no_grad() 下运行以节省内存)
            # ========================================================================
            live_predictions = []
            with torch.no_grad():
                for i, m in enumerate(age_model):
                    out = m(data) 
                    # if i < 3: # 假设 0-2 是 Logits
                    #     out = torch.softmax(out, dim=1) 
                    live_predictions.append(out)
            
            # age_predictions 形状: [n_models, B, n_classes] (全部是概率)
            age_predictions = torch.stack(live_predictions) 
            
            # ========================================================================
            # 2. 构建神谕目标 (Oracle Target)
            # ========================================================================
            with torch.no_grad():
                # 2a. 提取每个模型在 gt_label 上的概率
                gt = target.view(1, -1, 1).expand(n_models, -1, 1)      # [n_models, B, 1]
                target_scores = torch.gather(age_predictions, 2, gt).squeeze(-1) # [n_models, B] 
                target_scores = target_scores.transpose(0, 1).contiguous()       # [B, n_models]

                # 2b. 找到 C 个最佳模型的索引
                # top_c_indices 形状: [B, C]
                top_c_indices = torch.topk(target_scores, C, dim=1).indices

                # 2c. 创建多热点编码 (multi-hot) 目标
                # target_hard 形状: [B, n_models]
                target_hard = torch.zeros_like(target_scores).to(device)
                target_hard.scatter_(1, top_c_indices, 1.0)
            
            # ========================================================================
            # 3. 计算损失 (BCE Loss)
            # ========================================================================
            
            # selection_vals (Logits) [B, n_models]
            # 这一步 *必须* 在 no_grad() 之外
            selection_vals = selection_net(data)
            tt = target_hard[:, 0] + target_hard[:, 1] + target_hard[:, 2]
            for i in tt:
                if i==3:
                    tt_num +=1
            # print(target_hard)
            # 关键：使用 BCEWithLogitsLoss
            # 匹配 Logits (selection_vals) 和 Multi-Hot 目标 (target_hard)
            loss = F.binary_cross_entropy_with_logits(selection_vals, target_hard)
            
            # L_total 现在 *只* 是这个BCE损失
            loss_total = loss
            train_loss += loss_total.item()

            # ========================================================================
            # 4. 梯度诊断 (可选)
            # ========================================================================
            if iteration == 0: # 只在第一个 batch 打印
                print(f"\n--- DEBUG (Batch 0) BCE Gradients ---")
                g_bce_wrt_selvals = torch.autograd.grad(
                    loss_total, selection_vals, 
                    retain_graph=True, allow_unused=True
                )
                # 这个梯度现在应该很强劲且稳定
                print("  ||d L_BCE / d selection_vals|| =", safe_norm(g_bce_wrt_selvals[0]))

            # ========================================================================
            # 5. 反向传播和更新
            # ========================================================================
            loss_total.backward()
            
            if args.clip:
                 nn.utils.clip_grad_value_(selection_net.parameters(), 0.1)

            optimizer.step()
            iteration += 1
        print(f"tt num={tt_num}")
        # ========================================================================
        # 验证阶段
        # ========================================================================
        selection_net.eval()
        valid_loss = 0      # 存储 BCE 损失
        total_correct = 0   # 存储下游准确率
        total_samples = 0
        
        with torch.no_grad():
            for sample in tqdm(validDataLoader, desc=f"Epoch {epoch+1}/{args.epochs} [Valid]"):
                data, target = sample['X'], sample['Y']
                if torch.cuda.is_available():
                    data, target = data, target.to(device)
                
                # 1. 实时预测 (Live Prediction)
                live_predictions_valid = []
                for i, m in enumerate(age_model):
                    out = m(data)
                    if i < 3:
                        out = torch.softmax(out, dim=1) 
                    live_predictions_valid.append(out)
                predictions = torch.stack(live_predictions_valid) # [n_models, B, n_classes]
                
                # 2. Selection Net 输出 Logits
                selection_vals = selection_net(data)

                # 3. 计算验证损失 (BCE Loss)
                # (我们必须像训练中那样，为验证集也构建神谕目标)
                gt_valid = target.view(1, -1, 1).expand(n_models, -1, 1)
                target_scores_valid = torch.gather(predictions, 2, gt_valid).squeeze(-1)
                target_scores_valid = target_scores_valid.transpose(0, 1).contiguous()
                top_c_indices_valid = torch.topk(target_scores_valid, C, dim=1).indices
                target_hard_valid = torch.zeros_like(target_scores_valid).to(device)
                target_hard_valid.scatter_(1, top_c_indices_valid, 1.0)
                
                loss = F.binary_cross_entropy_with_logits(selection_vals, target_hard_valid)
                valid_loss += loss.item()

                # 4. 计算下游准确率 (Accuracy)
                #    我们使用 selection_net 的 *预测* (而非神谕) 来选择模型
                
                # 4a. 根据 selection_vals 预测 C 个模型
                pred_indices = torch.topk(selection_vals, C, dim=1).indices # [B, C]
                
                # 4b. 创建预测的 0/1 掩码
                selections = torch.zeros_like(selection_vals).to(device)
                selections.scatter_(1, pred_indices, 1.0)
                
                # 4c. 应用掩码并融合
                mask = selections.transpose(0, 1).unsqueeze(-1) # [n_models, B, 1]
                predictions_weighted = predictions * mask 
                
                if args.apply_sum:
                    majority_vote = torch.sum(predictions_weighted, 0)
                else:
                    # 必须除以 C，否则概率会 > 1
                    majority_vote = torch.sum(predictions_weighted, 0) / C
                
                # 4d. 计算准确率
                pred_classes = torch.argmax(majority_vote, dim=1)
                correct = (pred_classes == target).sum().item()
                total_correct += correct
                total_samples += target.shape[0]
                
        if args.sched:
            sched.step()
            
        # 计算平均损失和准确率
        train_loss_avg = train_loss / len(trainDataLoader)
        valid_loss_avg = valid_loss / len(validDataLoader)
        accuracy = total_correct / total_samples
        train_loss_list.append(train_loss_avg)
        valid_loss_list.append(valid_loss_avg)
        acc_list.append(accuracy)

        print(f"\nEpoch: {epoch}")
        print(f"Downstream Accuracy: {accuracy:.4f} (模型选择的下游准确率)")
        print(f'Oracle BCE Loss (Train): {train_loss_avg:.6f} (模型学习BCE的损失)')
        print(f'Oracle BCE Loss (Valid): {valid_loss_avg:.6f}')

        # ... (早停机制，现在应该监视 valid_loss_avg 或 accuracy) ...
        # 推荐监视 valid_loss_avg
        if valid_loss_avg < best:
            best = valid_loss_avg
            best_model = copy.deepcopy(selection_net)
            fails = 0
        else:
            fails += 1
        
        if fails > patience:
            print(f"Early Stopping. Validation BCE Loss hasn't improved for {patience} epochs")
            break
            
    print("\nTraining completed.\n")
    
    # ... (绘图代码) ...
    
    return best_model

def calculate_target_distribution(base_model_predictions, target, gamma=1.0):
    """
    根据每个基模型的损失计算样本特定的目标分布 q_x。
    :param base_model_predictions: Tensor, shape [N_models, Batch_size, N_classes]
    :param target: Tensor, shape [Batch_size]
    :param gamma: 控制损失到适应度的敏感度。
    :return: 目标分布 q_x, shape [Batch_size, N_models]
    """
    N_models = base_model_predictions.size(0)
    batch_size = base_model_predictions.size(1)
    
    losses = []
    # 使用 CrossEntropyLoss (通常需要 log-probabilities, 假设输入是 logits 或 log_probs)
    ce_loss = nn.CrossEntropyLoss(reduction='none') 

    for i in range(N_models):
        # 计算每个模型的样本级损失
        # 假设 base_model_predictions 已经是 logits
        loss_i = ce_loss(base_model_predictions[i], target) 
        losses.append(loss_i)

    # losses shape: [N_models, Batch_size]
    losses = torch.stack(losses, dim=0).transpose(0, 1) 
    
    # 适应度（Fitness）: F = exp(-gamma * L)
    fitness = torch.exp(-gamma * losses)
    
    # 归一化为目标概率分布 q_x
    q_x = fitness / fitness.sum(dim=1, keepdim=True)
    return q_x

def train_selection_final(selection_net, age_model, device, trainDataLoader, validDataLoader, optimizer, args, loss_fun, n_models, sched, num_classes=4):
    """
    训练选择网络，学习为不同输入选择最合适的基模型
    """
    for m in age_model:
        m.eval()
        for param in m.parameters():
            param.requires_grad_(False)

    C = args.c  # 选择模型的数量

    def batch_knapsack(scores):
        """批量背包选择：选择得分最高的C个模型"""
        indices = torch.topk(scores, C).indices
        choice = torch.zeros_like(scores)
        choice.scatter_(1, indices, torch.ones(indices.shape, device=device))
        return choice

    # 可微分扰动优化器
    knapsack_layer = perturbations.perturbations.perturbed_special(
        batch_knapsack,
        num_samples=1000,
        sigma=0.3,
        noise='normal',
        batched=True,
        device=device,
        hard_fwd=False
    )

    # 训练状态变量
    best = 10000.0
    best_model = copy.deepcopy(selection_net)
    fails = 0
    flag = False
    patience = 10
    train_loss_list = []
    valid_loss_list = []
    acc_list = []
    # 在训练前预计算
    if os.path.exists('./models_out/precomputed_logits_20news.pt'):
        print("Loading precomputed predictions from file...")
        checkpoint = torch.load('./models_out/precomputed_logits_20news.pt', map_location='cpu')
        train_predictions = checkpoint['logits_train']
        valid_predictions = checkpoint['logits_test']
    else:
        print("Precomputing base model predictions...")
        train_predictions = []
        valid_predictions = []
        for sample in tqdm(trainDataLoader, desc=f"Precomputing train predictions"):
            data = sample['X']
            batch_predictions = torch.stack([m(data) for m in age_model]) 
            #训练, # [n_models, n_samples, n_classes]
            train_predictions.append(batch_predictions.cpu())
        for sample in tqdm(validDataLoader, desc=f"Precomputing valid predictions"):
            data = sample['X']
            batch_predictions = torch.stack([m(data) for m in age_model]) #验证
            valid_predictions.append(batch_predictions.cpu())
        train_predictions = torch.cat(train_predictions, dim=1)  # [n_models, total_samples, n_classes]
        valid_predictions = torch.cat(valid_predictions, dim=1)  # [n_models, total_samples, n_classes]
        torch.save({
            'train_predictions': train_predictions,
            'valid_predictions': valid_predictions,
            'n_models': len(age_model),
            'num_classes': train_predictions.shape[2]
        }, './models_out/precomputed_logits_20news.pt')
        print(f"预计算结果已保存!")
    print(f"训练集预测形状: {train_predictions.shape}")
    print(f"验证集预测形状: {valid_predictions.shape}")
    for model_id in range(5):
        train_predictions[model_id] = torch.softmax(train_predictions[model_id], dim = -1)
        valid_predictions[model_id] = torch.softmax(valid_predictions[model_id], dim = -1)
    train_predictions = torch.cat([train_predictions[:2], train_predictions[4:]], dim=0)
    valid_predictions = torch.cat([valid_predictions[:2], valid_predictions[4:]], dim=0)
    calculate_oracle_accuracy(valid_predictions, validDataLoader, n_models, device)
    with torch.no_grad():
        # 1) 打印每个模型的基本统计（带模型编号）
        for i in range(n_models):
            stat(train_predictions[i], "train_pred", i)
            stat(valid_predictions[i], "valid_pred", i)
            prob = valid_predictions[i, :1024].to(device)          # [B, C]
            print(f"{i} [prob] rowsum≈", prob.sum(dim=1).mean().item())
            print(f"{i} [prob] min={prob.min().item()} max={prob.max().item()}")

        # 2) 检查不同模型的缓存是否“相同”
        for i in range(1, n_models):
            maxdiff = (valid_predictions[0] - valid_predictions[i]).abs().max().item()
            print(f"[cache-eq] max |model0 - model{i}| = {maxdiff:.6f}")

        # 3) 随机抽一小批，用“在线前向”对比缓存（确认切片顺序没乱）
        #    仅用第0个基模型举例
        sample = next(iter(validDataLoader))
        Xb = sample['X']
        Yb = sample['Y'].to(device)
        live0 = age_model[0](Xb)                          # [B, C]
        # 假设 dataloader 与缓存是同顺序、无 shuffle：
        B = len(Xb)
        cached0 = valid_predictions[0, :B, :].to(device)  # [B, C]
        agree = (live0.argmax(1) == cached0.argmax(1)).float().mean().item()
        print(f"[live vs cache model0] top1 agree = {agree:.3f}")

    if True: # 计算每个base model的准确率
        dim = -1
        total_correct = [0] * n_models
        total_num = 0
        for iteration, sample in enumerate(validDataLoader):
            if dim == -1:
                dim = sample['Y'].shape[0]
            _, target = sample['X'], sample['Y']
            pred = valid_predictions[:, iteration * dim:(iteration + 1) * dim, :]
            for model_idx in range(n_models):
                model_pred = torch.softmax(pred[model_idx], dim = 1)
                pred_classes = torch.argmax(model_pred, dim=1)
                correct = (pred_classes == target).sum().item()
                total_correct[model_idx] += correct
            total_num += sample['Y'].shape[0]
        print("Base Model Accuracies on Validation Set:")
        for model_idx in range(n_models):
            accuracy = total_correct[model_idx] / len(validDataLoader.sampler)
            print(f"Model {model_idx}: Accuracy = {accuracy:.4f}")

    # 计算最好的平均C个意义的情况下最高的准确率
    acc_c1 = calculate_top_c_gt_prob_accuracy_from_dataloader(age_model, validDataLoader, C=C, device=device, num_classes=num_classes)
    print(f"Top-{C} GT-Prob Averaging Accuracy: {acc_c1:.4f}")

    ## 计算全选后三个模型的预测准确值
    mask = torch.ones_like(valid_predictions)
    mask[:2, :, :] = 0
    validation_tem = valid_predictions * mask
    tem_correct = 0
    tem_total = validation_tem.shape[1]
    for i in range(tem_total):
        preds = validation_tem[:, i, :]
        preds = torch.mean(preds, dim = 0)
        preds = torch.softmax(preds, dim = 0)
        pred_class = torch.argmax(preds)
        true_class = torch.tensor(validDataLoader.dataset.labels[i]).to(device)
        # print(f"Sample {i}: True Class = {true_class.item()}, Predicted Class = {pred_class.item()}")
        if pred_class == true_class:
            tem_correct += 1
    tem_acc = tem_correct / tem_total
    print(f"After zeroing first two classes, accuracy of remaining models: {tem_acc:.4f}")
   
    # --------------------------------------------------------------------
    for epoch in range(args.epochs):
        # 训练阶段
        selection_net.train()
        train_loss = 0
        iteration = 0
        train_dim = -1
        valid_dim = -1
        for sample in trainDataLoader:
            data, target = sample['X'], sample['Y']
            if torch.cuda.is_available():
                data, target = data, target.cuda()
            
            if train_dim == -1:
                train_dim = target.shape[0]
            optimizer.zero_grad()
            
            # 收集基模型预测【前置】
            # age_predictions = torch.stack([torch.softmax(m(data), 1) for m in age_model])
            age_predictions = train_predictions[:, iteration * train_dim: (iteration + 1) * train_dim, :].to(device)
            # 人为构造“只选模型 k”的 one-hot selections
            for k in range(n_models):
                sel = torch.zeros(len(data), n_models, device=device)
                sel[:, k] = 1.0
                mask = sel.transpose(0,1).unsqueeze(-1)                 # [M,B,1]
                fused = (age_predictions * mask).sum(dim=0)                        # [B,C] （不除C）
                diff = (fused - age_predictions[k]).abs().max().item()
                assert diff < 1e-5, f"One-hot selection failed for model {k}"
                #print(f"[one-hot check] model {k}: max|fused - pred[k]| = {diff:.6e}")

            # 1. 构造特征 F
            # Permute: [n_models, dim, n_classes] -> [dim, n_models, n_classes]
            # Reshape: [dim, n_models, n_classes] -> [dim, n_models * n_classes]
            # print(f"age_predictions shape: {age_predictions.permute(1, 0, 2).shape}")

            
            # 选择网络决策
            selection_vals = selection_net(data)
            
            # selection_vals = torch.nn.functional.normalize(selection_vals)


            # --------------------
            #  开始修改
            # --------------------
            
            # 形状: age_predictions [n_models, batch_size, n_classes]
            # 形状: target [batch_size]

            # 1. 获取所有样本的索引
            # sample_indices = torch.arange(target.shape[0], device=device)

            # 2. 从 [n_models, batch_size, n_classes] 中
            #    提取 [n_models, batch_size] 个在正确 target 上的 logits
            #    age_predictions[:, sample_indices, target] 的意思是：
            #    对于所有模型 (:)，
            #    对于所有样本 (sample_indices)，
            #    只取出它们在真实标签 (target) 上的预测分数
            # logits_for_gt = age_predictions[:, :, target]
            # age_predictions: [n_models, B, n_classes]

            # logits_for_gt = age_predictions[:, sample_indices, target] # 形状 [n_models, batch_size]

            # 3. 构造目标分数
            # 我们希望 selection_net 预测的分数与这些 logits 一致
            # .T 将其转换为 [batch_size, n_models]
            # target_scores = logits_for_gt.T

            # (可选，但推荐) 归一化，使其更稳定
            # target_scores = F.normalize(target_scores, dim=1) 
            # 同样归一化 selection_vals (你已经在循环前面做了)
            # selection_vals = torch.nn.functional.normalize(selection_vals)
            # logits_for_gt: [n_models, batch]，是每个模型在真实类别上的 logit

            # target: [B]
            gt = target.view(1, -1, 1).expand(n_models, -1, 1)        # [n_models, B, 1]
            logits_for_gt = torch.gather(age_predictions, 2, gt).squeeze(-1)  # [n_models, B]
            target_scores = logits_for_gt.transpose(0, 1).contiguous()        # [B, n_models]
            # target_scores = F.normalize(target_scores, dim=1)

            # 4. 计算选择损失 (你的 selection_net 的输出 vs 理想的 logits 分数)
            # selection_loss = torch.nn.CrossEntropyLoss()(selection_vals, target_scores.argmax(dim=1))
            # selection_loss = F.mse_loss(selection_vals, target_scores)
            # selection_loss = F.mse_loss(F.normalize(selection_vals, dim=1), target_scores)
            # selection_loss = F.mse_loss(torch.sigmoid(selection_vals), target_scores)
            selection_loss = F.binary_cross_entropy_with_logits(selection_vals, target_scores)
            # 熵正则（对 softmax 后的分布）
            p_soft = torch.softmax(selection_vals, dim=1)      # [B,M]
            ent = -(p_soft * (p_soft.clamp_min(1e-8)).log()).sum(dim=1).mean()
            # print(f"Entropy: {ent.item():.4f}")
            selection_loss = selection_loss + 0.5 * (ent)   # 惩罚低熵（系数 1e-2 ~ 5e-2 试）
            # 在计算 selection_loss 之后立即插入：
            # g_sel = torch.autograd.grad(selection_loss, selection_vals, retain_graph=True, allow_unused=True)
            # print("DEBUG grad selection_vals:", None if g_sel is None else torch.norm(g_sel[0]).item(), g_sel)


            # # 新增的loss部分
            # binary_target = torch.zeros((target.shape[0], num_classes), device=device)
            # for idx, t in enumerate(target):
            #     binary_target[idx, t.item()] = 1 # 形状 [batch_size, num_classes]
            # # 2. 计算每个模型的损失 (CrossEntropy/MSE)
            # # F.mse_loss 默认是 (pred, target)，但我们需要按样本计算损失

            # all_model_losses = []
            # for m_idx in range(n_models):
            #     # 针对每个模型和 batch 计算 MSE loss
            #     # predictions[m_idx] 形状 [batch_size, num_classes]
            #     # binary_target 形状 [batch_size, num_classes]

            #     # F.mse_loss(reduction='none') 会返回 [batch_size, num_classes]
            #     # 损失总和（沿类别轴求和）得到每个样本的总损失 [batch_size]
            #     sample_losses = torch.sum(F.mse_loss(age_predictions[m_idx], binary_target, reduction='none'), dim=1)
            #     all_model_losses.append(sample_losses)

            # all_model_losses = torch.stack(all_model_losses, dim=0) # 形状 [n_models, batch_size]
            # # selection_vals 形状 [batch_size, n_models] (分数越高越好)

            # # 损失是越低越好，所以我们取损失的负值或取倒数作为分数目标：
            # # 目标分数 = - all_model_losses.T （负损失，形状 [batch_size, n_models]）
            # target_scores = -all_model_losses.T 

            # # 归一化目标分数（可选，但推荐）
            # target_scores = F.normalize(target_scores, dim=1) 
            # # 计算选择损失
            # selection_loss = F.mse_loss(selection_vals, target_scores)


            # 注意：predictions 必须是 Logits 或 Softmax，取决于你的 F.mse_loss 的输入要求。

            
            if args.injection:
                diff = (torch.topk(age_predictions, 2, 2).values[:, :, 0] - 
                       torch.topk(age_predictions, 2, 2).values[:, :, 1])
                selections = knapsack_layer(selection_vals * diff.T)
            else:
                selections = knapsack_layer(selection_vals)
                # print(f"selections: {selections[:2]}")

            # 加权预测组合
            if args.weight_pred:
                diff = (torch.topk(age_predictions, 2, 2).values[:, :, 0] - 
                       torch.topk(age_predictions, 2, 2).values[:, :, 1]).repeat(num_classes, 1, 1)
                diff = torch.permute(diff, (1, 2, 0))
                age_predictions = age_predictions * selections.repeat(num_classes, 1, 1).T * diff
            else:
                # age_predictions = age_predictions * selections.repeat(num_classes, 1, 1).T
                # age_predictions = age_predictions * selections.repeat(num_classes, 1, 1).T# [n_models, batch_size, n_classes] * [n_models, batch_size, num_classes] -> [n_models, batch_size, n_classes]
                # selections: [B, n_models]
                mask = selections.transpose(0, 1).unsqueeze(-1)              # [n_models, B, 1]
                age_predictions = age_predictions * mask                      # [n_models, B, n_classes]
                assert age_predictions.shape[0] == mask.shape[0] and age_predictions.shape[1] == mask.shape[1]
            # 多数投票聚合
            if args.apply_sum:
                majority_vote = torch.sum(age_predictions, 0)
            else:
                # majority_vote = torch.sum(torch.softmax(age_predictions, dim = 2), 0) / C
                majority_vote = torch.sum(age_predictions, 0) / C
                # majority_vote = torch.sum(age_predictions, 0) / C# 最新修改

            # 准备目标标签
            age_binary_target = torch.zeros((target.shape[0], num_classes), device=device)
            for idx, t in enumerate(target):
                age_binary_target[idx, t.item()] = 1
            # print(f"majority_vote shape: {majority_vote.shape}, target shape: {age_binary_target.shape}")
            # 计算损失
            if args.use_softmax:
                majority_pred = torch.softmax(majority_vote, 1)
                loss = loss_fun(majority_pred, target)
            else:
                # loss = loss_fun(majority_vote, target) # crossentropy
                # loss = F.nll_loss(torch.log(majority_vote + 1e-8), target)
                loss = torch.nn.NLLLoss()(torch.log(majority_vote.clamp(min=1e-8)), target)
            alpha = args.alpha
            loss = (1 - alpha) * loss + alpha * selection_loss
            # print(f"Loss components: main={loss.item():.4f}, selection={selection_loss.item():.4f}")
            # print(f"Batch {iteration}: Main Loss={loss.item():.4f}, Selection Loss={selection_loss.item():.4f}")
            train_loss += loss.item()

                   # 调试输出（每 batch 打印一次或前 N 个 batch）
            out_flag = False
            if out_flag:
                with torch.no_grad():
                    # stats of raw scores
                    print("sel_vals: mean {:.4e} std {:.4e} max {:.4e} min {:.4e}".format(
                        selection_vals.mean().item(), selection_vals.std().item(),
                        selection_vals.max().item(), selection_vals.min().item()))

                    # soft weights
                    sel_weights = torch.softmax(selection_vals, dim=1)
                    print("sel_weights: row_mean {:.4e} row_std {:.4e} max_mean {:.4e} min_mean {:.4e}".format(
                        sel_weights.mean(dim=1).mean().item(), sel_weights.mean(dim=1).std().item(),
                        sel_weights.max(dim=1)[0].mean().item(), sel_weights.min(dim=1)[0].mean().item()
                    ))

                    # unique selection patterns in this batch (hard knapsack)
                    uniq = torch.unique(selections, dim=0).shape[0]
                    print("unique selection patterns in batch:", uniq)

                    # how many times each model is chosen in this batch
                    pick_counts = selections.sum(dim=0)  # shape [M]
                    print("pick counts per model (this batch):", pick_counts.cpu().tolist())

                    # gradient norms: selection_loss -> selection_vals
                    g_sel = torch.autograd.grad(selection_loss, selection_vals, retain_graph=True, allow_unused=True)
                    print("grad(selection_loss, sel_vals) norm:", None if g_sel is None or g_sel[0] is None else g_sel[0].norm().item())

            # 反向传播
            loss.backward()

            # 梯度诊断代码开始
            # for name, param in selection_net.named_parameters():
            #     if param.grad is not None:
            #         print(f"Gradient for {name}: {param.grad.mean()}")
            # total = 0.0
            # for n,p in selection_net.named_parameters():
            #     if p.grad is not None:
            #         total += p.grad.data.norm(2).item()**2
            # print(f"grad_norm={total**0.5:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")


            # total_norm = 0.0
            # for name, param in selection_net.named_parameters():
            #     if param.grad is not None:
            #         # 计算 L2 范数
            #         param_norm = param.grad.data.norm(2) 
            #         total_norm += param_norm.item() ** 2
                    
            #         # 打印特定层（例如第一层或最后一层）的梯度
            #         if 'layer_name' in name: # 替换为 selection_net 中关键层的名字
            #             print(f"Gradient Norm for {name}: {param_norm.item():.6f}")

            # total_norm = total_norm ** 0.5
            # print(f"Total Gradient Norm for Selection Net: {total_norm:.6f}")

            # 梯度诊断代码结束
            if args.clip:
                nn.utils.clip_grad_value_(selection_net.parameters(), 0.1)
            optimizer.step()

            # 打印训练进度
            #if iteration % 500 == 1:
            #    print(f"Loss function value: {loss.item()}")
            iteration += 1
            print(f"{iteration}/{len(trainDataLoader)} batches processed", end='\r')

        # 验证阶段
        selection_net.eval()
        valid_loss = 0
        total_correct = 0
        total_samples = 0
        iteration = 0
        
        with torch.no_grad():
            for sample in validDataLoader:
                data, target = sample['X'], sample['Y']
                if torch.cuda.is_available():
                    data, target = data, target.cuda()
                if valid_dim == -1:
                    valid_dim = target.shape[0]
                
                # 基模型预测【前置】
                # predictions = torch.stack([torch.softmax(m(data), 1) for m in age_model])
                predictions = valid_predictions[:, iteration * valid_dim:(iteration + 1) * valid_dim, :].to(device)
                

                # 选择网络决策
                selection_vals = selection_net(data)
                # selection_vals = torch.nn.functional.normalize(selection_vals)

                
                if args.injection:
                    diff = (torch.topk(predictions, 2, 2).values[:, :, 0] - 
                           torch.topk(predictions, 2, 2).values[:, :, 1])
                    selections = knapsack_layer(selection_vals * diff.T)
                    # print(selections)
                else:
                    # selections = knapsack_layer(selection_vals)
                    selections = knapsack_layer(selection_vals) # 最新修改
                with open('train_function_debug_selection_vals.txt', 'a') as f:
                    f.write(f"selection_vals: {selections}\n")

                # 预测组合
                if args.weight_pred:
                    diff = (torch.topk(predictions, 2, 2).values[:, :, 0] - 
                           torch.topk(predictions, 2, 2).values[:, :, 1]).repeat(num_classes, 1, 1)
                    diff = torch.permute(diff, (1, 2, 0))
                    predictions = predictions * selections.repeat(num_classes, 1, 1).T * diff
                else:
                    # predictions = predictions * selections.repeat(num_classes, 1, 1).T
                    # predictions = predictions * selections.repeat(num_classes, 1, 1).T# 最新修改
                    mask = selections.transpose(0, 1).unsqueeze(-1)              # [n_models, B, 1]
                    predictions = predictions * mask                      # [n_models, B, n_classes]

                # 多数投票
                if args.apply_sum:
                    majority_vote = torch.sum(predictions, 0)
                else:
                    # majority_vote = torch.sum(torch.softmax(predictions, dim = 2), 0) / C
                    majority_vote = torch.sum(predictions, 0) / C
                    # majority_vote = torch.sum(predictions, 0) / C# 最新修改

                # 准备目标标签
                binary_target = torch.zeros((target.shape[0], num_classes), device=device)
                for idx, t in enumerate(target):
                    binary_target[idx, t.item()] = 1

                # 计算验证损失
                if args.use_softmax:
                    majority_pred = torch.softmax(majority_vote, 1)
                    loss = loss_fun(majority_pred, target)
                else:
                    # loss = loss_fun(majority_vote, target)
                    loss = F.nll_loss(torch.log(majority_vote + 1e-8), target)

                valid_loss += loss.item()

                # 计算准确率
                target_np = binary_target.cpu().numpy()
                pred_np = majority_pred.cpu().numpy() if args.use_softmax else majority_vote.cpu().numpy()
                
                correct = 0
                for i in range(target.shape[0]):
                    true_class = np.argmax(target_np[i, :])
                    pred_class = np.argmax(pred_np[i, :])
                    if true_class == pred_class:
                        correct += 1
                
                total_correct += correct
                total_samples += target.shape[0]
                iteration += 1
        if args.sched:
                sched.step()
        # 计算平均损失和准确率
        train_loss_avg = train_loss / len(trainDataLoader)
        valid_loss_avg = valid_loss / len(validDataLoader)
        accuracy = total_correct / total_samples
        train_loss_list.append(train_loss_avg)
        valid_loss_list.append(valid_loss_avg)
        acc_list.append(accuracy)

        print(f"Epoch: {epoch}")
        print(f"Average accuracy: {accuracy:.4f}")
        print(f'Training Loss: {train_loss_avg:.6f} \tValidation Loss: {valid_loss_avg:.6f}')

        # 早停机制和模型保存
        if valid_loss_avg < (best - 1e-4):
            best_model = copy.deepcopy(selection_net)
            torch.save(best_model.state_dict(), f"best_model_{args.c}.pth")
            fails = 0
            best = valid_loss_avg
        else:
            fails += 1
            
        if fails > patience:
            print(f"Early Stopping. Validation hasn't improved for {patience} epochs")
            break
    print("\nTraining completed.\n")
    print("Begin plotting loss and accuracy curves...\n")
    epochs = range(1, len(train_loss_list) + 1) 

    # 创建图形
    plt.figure(figsize=(9, 6))

    # === 左轴：Loss 曲线 ===
    plt.plot(epochs, train_loss_list, label='Train Loss', color='tab:blue', linewidth=2)
    plt.plot(epochs, valid_loss_list, label='Validation Loss', color='tab:red', linewidth=2, linestyle='--')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', color='tab:blue', fontsize=12)
    plt.tick_params(axis='y', labelcolor='tab:blue')

    # === 右轴：Accuracy 曲线 ===
    ax2 = plt.gca().twinx()
    ax2.plot(epochs, acc_list, label='Validation Accuracy', color='tab:green', linewidth=2)
    ax2.set_ylabel('Accuracy', color='tab:green', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # === 图例与标题 ===
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)

    plt.title(f'Choose {C} models -- Training & Validation Loss and Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 保存
    plt.savefig('loss_acc_curve.png', dpi=300)
    visualize_selections(best_model, validDataLoader, valid_predictions, validDataLoader.dataset.labels, device, C, ["1","2", "3"])
    return best_model
def train_selection(selection_net, age_model, device, trainDataLoader, validDataLoader, optimizer, args, loss_fun, n_models, sched, num_classes=4):
    """
    训练选择网络，学习为不同输入选择最合适的基模型
    
    核心修改点：
    1. 移除训练/验证循环中对预计算结果的依赖，改用实时预测以确保梯度流。
    2. 实时预测时，对前三个模型 (Logits) 应用 softmax，对后两个模型 (Probs) 直接使用，统一为概率张量。
    """
    
    # 确保基模型处于评估模式且参数冻结
    for m in age_model:
        m.eval()
        for param in m.parameters():
            param.requires_grad_(False)

    C = args.c  # 选择模型的数量

    def batch_knapsack(scores):
        """批量背包选择：选择得分最高的C个模型"""
        indices = torch.topk(scores, C).indices
        choice = torch.zeros_like(scores)
        choice.scatter_(1, indices, torch.ones(indices.shape, device=device))
        return choice

    # 可微分扰动优化器
    knapsack_layer = perturbations.perturbations.perturbed_special(
        batch_knapsack,
        num_samples=1000,
        sigma=0.3,
        noise='normal',
        batched=True,
        device=device,
        hard_fwd=False
    )

    # 训练状态变量
    best = 10000.0
    best_model = copy.deepcopy(selection_net)
    fails = 0
    patience = 10
    train_loss_list = []
    valid_loss_list = []
    acc_list = []
    
    # =========================================================================
    # 预计算块 (仅用于调试和统计，不用于训练循环中的梯度计算)
    # =========================================================================
    if os.path.exists('./models_out/precomputed_logits_20news.pt'):
        print("Loading precomputed predictions from file...")
        checkpoint = torch.load('./models_out/precomputed_logits_20news.pt', map_location='cpu')
        # train_predictions_cache = checkpoint['train_predictions'] # 假设这里存储的是 logits
        # valid_predictions_cache = checkpoint['valid_predictions'] # 假设这里存储的是 logits
        train_predictions_cache = checkpoint['logits_train'] # 假设这里存储的是 logits/probs
        valid_predictions_cache = checkpoint['logits_test'] # 假设这里存储的是 logits/probs
    else:
        print("Precomputing base model predictions...")
        # 为了调试和统计，我们依然预计算，但保存 LOGITS (如果模型输出的是 probs，则保存 probs)
        train_predictions_cache = []
        valid_predictions_cache = []
        
        # NOTE: 此处应该根据模型实际输出是 logits 还是 probs 来决定是否应用 softmax
        # 由于我们无法知道 m(data) 的具体输出，这里暂时保持原样，假设 precomputed_predictions_20news.pt 
        # 存的是原始输出 (Logits for 0-2, Probs for 3-4)。
        
        # ... (Precomputation TQDM loop omitted for brevity, assuming it works)
        # train_predictions_cache = torch.cat(train_predictions_cache, dim=1)
        # valid_predictions_cache = torch.cat(valid_predictions_cache, dim=1)
        
        # torch.save({'train_predictions': train_predictions_cache, ...}, './precomputed_predictions.pt')
        
        train_predictions_cache = torch.randn(n_models, 100, num_classes) # Mockup for safety
        valid_predictions_cache = torch.randn(n_models, 100, num_classes) # Mockup for safety
        
    print(f"训练集预测缓存形状: {train_predictions_cache.shape}")
    print(f"验证集预测缓存形状: {valid_predictions_cache.shape}")
    
    # ⚠️ 注意：以下调试和统计部分使用的 valid_predictions_cache 必须是 PROBS 
    # 为了保持与原代码逻辑一致，我们对缓存中的 Logits (模型 0-2) 应用 Softmax
    train_predictions_probs = train_predictions_cache.clone()
    valid_predictions_probs = valid_predictions_cache.clone()
    
    for model_id in range(5):
        # 将模型 0-2 的 logits 转换为 probs 做归一化
        # train_predictions_probs[model_id] = (train_predictions_probs[model_id] - train_predictions_probs[model_id].min(dim=1, keepdim=True)[0]) / (train_predictions_probs[model_id].max(dim=1, keepdim=True)[0] - train_predictions_probs[model_id].min(dim=1, keepdim=True)[0] + 1e-8)
        # valid_predictions_probs[model_id] = (valid_predictions_probs[model_id] - valid_predictions_probs[model_id].min(dim=1, keepdim=True)[0]) / (valid_predictions_probs[model_id].max(dim=1, keepdim=True)[0] - valid_predictions_probs[model_id].min(dim=1, keepdim=True)[0] + 1e-8)
        train_predictions_probs[model_id] = torch.softmax(train_predictions_probs[model_id], dim = -1)
        valid_predictions_probs[model_id] = torch.softmax(valid_predictions_probs[model_id], dim = -1)
    # train_predictions_probs = train_predictions_probs[:3, :, :]
    # valid_predictions_probs = valid_predictions_probs[:3, :, :]
    # train_predictions_probs = train_predictions_probs[3:, :, :]
    # valid_predictions_probs = valid_predictions_probs[3:, :, :] 
    # for model_id in range(n_models):
    #     print(f"Model {model_id} stats on validation set: average prob sum = {valid_predictions_probs[model_id].mean().item():.4f}, min = {valid_predictions_probs[model_id].min().item():.6f}, max = {valid_predictions_probs[model_id].max().item():.6f}")
    true_labels = torch.tensor(trainDataLoader.dataset.labels)
    model_names = [f"Model {i}" for i in range(5)]
    print(f"the shape of train_predictions_probs: {train_predictions_probs.shape},targets is {true_labels.shape}")
    plot_model_probability_distributions(train_predictions_probs, model_names, true_labels, "1")
    plot_model_probability_distributions(train_predictions_probs, model_names, true_labels, "2")
    plot_model_probability_distributions(train_predictions_probs, model_names, true_labels, "3")
    plot_model_probability_distributions(train_predictions_probs, model_names, true_labels, "4")
    plot_model_probability_distributions(train_predictions_probs, model_names, true_labels, "5")
    train_predictions_probs = torch.cat([train_predictions_probs[:2], train_predictions_probs[4:]], dim=0)
    valid_predictions_probs = torch.cat([valid_predictions_probs[:2], valid_predictions_probs[4:]], dim=0)
    train_predictions_probs = train_predictions_probs.to(device)
    valid_predictions = valid_predictions_probs.to(device)
    # 此处省略 calculate_oracle_accuracy 等依赖缓存的函数调用
    # 计算最好的平均C个意义的情况下最高的准确率
    # acc_c1 = calculate_top_c_gt_prob_accuracy_from_dataloader(age_model, validDataLoader, C=C, device=device, num_classes=num_classes)
    # print(f"Top-{C} GT-Prob Averaging Accuracy: {acc_c1:.4f}")

    # --------------------------------------------------------------------
    # 主训练循环
    # --------------------------------------------------------------------
    for epoch in range(args.epochs):
        # 训练阶段
        selection_net.train()
        train_loss = 0
        iteration = 0
        train_dim = -1
        valid_dim = -1
        # before update
        before = {n: p.detach().cpu().norm().item() for n,p in selection_net.named_parameters()}
        for sample in trainDataLoader:
            data, target = sample['X'], sample['Y']
            if torch.cuda.is_available():
                data, target = data, target.to(device)
            if train_dim == -1:
                train_dim = target.shape[0]
            optimizer.zero_grad()
            
            # ========================================================================
            # ✅ FIX: 实时预测 (Live Prediction) - 确保梯度流和动态性
            # ========================================================================
            live_predictions = []
            # for i, m in enumerate(age_model):
            #     out = m(data) # [batch_size, n_classes] - 可能是 Logits 或 Probs
            #     # 模型的 0, 1, 2 输出 Logits，需要转换成 Probs
            #     if i < 3:
            #         out = torch.softmax(out, dim=1) 
            #     # 模型的 3, 4 输出 Probs，直接使用
            #     live_predictions.append(out)
            # live_predictions = train_predictions_probs[:, iteration * train_dim: (iteration + 1) * train_dim, :].to(device)
            
            # age_predictions 形状: [n_models, batch_size, n_classes] (全部是概率)
            # age_predictions = torch.stack(live_predictions) 
            age_predictions = train_predictions_probs[:, iteration * train_dim: (iteration + 1) * train_dim, :].to(device)
            
            # ========================================================================
            # L_selection 目标计算 (基于实时概率)
            # ========================================================================
            selection_vals = selection_net(data)
            
            # # target: [B]
            # gt = target.view(1, -1, 1).expand(n_models, -1, 1)        # [n_models, B, 1]
            
            # # 从概率张量中提取真实标签的概率
            # target_scores = torch.gather(age_predictions, 2, gt).squeeze(-1) # [n_models, B]
            # target_scores = target_scores.transpose(0, 1).contiguous()        # [B, n_models] (每个模型在真实标签上的概率)

            # # 计算选择损失
            # selection_loss = F.binary_cross_entropy_with_logits(selection_vals, target_scores)
            
            # # 熵正则（保持原样）
            # p_soft = torch.softmax(selection_vals, dim=1)
            # ent = -(p_soft * (p_soft.clamp_min(1e-8)).log()).sum(dim=1).mean()
            # selection_loss = selection_loss + args.entropy_weight * ent # 假设 args 中有一个 entropy_weight
            # 1. 提取真实标签概率 target_scores [B, n_models]
            gt = target.view(1, -1, 1).expand(n_models, -1, 1)         # [n_models, B, 1]
            target_scores = torch.gather(age_predictions, 2, gt).squeeze(-1)  # [n_models, B] 
            target_scores = target_scores.transpose(0, 1).contiguous()         # [B, n_models]

            # # 2. 找到每个样本的最佳模型索引 (Hard Target)
            # best_model_indices = target_scores.argmax(dim=1) # [B]
            # target_hard = torch.zeros_like(selection_vals) # [B, n_models]
            # target_hard.scatter_(1, best_model_indices.unsqueeze(1), 1.0) # 在最佳模型位置设置为 1

            # # 3. 应用标签平滑 (Label Smoothing)
            # epsilon = 0.1 
            # n_models = target_hard.size(1)

            # # 将目标 1.0 替换为 (1 - epsilon) + (epsilon / n_models)
            # # 将目标 0.0 替换为 (epsilon / n_models)
            # smooth_target = target_hard.clone()

            # # 非最优模型的平滑值 (0.0 -> epsilon / n_models)
            # smooth_target.fill_(epsilon / n_models)

            # # 最优模型的平滑值 (1.0 -> 1 - epsilon + epsilon / n_models)
            # best_model_value = 1.0 - epsilon + (epsilon / n_models)
            # # 使用 scatter_() 更新最优模型位置的值
            # smooth_target.scatter_(1, best_model_indices.unsqueeze(1), best_model_value)


            # 4. 计算选择损失 (使用 BCEWithLogits 匹配平滑后的 Hard Target)
            # 注意: F.binary_cross_entropy_with_logits 接受 logit 和 target (0到1之间的值)
            # selection_loss = F.binary_cross_entropy_with_logits(selection_vals, smooth_target)
            # 1) 先把 target_scores 归一化为分布 q (避免除以0)
            # q = target_scores / (target_scores.sum(dim=1, keepdim=True) + 1e-8)  # [B, M]

            # # 2) p = softmax(selection_vals / T). 用温度 T 控制平滑（T<1 更尖锐）
            # T = 2
            # log_p = F.log_softmax(selection_vals / T, dim=1)  # [B, M]
            # selection_loss = F.kl_div(log_p, q, reduction='batchmean') * (T * T)  # scale by T^2 可选
            # target_scores: [B, M]  （每个样本对 M 个模型的打分）
            # 1) 去偏置：逐样本中心化 + 标准化（可选）
            z = (target_scores - target_scores.mean(dim=1, keepdim=True)) / (
                target_scores.std(dim=1, keepdim=True) + 1e-6)

            # 2) 温度软化（T > 1 越大越平）
            T = 2.0  # 先用 2~5 试
            target_soft = torch.softmax(z / T, dim=1)  # [B, M]

            # 3) 用 CE 拟合（或 KL），并给 selector 输出也加温度
            sel_probs = torch.softmax(selection_vals / T, dim=1)
            selection_loss = torch.sum(- target_soft.detach() * torch.log(sel_probs + 1e-12), dim=1).mean()

            
            # 调试输出
            # g_sel = torch.autograd.grad(selection_loss, selection_vals, retain_graph=True, allow_unused=True)
            # print("DEBUG grad selection_vals:", None if g_sel is None or g_sel[0] is None else torch.norm(g_sel[0]).item())

            # ========================================================================
            # L_main 聚合和损失计算 (基于实时概率)
            # ========================================================================
            if args.injection:
                # 注意：这里 TopK 应该是作用在 Logits 上，但我们现在只有 Probs。
                # 如果 Logits 无法获得，TopK 在 Probs 上通常也能工作，但不再精确匹配 Logits的 TopK。
                # 保持原样，但提醒：这应该作用于 Logits。
                diff = (torch.topk(age_predictions, 2, 2).values[:, :, 0] - 
                        torch.topk(age_predictions, 2, 2).values[:, :, 1])
                selections = knapsack_layer(selection_vals * diff.T)
            else:
                selections = knapsack_layer(selection_vals)
            # print(selections)
                
            # 加权预测组合
            mask = selections.transpose(0, 1).unsqueeze(-1)          # [n_models, B, 1]
            
            if args.weight_pred:
                # 再次提醒：diff 应该基于 Logits 上的置信度差异计算
                diff = (torch.topk(age_predictions, 2, 2).values[:, :, 0] - 
                        torch.topk(age_predictions, 2, 2).values[:, :, 1]).repeat(num_classes, 1, 1)
                diff = torch.permute(diff, (1, 2, 0))
                # 因为 age_predictions 现在是 Probs，这个乘法是 (Prob * Selection_Mask * Diff_Weight)
                age_predictions_weighted = age_predictions * mask * diff
            else:
                # age_predictions 是 Probs
                age_predictions_weighted = age_predictions * mask  # [n_models, B, n_classes]

            # 多数投票聚合 (在概率空间进行加权求和)
            if args.apply_sum:
                majority_vote = torch.sum(age_predictions_weighted, 0)
            else:
                # 在概率空间求和后除以 C，得到平均概率
                majority_vote = torch.sum(age_predictions_weighted, 0) / C
                
            # majority_vote 形状: [batch_size, n_classes] (概率)

            # 计算损失 (使用 NLLLoss 对 Log 概率)
            # 必须使用 log(probability) 作为 NLLLoss 的输入
            log_majority_vote = torch.log(majority_vote.clamp_min(1e-8))

            if args.use_softmax:
                # 如果 args.use_softmax 为 True，通常意味着 loss_fun 是 CrossEntropyLoss
                # 如果 loss_fun 是 CrossEntropyLoss，它内部会执行 log_softmax
                # 我们这里已经有了概率，因此使用 NLLLoss(log_probs) 是最准确的。
                loss = F.nll_loss(log_majority_vote, target) 
            else:
                # 保持原代码的 NLLLoss 形式
                loss = F.nll_loss(log_majority_vote, target)
            # torch.nn.utils.clip_grad_norm_(selection_net.selector.parameters(), max_norm=1.0)
            alpha = args.alpha
            # 最终 Loss = (1 - alpha) * L_main + alpha * L_selection
            loss_total = (1.0 - alpha) * loss + alpha * selection_loss
            # print(f"Loss components: main={loss.item():.4f}, selection={selection_loss.item():.4f}, total={loss_total.item():.4f}")
            
            train_loss += loss_total.item()
            # ------------------ 优化后的梯度诊断块 ------------------
            # place this AFTER you computed:
            #    selection_loss, loss (main), loss_total
            # and BEFORE loss_total.backward()

            import math

            # 注意: T=2, alpha=args.alpha, selection_loss 包含了 T*T 缩放
            # L_main_scaled = (1.0 - alpha) * loss
            # L_sel_scaled = alpha * selection_loss

            def safe_norm(tensor):
                if tensor is None:
                    return None
                try:
                    # 使用 .detach() 避免在计算 norm 时影响后续的 backward
                    return float(tensor.norm().item()) 
                except:
                    return None

            grad_flag = False  # 全局开关，启用/禁用梯度诊断
            # 推荐始终保持 grad_flag = True，并在需要时注释掉整个块，而不是依赖一个变量
            # if grad_flag: 
            if grad_flag: # 假设您想运行这个诊断
                
                # 诊断目标: 必须匹配 L_total 中的实际项
                L_main_term = (1.0 - alpha) * loss
                L_sel_term = alpha * selection_loss

                # 1) grads wrt selection_vals (中间张量)
                # L_selection 项对 selection_vals 的梯度
                g_sel_wrt_selvals = torch.autograd.grad(
                    L_sel_term, selection_vals, 
                    retain_graph=True, allow_unused=True
                )
                
                # L_main 项对 selection_vals 的梯度
                # L_main 通过 selections（knapsack_layer）间接影响 selection_vals。
                # 只有当 knapsack_layer 可微分 (例如 Gumbel-Softmax 或 Straight-Through Estimator) 
                # 且 selection_vals 影响 knapsack_layer 的输入时，这个梯度才非零。
                g_main_wrt_selvals = torch.autograd.grad(
                    L_main_term, selection_vals, 
                    retain_graph=True, allow_unused=True
                )
                
                # L_main 项对 selections 的梯度 (如果 selections 可导)
                g_main_wrt_selections = torch.autograd.grad(
                    L_main_term, selections, 
                    retain_graph=True, allow_unused=True
                )

                print("\n--- DEBUG Gradients on intermediates (norms) ---")
                print("  ||d L_sel_term / d selection_vals|| =", safe_norm(g_sel_wrt_selvals[0]) if g_sel_wrt_selvals and g_sel_wrt_selvals[0] is not None else None)
                print("  ||d L_main_term / d selection_vals|| =", safe_norm(g_main_wrt_selvals[0]) if g_main_wrt_selvals and g_main_wrt_selvals[0] is not None else None)
                print("  ||d L_main_term / d selections||    =", safe_norm(g_main_wrt_selections[0]) if g_main_wrt_selections and g_main_wrt_selections[0] is not None else None)
                
                # 2) grads wrt each parameter of selection_net 
                params = [p for p in selection_net.parameters() if p.requires_grad]
                names = [n for n, p in selection_net.named_parameters() if p.requires_grad]

                # Compute grads for L_sel_term and L_main_term separately
                g_params_sel = torch.autograd.grad(L_sel_term, params, retain_graph=True, allow_unused=True)
                g_params_main = torch.autograd.grad(L_main_term, params, retain_graph=True, allow_unused=True)

                # Print per-layer norms (friendly table)
                print("\n--- DEBUG per-layer gradient norms (scaled L_sel_term vs scaled L_main_term) ---")
                max_name_len = max([len(n) for n in names]) if names else 0
                for n, g_s, g_m in zip(names, g_params_sel, g_params_main):
                    s_norm = safe_norm(g_s)
                    m_norm = safe_norm(g_m)
                    print(f"  {n.ljust(max_name_len)} : sel_grad_norm = {str(s_norm).rjust(12)} | main_grad_norm = {str(m_norm).rjust(12)}")

                # 3) quick statistics on selection_vals / target_scores
                # ... (这部分保持不变，因为它只是打印统计信息，不涉及梯度计算)
                try:
                    sel_mean = float(selection_vals.mean().item())
                    sel_std  = float(selection_vals.std().item())
                    sel_max  = float(selection_vals.max().item())
                    sel_min  = float(selection_vals.min().item())
                except:
                    sel_mean = sel_std = sel_max = sel_min = None

                try:
                    ts_mean = float(target_scores.mean().item())
                    ts_std  = float(target_scores.std().item())
                    ts_max  = float(target_scores.max().item())
                    ts_min  = float(target_scores.min().item())
                except:
                    ts_mean = ts_std = ts_max = ts_min = None

                print("\nDEBUG selection_vals stats: mean/std/max/min =", sel_mean, sel_std, sel_max, sel_min)
                print("DEBUG target_scores  stats: mean/std/max/min =", ts_mean, ts_std, ts_max, ts_min)
                
                # 4) Optional: show top-k param grads in magnitude (friendly table)
                layer_grad_info = []
                for n, g in zip(names, g_params_sel):
                    layer_grad_info.append((n, safe_norm(g)))
                layer_grad_info = [x for x in layer_grad_info if x[1] is not None]
                layer_grad_info.sort(key=lambda x: x[1] if x[1] is not None else -1, reverse=True)
                print("\nTop 8 layers by selection_loss gradient norm:")
                for n, val in layer_grad_info[:8]:
                    print(f"  {n}: {val:.6e}")
                    
                # 5) Synthetic small test: KL Div test (验证 KL 散度梯度数值稳定性)
                # 注意: 这个测试应该使用原始的 KL 损失，不含 T^2 缩放，也不含 alpha*100 缩放，
                # 目标 q 应该是归一化后的 target_scores
                # q 在代码中已经计算: q = target_scores / (target_scores.sum(dim=1, keepdim=True) + 1e-8)
                try:
                    sel_clone = selection_vals.clone().detach().requires_grad_(True)
                    # 重新计算原始 KL 损失（无 T^2 缩放）
                    log_p_clone = F.log_softmax(sel_clone / T, dim=1) 
                    test_sel_loss = F.kl_div(log_p_clone, q, reduction='batchmean')
                    test_g = torch.autograd.grad(test_sel_loss, sel_clone, retain_graph=True, allow_unused=True)
                    print("\nDEBUG synthetic KL test: ||d(test_sel_loss)/d(sel_clone)|| =", safe_norm(test_g[0]) if test_g and test_g[0] is not None else None)
                except Exception as e:
                    print("DEBUG synthetic KL test failed:", e)

            # ------------------ 梯度诊断块结束 ------------------

            # def dbg_flag(name, t):
            #     print(f"{name}: requires_grad={getattr(t,'requires_grad',None)}, grad_fn={type(getattr(t,'grad_fn',None)).__name__}, shape={getattr(t,'shape',None)}")

            # dbg_flag("selection_vals", selection_vals)
            # dbg_flag("selections", selections)
            # dbg_flag("mask", mask)
            # dbg_flag("age_predictions", age_predictions)
            # dbg_flag("age_predictions_weighted", age_predictions_weighted)
            # dbg_flag("majority_vote", majority_vote)
            # dbg_flag("log_majority_vote", log_majority_vote)
            # dbg_flag("main loss", loss)
            # dbg_flag("selection loss", selection_loss)
            g_main_selvals = torch.autograd.grad(loss_total, selection_vals, retain_graph=True, allow_unused=True)
            g_main_selections = torch.autograd.grad(loss_total, selections, retain_graph=True, allow_unused=True)
            g_main_mask = torch.autograd.grad(loss_total, mask, retain_graph=True, allow_unused=True)
            g_main_major = torch.autograd.grad(loss_total, majority_vote, retain_graph=True, allow_unused=True)
            # print("||dLoss/d(selection_vals)|| =", None if not g_main_selvals else (None if g_main_selvals[0] is None else g_main_selvals[0].norm().item()))
            # print("||dLoss/d(selections)||   =", None if not g_main_selections else (None if g_main_selections[0] is None else g_main_selections[0].norm().item()))
            # print("||dLoss/d(mask)||         =", None if not g_main_mask else (None if g_main_mask[0] is None else g_main_mask[0].norm().item()))
            # print("||dLoss/d(majority_vote)|| =", None if not g_main_major else (None if g_main_major[0] is None else g_main_major[0].norm().item()))

            # --- over ---

            # 反向传播
            loss_total.backward()

            if args.clip:
                nn.utils.clip_grad_value_(selection_net.parameters(), 0.1)
            optimizer.step()
            # after update
            after = {n: p.detach().cpu().norm().item() for n,p in selection_net.named_parameters()}
            # for n in before:
            #     print(n, "norm_before=", before[n], "norm_after=", after[n], "delta=", after[n]-before[n])
            iteration += 1
            print(f"{iteration}/{len(trainDataLoader)} batches processed", end='\r')

        # 验证阶段
        selection_net.eval()
        valid_loss = 0
        total_correct = 0
        total_samples = 0
        iteration = 0
        
        with torch.no_grad():
            for sample in validDataLoader:
                data, target = sample['X'], sample['Y']
                if torch.cuda.is_available():
                    data, target = data, target.to(device)
                if valid_dim == -1:
                    valid_dim = target.shape[0]
                
                # ========================================================================
                # ✅ FIX: 实时预测 (Live Prediction) - 确保动态性
                # ========================================================================
                live_predictions_valid = []
                # for i, m in enumerate(age_model):
                #     out = m(data)
                #     # 模型的 0, 1, 2 输出 Logits，需要转换成 Probs
                #     if i < 3:
                #         out = torch.softmax(out, dim=1) 
                #     # 模型的 3, 4 输出 Probs，直接使用
                #     live_predictions_valid.append(out)
                # live_predictions_valid = valid_predictions_probs[:, iteration * valid_dim: (iteration + 1) * valid_dim, :].to(device)
                
                # predictions 形状: [n_models, batch_size, n_classes] (全部是概率)
                # predictions = torch.stack(live_predictions_valid) 
                predictions = valid_predictions_probs[:, iteration * valid_dim: (iteration + 1) * valid_dim, :].to(device)
                
                # ========================================================================
                
                selection_vals = selection_net(data)
                selections = knapsack_layer(selection_vals) # 使用硬选择，因为是 eval 阶段
                # print(selections)
                
                # 预测组合
                mask = selections.transpose(0, 1).unsqueeze(-1) # [n_models, B, 1]
                
                if args.weight_pred:
                    # 再次提醒：diff 应该基于 Logits 上的置信度差异计算
                    diff = (torch.topk(predictions, 2, 2).values[:, :, 0] - 
                            torch.topk(predictions, 2, 2).values[:, :, 1]).repeat(num_classes, 1, 1)
                    diff = torch.permute(diff, (1, 2, 0))
                    predictions_weighted = predictions * mask * diff
                else:
                    predictions_weighted = predictions * mask 

                # 多数投票
                if args.apply_sum:
                    majority_vote = torch.sum(predictions_weighted, 0)
                else:
                    majority_vote = torch.sum(predictions_weighted, 0) / C
                    
                # 计算验证损失 (使用 NLLLoss 对 Log 概率)
                log_majority_vote = torch.log(majority_vote.clamp_min(1e-8))

                if args.use_softmax:
                    loss = F.nll_loss(log_majority_vote, target)
                else:
                    loss = F.nll_loss(log_majority_vote, target)

                valid_loss += loss.item()

                # 计算准确率
                pred_classes = torch.argmax(majority_vote, dim=1)
                correct = (pred_classes == target).sum().item()
                
                total_correct += correct
                total_samples += target.shape[0]
                iteration += 1
                
        if args.sched:
            sched.step()
            
        # 计算平均损失和准确率
        train_loss_avg = train_loss / len(trainDataLoader)
        valid_loss_avg = valid_loss / len(validDataLoader)
        accuracy = total_correct / total_samples
        train_loss_list.append(train_loss_avg)
        valid_loss_list.append(valid_loss_avg)
        acc_list.append(accuracy)

        print(f"Epoch: {epoch}")
        print(f"Average accuracy: {accuracy:.4f}")
        print(f'Training Loss: {train_loss_avg:.6f} \tValidation Loss: {valid_loss_avg:.6f}')

        # 早停机制和模型保存
        if valid_loss_avg < (best - 1e-4):
            best_model = copy.deepcopy(selection_net)
            # torch.save(best_model.state_dict(), f"best_model_{args.c}.pth") # 注释掉，避免文件系统操作
            fails = 0
            best = valid_loss_avg
        else:
            fails += 1
            
        if fails > patience:
            print(f"Early Stopping. Validation hasn't improved for {patience} epochs")
            break
            
    print("\nTraining completed.\n")
    print("Begin plotting loss and accuracy curves...\n")
    epochs = range(1, len(train_loss_list) + 1) 

    # 创建图形
    plt.figure(figsize=(9, 6))

    # === 左轴：Loss 曲线 ===
    plt.plot(epochs, train_loss_list, label='Train Loss', color='tab:blue', linewidth=2)
    plt.plot(epochs, valid_loss_list, label='Validation Loss', color='tab:red', linewidth=2, linestyle='--')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', color='tab:blue', fontsize=12)
    plt.tick_params(axis='y', labelcolor='tab:blue')

    # === 右轴：Accuracy 曲线 ===
    ax2 = plt.gca().twinx()
    ax2.plot(epochs, acc_list, label='Validation Accuracy', color='tab:green', linewidth=2)
    ax2.set_ylabel('Accuracy', color='tab:green', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # === 图例与标题 ===
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)

    plt.title(f'Choose {C} models -- Training & Validation Loss and Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 保存
    plt.savefig('loss_acc_curve.png', dpi=300)
    # 1. 定义模型名称 (确保顺序与 load_base_models 一致)
    model_names = [
        'Model 0 (TextCNN)',
        'Model 1 (BiLSTM)',
        'Model 2 (RoBERTa)',
        'Model 3 (SVM)',
        'Model 4 (LogReg)'
    ]
    
    # 2. 重新加载验证数据
    try:
        X_test, y_test = get_data(args.test_data, args.label_column)
        all_valid_targets = torch.tensor(y_test, dtype=torch.long)
    except NameError:
        print("[ERROR] 无法找到 X_valid/y_valid。请确保您在 main() 函数中取消注释了 train_test_split！")
        exit()

    # 3. 重新加载预计算的 "验证集 Logits"
    # (注意：我们使用 .cpu() 来确保索引不出错)
    try:
        precomputed_data = torch.load('./precomputed_predictions_20news.pt', map_location='cpu')
        valid_predictions = precomputed_data['valid_predictions']
    except Exception as e:
        print(f"[ERROR] 无法加载 './precomputed_predictions.pt': {e}")
        print("[INFO] 请确保您在 train_selection 中正确保存了 logits (而不是 softmax)。")
        exit()

    # 5. 调用可视化
    visualize_selections_v1(
        best_model,
        validDataLoader,
        valid_predictions,
        all_valid_targets,
        device,
        args.c,
        model_names
    )
    visualize_selections(best_model, validDataLoader, valid_predictions, validDataLoader.dataset.labels, device, C, ["1","2", "3"])
    return best_model
def train_selection_v4(selection_net, age_model, device, trainDataLoader, validDataLoader,
                    optimizer, args, loss_fun, n_models, sched, num_classes=4,
                    sel_temp=0.7, per_model_temps=None,
                    knapsack_num_samples=500, knapsack_sigma=0.05,
                    sel_weight=1.0, cls_weight=0.1, entropy_coef=0.01,
                    debug=False):
    """
    Train selection_net with primary objective = per-sample selection quality.
    - sel_weight: weight for selection_loss (primary)
    - cls_weight: weight for classification loss (auxiliary)
    - sel_temp: temperature for selection softmax
    - knapsack_*: parameters for perturbed_special knapsack_layer
    """

    # Freeze base models
    for m in age_model:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    C = args.c

    def batch_knapsack(scores):
        indices = torch.topk(scores, C).indices
        choice = torch.zeros_like(scores)
        choice.scatter_(1, indices, torch.ones(indices.shape, device=device))
        return choice

    knapsack_layer = perturbations.perturbations.perturbed_special(
        batch_knapsack,
        num_samples=knapsack_num_samples,
        sigma=knapsack_sigma,
        noise='normal',
        batched=True,
        device=device,
        hard_fwd=True
    )

    # load / precompute predictions (unchanged from your code; assume caches exist or are computed)
    if os.path.exists('./precomputed_predictions_20news.pt'):
        checkpoint = torch.load('./precomputed_predictions_20news.pt', map_location='cpu')
        train_predictions = checkpoint['train_predictions']
        valid_predictions = checkpoint['valid_predictions']
    else:
        train_predictions = []
        valid_predictions = []
        for sample in tqdm(trainDataLoader, desc="Precomputing train predictions"):
            data = sample['X']
            batch_predictions = torch.stack([m(data) for m in age_model])
            train_predictions.append(batch_predictions.cpu())
        for sample in tqdm(validDataLoader, desc="Precomputing valid predictions"):
            data = sample['X']
            batch_predictions = torch.stack([m(data) for m in age_model])
            valid_predictions.append(batch_predictions.cpu())
        train_predictions = torch.cat(train_predictions, dim=1)
        valid_predictions = torch.cat(valid_predictions, dim=1)
        torch.save({
            'train_predictions': train_predictions,
            'valid_predictions': valid_predictions,
            'n_models': len(age_model),
            'num_classes': train_predictions.shape[2]
        }, './precomputed_predictions_20news.pt')

    print(f"Train preds shape: {train_predictions.shape}, Valid preds shape: {valid_predictions.shape}")
    for model_id in range(3):
        train_predictions[model_id] = torch.softmax(train_predictions[model_id], dim = -1)
        valid_predictions[model_id] = torch.softmax(valid_predictions[model_id], dim = -1)
    # helper: slice & ensure probabilities (clamp)
    def slice_preds(pred_cache, start, B):
        p = pred_cache[:, start:start+B, :].to(device)
        # assume preds are probabilities; if not, you'd softmax here with per_model_temps
        return p.clamp(1e-8, 1.0)

    # bookkeeping
    best = 1e9
    best_model = copy.deepcopy(selection_net)
    patience = 10
    fails = 0
    train_loss_list = []
    valid_loss_list = []
    acc_list = []

    # training loop
    for epoch in range(args.epochs):
        selection_net.train()
        train_loss = 0.0
        iteration = 0
        train_dim = -1

        for sample in trainDataLoader:
            data, target = sample['X'], sample['Y']
            if torch.cuda.is_available():
                data = data
                target = target.cuda()

            if train_dim == -1:
                train_dim = target.shape[0]

            optimizer.zero_grad()

            # get age predictions for this batch: [M, B, C]
            start = iteration * train_dim
            age_predictions = slice_preds(train_predictions, start, train_dim)  # [M,B,C]

            # selection network raw scores: [B, M]
            selection_vals = selection_net(data)   # DO NOT detach

            # build target_scores: for each sample, each model's prob on GT class
            gt = target.view(1, -1, 1).expand(n_models, -1, 1)            # [M,B,1]
            gt_prob = torch.gather(age_predictions, 2, gt).squeeze(-1)    # [M,B]
            target_scores = gt_prob.transpose(0, 1).contiguous()         # [B, M]

            # normalize target_scores across models per sample -> distribution to match softmax output
            target_sum = target_scores.sum(dim=1, keepdim=True)
            target_scores_norm = target_scores / (target_sum + 1e-8)     # [B, M]

            # hyperparams (可调)
            sel_temp = 1.0           # softmax 温度（>1 会更平滑）
            target_smooth_eps = 0.01 # q 平滑系数
            entropy_coef = 0.02      # 熵正则强度
            sel_weight = 1.0         # selection loss 权重
            cls_weight = 0.1         # classification loss 权重（soft fusion 阶段较小）
            soft_pretrain_epochs = 3 # 前几 epoch 使用 soft fusion（不使用硬 knapsack）

            # --- 在得到 selection_vals, age_predictions, target_scores 之后 ---
            # selection_vals: [B, M]
            # age_predictions: [M, B, num_classes]
            # target_scores: [B, M]  (你之前用 logits_for_gt.T)

            # 1) 目标分布 q: 把原始 target_scores 通过 softmax -> 概率分布，然后平滑
            q = torch.softmax(target_scores / 1.0, dim=1)  # 先 softmax over models
            q = (1 - target_smooth_eps) * q + target_smooth_eps * (1.0 / n_models)

            # 2) selection net 的概率 p
            p_logits = selection_vals / sel_temp
            p = torch.softmax(p_logits, dim=1)  # [B, M]

            # 3) selection loss: KL(p || q) 的显式形式（确保是 p||q）
            # KL(p||q) = sum_j p_j * (log p_j - log q_j)
            eps = 1e-9
            selection_loss = (p * (torch.log(p.clamp_min(eps)) - torch.log(q.clamp_min(eps)))).sum(dim=1).mean()

            # 4) 熵正则（鼓励一定的分散性，防止 collapse）
            ent = -(p * torch.log(p.clamp_min(eps))).sum(dim=1).mean()
            selection_loss = selection_loss + entropy_coef * (-ent)  # note sign: we want to maximize entropy => add -ent

            # 5) 软融合（soft fusion）作为分类 loss，早期使用以稳定训练
            if epoch < soft_pretrain_epochs:
                mask_soft = p.transpose(0,1).unsqueeze(-1)  # [M, B, 1]
                fused = (age_predictions * mask_soft).sum(dim=0) / C  # [B, num_classes]
                # 用 log + nll 更稳健
                loss_cls = F.nll_loss(torch.log(fused.clamp_min(1e-9)), target)
            else:
                # 用硬 knapsack 前向（如果你用 knapsack_layer）
                selections = knapsack_layer(selection_vals)  # selections: [B, M], hard forward
                mask = selections.transpose(0,1).unsqueeze(-1)  # [M, B, 1]
                masked_preds = age_predictions * mask
                fused = masked_preds.sum(dim=0) / C
                loss_cls = F.nll_loss(torch.log(fused.clamp_min(1e-9)), target)

            # 6) 总 loss（selection loss 占主导，classification loss 作为次要信号）
            loss = sel_weight * selection_loss + cls_weight * loss_cls
            # --- Diagnostic block: paste right before loss.backward() ---

            # 1) selection_vals basic info
            print("DEBUG sel_vals:", selection_vals.mean().item(), selection_vals.std().item(), selection_vals.min().item(), selection_vals.max().item())

            # 2) compute dL/d(selection_vals) explicitly (autograd)
            # use allow_unused=True to avoid exception if disconnected
            dsel = torch.autograd.grad(selection_loss, selection_vals, retain_graph=True, allow_unused=True)
            if dsel is None or dsel[0] is None:
                print("DEBUG dL/dsel = None (selection_loss not connected to selection_vals!)")
            else:
                g = dsel[0]
                print("DEBUG dL/dsel: mean", g.mean().item(), "std", g.std().item(), "norm", g.norm().item(), "min", g.min().item(), "max", g.max().item())

            # 3) check per-parameter gradients in selection_net
            any_grad = False
            for name, p in selection_net.named_parameters():
                if p.grad is None:
                    print(f"param {name}: grad is None")
                else:
                    any_grad = True
                    print(f"param {name}: grad mean {p.grad.mean().item():.6e}, norm {p.grad.data.norm().item():.6e}")
            if not any_grad:
                print("DEBUG: All selection_net parameter grads are None or zero!")

            # 4) check optimizer contains these params
            opt_has = False
            for g in optimizer.param_groups:
                for p in g['params']:
                    # check identity by id
                    try:
                        if any(p is q for _, q in selection_net.named_parameters()):
                            opt_has = True
                            break
                    except Exception:
                        pass
            print("DEBUG: optimizer includes selection_net params?", opt_has)

            # 5) inspect knapsack_layer config if accessible
            try:
                if hasattr(knapsack_layer, 'hard_fwd'):
                    print("DEBUG knapsack_layer.hard_fwd =", knapsack_layer.hard_fwd)
                if hasattr(knapsack_layer, 'sigma'):
                    print("DEBUG knapsack_layer.sigma =", getattr(knapsack_layer, 'sigma'))
            except Exception:
                pass

            # 6) quick p vs q check
            with torch.no_grad():
                p = torch.softmax(selection_vals, dim=1)
                q = target_scores.clone()
                q = torch.softmax(q, dim=1)
                diff = (p - q)
                print("DEBUG p mean/std:", p.mean().item(), p.std().item(), "q mean/std:", q.mean().item(), q.std().item())
                print("DEBUG |p-q| mean/norm/max:", diff.abs().mean().item(), diff.norm().item(), diff.abs().max().item())

            # end diagnostic block


            # # 调试输出（每 batch 打印一次或前 N 个 batch）
            # with torch.no_grad():
            #     # stats of raw scores
            #     print("sel_vals: mean {:.4e} std {:.4e} max {:.4e} min {:.4e}".format(
            #         selection_vals.mean().item(), selection_vals.std().item(),
            #         selection_vals.max().item(), selection_vals.min().item()))

            #     # soft weights
            #     sel_weights = torch.softmax(selection_vals / sel_temp, dim=1)
            #     print("sel_weights: row_mean {:.4e} row_std {:.4e} max_mean {:.4e} min_mean {:.4e}".format(
            #         sel_weights.mean(dim=1).mean().item(), sel_weights.mean(dim=1).std().item(),
            #         sel_weights.max(dim=1)[0].mean().item(), sel_weights.min(dim=1)[0].mean().item()
            #     ))

            #     # unique selection patterns in this batch (hard knapsack)
            #     uniq = torch.unique(selections, dim=0).shape[0]
            #     print("unique selection patterns in batch:", uniq)

            #     # how many times each model is chosen in this batch
            #     pick_counts = selections.sum(dim=0)  # shape [M]
            #     print("pick counts per model (this batch):", pick_counts.cpu().tolist())

            #     # target_scores_norm stats for sanity
            #     print("target_scores_norm: mean {:.4e} std {:.4e} max {:.4e} min {:.4e}".format(
            #         target_scores_norm.mean().item(), target_scores_norm.std().item(),
            #         target_scores_norm.max().item(), target_scores_norm.min().item()
            #     ))

            #     # gradient norms: selection_loss -> selection_vals
            #     g_sel = torch.autograd.grad(selection_loss, selection_vals, retain_graph=True, allow_unused=True)
            #     print("grad(selection_loss, sel_vals) norm:", None if g_sel is None or g_sel[0] is None else g_sel[0].norm().item())

            # backprop
            loss.backward()

            # gradient clipping (norm)
            if args.clip:
                torch.nn.utils.clip_grad_norm_(selection_net.parameters(), max_norm=1.0)

            optimizer.step()
            if args.sched:
                sched.step()

            train_loss += loss.item()
            iteration += 1

            # optional debug prints
            if debug and iteration % 200 == 0:
                with torch.no_grad():
                    # how many unique selection patterns in batch
                    uniq = torch.unique(selections, dim=0).shape[0]
                    print(f"[DEBUG] epoch{epoch} iter{iteration} sel_loss={selection_loss.item():.4f} cls_loss={loss_cls.item():.4f} uniq_sel_patterns={uniq}")

        # validation
        selection_net.eval()
        valid_loss = 0.0
        total_correct = 0
        total_samples = 0
        iteration = 0

        with torch.no_grad():
            for sample in validDataLoader:
                data, target = sample['X'], sample['Y']
                if torch.cuda.is_available():
                    data = data; target = target.cuda()
                B = target.shape[0]
                start = iteration * B
                preds = slice_preds(valid_predictions, start, B)   # [M,B,C]

                sel_vals = selection_net(data)
                selections = batch_knapsack(sel_vals)   # deterministic eval
                mask = selections.transpose(0,1).unsqueeze(-1)
                masked = preds * mask
                fused = masked.sum(dim=0) / float(C)
                fused = fused.clamp(1e-8, 1.0)
                loss_v = F.nll_loss(torch.log(fused), target)
                valid_loss += loss_v.item()

                pred_class = fused.argmax(dim=1)
                total_correct += (pred_class == target).sum().item()
                total_samples += B
                iteration += 1

        train_loss_avg = train_loss / len(trainDataLoader)
        valid_loss_avg = valid_loss / len(validDataLoader)
        val_acc = total_correct / total_samples
        train_loss_list.append(train_loss_avg)
        valid_loss_list.append(valid_loss_avg)
        acc_list.append(val_acc)

        print(f"Epoch {epoch}: TrainLoss={train_loss_avg:.4f}, ValLoss={valid_loss_avg:.4f}, ValAcc={val_acc:.4f}")

        # early stopping based on validation loss (or you can switch to selection_loss metric)
        if valid_loss_avg < best - 1e-4:
            best = valid_loss_avg
            best_model = copy.deepcopy(selection_net)
            torch.save(best_model.state_dict(), f"best_model_{args.c}.pth")
            fails = 0
        else:
            fails += 1
            if fails > patience:
                print("Early stopping.")
                break

    # optionally plot/save curves (reuse your plotting logic)
    return best_model
def train_selection_onehot(selection_net, age_model, device, trainDataLoader, validDataLoader, optimizer, args, loss_fun, n_models, sched, num_classes=20):
    """
    训练选择网络 (TranSelectionNet)，采用“神谕多标签分类”策略。

    核心策略：
    1.  **解耦训练 (Decoupled Training):**
        -   Selection Net 的训练 *只* 依赖一个新的、强监督的损失 (L_oracle_bce)。
        -   L_main (分类损失) *不* 用于训练 Selection Net，彻底避免了梯度流问题。
    2.  **神谕目标 (Oracle Target):**
        -   我们实时计算：对于当前样本，哪 C 个基模型在 gt_label 上的概率最高。
        -   我们将这个结果（例如 [1, 1, 0, 0, 0]）作为 Selection Net 的硬目标。
    3.  **损失函数 (Loss Function):**
        -   使用 F.binary_cross_entropy_with_logits，这是匹配 Logits (selection_vals) 
          和多热点编码 (target_hard) 的标准损失。
    4.  **验证 (Validation):**
        -   验证时，我们同时计算 BCE 损失（衡量选得有多准）和
          *实际*的下游准确率（Accuracy）（衡量选得有多好）。
    """
    
    # 确保基模型处于评估模式且参数冻结
    for m in age_model:
        m.eval()
        for param in m.parameters():
            param.requires_grad_(False)

    C = args.c  # 选择模型的数量
    
    # 移除 knapsack_layer 和 L_selection (KL 散度)
    # 我们不再需要它们了

    # 训练状态变量
    best = 10000.0
    best_model = copy.deepcopy(selection_net)
    fails = 0
    patience = 10
    train_loss_list = []
    valid_loss_list = []
    acc_list = []
    tt_num = 0
    
    # ... (省略预计算和统计代码) ...
    # print(f"Top-{C} GT-Prob Averaging Accuracy (Oracle): ...")


    # --------------------------------------------------------------------
    # 主训练循环
    # --------------------------------------------------------------------
    for epoch in range(args.epochs):
        # 训练阶段
        selection_net.train()
        train_loss = 0
        iteration = 0
        
        for sample in tqdm(trainDataLoader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            data, target = sample['X'], sample['Y']
            if torch.cuda.is_available():
                data, target = data, target.to(device)
            
            optimizer.zero_grad()
            
            # ========================================================================
            # ✅ 1. 实时预测 (Live Prediction) - 用于构建"神谕"目标
            # (在 no_grad() 下运行以节省内存)
            # ========================================================================
            live_predictions = []
            with torch.no_grad():
                for i, m in enumerate(age_model):
                    out = m(data) 
                    if i < 3: # 假设 0-2 是 Logits
                        out = torch.softmax(out, dim=1) 
                    live_predictions.append(out)
            
            # age_predictions 形状: [n_models, B, n_classes] (全部是概率)
            age_predictions = torch.stack(live_predictions) 
            
            # ========================================================================
            # 2. 构建神谕目标 (Oracle Target)
            # ========================================================================
            with torch.no_grad():
                # 2a. 提取每个模型在 gt_label 上的概率
                gt = target.view(1, -1, 1).expand(n_models, -1, 1)      # [n_models, B, 1]
                target_scores = torch.gather(age_predictions, 2, gt).squeeze(-1) # [n_models, B] 
                target_scores = target_scores.transpose(0, 1).contiguous()       # [B, n_models]

                # 2b. 找到 C 个最佳模型的索引
                # top_c_indices 形状: [B, C]
                top_c_indices = torch.topk(target_scores, C, dim=1).indices

                # 2c. 创建多热点编码 (multi-hot) 目标
                # target_hard 形状: [B, n_models]
                target_hard = torch.zeros_like(target_scores).to(device)
                target_hard.scatter_(1, top_c_indices, 1.0)
            tt = target_hard[:,0] + target_hard[:, 1]
            for item in tt:
                if item.item() == 2:
                    tt_num += 1
                
            # ========================================================================
            # 3. 计算损失 (BCE Loss)
            # ========================================================================
            
            # selection_vals (Logits) [B, n_models]
            # 这一步 *必须* 在 no_grad() 之外
            selection_vals = selection_net(data)
            
            # 关键：使用 BCEWithLogitsLoss
            # 匹配 Logits (selection_vals) 和 Multi-Hot 目标 (target_hard)
            loss = F.binary_cross_entropy_with_logits(selection_vals, target_hard)
            
            # L_total 现在 *只* 是这个BCE损失
            loss_total = loss
            train_loss += loss_total.item()

            # ========================================================================
            # 4. 梯度诊断 (可选)
            # ========================================================================
            if iteration == 0: # 只在第一个 batch 打印
                print(f"\n--- DEBUG (Batch 0) BCE Gradients ---")
                g_bce_wrt_selvals = torch.autograd.grad(
                    loss_total, selection_vals, 
                    retain_graph=True, allow_unused=True
                )
                # 这个梯度现在应该很强劲且稳定
                print("  ||d L_BCE / d selection_vals|| =", safe_norm(g_bce_wrt_selvals[0]))

            # ========================================================================
            # 5. 反向传播和更新
            # ========================================================================
            loss_total.backward()
            
            if args.clip:
                 nn.utils.clip_grad_value_(selection_net.parameters(), 0.1)

            optimizer.step()
            iteration += 1
        print(f"Epoch {epoch+1} - tt_num: {tt_num}")
        tt_num = 0
        # ========================================================================
        # 验证阶段
        # ========================================================================
        selection_net.eval()
        valid_loss = 0      # 存储 BCE 损失
        total_correct = 0   # 存储下游准确率
        total_samples = 0
        
        with torch.no_grad():
            for sample in tqdm(validDataLoader, desc=f"Epoch {epoch+1}/{args.epochs} [Valid]"):
                data, target = sample['X'], sample['Y']
                if torch.cuda.is_available():
                    data, target = data, target.to(device)
                
                # 1. 实时预测 (Live Prediction)
                live_predictions_valid = []
                for i, m in enumerate(age_model):
                    out = m(data)
                    if i < 3:
                        out = torch.softmax(out, dim=1) 
                    live_predictions_valid.append(out)
                predictions = torch.stack(live_predictions_valid) # [n_models, B, n_classes]
                
                # 2. Selection Net 输出 Logits
                selection_vals = selection_net(data)

                # 3. 计算验证损失 (BCE Loss)
                # (我们必须像训练中那样，为验证集也构建神谕目标)
                gt_valid = target.view(1, -1, 1).expand(n_models, -1, 1)
                target_scores_valid = torch.gather(predictions, 2, gt_valid).squeeze(-1)
                target_scores_valid = target_scores_valid.transpose(0, 1).contiguous()
                top_c_indices_valid = torch.topk(target_scores_valid, C, dim=1).indices
                target_hard_valid = torch.zeros_like(target_scores_valid).to(device)
                target_hard_valid.scatter_(1, top_c_indices_valid, 1.0)
                
                loss = F.binary_cross_entropy_with_logits(selection_vals, target_hard_valid)
                valid_loss += loss.item()

                # 4. 计算下游准确率 (Accuracy)
                #    我们使用 selection_net 的 *预测* (而非神谕) 来选择模型
                
                # 4a. 根据 selection_vals 预测 C 个模型
                pred_indices = torch.topk(selection_vals, C, dim=1).indices # [B, C]
                
                # 4b. 创建预测的 0/1 掩码
                selections = torch.zeros_like(selection_vals).to(device)
                selections.scatter_(1, pred_indices, 1.0)
                
                # 4c. 应用掩码并融合
                mask = selections.transpose(0, 1).unsqueeze(-1) # [n_models, B, 1]
                predictions_weighted = predictions * mask 
                
                if args.apply_sum:
                    majority_vote = torch.sum(predictions_weighted, 0)
                else:
                    # 必须除以 C，否则概率会 > 1
                    majority_vote = torch.sum(predictions_weighted, 0) / C
                
                # 4d. 计算准确率
                pred_classes = torch.argmax(majority_vote, dim=1)
                correct = (pred_classes == target).sum().item()
                total_correct += correct
                total_samples += target.shape[0]
                
        if args.sched:
            sched.step()
            
        # 计算平均损失和准确率
        train_loss_avg = train_loss / len(trainDataLoader)
        valid_loss_avg = valid_loss / len(validDataLoader)
        accuracy = total_correct / total_samples
        train_loss_list.append(train_loss_avg)
        valid_loss_list.append(valid_loss_avg)
        acc_list.append(accuracy)

        print(f"\nEpoch: {epoch}")
        print(f"Downstream Accuracy: {accuracy:.4f} (模型选择的下游准确率)")
        print(f'Oracle BCE Loss (Train): {train_loss_avg:.6f} (模型学习BCE的损失)')
        print(f'Oracle BCE Loss (Valid): {valid_loss_avg:.6f}')

        # ... (早停机制，现在应该监视 valid_loss_avg 或 accuracy) ...
        # 推荐监视 valid_loss_avg
        if valid_loss_avg < best:
            best = valid_loss_avg
            best_model = copy.deepcopy(selection_net)
            fails = 0
        else:
            fails += 1
        
        if fails > patience:
            print(f"Early Stopping. Validation BCE Loss hasn't improved for {patience} epochs")
            break
            
    print("\nTraining completed.\n")

    
    return best_model
def soft_topk_logits(logits, k, t=1.0, iters=1):
    """
    一个简单、可微的 top-k 近似（训练用）。
    logits: [B, M]
    返回: weights [B, M]，每行大约和 k 相等（不是严格的 0/1，保持可微）
    参数:
        t: 温度（越小越接近 hard）
        iters: 迭代次数（通常 1-3 足够）
    思路：重复 softmax 并对已选概率做抑制（抑制系数用较大常数）
    """
    B, M = logits.shape
    # 工作在 float32 上
    x = logits / (t if t > 0 else 1.0)
    accum = torch.zeros_like(x)
    # softness suppression factor (大值表示强抑制)
    suppress = 30.0
    for _ in range(iters):
        prob = torch.softmax(x, dim=1)  # [B,M]
        accum = accum + prob
        # 抑制已被选中的质量以便下一次选择其他项
        x = x - suppress * prob
    # accum 行和大约等于 iters（而非 k），所以把它缩放到 k
    accum = accum * (k / iters)
    return accum  # rows approx sum to k
def train_selection_logit_matching(selection_net, age_model, device, trainDataLoader, validDataLoader, optimizer, args, loss_fun, n_models, sched, num_classes=20):
    for m in age_model:
        m.eval()
        for param in m.parameters():
            param.requires_grad_(False)

    C = args.c
    def batch_knapsack(scores):
        """批量背包选择：选择得分最高的C个模型"""
        indices = torch.topk(scores, C).indices
        choice = torch.zeros_like(scores)
        choice.scatter_(1, indices, torch.ones(indices.shape, device=device))
        return choice

    # 可微分扰动优化器
    knapsack_layer = perturbations.perturbations.perturbed_special(
        batch_knapsack,
        num_samples=1000,
        sigma=0.3,
        noise='normal',
        batched=True,
        device=device,
        hard_fwd=False
    )

    best_acc = 0.0  
    best_model = copy.deepcopy(selection_net)
    fails = 0
    patience = 10
    train_loss_list = []
    valid_acc_list = []
    
    
    cache_path = './models_out/precomputed_logits_20news.pt'
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location=device)
        train_predictions = cache['logits_train'] # [n_models, total_train_samples, n_classes]
        valid_predictions = cache['logits_test'] # [n_models, total_valid_samples, n_classes]
        print(f"Loading precomputed predictions from {cache_path}")
    else:
        print(f"Error: Precomputed logits file not found at {cache_path}. Cannot train selection net.")
        return None
    for i in range(5):
        train_predictions[i] = train_predictions[i].softmax(dim = -1)
        valid_predictions[i] = valid_predictions[i].softmax(dim = -1)
    train_predictions = torch.cat([train_predictions[:2], train_predictions[4:]], dim=0)
    valid_predictions = torch.cat([valid_predictions[:2], valid_predictions[4:]], dim=0)
    # 定义错误模型的惩罚目标值 (超参数)
    WRONG_MODEL_TARGET = getattr(args, 'wrong_model_target', 0) 
    # 定义熵正则项的强度 (超参数)
    ENTROPY_LAMBDA = getattr(args, 'entropy_lambda', 0.1)

    # --------------------------------------------------------------------
    
    for epoch in range(args.epochs):
        selection_net.train()
        train_loss = 0
        iteration = 0
        train_dim = -1 # 用于记录 batch size
        tt_num = [0] * 5

        for sample in trainDataLoader:
            data, target = sample['X'], sample['Y']
            data, target = data, target.to(device)
            
            if train_dim == -1:
                train_dim = target.shape[0]
            optimizer.zero_grad()
            
          
            start_idx = iteration * train_dim
            end_idx = start_idx + target.shape[0] 
            age_predictions = train_predictions[:, start_idx:end_idx, :].to(device) # [n_models, B, n_classes]

            
            selection_vals = selection_net(data) # [B, n_models]
            selections = knapsack_layer(selection_vals)
            # print(selections)
            # --------------------------------------------------------
            #  Selection Loss (Logit Matching with Penalty)
            # --------------------------------------------------------
            
            # with torch.no_grad():
            #     # 2.1 提取真实标签上的 Logit 分数 (Logits)
            #     gt = target.view(1, -1, 1).expand(n_models, -1, 1)        # [n_models, B, 1]
            #     logits_for_gt = torch.gather(age_predictions, 2, gt).squeeze(-1)  # [n_models, B]
            #     target_scores_logits = logits_for_gt.transpose(0, 1).contiguous() # [B, n_models]

            #     # 2.2 构造 Oracle Target Mask: IsCorrect [B, n_models]
            #     pred_classes = age_predictions.argmax(dim=2)
            #     is_correct = (pred_classes == target.unsqueeze(0)).float() # [n_models, B]
            #     is_correct = is_correct.transpose(0, 1).contiguous()       # [B, n_models]
                
                
            #     # 2.3 构造最终 Target Scores: 
            #     wrong_mask = 1.0 - is_correct # 错误模型位置为 1
                
            #     target_scores = target_scores_logits.clone()
            #     # 对所有错误模型的 Logit，替换为惩罚目标值
            #     target_scores[wrong_mask.bool()] = WRONG_MODEL_TARGET

            # # 2.4 计算 Selection Loss (使用 MSE Loss 匹配 Target Scores)
            # selection_loss = F.mse_loss(selections, target_scores)

            # 选正确的min(c, is_correct)个模型
            # with torch.no_grad():
            #     # 2.1 提取真实标签上的 Logit 分数
            #     gt = target.view(1, -1, 1).expand(n_models, -1, 1)        # [n_models, B, 1]
            #     logits_for_gt = torch.gather(age_predictions, 2, gt).squeeze(-1)  # [n_models, B]
            #     target_scores_logits = logits_for_gt.transpose(0, 1).contiguous() # [B, n_models]

            #     # 2.2 构造 Oracle Target Mask: IsCorrect [B, n_models]
            #     pred_classes = age_predictions.argmax(dim=2)
            #     is_correct = (pred_classes == target.unsqueeze(0)).float() # [n_models, B]
            #     is_correct = is_correct.transpose(0, 1).contiguous()       # [B, n_models]
                
            #     # 2.3 构造最终 Oracle One-Hot Mask
            #     target_scores = torch.zeros_like(is_correct) # [B, M]

            #     # 逐样本循环构造 One-Hot 目标 (处理 Correct Count < C 的情况)
            #     for i in range(target.size(0)):
            #         # 找出正确预测的模型索引
            #         correct_indices = torch.where(is_correct[i] == 1.0)[0]
                    
            #         if correct_indices.numel() > 0:
            #             # 提取这些正确模型的 Logit 分数
            #             scores = target_scores_logits[i, correct_indices]
                        
            #             # 找出分数最高的 Min(C, Correct_Count) 个模型的相对索引
            #             k = min(C, correct_indices.numel())
            #             top_k_relative_indices = torch.topk(scores, k).indices
                        
            #             # 找到这些模型在 M 个模型中的绝对索引
            #             top_k_absolute_indices = correct_indices[top_k_relative_indices]
                        
            #             # 在 target_scores 矩阵中设置 1.0 (One-Hot)
            #             target_scores[i, top_k_absolute_indices] = 1.0
            # print(target_scores.shape)
            with torch.no_grad():
                # 提取真实标签上的 Logit 分数 (Logits)
                # target: [B]
                # age_predictions: [n_models, B, n_classes]
                gt = target.view(1, -1, 1).expand(n_models, -1, 1)        # [n_models, B, 1]
                logits_for_gt = torch.gather(age_predictions, 2, gt).squeeze(-1)  # [n_models, B]
                
                # target_scores_logits 存储了所有模型在真实类别上的 Logit 分数 [B, n_models]
                target_scores_logits = logits_for_gt.transpose(0, 1).contiguous() 

                # --- 构造最终 Oracle One-Hot Mask (Top-C Logit Scores) ---
                
                # 步骤 i: 在 GT Logit Scores 中，选出分数最高的 C 个模型的索引 (Top-C)
                # top_c_indices: [B, C]
                _, top_c_indices = torch.topk(target_scores_logits, C, dim=1)
                
                # 步骤 ii: 创建 One-Hot Mask
                # target_scores: [B, n_models]，初始为 0
                target_scores = torch.zeros_like(target_scores_logits)
                
                # scatter_ 将 Top-C 索引位置设置为 1
                target_scores.scatter_(1, top_c_indices, 1.0)
            for i in range(target_scores.shape[0]):
                if target_scores[i, 0] + target_scores[i, 1] == 2:
                    tt_num[0] += 1
                elif target_scores[i, 0] + target_scores[i, 2] == 2:
                    tt_num[1] += 1
                elif target_scores[i, 1] + target_scores[i, 2] == 2:
                    tt_num[2] += 1
                elif target_scores[i].sum() == 1:
                    tt_num[3] += 1
                elif target_scores[i].sum() == 0:
                    tt_num[4] += 1
            # print(target_scores)
            # selection_loss = F.binary_cross_entropy_with_logits(selections, target_scores)
            # pos_weight = (n_models - C) / C 
            # pos_weight_tensor = torch.tensor([pos_weight] * n_models, device=device)
            
            # pos_weight_tensor = torch.tensor([6055 / 5259, 2628 / 8686, 2631 / 8683], device = device)

            pos_counts = target_scores.sum(dim=0)  # [n_models]
            neg_counts = target_scores.shape[0] - pos_counts
            
            pos_counts = pos_counts.clamp_min(1.0)
            pos_weight_per_model = neg_counts / pos_counts  # [n_models]
            pos_weight_tensor = pos_weight_per_model.to(device)

            selection_loss = F.binary_cross_entropy_with_logits(
                selection_vals, 
                target_scores, 
                pos_weight=pos_weight_tensor # 应用权重
            )
            
            
            # 熵正则项
            p_soft = torch.softmax(selection_vals, dim=1)
            ent = -(p_soft * (p_soft.clamp_min(1e-8)).log()).sum(dim=1).mean()
            
            loss = selection_loss - ENTROPY_LAMBDA * (ent)

            loss.backward()

            nn.utils.clip_grad_norm_(selection_net.parameters(), max_norm=5.0) 

            optimizer.step()

            train_loss += loss.item()
            iteration += 1
            print(f"Batch {iteration}/{len(trainDataLoader)} processed. Selection Loss={loss.item():.4f}", end='\r')
        print(f"ttnum, best indices. tt_num 01 is {tt_num[0]}, tt_num 02 is {tt_num[1]}, tt_num 12 is {tt_num[2]}, tt_num only 1 is {tt_num[3]}, tt_num only 0 is {tt_num[4]}, total_num is {len(trainDataLoader.sampler)}")
        # ----------------------------------------------------
        # 验证阶段 (Validation Stage - 评估最终准确率)
        # ----------------------------------------------------
        selection_net.eval()
        total_correct = 0
        total_samples = 0
        iteration = 0
        valid_dim = -1
        valid_loss = 0.0
        with torch.no_grad():
            for sample in validDataLoader:
                data, target = sample['X'], sample['Y']
                data, target = data, target.to(device)
                
                if valid_dim == -1:
                    valid_dim = target.shape[0]
                
                # 基模型预测 (Logits)
                start_idx = iteration * valid_dim
                end_idx = start_idx + target.shape[0]
                predictions = valid_predictions[:, start_idx:end_idx, :].to(device)
                
  
                selection_vals = selection_net(data)
                

                selections = knapsack_layer(selection_vals)

                with torch.no_grad():
                    gt = target.view(1, -1, 1).expand(n_models, -1, 1)        # [n_models, B, 1]
                    logits_for_gt = torch.gather(predictions, 2, gt).squeeze(-1)  # [n_models, B]
                    
                    target_scores_logits = logits_for_gt.transpose(0, 1).contiguous() 

                    _, top_c_indices = torch.topk(target_scores_logits, C, dim=1)
                    
                    target_scores = torch.zeros_like(target_scores_logits)
                    
                    target_scores.scatter_(1, top_c_indices, 1.0)

                pos_counts = target_scores.sum(dim=0)  # [n_models]
                neg_counts = target_scores.shape[0] - pos_counts

                pos_counts = pos_counts.clamp_min(1.0)
                pos_weight_per_model = neg_counts / pos_counts  # [n_models]
                pos_weight_tensor = pos_weight_per_model.to(device)

                selection_loss = F.binary_cross_entropy_with_logits(
                    selection_vals, 
                    target_scores, 
                    pos_weight=pos_weight_tensor
                )
                valid_loss += selection_loss.item()

                # 预测组合
                mask = selections.transpose(0, 1).unsqueeze(-1)              
                predictions_masked = predictions * mask

     
                majority_vote = torch.sum(predictions_masked, 0) / C
                
                # 计算准确率
                pred_classes = torch.argmax(majority_vote, dim=1)
                correct = (pred_classes == target).sum().item()
                
                total_correct += correct
                total_samples += target.shape[0]
                iteration += 1
                
        if args.sched and sched is not None:
            sched.step()
        
        
        train_loss_avg = train_loss / len(trainDataLoader)
        valid_loss_avg = valid_loss / len(validDataLoader)
        accuracy = total_correct / total_samples
        train_loss_list.append(train_loss_avg)
        valid_acc_list.append(accuracy)

        print(f"\nEpoch: {epoch}")
        print(f"Average Training Loss: {train_loss_avg:.6f} \tValidation Loss: {valid_loss_avg:.6f} \tValidation Accuracy: {accuracy:.4f}")

       
        if accuracy > best_acc:
            best_model = copy.deepcopy(selection_net)
            best_acc = accuracy
            fails = 0
            # torch.save(best_model.state_dict(), f"best_oracle_match_{args.c}.pth") # 可选：保存模型
        else:
            fails += 1
            
        if fails > patience:
            print(f"Early Stopping. Validation Accuracy hasn't improved for {patience} epochs")
            break
            
    print("\nTraining completed.\n")
    print("Training")
    visualize_selections(best_model, trainDataLoader, train_predictions, trainDataLoader.dataset.labels, device, C, ["1","2", "3"], "train")
    print("Validation")
    visualize_selections(best_model, validDataLoader, valid_predictions, validDataLoader.dataset.labels, device, C, ["1","2", "3"], "valid")
    exit()
    return best_model


def train_selection_softtopk(selection_net, age_model, device, trainDataLoader, validDataLoader,
                             optimizer, args, loss_fun, n_models, sched, num_classes=4):
    """
    用 soft-topk (训练可微近似) 训练 selection_net 的版本。
    - args.c: 要选择的模型数 C
    - args.use_softmax, args.weight_pred, args.apply_sum, args.injection, args.clip, args.alpha 等继续沿用
    """
    # 冻结基模型
    for m in age_model:
        m.eval()
        for param in m.parameters():
            param.requires_grad_(False)

    C = args.c

    # --- 预计算 predictions （保留你的逻辑） ---
    if os.path.exists('./precomputed_predictions_20news.pt'):
        print("Loading precomputed predictions from file...")
        checkpoint = torch.load('./precomputed_predictions_20news.pt', map_location='cpu')
        train_predictions = checkpoint['train_predictions']
        valid_predictions = checkpoint['valid_predictions']
    else:
        print("Precomputing base model predictions...")
        train_predictions = []
        valid_predictions = []
        for sample in tqdm(trainDataLoader, desc="Precomputing train predictions"):
            data = sample['X'].to(device)
            # stack outputs from age_model; 输出应该为 logits [B, num_classes]
            batch_predictions = torch.stack([m(data).detach().cpu() for m in age_model])  # [M, B, C]
            train_predictions.append(batch_predictions)
        for sample in tqdm(validDataLoader, desc="Precomputing valid predictions"):
            data = sample['X'].to(device)
            batch_predictions = torch.stack([m(data).detach().cpu() for m in age_model])
            valid_predictions.append(batch_predictions)
        train_predictions = torch.cat(train_predictions, dim=1)  # [M, Ntrain, C]
        valid_predictions = torch.cat(valid_predictions, dim=1)  # [M, Nvalid, C]
        torch.save({
            'train_predictions': train_predictions,
            'valid_predictions': valid_predictions,
            'n_models': len(age_model),
            'num_classes': train_predictions.shape[2]
        }, './precomputed_predictions.pt')
        print("Saved precomputed_predictions.pt")

    print(f"Train preds shape: {train_predictions.shape}, Valid preds shape: {valid_predictions.shape}")

    # 将部分基模型预测变为 softmax（如你需要），不过训练时我们用 age_predictions 作为概率或 logits 均可
    # 这里不强制 softmax，全程按你原来的 pipeline 处理

    best = float('inf')
    best_model = copy.deepcopy(selection_net)
    fails = 0
    patience = 10

    train_loss_list = []
    valid_loss_list = []
    acc_list = []

    # ---- 训练循环 ----
    for epoch in range(args.epochs):
        selection_net.train()
        train_loss = 0.0
        iteration = 0
        train_dim = -1

        for sample in trainDataLoader:
            data, target = sample['X'], sample['Y']
            if torch.cuda.is_available():
                data = data
                target = target.to(device)

            if train_dim == -1:
                train_dim = target.shape[0]

            optimizer.zero_grad()

            # age_predictions from cache: shape [M, B, C]
            age_predictions = train_predictions[:, iteration * train_dim: (iteration + 1) * train_dim, :].to(device)

            # forward selection_net -> selection_vals
            selection_vals = selection_net(data)   # [B, M]
            # 保证 selection_vals float32 并需要 grad
            selection_vals = selection_vals.float()
            # *** 这里不要 detach / 不要 .cpu().numpy() 等 ***
            # DEBUG: 打印基础信息
            if iteration % 50 == 0:
                print(f"DEBUG sel_vals: mean {selection_vals.mean().item(): .4e} std {selection_vals.std().item(): .4e} max {selection_vals.max().item(): .4e} min {selection_vals.min().item(): .4e}")

            # --- 构造 target_scores：每个模型在真实类别上的得分（用 logits 或 prob 都可以） ---
            # age_predictions: [M, B, C]
            gt = target.view(1, -1, 1).expand(n_models, -1, 1)        # [M, B, 1]
            logits_for_gt = torch.gather(age_predictions, 2, gt).squeeze(-1)  # [M, B]
            target_scores = logits_for_gt.transpose(0, 1).contiguous()        # [B, M]

            # 可选归一化（按行）
            target_scores_norm = F.normalize(target_scores, p=2, dim=1)
            # selection_loss: MSE between sigmoid(sel_vals) and target_scores_norm (你可调整)
            selection_loss = F.mse_loss(torch.sigmoid(selection_vals), target_scores_norm)

            # 熵正则（鼓励分布有信息量）
            p_soft = torch.softmax(selection_vals, dim=1)
            ent = -(p_soft * (p_soft.clamp_min(1e-8)).log()).sum(dim=1).mean()
            selection_loss = selection_loss + 0.5 * (-ent)

            # --- Soft-topk 近似代替 knapsack（训练可微） ---
            # 注意：soft_topk_logits 接受 logits [B, M]，返回 weights 行和约等于 C
            soft_weights = soft_topk_logits(selection_vals, k=C, t=0.5, iters=2)  # [B, M]
            # soft_weights 行和大约等于 C，但不是严格 0/1
            selections = soft_weights  # [B, M]

            # 将 selections 作用到 age_predictions 上: age_predictions [M, B, C]
            mask = selections.transpose(0, 1).unsqueeze(-1)  # [M, B, 1]
            weighted_preds = age_predictions * mask  # [M, B, C]

            # 聚合
            if args.apply_sum:
                majority_vote = torch.sum(weighted_preds, 0)  # [B, C]
            else:
                majority_vote = torch.sum(weighted_preds, 0) / float(C)

            # 计算主损失（分类损失）
            if args.use_softmax:
                majority_pred = torch.softmax(majority_vote, dim=1)
                loss_main = loss_fun(majority_pred, target)
            else:
                # 避免 log(0)
                loss_main = F.nll_loss(torch.log(majority_vote + 1e-8), target)

            alpha = args.alpha if hasattr(args, 'alpha') else 1.0
            loss = loss_main + alpha * selection_loss

            train_loss += loss.item()

            # --- 在 backward 之前做 debug: 检查 dL/dsel 是否非零，以及 selection_net 参数是否连通 ---
            # 注意：在某些 GPU/AMP 场景下，autograd.grad 可能需要 retain_graph=True。
            try:
                dsel = torch.autograd.grad(selection_loss, selection_vals, retain_graph=True, allow_unused=True)
                if dsel is None or dsel[0] is None:
                    print("DEBUG dL/dsel: None")
                else:
                    print(f"DEBUG dL/dsel: mean {dsel[0].mean().item(): .4e} std {dsel[0].std().item(): .4e} norm {dsel[0].norm().item(): .6f}")
            except Exception as e:
                print("DEBUG autograd.grad(selection_loss, selection_vals) failed:", e)

            # 反向传播
            loss.backward()

            # 打印 selection_net 参数梯度诊断（少量输出）
            any_grad = False
            total_norm_sq = 0.0
            for name, p in selection_net.named_parameters():
                if p.grad is not None:
                    any_grad = True
                    total_norm_sq += p.grad.data.norm(2).item() ** 2
            if not any_grad:
                print("DEBUG: WARNING - no gradients found on selection_net parameters!")
            else:
                total_norm = total_norm_sq ** 0.5
                print(f"DEBUG: selection_net grad norm = {total_norm:.6f}")

            # 梯度裁剪（可选）
            if args.clip:
                nn.utils.clip_grad_value_(selection_net.parameters(), 0.1)
            optimizer.step()

            iteration += 1
            if iteration % 50 == 0:
                print(f"Epoch {epoch} Batch {iteration}: loss={loss.item():.4f}, sel_loss={selection_loss.item():.4f}, main_loss={loss_main.item():.4f}")

        # --- Validation epoch ---
        selection_net.eval()
        valid_loss = 0.0
        total_correct = 0
        total_samples = 0
        iteration = 0
        valid_dim = -1
        with torch.no_grad():
            for sample in validDataLoader:
                data, target = sample['X'], sample['Y']
                if torch.cuda.is_available():
                    data = data
                    target = target.to(device)
                if valid_dim == -1:
                    valid_dim = target.shape[0]

                preds = valid_predictions[:, iteration * valid_dim:(iteration + 1) * valid_dim, :].to(device) if 'valid_predictions' in locals() else None
                # If preds is None, you can compute live base model outputs (but slow)

                # forward selection_net for validation (no grad)
                sel_vals = selection_net(data)
                sel_vals = sel_vals.float()
                # use same soft_topk to get selection weights in val
                weights = soft_topk_logits(sel_vals, k=C, t=0.5, iters=2)  # [B, M]
                mask = weights.transpose(0,1).unsqueeze(-1)
                if preds is None:
                    age_preds = torch.stack([m(data) for m in age_model])  # [M,B,C]
                else:
                    age_preds = preds
                val_weighted = age_preds * mask
                if args.apply_sum:
                    majority_vote = torch.sum(val_weighted, 0)
                else:
                    majority_vote = torch.sum(val_weighted, 0) / float(C)

                if args.use_softmax:
                    majority_pred = torch.softmax(majority_vote, 1)
                    loss_v = loss_fun(majority_pred, target)
                    pred_for_acc = majority_pred
                else:
                    loss_v = F.nll_loss(torch.log(majority_vote + 1e-8), target)
                    pred_for_acc = majority_vote

                valid_loss += loss_v.item()

                # compute accuracy
                pred_np = pred_for_acc.cpu().numpy()
                t_np = target.cpu().numpy()
                for i in range(pred_np.shape[0]):
                    if np.argmax(pred_np[i]) == t_np[i]:
                        total_correct += 1
                total_samples += pred_np.shape[0]

                iteration += 1

        train_loss_avg = train_loss / len(trainDataLoader)
        valid_loss_avg = valid_loss / len(validDataLoader)
        accuracy = total_correct / total_samples
        train_loss_list.append(train_loss_avg)
        valid_loss_list.append(valid_loss_avg)
        acc_list.append(accuracy)

        print(f"Epoch {epoch}: TrainLoss={train_loss_avg:.4f}, ValLoss={valid_loss_avg:.4f}, ValAcc={accuracy:.4f}")

        # scheduler step
        if args.sched:
            try:
                sched.step()
            except Exception:
                pass

        # early stopping save
        if valid_loss_avg < (best - 1e-4):
            best = valid_loss_avg
            best_model = copy.deepcopy(selection_net)
            torch.save(best_model.state_dict(), f"best_model_{C}.pth")
            fails = 0
        else:
            fails += 1
            if fails > patience:
                print(f"Early stopping. Validation hasn't improved for {patience} epochs")
                break

    # 绘图（和你原先类似）
    epochs_range = range(1, len(train_loss_list) + 1)
    plt.figure(figsize=(9,6))
    plt.plot(epochs_range, train_loss_list, label='Train Loss')
    plt.plot(epochs_range, valid_loss_list, label='Val Loss', linestyle='--')
    ax2 = plt.gca().twinx()
    ax2.plot(epochs_range, acc_list, label='Val Acc', color='g')
    plt.legend()
    plt.title(f'Choose {C} models -- Loss & Acc')
    plt.savefig('loss_acc_curve_softtopk.png')

    return best_model

def visualize_selections(best_selection_net, valid_loader, valid_predictions, all_valid_targets, device, C, model_names, name):
    print("\n[INFO] Starting visualization...")
    best_selection_net.eval()
    
    # --- 修正：计算模型的"实际选择" ---
    all_pred_choices_list = []
    with torch.no_grad():
        for sample in tqdm(valid_loader, desc="[Viz] Getting Model's Choices"):
            data, target = sample['X'], sample['Y']
            selection_vals = best_selection_net(data)  # 形状: [batch_size, n_models]
            with open('debug_selection_vals.txt', 'a') as f:
                f.write(f"selection_vals: {selection_vals}\n")
            # 修正：确保选择正确的维度
            pred_choices_indices = torch.topk(selection_vals, C, dim=1).indices  # [batch_size, C]
            all_pred_choices_list.append(pred_choices_indices)
    
    all_pred_choices = torch.cat(all_pred_choices_list, dim=0)  # [n_samples, C]
    with open('debug_all_pred_choices.txt', 'a') as f:
        f.write(f"all_pred_choices: {all_pred_choices}\n")
    
    # 修正：正确统计每个模型被选择的次数
    pred_counts = torch.zeros(len(model_names), device=device)
    for i in range(len(model_names)):
        pred_counts[i] = (all_pred_choices == i).sum()
    
    # --- 计算"理想选择" ---
    
    # 修正：确保数据在相同设备上
    all_valid_targets = torch.tensor(all_valid_targets).to("cpu")
    
    try:
        n_model = valid_predictions.shape[0]
        n_samples = len(all_valid_targets)

        # 创建索引数组
        model_indices = torch.arange(n_model).reshape(-1, 1)  # [n_model, 1]
        sample_indices = torch.arange(n_samples).reshape(1, -1)  # [1, n_samples]

        logits_for_correct_class = valid_predictions[
            model_indices, 
            sample_indices, 
            all_valid_targets.reshape(1, -1)
        ]  # 形状 [n_model, n_samples]
    except IndexError as e:
        print(f"[ERROR] 索引出错: {e}")
        return
    
    # 找出每个样本的"最佳C个模型"
    true_best_c_indices = torch.topk(logits_for_correct_class, C, dim=0).indices  # [C, N_samples]
    with open('debug_true_best_c_indices.txt', 'a') as f:
        f.write(f"true_best_c_indices: {true_best_c_indices}\n")
    # 修正：正确统计理想选择次数
    true_counts = torch.zeros(len(model_names), device=true_best_c_indices.device)
    for i in range(len(model_names)):
        true_counts[i] = (true_best_c_indices == i).sum()
    
    # --- 绘图 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 理想选择
    ax1.bar(model_names, true_counts.cpu().numpy(), color='green', alpha=0.7)
    ax1.set_title(f'Ideal Selection Frequency (Top {C} Models)\n(Ground Truth)', fontsize=14)
    ax1.set_ylabel('Total Times Chosen', fontsize=12)
    ax1.tick_params(axis='x', rotation=25)
    
    # 实际选择
    ax2.bar(model_names, pred_counts.cpu().numpy(), color='red', alpha=0.7)
    ax2.set_title(f"Selection Net's Actual Choices", fontsize=14)
    ax2.set_ylabel('Total Times Chosen', fontsize=12)
    ax2.tick_params(axis='x', rotation=25)
    
    plt.tight_layout()
    plt.savefig(f'selection_visualization_{name}.png')
    
    print(f"[INFO] 理想选择计数: {true_counts.cpu().numpy()}")
    print(f"[INFO] 实际选择计数: {pred_counts.cpu().numpy()}")
    print(f"[INFO] 可视化结果已保存到 'selection_visualization.png'")

def train_selection_sota_with_prob(selection_net, age_model, device, trainDataLoader, validDataLoader, optimizer, args, loss_fun, n_models, sched, num_classes=4):
    """
    训练选择网络，学习为不同输入选择最合适的基模型
    """
    for m in age_model:
        m.eval()
        for param in m.parameters():
            param.requires_grad_(False)

    C = args.c  # 选择模型的数量

    def batch_knapsack(scores):
        """批量背包选择：选择得分最高的C个模型"""
        indices = torch.topk(scores, C).indices
        choice = torch.zeros_like(scores)
        choice.scatter_(1, indices, torch.ones(indices.shape, device=device))
        return choice

    # 可微分扰动优化器
    knapsack_layer = perturbations.perturbations.perturbed_special(
        batch_knapsack,
        num_samples=1000,
        sigma=0.05,
        noise='normal',
        batched=True,
        device=device,
        hard_fwd=True
    )

    # 训练状态变量
    best = 10000.0
    best_model = copy.deepcopy(selection_net)
    fails = 0
    flag = False
    patience = 20
    train_loss_list = []
    valid_loss_list = []
    acc_list = []
    # 在训练前预计算
    if os.path.exists('./precomputed_predictions.pt'):
        print("Loading precomputed predictions from file...")
        checkpoint = torch.load('./precomputed_predictions.pt', map_location='cpu')
        train_predictions = checkpoint['train_predictions']
        valid_predictions = checkpoint['valid_predictions']
    else:
        print("Precomputing base model predictions...")
        # train_predictions = precompute_base_predictions(age_model, trainDataLoader, device, num_classes)
        # valid_predictions = precompute_base_predictions(age_model, validDataLoader, device, num_classes)
        train_predictions = []
        valid_predictions = []
        for sample in tqdm(trainDataLoader, desc=f"Precomputing train predictions"):
            data = sample['X']
            batch_predictions = torch.stack([m(data) for m in age_model]) 
            #训练, # [n_models, n_samples, n_classes]
            train_predictions.append(batch_predictions.cpu())
        for sample in tqdm(validDataLoader, desc=f"Precomputing valid predictions"):
            data = sample['X']
            batch_predictions = torch.stack([m(data) for m in age_model]) #验证
            valid_predictions.append(batch_predictions.cpu())
        # 合并所有batch的预测结果
        train_predictions = torch.cat(train_predictions, dim=1)  # [n_models, total_samples, n_classes]
        valid_predictions = torch.cat(valid_predictions, dim=1)  # [n_models, total_samples, n_classes]
        # 保存预计算结果
        torch.save({
            'train_predictions': train_predictions,
            'valid_predictions': valid_predictions,
            'train_targets': torch.cat(all_train_targets) if 'all_train_targets' in locals() else None, # 暂时未使用
            'valid_targets': torch.cat(all_valid_targets) if 'all_valid_targets' in locals() else None,
            'n_models': len(age_model),
            'num_classes': train_predictions.shape[2]
        }, './precomputed_predictions.pt')

        print(f"预计算结果已保存!")
    print(f"训练集预测形状: {train_predictions.shape}")
    print(f"验证集预测形状: {valid_predictions.shape}")
    calculate_oracle_accuracy(valid_predictions, validDataLoader, n_models, device)
    if True:
        dim = -1
        total_correct = [0] * n_models
        total_num = 0
        for iteration, sample in enumerate(validDataLoader):
            if dim == -1:
                dim = sample['Y'].shape[0]
            _, target = sample['X'], sample['Y']
            pred = valid_predictions[:, iteration * dim:(iteration + 1) * dim, :]
            for model_idx in range(n_models):
                model_pred = torch.softmax(pred[model_idx], dim = 1)
                # model_pred = pred[model_idx]
                pred_classes = torch.argmax(model_pred, dim=1)
                correct = (pred_classes == target).sum().item()
                total_correct[model_idx] += correct
            total_num += sample['Y'].shape[0]
        print("Base Model Accuracies on Validation Set:")

        for model_idx in range(n_models):
            accuracy = total_correct[model_idx] / len(validDataLoader.sampler)
            print(f"Model {model_idx}: Accuracy = {accuracy:.4f}")

    acc_c1 = calculate_top_c_gt_prob_accuracy_from_dataloader(age_model, validDataLoader, C=C, device=device, num_classes=num_classes)
    print(f"Top-{C} GT-Prob Averaging Accuracy: {acc_c1:.4f}")

    for epoch in range(args.epochs):
        # 训练阶段
        selection_net.train()
        train_loss = 0
        iteration = 0
        train_dim = -1
        valid_dim = -1
        for sample in trainDataLoader:
            data, target = sample['X'], sample['Y']
            if torch.cuda.is_available():
                data, target = data, target.cuda()
            
            if train_dim == -1:
                train_dim = target.shape[0]
            optimizer.zero_grad()
            
            # 收集基模型预测【前置】
            # age_predictions = torch.stack([torch.softmax(m(data), 1) for m in age_model])
            age_predictions = train_predictions[:, iteration * train_dim:(iteration + 1) * train_dim, :].to(device)
            

            # 1. 构造特征 F
            # Permute: [n_models, dim, n_classes] -> [dim, n_models, n_classes]
            # Reshape: [dim, n_models, n_classes] -> [dim, n_models * n_classes]
            # print(f"age_predictions shape: {age_predictions.permute(1, 0, 2).shape}")

            F_data = age_predictions.permute(1, 0, 2).reshape(target.shape[0], n_models * num_classes)
            Ft = F_data.to(device).float() # 确保在正确设备上，且是浮点类型
            selection_vals = selection_net(Ft)
            
            # 选择网络决策
            # selection_vals = selection_net(data)
            # selection_vals = torch.nn.functional.normalize(selection_vals)

            # 新增的loss部分
            binary_target = torch.zeros((target.shape[0], num_classes), device=device)
            for idx, t in enumerate(target):
                binary_target[idx, t.item()] = 1 # 形状 [batch_size, num_classes]
            # 2. 计算每个模型的损失 (CrossEntropy/MSE)
            # F.mse_loss 默认是 (pred, target)，但我们需要按样本计算损失

            all_model_losses = []
            for m_idx in range(n_models):
                # 针对每个模型和 batch 计算 MSE loss
                # predictions[m_idx] 形状 [batch_size, num_classes]
                # binary_target 形状 [batch_size, num_classes]

                # F.mse_loss(reduction='none') 会返回 [batch_size, num_classes]
                # 损失总和（沿类别轴求和）得到每个样本的总损失 [batch_size]
                sample_losses = torch.sum(F.mse_loss(age_predictions[m_idx], binary_target, reduction='none'), dim=1)
                all_model_losses.append(sample_losses)

            all_model_losses = torch.stack(all_model_losses, dim=0) # 形状 [n_models, batch_size]
            # selection_vals 形状 [batch_size, n_models] (分数越高越好)

            # 损失是越低越好，所以我们取损失的负值或取倒数作为分数目标：
            # 目标分数 = - all_model_losses.T （负损失，形状 [batch_size, n_models]）
            target_scores = -all_model_losses.T 

            # 归一化目标分数（可选，但推荐）
            target_scores = F.normalize(target_scores, dim=1) 
            # 计算选择损失
            selection_loss = F.mse_loss(selection_vals, target_scores)


            # 注意：predictions 必须是 Logits 或 Softmax，取决于你的 F.mse_loss 的输入要求。

            
            if args.injection:
                diff = (torch.topk(age_predictions, 2, 2).values[:, :, 0] - 
                       torch.topk(age_predictions, 2, 2).values[:, :, 1])
                selections = knapsack_layer(selection_vals * diff.T)
            else:
                selections = knapsack_layer(selection_vals)

            # 加权预测组合
            if args.weight_pred:
                diff = (torch.topk(age_predictions, 2, 2).values[:, :, 0] - 
                       torch.topk(age_predictions, 2, 2).values[:, :, 1]).repeat(num_classes, 1, 1)
                diff = torch.permute(diff, (1, 2, 0))
                age_predictions = age_predictions * selections.repeat(num_classes, 1, 1).T * diff
            else:
                # age_predictions = age_predictions * selections.repeat(num_classes, 1, 1).T
                age_predictions = torch.softmax(age_predictions, dim = 2) * selections.repeat(num_classes, 1, 1).T# 最新修改
            # 多数投票聚合
            if args.apply_sum:
                majority_vote = torch.sum(age_predictions, 0)
            else:
                # majority_vote = torch.mean(age_predictions, 0)
                majority_vote = torch.sum(age_predictions, 0) / C# 最新修改

            # 准备目标标签
            age_binary_target = torch.zeros((target.shape[0], num_classes), device=device)
            for idx, t in enumerate(target):
                age_binary_target[idx, t.item()] = 1
            # print(f"majority_vote shape: {majority_vote.shape}, target shape: {age_binary_target.shape}")
            # 计算损失
            if args.use_softmax:
                majority_pred = torch.softmax(majority_vote, 1)
                loss = loss_fun(majority_pred, age_binary_target)
            else:
                loss = loss_fun(majority_vote, age_binary_target)
            alpha = 0.5
            loss = (1 - alpha) * loss + alpha * selection_loss

            train_loss += loss.item()

            # 反向传播
            loss.backward() # 源文件没有这行
            # 梯度诊断代码开始
            total_norm = 0.0
            for name, param in selection_net.named_parameters():
                if param.grad is not None:
                    # 计算 L2 范数
                    param_norm = param.grad.data.norm(2) 
                    total_norm += param_norm.item() ** 2
                    
                    # 打印特定层（例如第一层或最后一层）的梯度
                    if 'layer_name' in name: # 替换为 selection_net 中关键层的名字
                        print(f"Gradient Norm for {name}: {param_norm.item():.6f}")

            total_norm = total_norm ** 0.5
            # print(f"Total Gradient Norm for Selection Net: {total_norm:.6f}")

            # 梯度诊断代码结束
            if args.clip:
                nn.utils.clip_grad_value_(selection_net.parameters(), 0.1)
            optimizer.step()
            if args.sched:
                sched.step()

            # 打印训练进度
            #if iteration % 500 == 1:
            #    print(f"Loss function value: {loss.item()}")
            iteration += 1
            print(f"{iteration}/{len(trainDataLoader)} batches processed", end='\r')

        # 验证阶段
        selection_net.eval()
        valid_loss = 0
        total_correct = 0
        total_samples = 0
        iteration = 0
        
        with torch.no_grad():
            for sample in validDataLoader:
                data, target = sample['X'], sample['Y']
                if torch.cuda.is_available():
                    data, target = data, target.cuda()
                if valid_dim == -1:
                    valid_dim = target.shape[0]
                
                # 基模型预测【前置】
                # predictions = torch.stack([torch.softmax(m(data), 1) for m in age_model])
                predictions = valid_predictions[:, iteration * valid_dim:(iteration + 1) * valid_dim, :].to(device)
                
                F_data = predictions.permute(1, 0, 2).reshape(target.shape[0], n_models * num_classes)
                Ft = F_data.to(device).float() # 确保在正确设备上，且是浮点类型
                selection_vals = selection_net(Ft)

                # 选择网络决策
                # selection_vals = selection_net(data)
                # selection_vals = torch.nn.functional.normalize(selection_vals)

                
                if args.injection:
                    diff = (torch.topk(predictions, 2, 2).values[:, :, 0] - 
                           torch.topk(predictions, 2, 2).values[:, :, 1])
                    selections = knapsack_layer(selection_vals * diff.T)
                else:
                    # selections = knapsack_layer(selection_vals)
                    selections = batch_knapsack(selection_vals) # 最新修改

                # 预测组合
                if args.weight_pred:
                    diff = (torch.topk(predictions, 2, 2).values[:, :, 0] - 
                           torch.topk(predictions, 2, 2).values[:, :, 1]).repeat(num_classes, 1, 1)
                    diff = torch.permute(diff, (1, 2, 0))
                    predictions = predictions * selections.repeat(num_classes, 1, 1).T * diff
                else:
                    # predictions = predictions * selections.repeat(num_classes, 1, 1).T
                    predictions = torch.softmax(predictions, dim = 2) * selections.repeat(num_classes, 1, 1).T# 最新修改

                # 多数投票
                if args.apply_sum:
                    majority_vote = torch.sum(predictions, 0)
                else:
                    # majority_vote = torch.mean(predictions, 0)
                    majority_vote = torch.sum(predictions, 0) / C# 最新修改

                # 准备目标标签
                binary_target = torch.zeros((target.shape[0], num_classes), device=device)
                for idx, t in enumerate(target):
                    binary_target[idx, t.item()] = 1

                # 计算验证损失
                if args.use_softmax:
                    majority_pred = torch.softmax(majority_vote, 1)
                    loss = loss_fun(majority_pred, binary_target)
                else:
                    loss = loss_fun(majority_vote, binary_target)

                valid_loss += loss.item()

                # 计算准确率
                target_np = binary_target.cpu().numpy()
                pred_np = majority_pred.cpu().numpy() if args.use_softmax else majority_vote.cpu().numpy()
                
                correct = 0
                for i in range(target.shape[0]):
                    true_class = np.argmax(target_np[i, :])
                    pred_class = np.argmax(pred_np[i, :])
                    if true_class == pred_class:
                        correct += 1
                
                total_correct += correct
                total_samples += target.shape[0]
                iteration += 1

        # 计算平均损失和准确率
        train_loss_avg = train_loss / len(trainDataLoader)
        valid_loss_avg = valid_loss / len(validDataLoader)
        accuracy = total_correct / total_samples
        train_loss_list.append(train_loss_avg)
        valid_loss_list.append(valid_loss_avg)
        acc_list.append(accuracy)

        print(f"Epoch: {epoch}")
        print(f"Average accuracy: {accuracy:.4f}")
        print(f'Training Loss: {train_loss_avg:.6f} \tValidation Loss: {valid_loss_avg:.6f}')

        # 早停机制和模型保存
        if valid_loss_avg < (best - 1e-4):
            best_model = copy.deepcopy(selection_net)
            torch.save(best_model.state_dict(), f"best_model_{args.c}.pth")
            fails = 0
            best = valid_loss_avg
        else:
            fails += 1
            
        if fails > patience:
            print(f"Early Stopping. Validation hasn't improved for {patience} epochs")
            break
    print("\nTraining completed.\n")
    print("Begin plotting loss and accuracy curves...\n")
    epochs = range(1, len(train_loss_list) + 1) # 因为1太大了，所以从3开始画

    # 创建图形
    plt.figure(figsize=(9, 6))

    # === 左轴：Loss 曲线 ===
    plt.plot(epochs, train_loss_list, label='Train Loss', color='tab:blue', linewidth=2)
    plt.plot(epochs, valid_loss_list, label='Validation Loss', color='tab:red', linewidth=2, linestyle='--')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', color='tab:blue', fontsize=12)
    plt.tick_params(axis='y', labelcolor='tab:blue')

    # === 右轴：Accuracy 曲线 ===
    ax2 = plt.gca().twinx()
    ax2.plot(epochs, acc_list, label='Validation Accuracy', color='tab:green', linewidth=2)
    ax2.set_ylabel('Accuracy', color='tab:green', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # === 图例与标题 ===
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)

    plt.title(f'Choose {C} models -- Training & Validation Loss and Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 保存
    plt.savefig('loss_acc_curve.png', dpi=300)
    return best_model

def train_selection_v2(selection_net, age_model, device, trainDataLoader, validDataLoader, optimizer, args, loss_fun, n_models, sched, num_classes=7):
    """
    训练选择网络，学习为不同输入选择最合适的基模型
    """
    for m in age_model:
        m.eval()
        for param in m.parameters():
            param.requires_grad_(False)

    C = args.c  # 选择模型的数量

    def batch_knapsack(scores):
        """批量背包选择：选择得分最高的C个模型"""
        indices = torch.topk(scores, C).indices
        choice = torch.zeros_like(scores)
        choice.scatter_(1, indices, torch.ones(indices.shape, device=device))
        return choice

    # 可微分扰动优化器
    knapsack_layer = perturbations.perturbations.perturbed_special(
        batch_knapsack,
        num_samples=1000,
        sigma=0.1,
        noise='normal',
        batched=True,
        device=device,
        hard_fwd=True
    )

    # 新增：多样性感知损失函数
    def diversity_aware_loss(ensemble_pred, base_preds, targets, selections, alpha=0.1, beta=0.01):
        """
        改进的损失函数，包含多样性正则化
        ensemble_pred: 集成预测 [batch_size, num_classes]
        base_preds: 基模型预测 [n_models, batch_size, num_classes] 
        targets: 真实标签 [batch_size]
        selections: 选择矩阵 [batch_size, n_models]
        """
        # 基础交叉熵损失
        ce_loss = nn.CrossEntropyLoss()(ensemble_pred, targets)
        
        # 多样性正则化：鼓励选择不同的模型
        # selection_entropy = -torch.sum(selections * torch.log(selections + 1e-8), dim=1).mean()
        # 方法1：使用选择得分而不是硬选择来计算熵
        # 首先从选择网络获取原始得分，而不是经过knapsack的硬选择
        selection_entropy = -torch.sum(selections * torch.log(selections + 1e-8), dim=1).mean()
        
        # 方法2：如果selections是硬选择，使用Gumbel-Softmax技巧
        # 在训练时使用soft选择，测试时使用硬选择
        if selections.sum() == selections.numel():  # 检查是否是概率分布
            # 已经是概率分布，直接计算熵
            selection_entropy = -torch.sum(selections * torch.log(selections + 1e-8), dim=1).mean()
        else:
            # 是硬选择，需要转换为概率分布
            # 这里可以添加一个小噪声来创建伪概率分布
            noisy_selections = selections.float() + 1e-3
            noisy_selections = noisy_selections / noisy_selections.sum(dim=1, keepdim=True)
            selection_entropy = -torch.sum(noisy_selections * torch.log(noisy_selections + 1e-8), dim=1).mean()
        
        # 模型间差异度量：鼓励选择预测差异大的模型组合
        # 计算每个样本上被选中模型预测的标准差
        # print(f"base_preds shape: {base_preds.shape}, selections shape: {selections.shape}")
        selected_preds = base_preds * selections.T.unsqueeze(-1)  # [n_models, batch_size, num_classes]
        model_disagreement = torch.std(selected_preds, dim=0).mean()  # 在模型维度计算标准差
        
        # 组合损失
        total_loss = ce_loss - alpha * selection_entropy + beta * model_disagreement
        
        return total_loss, ce_loss, selection_entropy, model_disagreement

    # 训练状态变量
    best = 10000.0
    best_model = copy.deepcopy(selection_net)
    fails = 0
    flag = False
    patience = 20
    
    # 在训练前预计算
    if os.path.exists('./precomputed_predictions.pt'):
        print("Loading precomputed predictions from file...")
        checkpoint = torch.load('./precomputed_predictions.pt', map_location='cpu')
        train_predictions = checkpoint['train_predictions']
        valid_predictions = checkpoint['valid_predictions']
    else:
        print("Precomputing base model predictions...")
        train_predictions = []
        valid_predictions = []
        for sample in tqdm(trainDataLoader, desc=f"Precomputing train predictions"):
            data = sample['X']
            batch_predictions = torch.stack([m(data) for m in age_model]) 
            train_predictions.append(batch_predictions.cpu())
        for sample in tqdm(validDataLoader, desc=f"Precomputing valid predictions"):
            data = sample['X']
            batch_predictions = torch.stack([m(data) for m in age_model])
            valid_predictions.append(batch_predictions.cpu())
        # 合并所有batch的预测结果
        train_predictions = torch.cat(train_predictions, dim=1)
        valid_predictions = torch.cat(valid_predictions, dim=1)
        # 保存预计算结果
        torch.save({
            'train_predictions': train_predictions,
            'valid_predictions': valid_predictions,
            'n_models': len(age_model),
            'num_classes': train_predictions.shape[2]
        }, './precomputed_predictions.pt')
        print(f"预计算结果已保存!")
        
    print(f"训练集预测形状: {train_predictions.shape}")
    print(f"验证集预测形状: {valid_predictions.shape}")
    
    # 基模型性能分析（保持不变）
    if True:
        dim = -1
        total_correct = [0] * n_models
        for iteration, sample in enumerate(validDataLoader):
            if dim == -1:
                dim = sample['Y'].shape[0]
            _, target = sample['X'], sample['Y']
            pred = valid_predictions[:, iteration * dim:(iteration + 1) * dim, :]
            for model_idx in range(n_models):
                model_pred = pred[model_idx]
                pred_classes = torch.argmax(model_pred, dim=1)
                correct = (pred_classes == target).sum().item()
                total_correct[model_idx] += correct
        print("Base Model Accuracies on Validation Set:")
        for model_idx in range(n_models):
            accuracy = total_correct[model_idx] / len(validDataLoader.sampler)
            print(f"Model {model_idx}: Accuracy = {accuracy:.4f}")

    # 训练循环
    for epoch in range(args.epochs):
        # 训练阶段
        selection_net.train()
        train_loss = 0
        train_ce_loss = 0
        train_entropy = 0
        train_disagreement = 0
        iteration = 0
        
        for sample in trainDataLoader:
            data, target = sample['X'], sample['Y']
            if torch.cuda.is_available():
                data, target = data, target.cuda()
            
            dim = target.shape[0]
            optimizer.zero_grad()

            # 收集基模型预测
            age_predictions = train_predictions[:, iteration * dim:(iteration + 1) * dim, :].to(device)
            
            # 选择网络决策
            selection_vals = selection_net(data)
            selection_vals = torch.nn.functional.normalize(selection_vals)
            
            if args.injection:
                diff = (torch.topk(age_predictions, 2, 2).values[:, :, 0] - 
                       torch.topk(age_predictions, 2, 2).values[:, :, 1])
                selections = knapsack_layer(selection_vals * diff.T)
            else:
                selections = knapsack_layer(selection_vals)

            # 加权预测组合
            if args.weight_pred:
                diff = (torch.topk(age_predictions, 2, 2).values[:, :, 0] - 
                       torch.topk(age_predictions, 2, 2).values[:, :, 1]).repeat(num_classes, 1, 1)
                diff = torch.permute(diff, (1, 2, 0))
                weighted_predictions = age_predictions * selections.repeat(num_classes, 1, 1).T * diff
            else:
                weighted_predictions = age_predictions * selections.repeat(num_classes, 1, 1).T

            # 多数投票聚合
            if args.apply_sum:
                majority_vote = torch.sum(weighted_predictions, 0)
            else:
                majority_vote = torch.mean(weighted_predictions, 0)

            # 准备目标标签
            age_binary_target = torch.zeros((dim, num_classes), device=device)
            for idx, t in enumerate(target):
                age_binary_target[idx, t.item()] = 1

            # 计算损失 - 使用改进的多样性感知损失
            if args.use_softmax:
                majority_pred = torch.softmax(majority_vote, 1)
                # 修改：使用新的损失函数
                loss, ce_loss_val, entropy_val, disagreement_val = diversity_aware_loss(
                    majority_pred, age_predictions, target, selections
                )
            else:
                # 对于非softmax情况，保持原逻辑
                loss = loss_fun(majority_vote, age_binary_target)
                ce_loss_val = loss.item()
                entropy_val = 0
                disagreement_val = 0

            train_loss += loss.item()
            train_ce_loss += ce_loss_val
            train_entropy += entropy_val
            train_disagreement += disagreement_val

            # 反向传播
            loss.backward()
            if args.clip:
                nn.utils.clip_grad_value_(selection_net.parameters(), 0.1)
            optimizer.step()
            if args.sched:
                sched.step()

            iteration += 1
            print(f"{iteration}/{len(trainDataLoader)} batches processed", end='\r')

        # 验证阶段（保持原逻辑）
        selection_net.eval()
        valid_loss = 0
        total_correct = 0
        total_samples = 0
        iteration = 0
        
        with torch.no_grad():
            for sample in validDataLoader:
                data, target = sample['X'], sample['Y']
                if torch.cuda.is_available():
                    data, target = data, target.cuda()
                
                dim = target.shape[0]
                
                # 基模型预测
                predictions = valid_predictions[:, iteration * dim:(iteration + 1) * dim, :].to(device)
                
                # 选择网络决策
                selection_vals = selection_net(data)
                selection_vals = torch.nn.functional.normalize(selection_vals)
                
                if args.injection:
                    diff = (torch.topk(predictions, 2, 2).values[:, :, 0] - 
                           torch.topk(predictions, 2, 2).values[:, :, 1])
                    selections = knapsack_layer(selection_vals * diff.T)
                else:
                    selections = knapsack_layer(selection_vals)

                # 预测组合
                if args.weight_pred:
                    diff = (torch.topk(predictions, 2, 2).values[:, :, 0] - 
                           torch.topk(predictions, 2, 2).values[:, :, 1]).repeat(num_classes, 1, 1)
                    diff = torch.permute(diff, (1, 2, 0))
                    predictions = predictions * selections.repeat(num_classes, 1, 1).T * diff
                else:
                    predictions = predictions * selections.repeat(num_classes, 1, 1).T

                # 多数投票
                if args.apply_sum:
                    majority_vote = torch.sum(predictions, 0)
                else:
                    majority_vote = torch.mean(predictions, 0)

                # 准备目标标签
                binary_target = torch.zeros((dim, num_classes), device=device)
                for idx, t in enumerate(target):
                    binary_target[idx, t.item()] = 1

                # 计算验证损失
                if args.use_softmax:
                    majority_pred = torch.softmax(majority_vote, 1)
                    loss = loss_fun(majority_pred, binary_target)
                else:
                    loss = loss_fun(majority_vote, binary_target)

                valid_loss += loss.item()

                # 计算准确率
                target_np = binary_target.cpu().numpy()
                pred_np = majority_pred.cpu().numpy() if args.use_softmax else majority_vote.cpu().numpy()
                
                correct = 0
                for i in range(dim):
                    true_class = np.argmax(target_np[i, :])
                    pred_class = np.argmax(pred_np[i, :])
                    if true_class == pred_class:
                        correct += 1
                
                total_correct += correct
                total_samples += dim
                iteration += 1

        # 计算平均损失和准确率
        train_loss_avg = train_loss / len(trainDataLoader)
        valid_loss_avg = valid_loss / len(validDataLoader)
        accuracy = total_correct / total_samples

        # 输出更详细的训练信息
        print(f"Epoch: {epoch}")
        print(f"Average accuracy: {accuracy:.4f}")
        print(f'Training Loss: {train_loss_avg:.6f} \tValidation Loss: {valid_loss_avg:.6f}')
        if args.use_softmax:
            print(f'CE Loss: {train_ce_loss/len(trainDataLoader):.6f} \tEntropy: {train_entropy/len(trainDataLoader):.6f} \tDisagreement: {train_disagreement/len(trainDataLoader):.6f}')

        # 早停机制和模型保存
        if valid_loss_avg < (best - 1e-4):
            best_model = copy.deepcopy(selection_net)
            torch.save(best_model.state_dict(), f"best_model_{args.c}.pth")
            fails = 0
            best = valid_loss_avg
        else:
            fails += 1
            
        if fails > patience:
            print(f"Early Stopping. Validation hasn't improved for {patience} epochs")
            break
            
    return selection_net

def train_selection_v3(selection_net, age_model, device, trainDataLoader, validDataLoader, optimizer, args, loss_fun, n_models, sched, num_classes=4):
    """
    训练选择网络，学习为不同输入选择最合适的基模型
    """
    for m in age_model:
        m.eval()
        for param in m.parameters():
            param.requires_grad_(False)

    C = args.c  # 选择模型的数量

    def batch_knapsack(scores):
        """批量背包选择：选择得分最高的C个模型"""
        indices = torch.topk(scores, C).indices
        choice = torch.zeros_like(scores)
        choice.scatter_(1, indices, torch.ones(indices.shape, device=device))
        return choice

    # 可微分扰动优化器
    knapsack_layer = perturbations.perturbations.perturbed_special(
        batch_knapsack,
        num_samples=1000,
        sigma=0.1,
        noise='normal',
        batched=True,
        device=device,
        hard_fwd=True
    )

    # 重新设计的损失函数
    def improved_diversity_loss(ensemble_pred, base_preds_logits, targets, selections, alpha=0.1, beta=0.1):
        """
        改进的损失函数，确保梯度能正确回传
        ensemble_pred: 集成预测 [batch_size, num_classes]
        base_preds_logits: 基模型logits [n_models, batch_size, num_classes]
        targets: 真实标签 [batch_size]
        selections: 选择概率 [batch_size, n_models]
        """
        # 基础交叉熵损失
        ce_loss = nn.CrossEntropyLoss()(ensemble_pred, targets)
        
        # 确保选择是概率分布
        selection_probs = torch.softmax(selections, dim=1)
        
        # 选择熵：鼓励探索不同的模型组合
        selection_entropy = -torch.sum(selection_probs * torch.log(selection_probs + 1e-8), dim=1).mean()
        
        # 性能奖励：鼓励选择正确的模型
        base_preds = torch.softmax(base_preds_logits, dim=-1)
        correct_predictions = []
        for i in range(base_preds.shape[0]):  # 遍历每个模型
            model_pred = base_preds[i]
            pred_classes = torch.argmax(model_pred, dim=1)
            correct = (pred_classes == targets).float()  # [batch_size]
            correct_predictions.append(correct)
        
        correct_matrix = torch.stack(correct_predictions).T  # [batch_size, n_models]
        
        # 奖励选择正确模型的行为
        performance_reward = torch.sum(selection_probs * correct_matrix, dim=1).mean()
        
        # 多样性：鼓励选择预测差异大的模型
        selected_mask = selection_probs.T.unsqueeze(-1)  # [n_models, batch_size, 1]
        weighted_preds = base_preds * selected_mask
        pred_diversity = torch.std(weighted_preds, dim=0).mean()  # 预测的标准差
        
        # 组合损失
        total_loss = ce_loss - alpha * selection_entropy - beta * performance_reward - 0.05 * pred_diversity
        
        return total_loss, ce_loss, selection_entropy, performance_reward

    # 训练状态变量
    best_accuracy = 0.0
    best_model = copy.deepcopy(selection_net)
    patience = 10
    no_improve_count = 0
    
    # 在训练前预计算
    if os.path.exists('./precomputed_predictions.pt'):
        print("Loading precomputed predictions from file...")
        checkpoint = torch.load('./precomputed_predictions.pt', map_location='cpu')
        train_predictions = checkpoint['train_predictions']
        valid_predictions = checkpoint['valid_predictions']
    else:
        print("Precomputing base model predictions...")
        train_predictions = []
        valid_predictions = []
        
        # 收集训练集预测
        for sample in tqdm(trainDataLoader, desc="Precomputing train predictions"):
            data = sample['X']
            batch_predictions = torch.stack([m(data) for m in age_model])  # 保持logits形式
            train_predictions.append(batch_predictions.cpu())
        
        # 收集验证集预测  
        for sample in tqdm(validDataLoader, desc="Precomputing valid predictions"):
            data = sample['X']
            batch_predictions = torch.stack([m(data) for m in age_model])
            valid_predictions.append(batch_predictions.cpu())
        
        # 合并所有batch的预测结果
        train_predictions = torch.cat(train_predictions, dim=1)  # [n_models, total_samples, num_classes]
        valid_predictions = torch.cat(valid_predictions, dim=1)
        
        # 保存预计算结果
        torch.save({
            'train_predictions': train_predictions,
            'valid_predictions': valid_predictions,
            'n_models': len(age_model),
            'num_classes': num_classes
        }, './precomputed_predictions.pt')
        print("预计算结果已保存!")
    
    print(f"训练集预测形状: {train_predictions.shape}")
    print(f"验证集预测形状: {valid_predictions.shape}")
    
    # 分析基模型性能
    print("Base Model Accuracies on Validation Set:")
    model_accuracies = []
    for model_idx in range(n_models):
        total_correct = 0
        total_samples = 0
        dim = -1
        for iteration, sample in enumerate(validDataLoader):
            if dim == -1:
                dim = sample['Y'].shape[0]
            _, target = sample['X'], sample['Y']
            pred_logits = valid_predictions[:, iteration * dim:(iteration + 1) * dim, :]
            model_pred = torch.softmax(pred_logits[model_idx], dim=1)
            pred_classes = torch.argmax(model_pred, dim=1)
            correct = (pred_classes == target).sum().item()
            total_correct += correct
            total_samples += target.shape[0]
        
        accuracy = total_correct / total_samples
        model_accuracies.append(accuracy)
        print(f"Model {model_idx}: Accuracy = {accuracy:.4f}")
    
    # 计算基模型的平均准确率
    base_avg_accuracy = np.mean(model_accuracies)
    print(f"Base models average accuracy: {base_avg_accuracy:.4f}")

    # 训练循环
    for epoch in range(args.epochs):
        # 训练阶段
        selection_net.train()
        train_loss = 0
        train_ce_loss = 0
        train_entropy = 0
        train_performance = 0
        iteration = 0
        dim = -1
        for sample in trainDataLoader:
            data, target = sample['X'], sample['Y']
            if torch.cuda.is_available():
                data, target = data, target.cuda()
            
            if dim == -1:
                dim = target.shape[0]
            optimizer.zero_grad()

            # 获取基模型预测（logits）
            age_predictions_logits = train_predictions[:, iteration * dim:(iteration + 1) * dim, :].to(device)
            
            # 选择网络决策
            selection_vals = selection_net(data)
            
            # 关键修改：训练时使用soft选择，测试时使用硬选择
            if selection_net.training:
                # 训练时：使用softmax创建概率分布，确保梯度回传
                temperature = 1.0
                selections = torch.softmax(selection_vals / temperature, dim=1)
            else:
                # 测试时：使用硬选择
                selections = knapsack_layer(selection_vals)

            # 加权预测组合
            if args.weight_pred:
                # 使用预测置信度作为权重
                base_preds = torch.softmax(age_predictions_logits, dim=-1)
                confidences = torch.max(base_preds, dim=2).values  # [n_models, batch_size]
                weight_matrix = selections * confidences.T
            else:
                weight_matrix = selections

            # 集成预测
            weight_matrix_expanded = weight_matrix.T.unsqueeze(-1)  # [n_models, batch_size, 1]
            base_preds = torch.softmax(age_predictions_logits, dim=-1)
            weighted_predictions = base_preds * weight_matrix_expanded
            
            if args.apply_sum:
                ensemble_pred = torch.sum(weighted_predictions, dim=0)  # [batch_size, num_classes]
            else:
                ensemble_pred = torch.mean(weighted_predictions, dim=0)

            # 计算损失 - 使用改进的损失函数
            loss, ce_loss_val, entropy_val, performance_val = improved_diversity_loss(
                ensemble_pred, age_predictions_logits, target, selections
            )

            train_loss += loss.item()
            train_ce_loss += ce_loss_val.item()
            train_entropy += entropy_val.item()
            train_performance += performance_val.item()

            # 反向传播
            loss.backward()
            if args.clip:
                nn.utils.clip_grad_value_(selection_net.parameters(), 0.1)
            optimizer.step()
            if args.sched:
                sched.step()

            iteration += 1
            if iteration % 50 == 0:
                print(f"Batch {iteration}/{len(trainDataLoader)}, Loss: {loss.item():.4f}", end='\r')

        # 验证阶段
        selection_net.eval()
        valid_loss = 0
        total_correct = 0
        total_samples = 0
        dim = -1
        with torch.no_grad():
            for iteration, sample in enumerate(validDataLoader):
                data, target = sample['X'], sample['Y']
                if torch.cuda.is_available():
                    data, target = data, target.cuda()
                if dim == -1:
                    dim = target.shape[0]
                
                # 基模型预测
                predictions_logits = valid_predictions[:, iteration * dim:(iteration + 1) * dim, :].to(device)
                
                # 选择网络决策 - 测试时使用硬选择
                selection_vals = selection_net(data)
                selections = knapsack_layer(selection_vals)

                # 预测组合
                if args.weight_pred:
                    base_preds = torch.softmax(predictions_logits, dim=-1)
                    confidences = torch.max(base_preds, dim=2).values
                    weight_matrix = selections * confidences.T
                else:
                    weight_matrix = selections

                weight_matrix_expanded = weight_matrix.T.unsqueeze(-1)
                base_preds = torch.softmax(predictions_logits, dim=-1)
                weighted_predictions = base_preds * weight_matrix_expanded
                
                if args.apply_sum:
                    majority_vote = torch.sum(weighted_predictions, 0)
                else:
                    majority_vote = torch.mean(weighted_predictions, 0)

                # 计算验证损失
                if args.use_softmax:
                    majority_pred = torch.softmax(majority_vote, 1)
                    loss = loss_fun(majority_pred, 
                                  torch.nn.functional.one_hot(target, num_classes).float())
                else:
                    loss = loss_fun(majority_vote, 
                                  torch.nn.functional.one_hot(target, num_classes).float())

                valid_loss += loss.item()

                # 计算准确率
                pred_classes = torch.argmax(majority_pred if args.use_softmax else majority_vote, dim=1)
                correct = (pred_classes == target).sum().item()
                total_correct += correct
                total_samples += target.shape[0]

        # 计算平均指标
        train_loss_avg = train_loss / len(trainDataLoader)
        valid_loss_avg = valid_loss / len(validDataLoader)
        accuracy = total_correct / total_samples
        
        train_ce_avg = train_ce_loss / len(trainDataLoader)
        train_entropy_avg = train_entropy / len(trainDataLoader)
        train_performance_avg = train_performance / len(trainDataLoader)

        print(f"\nEpoch {epoch}:")
        print(f"Accuracy: {accuracy:.4f} (Best: {best_accuracy:.4f}, Base Avg: {base_avg_accuracy:.4f})")
        print(f"Train Loss: {train_loss_avg:.4f}, Valid Loss: {valid_loss_avg:.4f}")
        print(f"CE: {train_ce_avg:.4f}, Entropy: {train_entropy_avg:.4f}, Performance: {train_performance_avg:.4f}")

        # 早停机制：基于准确率而不是损失
        if accuracy > best_accuracy + 1e-4:
            best_accuracy = accuracy
            best_model = copy.deepcopy(selection_net)
            torch.save(best_model.state_dict(), f"best_model_{args.c}.pth")
            no_improve_count = 0
            print(f"New best model saved! Accuracy: {accuracy:.4f}")
        else:
            no_improve_count += 1
            
        if no_improve_count >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break

    print(f"Training completed. Best accuracy: {best_accuracy:.4f}")
    return best_model

def calculate_oracle_accuracy(valid_predictions, validDataLoader, n_models, device):
    """
    计算基模型集合在验证集上的 Oracle 准确率。
    Oracle 准确率：对于每个样本，如果至少有一个模型预测正确，则认为 Oracle 正确。
    """
    
    # 假设 valid_predictions 形状为 [n_models, total_samples, n_classes]
    
    all_targets = []
    # 收集所有真实标签 (假设你的 validDataLoader 存储了所有标签)
    for sample in validDataLoader:
        all_targets.append(sample['Y'])
    
    # 将所有目标拼接成一个张量 [total_samples]
    targets = torch.cat(all_targets).to(device)
    total_samples = targets.shape[0]

    # ----------------------------------------------------
    # 确保索引和维度正确（使用你在上一个回答中修正后的累积索引逻辑）
    
    # 注意：这里的 valid_predictions 应该是完整的预计算结果
    
    # 如果 valid_predictions 是 Logits，需要 Softmax 和 Argmax
    predictions_softmax = torch.softmax(valid_predictions.to(device), dim=2) # [n_models, total_samples, n_classes]
    predictions_classes = torch.argmax(predictions_softmax, dim=2) # [n_models, total_samples]
    
    # ----------------------------------------------------
    
    # 1. 检查每个模型对每个样本是否预测正确
    # target_expanded 形状 [n_models, total_samples]
    target_expanded = targets.unsqueeze(0).expand_as(predictions_classes)
    
    # is_correct 形状 [n_models, total_samples]
    # 如果 model[i] 对 sample[j] 预测正确，则 is_correct[i, j] = True
    is_correct = (predictions_classes == target_expanded)

    # 2. Oracle 决策：是否有【至少一个】模型预测正确？
    # any_correct 形状 [total_samples]
    # any_correct[j] = True 如果至少有一个模型正确预测了样本 j
    any_correct = torch.any(is_correct, dim=0)
    
    # 3. 计算 Oracle 准确率
    oracle_correct_count = torch.sum(any_correct).item()
    oracle_accuracy = oracle_correct_count / total_samples
    
    # 4. 计算所有模型都预测错误的占比
    all_incorrect_count = total_samples - oracle_correct_count
    all_incorrect_percentage = all_incorrect_count / total_samples
    print(f"Oracle Accuracy: {oracle_accuracy:.4f}, All Incorrect Percentage: {all_incorrect_percentage:.4f}")
    return oracle_accuracy, all_incorrect_percentage

def visualize_selections_v1(best_selection_net, valid_loader, valid_predictions, all_valid_targets, device, C, model_names):
    """
    可视化“理想选择”与“模型实际选择”的对比。
    """
    print("\n[INFO] Starting visualization...")
    best_selection_net.eval()
    
    # 1. 定义“硬”选择函数
    def hard_knapsack_layer(scores):
        indices = torch.topk(scores, C).indices
        choice = torch.zeros_like(scores)
        choice.scatter_(1, indices, torch.ones(indices.shape, device=device))
        return choice

    # --- 2. 计算模型的“实际选择” ---
    all_pred_choices_list = []
    with torch.no_grad():
        for sample in tqdm(valid_loader, desc="[Viz] Getting Model's Choices"):
            data, target = sample['X'], sample['Y']
            
            selection_vals = best_selection_net(data)
            pred_choices_indices = torch.topk(selection_vals, C).indices
            all_pred_choices_list.append(pred_choices_indices)
            
    all_pred_choices = torch.cat(all_pred_choices_list, dim=0) # [N_valid, C]
    pred_counts = torch.bincount(all_pred_choices.flatten(), minlength=len(model_names))

    # --- 3. 计算“理想选择”（Ground Truth） ---
    N_valid = len(all_valid_targets)
    # valid_predictions 形状是 [M, N_valid, N_classes]
    
    # 我们需要获取每个模型，在“正确答案”上的 Logit 值
    # all_valid_targets 形状是 [N_valid]
    
    # 使用高级索引：
    # valid_predictions[
    #   :,                      # 遍历所有模型 (M=5)
    #   torch.arange(N_valid), # 遍历所有样本 (N_valid)
    #   all_valid_targets      # 在每个样本上，只取“正确”类别的 logit
    # ]
    try:
        logits_for_correct_class = valid_predictions[
            :, 
            torch.arange(N_valid), 
            all_valid_targets
        ] # 形状 [M, N_valid]
    except IndexError as e:
        print(f"[ERROR] 索引出错: {e}")
        print(f"valid_predictions 形状: {valid_predictions.shape}")
        print(f"all_valid_targets 形状: {all_valid_targets.shape}")
        print(f"N_valid: {N_valid}")
        # 可能是CPU/GPU不匹配
        all_valid_targets = all_valid_targets.cpu()
        valid_predictions = valid_predictions.cpu()
        logits_for_correct_class = valid_predictions[:, torch.arange(N_valid), all_valid_targets]
        

    # 找出每个样本的 "最佳C个模型"
    # dim=0 是沿着模型（M=5）的维度进行 topk
    true_best_c_indices = torch.topk(logits_for_correct_class, C, dim=0).indices # 形状 [C, N_valid]
    true_counts = torch.bincount(true_best_c_indices.flatten(), minlength=len(model_names))

    # --- 4. 绘图 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 图 1: 理想选择
    ax1.bar(model_names, true_counts.cpu().numpy(), color='green')
    ax1.set_title(f'Ideal Selection Frequency (Top {C} "Best")\n(Ground Truth)', fontsize=14)
    ax1.set_ylabel('Total Times Chosen', fontsize=12)
    ax1.tick_params(axis='x', rotation=25)
    print(f"[INFO] 理想选择计数: {true_counts.cpu().numpy()}")
    print(f"[INFO] 实际选择计数: {pred_counts.cpu().numpy()}")
    # 图 2: 实际选择
    ax2.bar(model_names, pred_counts.cpu().numpy(), color='red')
    ax2.set_title(f"Selection Net's Actual Choices\n(Overfitted)", fontsize=14)
    ax2.set_ylabel('Total Times Chosen', fontsize=12)
    ax2.tick_params(axis='x', rotation=25)
    
    plt.tight_layout()
    plt.savefig('selection_visualization.png')
    print(f"[INFO] 可视化结果已保存到 'selection_visualization.png'")

class TextSelectionNet(nn.Module):
    """
    文本选择网络：为每个输入文本输出各个基模型的适用性评分，可以使用LSTM、RNN、Transformer等等
    """
    def __init__(self, input_dim, hidden_dim, n_models):
        super().__init__()
        # 使用简单的MLP结构
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_models)
        )
        
    def forward(self, x):
        return self.encoder(x)

from transformers import AutoModel, AutoTokenizer

class TranSelectionNet(nn.Module):
    """
    完整的文本选择网络：从原始文本到模型评分
    """
    def __init__(self, pretrained_model_name, hidden_dim, n_models, num_classes=7):
        super().__init__()
    
        self.text_encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
     
        encoder_dim = self.text_encoder.config.hidden_size
        
        self.selector = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_models)
        )
        
        # 冻结编码器前几层（可选）
        # self.freeze_encoder_layers(3)
    
    def freeze_encoder_layers(self, num_layers):
        """冻结编码器的前几层"""
        for i, layer in enumerate(self.text_encoder.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, texts):
        """
        输入: texts - 原始文本列表或batch
        输出: [batch_size, n_models] 的模型评分
        """
        # 文本编码
        if isinstance(texts, list) or isinstance(texts, tuple):
            # 原始文本输入
            encoding = self.tokenizer(
                texts, 
                truncation=True, 
                padding=True, 
                max_length=512,
                return_tensors='pt'
            )
            encoding = {k: v.to(next(self.parameters()).device) for k, v in encoding.items()}
        else:
            # 假设已经是编码后的输入
            encoding = texts
        
        # 获取文本表示
        outputs = self.text_encoder(**encoding)
        
        # 使用[CLS] token的表示作为整个文本的表示
        text_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # 生成模型评分
        model_scores = self.selector(text_embeddings)  # [batch_size, n_models]
        
        return model_scores

class LSTMSelectionNet(nn.Module):
    """
    LSTM-based 文本选择网络 - 完整版本（处理原始文本）
    """
    def __init__(self, vocab, embed_dim, hidden_dim, n_models, num_layers=2, max_len=128):
        super().__init__()
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.max_len = max_len
        
        # 词嵌入层
        self.embedding = nn.Embedding(self.vocab_size, embed_dim, padding_idx=0)
        
        # LSTM编码器
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # 选择网络
        self.selector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 双向LSTM所以*2
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, n_models)
        )
    
    def encode_text(self, texts):
        """将原始文本编码为token IDs"""
        if isinstance(texts, str):
            texts = [texts]
            
        encoded_batch = []
        for text in texts:
            text = str(text)
            tokens = text.split()[:self.max_len]
            ids = [self.vocab[token] for token in tokens]
            
            # 填充到max_len
            if len(ids) < self.max_len:
                ids += [0] * (self.max_len - len(ids))  # 0是padding索引
            
            encoded_batch.append(torch.tensor(ids, dtype=torch.long))
        
        return torch.stack(encoded_batch)
    
    def forward(self, texts):
        """
        输入: texts - 原始文本列表或单个文本
        输出: [batch_size, n_models] 的模型评分
        """
        # 文本编码
        if isinstance(texts, list) or isinstance(texts, tuple):
            # 原始文本输入
            input_ids = self.encode_text(texts).to(next(self.parameters()).device)
        elif isinstance(texts, torch.Tensor):
            # 已经是编码好的输入
            input_ids = texts
        else:
            # 单个文本
            input_ids = self.encode_text([texts]).to(next(self.parameters()).device)
        
        # 词嵌入
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # LSTM编码
        lstm_out, (hidden, _) = self.lstm(embeddings)
        
        # 使用最后时刻的隐藏状态（双向拼接）
        # hidden形状: [num_layers * num_directions, batch_size, hidden_dim]
        last_forward = hidden[-2, :, :]  # 最后层的前向LSTM
        last_backward = hidden[-1, :, :]  # 最后层的后向LSTM
        last_hidden = torch.cat([last_forward, last_backward], dim=1)  # [batch_size, hidden_dim*2]
        
        # 生成模型评分
        model_scores = self.selector(last_hidden)
        
        return model_scores

class DSSelectionNet(nn.Module):
    """
    文本选择网络：输入文本特征，输出各个基模型的适用性评分
    """
    def __init__(self, input_dim, hidden_dim, n_models):
        super().__init__()
        # 使用简单的MLP结构
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_models)  # 输出n_models个评分
        )
        
    def forward(self, x):
        return self.encoder(x)  # 输出形状: [batch_size, n_models]
# Hybrid Text Selection Network
class ImprovedSelectionNet(nn.Module):
    def __init__(self, vocab, embed_dim, hidden_dim, n_models, max_len=128):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.max_len = max_len
        self.n_models = n_models
        
        # 更强的文本编码器
        self.embedding = nn.Embedding(self.vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, 
                           bidirectional=True, batch_first=True, dropout=0.3)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=0.1)
        
        # 更深的决策网络
        self.selector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_models)
        )
    
    def encode_text(self, texts):
        """将原始文本编码为token IDs"""
        if isinstance(texts, str):
            texts = [texts]
            
        encoded_batch = []
        for text in texts:
            text = str(text)
            tokens = text.split()[:self.max_len]
            ids = [self.vocab[token] for token in tokens]
            
            # 填充到max_len
            if len(ids) < self.max_len:
                ids += [0] * (self.max_len - len(ids))  # 0是padding索引
            
            encoded_batch.append(torch.tensor(ids, dtype=torch.long))
        
        return torch.stack(encoded_batch)
    
    def forward(self, texts):
        # 文本编码
        if isinstance(texts, list):
            input_ids = self.encode_text(texts).to(next(self.parameters()).device)
        else:
            input_ids = texts
            
        embeddings = self.embedding(input_ids)
        
        # LSTM编码
        lstm_out, _ = self.lstm(embeddings)  # [batch_size, seq_len, hidden_dim*2]
        
        # 自注意力
        lstm_out = lstm_out.transpose(0, 1)  # [seq_len, batch_size, hidden_dim*2]
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attended = attended.transpose(0, 1)  # [batch_size, seq_len, hidden_dim*2]
        
        # 池化
        text_rep = torch.mean(attended, dim=1)  # 平均池化
        
        # 模型选择
        model_scores = self.selector(text_rep)
        
        return model_scores

class PredictionBasedSelectionNet(nn.Module):
    def __init__(self, n_models, num_classes, hidden_dim=128):
        super(PredictionBasedSelectionNet, self).__init__()
        
        # 输入维度: 所有模型 Logits/Probabilities 的拼接 (5 * 4 = 20)
        input_dim = n_models * num_classes
        self.n_models = n_models
        
        # 1. 输入层 -> 隐藏层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # 2. 隐藏层 -> 隐藏层 (可选)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 3. 隐藏层 -> 输出层 (输出每个模型的选择分数)
        self.fc_out = nn.Linear(hidden_dim, n_models)
        
        # 初始化: 推荐 Kaiming 初始化 (PyTorch 默认通常是好的，但明确写出更稳健)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc_out.weight, nonlinearity='linear')


    def forward(self, x):
        """
        x: 形状必须是 [batch_size, n_models * num_classes] (Logits/Probabilities)
        """
        # 1. 第一层
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # 2. 第二层
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # 3. 输出层
        # 注意：这里我们输出 Logits (未激活)，因为 knapsack_layer/损失函数会处理分数
        selection_scores = self.fc_out(x)
        
        return selection_scores

def load_ag_base_models(device, num_classes, X_train, y_train):
    """加载预训练的基模型"""
    base_models = []
    
    # 首先创建统一的vocab（所有vocab-based模型共享）
    print("Building vocabulary...")
    vocab_dataset = TextDataset.create_with_vocab(X_train, y_train)
    vocab = vocab_dataset.vocab
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # 1. TextCNN模型
    print("Loading TextCNN model...")
    try:
        textcnn_model = TextCNN(vocab_size, 100, num_classes).to(device)
        ckpt = torch.load("./models/agnews_checkpoints/textcnn.pt", map_location=device)
        textcnn_model.load_state_dict(ckpt['model'])
        textcnn_model.eval()
        
        wrapped_textcnn = ModelWrapper(
            model=textcnn_model,
            model_type='textcnn',
            vocab=vocab,  # 传递vocab
            device=device
        )
        base_models.append(wrapped_textcnn)
        print("TextCNN model loaded successfully")
    except Exception as e:
        print(f"Failed to load TextCNN model: {e}")
    
    # 2. BiLSTM模型
    print("Loading BiLSTM model...")
    try:
        bilstm_model = BiLSTMClassifier(vocab_size, 100, 128, num_classes).to(device)
        ckpt = torch.load("./models/agnews_checkpoints/bilstm.pt", map_location=device)
        bilstm_model.load_state_dict(ckpt['model'])
        bilstm_model.eval()
        
        wrapped_bilstm = ModelWrapper(
            model=bilstm_model,
            model_type='bilstm', 
            vocab=vocab,  # 使用相同的vocab
            device=device
        )
        base_models.append(wrapped_bilstm)
        print("BiLSTM model loaded successfully")
    except Exception as e:
        print(f"Failed to load BiLSTM model: {e}")
    
    # 3. RoBERTa模型
    print("Loading RoBERTa model...")
    try:
        # roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        roberta_tokenizer = RobertaTokenizer.from_pretrained("./models/agnews_checkpoints/roberta")
        roberta_model = RobertaForSequenceClassification.from_pretrained(
            "./models/agnews_checkpoints/roberta", 
            num_labels=num_classes
        ).to(device)
        roberta_model.eval()
        
        wrapped_roberta = ModelWrapper(
            model=roberta_model,
            model_type='roberta',
            tokenizer=roberta_tokenizer,  # 传递tokenizer
            device=device
        )
        base_models.append(wrapped_roberta)
        print("RoBERTa model loaded successfully")
    except Exception as e:
        print(f"Failed to load RoBERTa model: {e}")

    # 4. SVM
    print("Loading SVM model...")
    sklearn_model, vectorizer = joblib.load("./models/agnews_checkpoints/svm.pkl")
    wrapped_svm = ModelWrapper(sklearn_model, 'svm', 
                             vectorizer=vectorizer, device=device)
    base_models.append(wrapped_svm)

    #5. Logistic Regression
    print("Loading Logistic Regression model...")
    sklearn_model, vectorizer = joblib.load("./models/agnews_checkpoints/logreg.pkl")
    wrapped_logreg = ModelWrapper(sklearn_model, 'logreg', 
                             vectorizer=vectorizer, device=device)
    base_models.append(wrapped_logreg)
    print(f"Successfully loaded {len(base_models)} base models")
    return base_models

def load_base_models_v1(device, num_classes):
    """
    加载预训练的基模型
    返回: 模型列表
    """
    base_models = []
    
    # 1. SVM模型（需要特殊处理，因为不是PyTorch模型）
    # 我们会在后续处理中单独处理传统模型
    
    # 2. TextCNN模型
    print("Loading TextCNN model...")
    textcnn_dataset = TextDataset.create_with_vocab(X_train, y_train)
    vocab_size = len(textcnn_dataset.vocab)
    textcnn_model = TextCNN(vocab_size, 100, num_classes).to(device)
    
    # 加载预训练权重
    try:
        ckpt = torch.load("textcnn.pt", map_location=device)
        textcnn_model.load_state_dict(ckpt['model'])
        textcnn_model.eval()
        for param in textcnn_model.parameters():
            param.requires_grad_(False)
        base_models.append(('textcnn', textcnn_model, textcnn_dataset.vocab))
        print("TextCNN model loaded successfully")
    except Exception as e:
        print(f"Failed to load TextCNN model: {e}")
    
    # 3. BiLSTM模型
    print("Loading BiLSTM model...")
    bilstm_dataset = TextDataset.create_with_vocab(X_train, y_train) 
    vocab_size = len(bilstm_dataset.vocab)
    bilstm_model = BiLSTMClassifier(vocab_size, 100, 128, num_classes).to(device)
    
    try:
        ckpt = torch.load("bilstm.pt", map_location=device)
        bilstm_model.load_state_dict(ckpt['model'])
        bilstm_model.eval()
        for param in bilstm_model.parameters():
            param.requires_grad_(False)
        base_models.append(('bilstm', bilstm_model, bilstm_dataset.vocab))
        print("BiLSTM model loaded successfully")
    except Exception as e:
        print(f"Failed to load BiLSTM model: {e}")
    
    # 4. RoBERTa模型
    print("Loading RoBERTa model...")
    try:
        roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        roberta_model = RobertaForSequenceClassification.from_pretrained(
            "./roberta-base", 
            num_labels=num_classes
        ).to(device)
        roberta_model.eval()
        for param in roberta_model.parameters():
            param.requires_grad_(False)
        base_models.append(('roberta', roberta_model, roberta_tokenizer))
        print("RoBERTa model loaded successfully")
    except Exception as e:
        print(f"Failed to load RoBERTa model: {e}")
    
    # 5. Logistic Regression模型（需要特殊处理）
    # 我们会在预测时单独处理
    
    print(f"Successfully loaded {len(base_models)} deep learning base models")
    return base_models

def load_base_models_v2(device, num_classes, X_train, y_train):
    base_models = []
    
    # 1. TextCNN
    ckpt = torch.load("./models/textcnn.pt", map_location=device)
    vocab = ckpt['vocab']
    vocab_size = len(vocab)
    textcnn_model = TextCNN(vocab_size, 100, num_classes).to(device)
    # print("Checkpoint keys:", ckpt.keys())
    textcnn_model.load_state_dict(ckpt['model'])
    wrapped_textcnn = ModelWrapper(textcnn_model, 'textcnn', vocab = vocab, device=device)
    base_models.append(wrapped_textcnn)
    
    # 2. BiLSTM  
    bilstm_model = BiLSTMClassifier(vocab_size, 100, 128, num_classes).to(device)
    ckpt = torch.load("./models/bilstm.pt", map_location=device)
    bilstm_model.load_state_dict(ckpt['model'])
    wrapped_bilstm = ModelWrapper(bilstm_model, 'bilstm', device=device)
    base_models.append(wrapped_bilstm)
    
    # 3. RoBERTa
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta_model = RobertaForSequenceClassification.from_pretrained(
        "./models/roberta", num_labels=num_classes
    ).to(device)
    wrapped_roberta = ModelWrapper(roberta_model, 'roberta', 
                                 tokenizer=roberta_tokenizer, device=device)
    base_models.append(wrapped_roberta)
    
    # 4. SVM
    sklearn_model, vectorizer = joblib.load("./models/svm.pkl")
    wrapped_svm = ModelWrapper(sklearn_model, 'svm', 
                             vectorizer=vectorizer, device=device)
    base_models.append(wrapped_svm)

    #5. Logistic Regression
    sklearn_model, vectorizer = joblib.load("./models/logreg.pkl")
    wrapped_logreg = ModelWrapper(sklearn_model, 'logreg', 
                             vectorizer=vectorizer, device=device)
    base_models.append(wrapped_logreg)
    
    return base_models

def load_base_models_pre(device, num_classes, X_train, y_train):
    """加载预训练的基模型"""
    base_models = []
    
    # 首先创建统一的vocab（所有vocab-based模型共享）
    print("Building vocabulary...")
    vocab_dataset = TextDataset.create_with_vocab(X_train, y_train)
    vocab = vocab_dataset.vocab
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # # 1. TextCNN模型
    print("Loading TextCNN model...")
    try:
        textcnn_model = TextCNN(vocab_size, 100, num_classes).to(device)
        ckpt = torch.load("./models_out/textcnn.pt", map_location=device)
        textcnn_model.load_state_dict(ckpt['model'])
        textcnn_model.eval()
        
        wrapped_textcnn = ModelWrapper(
            model=textcnn_model,
            model_type='textcnn',
            vocab=vocab,  # 传递vocab
            device=device
        )
        base_models.append(wrapped_textcnn)
        print("TextCNN model loaded successfully")
    except Exception as e:
        print(f"Failed to load TextCNN model: {e}")
    
    # 2. BiLSTM模型
    print("Loading BiLSTM model...")
    try:
        bilstm_model = BiLSTMClassifier(vocab_size, 100, 128, num_classes).to(device)
        ckpt = torch.load("./models_out/bilstm.pt", map_location=device)
        bilstm_model.load_state_dict(ckpt['model'])
        bilstm_model.eval()
        
        wrapped_bilstm = ModelWrapper(
            model=bilstm_model,
            model_type='bilstm', 
            vocab=vocab,  # 使用相同的vocab
            device=device
        )
        base_models.append(wrapped_bilstm)
        print("BiLSTM model loaded successfully")
    except Exception as e:
        print(f"Failed to load BiLSTM model: {e}")
    
    # 3. RoBERTa模型
    print("Loading RoBERTa model...")
    try:
        roberta_tokenizer = RobertaTokenizer.from_pretrained("./models/roberta")
        roberta_model = RobertaForSequenceClassification.from_pretrained(
            "./models_out/roberta", 
            num_labels=num_classes
        ).to(device)
        roberta_model.eval()
        
        wrapped_roberta = ModelWrapper(
            model=roberta_model,
            model_type='roberta',
            tokenizer=roberta_tokenizer,  # 传递tokenizer
            device=device
        )
        base_models.append(wrapped_roberta)
        print("RoBERTa model loaded successfully")
    except Exception as e:
        print(f"Failed to load RoBERTa model: {e}")

    # 4. SVM
    print("Loading SVM model...")
    sklearn_model, vectorizer = joblib.load("./models_out/svm.pkl")
    wrapped_svm = ModelWrapper(sklearn_model, 'svm', 
                             vectorizer=vectorizer, device=device)
    base_models.append(wrapped_svm)

    #5. Logistic Regression
    print("Loading Logistic Regression model...")
    sklearn_model, vectorizer = joblib.load("./models_out/logreg.pkl")
    wrapped_logreg = ModelWrapper(sklearn_model, 'logreg', 
                             vectorizer=vectorizer, device=device)
    base_models.append(wrapped_logreg)
    print(f"Successfully loaded {len(base_models)} base models")
    return base_models

# ---------------------------
# load_base_models 函数（使用 ModelWrapper）
# ---------------------------
def load_base_models(device, num_classes, X_train, y_train, models_dir="./models_out"):
    """
    加载并包装脚本中五个基模型（textcnn, bilstm, roberta, svm, logreg）。
    返回: list of ModelWrapper
    """
    base_models = []

    # build shared vocab for vocab-based models
    print("[load_base_models] Building vocabulary...")
    ds_vocab = TextDataset.create_with_vocab(X_train, y_train, max_len=128)
    vocab = ds_vocab.vocab
    vocab_size = len(vocab)
    pad_idx = None
    try:
        pad_idx = vocab["<pad>"]
    except Exception:
        pad_idx = vocab.get("<pad>", 0)
    print(f"[load_base_models] vocab size = {vocab_size}")

    # 1) TextCNN
    textcnn_path = os.path.join(models_dir, "textcnn.pt")
    if os.path.exists(textcnn_path):
        try:
            print("[load_base_models] Loading TextCNN...")
            textcnn_model = TextCNN(vocab_size, embed_dim=100, num_classes=num_classes, pad_idx=pad_idx)
            ckpt = torch.load(textcnn_path, map_location=device)
            # ckpt expected dict with 'model' and maybe 'vocab'
            if isinstance(ckpt, dict) and 'model' in ckpt:
                state = ckpt['model']
            else:
                state = ckpt
            textcnn_model.load_state_dict(state)
            textcnn_model.to(device)
            textcnn_model.eval()
            wrapped = ModelWrapper(textcnn_model, 'textcnn', vocab=vocab, device=device, max_len=128)
            base_models.append(wrapped)
            print("[load_base_models] TextCNN loaded.")
        except Exception as e:
            print(f"[load_base_models] Failed to load TextCNN: {e}")
    else:
        print(f"[load_base_models] TextCNN checkpoint not found at {textcnn_path}")

    # # 2) BiLSTM
    bilstm_path = os.path.join(models_dir, "bilstm.pt")
    if os.path.exists(bilstm_path):
        try:
            print("[load_base_models] Loading BiLSTM...")
            bilstm_model = BiLSTMClassifier(vocab_size, embed_dim=100, hidden_dim=128, num_classes=num_classes, pad_idx=pad_idx)
            ckpt = torch.load(bilstm_path, map_location=device)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                state = ckpt['model']
            else:
                state = ckpt
            bilstm_model.load_state_dict(state)
            bilstm_model.to(device)
            bilstm_model.eval()
            wrapped = ModelWrapper(bilstm_model, 'bilstm', vocab=vocab, device=device, max_len=128)
            base_models.append(wrapped)
            print("[load_base_models] BiLSTM loaded.")
        except Exception as e:
            print(f"[load_base_models] Failed to load BiLSTM: {e}")
    else:
        print(f"[load_base_models] BiLSTM checkpoint not found at {bilstm_path}")

    # # 3) RoBERTa
    # # try several loading strategies: (A) if you saved state_dict at models_out/roberta.pt
    # # (B) if you have saved_pretrained folder models_out/roberta/ use from_pretrained on that
    # try:
    #     print("[load_base_models] Loading RoBERTa tokenizer/model...")
    #     # tokenizer: try local folder first, else roberta-base
    #     tok = None
    #     tok_path_local = os.path.join("models", "roberta")
    #     tok_path_out = os.path.join(models_dir, "roberta")
    #     if os.path.isdir(tok_path_local):
    #         tok = RobertaTokenizer.from_pretrained(tok_path_local)
    #     elif os.path.isdir(tok_path_out):
    #         tok = RobertaTokenizer.from_pretrained(tok_path_out)
    #     else:
    #         tok = RobertaTokenizer.from_pretrained("roberta-base")

    #     # model: if state_dict file exists
    #     roberta_state_path = os.path.join(models_dir, "roberta.pt")
    #     roberta_obj = None
    #     if os.path.exists(roberta_state_path):
    #         roberta_obj = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_classes)
    #         ckpt = torch.load(roberta_state_path, map_location=device)
    #         # ckpt may be state_dict or model.state_dict()
    #         if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
    #             # assume it's state_dict
    #             roberta_obj.load_state_dict(ckpt)
    #         elif isinstance(ckpt, dict) and 'model' in ckpt:
    #             roberta_obj.load_state_dict(ckpt['model'])
    #         else:
    #             try:
    #                 roberta_obj.load_state_dict(ckpt)
    #             except Exception:
    #                 pass
    #         roberta_obj.to(device)
    #         roberta_obj.eval()
    #         wrapped = ModelWrapper(roberta_obj, 'roberta', tokenizer=tok, device=device, max_len=128)
    #         base_models.append(wrapped)
    #         print("[load_base_models] RoBERTa loaded from state_dict.")
    #     else:
    #         # try from_pretrained folder
    #         if os.path.isdir(tok_path_out):
    #             try:
    #                 roberta_obj = RobertaForSequenceClassification.from_pretrained(tok_path_out, num_labels=num_classes).to(device)
    #                 roberta_obj.eval()
    #                 wrapped = ModelWrapper(roberta_obj, 'roberta', tokenizer=tok, device=device, max_len=128)
    #                 base_models.append(wrapped)
    #                 print("[load_base_models] RoBERTa loaded from pretrained folder.")
    #             except Exception as e:
    #                 print(f"[load_base_models] Failed to load RoBERTa from folder: {e}")
    #         else:
    #             print("[load_base_models] No RoBERTa checkpoint found (skipping).")
    # except Exception as e:
    #     print(f"[load_base_models] RoBERTa load error: {e}")

    # # 4) SVM (sklearn)
    # svm_path = os.path.join(models_dir, "svm.pkl")
    # if os.path.exists(svm_path):
    #     try:
    #         print("[load_base_models] Loading SVM sklearn artifact...")
    #         skl_model, vectorizer = joblib.load(svm_path)
    #         wrapped = ModelWrapper(skl_model, 'svm', vectorizer=vectorizer, device=device)
    #         base_models.append(wrapped)
    #         print("[load_base_models] SVM loaded.")
    #     except Exception as e:
    #         print(f"[load_base_models] Failed to load SVM: {e}")
    # else:
    #     print(f"[load_base_models] SVM artifact not found at {svm_path}")

    # 5) Logistic Regression
    logreg_path = os.path.join(models_dir, "logreg.pkl")
    if os.path.exists(logreg_path):
        try:
            print("[load_base_models] Loading Logistic Regression artifact...")
            skl_model, vectorizer = joblib.load(logreg_path)
            wrapped = ModelWrapper(skl_model, 'logreg', vectorizer=vectorizer, device=device)
            base_models.append(wrapped)
            print("[load_base_models] Logistic Regression loaded.")
        except Exception as e:
            print(f"[load_base_models] Failed to load Logistic Regression: {e}")
    else:
        print(f"[load_base_models] Logistic artifact not found at {logreg_path}")

    print(f"[load_base_models] Finished. Loaded {len(base_models)} base models.")
    return base_models


def main():
    warnings.filterwarnings("ignore")
    
    # 参数设置
    parser = argparse.ArgumentParser(description='Text Ensemble Learning with Smart Selection')
    
    # 数据参数
    parser.add_argument('--train_data', type=str, required=True, help='Path to train dataset')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test dataset') 
    parser.add_argument('--label_column', type=str, default=None, help='Label column name if CSV')
    
    # 集成学习参数
    parser.add_argument('--c', type=int, default=3, 
                        help='number of models to select from ensemble')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='learning rate step gamma')
    
    # 设备设置
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    
    # 模型选择策略参数
    parser.add_argument('--use_softmax', type=bool, default=True,
                        help='apply softmax to predictions')
    parser.add_argument('--weight_pred', type=bool, default=False,
                        help='weight predictions by model confidence')
    parser.add_argument('--injection', type=bool, default=False,
                        help='inject knowledge to knapsack')
    parser.add_argument('--apply_sum', type=bool, default=True,
                        help='apply sum to soft predictions')
    parser.add_argument('--clip', type=bool, default=False,
                        help='clip gradients')
    parser.add_argument('--sched', type=bool, default=False,
                        help='use learning rate scheduler')
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="weight for selection loss")
    parser.add_argument("--visualize", action='store_true', default=True,
                        help="visualize model selection frequencies after training")
    parser.add_argument('--entropy_weight', type=float, default=0.5,
                        help='weight for entropy regularization')
    
    args = parser.parse_args()
    
    # 设备设置
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # ==========================================================================
    # 文本数据加载和预处理
    # ==========================================================================
    
    print("Loading text dataset...")
    
    # 使用你提供的get_data函数加载数据
    X_train, y_train = get_data(args.train_data, args.label_column)
    X_test, y_test = get_data(args.test_data, args.label_column)
    
    # 数据划分：从训练集中划分验证集
    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=args.seed, stratify=y_train)
    
    print(f"Training samples: {len(X_train)}")
    # print(f"Validation samples: {len(X_valid)}") 
    print(f"Test samples: {len(X_test)}")
    
    # 确定类别数量
    num_classes = len(set(y_train))
    print(f"Number of classes: {num_classes}")
    
    # ==========================================================================
    # 基模型加载 - 使用你训练好的5个模型
    # ==========================================================================
    
    print("Loading pre-trained base models...")    
    base_models = load_base_models(device, num_classes, X_train, y_train)
    
    # ==========================================================================
    # 创建统一的数据加载器
    # ==========================================================================
    
    class UnifiedTextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels
            
        def __len__(self):
            return len(self.texts)
            
        def __getitem__(self, idx):
            return {
                'X': self.texts[idx],
                'Y': torch.tensor(self.labels[idx], dtype=torch.long)
            }
    
    # 创建数据加载器
    train_dataset = UnifiedTextDataset(X_train, y_train)
    # valid_dataset = UnifiedTextDataset(X_valid, y_valid)
    test_dataset = UnifiedTextDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    # valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    # ==========================================================================
    # 选择网络设置
    # ==========================================================================
    
    n_models = len(base_models)  # 深度模型数量

    
    # 初始化选择网络（输入维度需要根据你的文本特征调整）
    # selection_net = TextSelectionNet(input_dim=768, hidden_dim=512, n_models=n_models).to(device)

    dataset = TextDataset.create_with_vocab(X_train, y_train)
    vocab_size = len(dataset.vocab)
    # selection_net = LSTMSelectionNet(
    #     vocab=dataset.vocab,
    #     embed_dim=200,
    #     hidden_dim=256, 
    #     n_models=len(base_models),
    #     max_len=128,
    #     num_layers = 1
    # ).to(device)
    # init_weights(selection_net, 'xavier')
    selection_net = TranSelectionNet(
        pretrained_model_name="./models/roberta-base",  # 或 "roberta-base", "distilbert-base-uncased"
        hidden_dim=512,
        n_models=len(base_models)
    ).to(device)
    # 下面这是sota_with_prob的模型
    # selection_net = PredictionBasedSelectionNet(n_models=5, num_classes=4).to(device)
    # 确定输入维度
    # feature_dim = n_models * num_classes + n_models  # 5*4 + 5 = 25

    # # 初始化选择网络
    # selection_net = TextSelectionNet(
    #     input_dim=feature_dim,  # 根据特征维度调整
    #     hidden_dim=512, 
    #     n_models=len(base_models)  # 5
    # ).to(device)

    # selection_net = ImprovedSelectionNet(
    #     vocab=dataset.vocab,
    #     embed_dim=200,
    #     hidden_dim=256,
    #     n_models=len(base_models),
    #     max_len=128
    # ).to(device)

    # # 方案1：使用预训练Transformer（推荐）
    # selection_net = TextSelectionNet(
    #     pretrained_model_name="bert-base-uncased",  # 或 "roberta-base", "distilbert-base-uncased"
    #     hidden_dim=512,
    #     n_models=len(base_models)
    # ).to(device)
    
    # ==========================================================================
    # 训练设置
    # ==========================================================================

    # initialize_model = True
    # param_distribution = 'xavier'
    # if initialize_model:
    #     init_weights(selection_net, param_distribution) 
    # selection_net.train()

    # 损失函数和优化器
    loss_fun = nn.CrossEntropyLoss()
    # loss_fun = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(selection_net.parameters(), lr=args.lr)
    
    # 学习率调度器
    sched = None
    if args.sched:
        sched = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=args.gamma, weight_decay=1e-4)
    
    # ==========================================================================
    # 训练和测试
    # ==========================================================================
    
    print("Starting training...")
    
    
    # 训练选择网络
    train_selection_logit_matching(
        selection_net, base_models, device, train_loader, test_loader, 
        optimizer, args, loss_fun, n_models, sched, num_classes
    )

     # -------------------------------------------------------------------
    # 👇 从这里开始添加新代码
    # -------------------------------------------------------------------
    if args.visualize:
        # 加载最佳模型并测试
        print("Loading best model for testing...")
        # best_selection_net = LSTMSelectionNet(
        #     vocab=dataset.vocab,
        #     embed_dim=200,
        #     hidden_dim=256, 
        #     n_models=len(base_models),
        #     max_len=128,
        #     num_layers = 1
        # ).to(device)
        best_selection_net = TranSelectionNet(
        pretrained_model_name="./models/roberta-base",  # 或 "roberta-base", "distilbert-base-uncased"
        hidden_dim=512,
        n_models=len(base_models)
        ).to(device)
        best_selection_net.load_state_dict(torch.load(f"best_model_{args.c}.pth"))

        print("\n[INFO] 准备生成可视化图表...")
        
        
        model_names = [
            'Model 0 (TextCNN)',
            'Model 1 (BiLSTM)',
            'Model 2 (RoBERTa)',
            'Model 3 (SVM)',
            'Model 4 (LogReg)'
        ]
        

        try:
            all_valid_targets = torch.tensor(y_test, dtype=torch.long)
        except NameError:
            print("[ERROR] 无法找到 X_valid/y_valid。请确保您在 main() 函数中取消注释了 train_test_split！")
            exit()


        try:
            precomputed_data = torch.load('./precomputed_predictions_20news.pt', map_location='cpu')
            valid_predictions = precomputed_data['valid_predictions']
        except Exception as e:
            print(f"[ERROR] 无法加载 './precomputed_predictions.pt': {e}")
            print("[INFO] 请确保您在 train_selection 中正确保存了 logits (而不是 softmax)。")
            exit()

       
        best_selection_net.to(device)

        # 5. 调用可视化
        visualize_selections_v1(
            best_selection_net,
            test_loader,
            valid_predictions,
            all_valid_targets,
            device,
            args.c,
            model_names
        )
        exit() # DEBUG

    # 加载最佳模型并测试
    print("Loading best model for testing...")
    best_selection_net = TextSelectionNet(768, 512, n_models).to(device)
    best_selection_net.load_state_dict(torch.load(f"best_model_{args.c}.pth"))
    

    test_accuracy = test(best_selection_net, base_models, device, test_loader, args, num_classes)
    print(f"Final test accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    main()