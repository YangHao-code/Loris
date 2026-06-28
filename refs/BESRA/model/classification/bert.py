#!/usr/bin/env python
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

import torch
import torch.nn.functional as F

from dataset.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier
from transformers import AutoConfig, AutoModel
import pdb


class BERT(torch.nn.Module):
    """BERT
    """
    def __init__(self, dataset, config):
        super(BERT, self).__init__()
        self.config = config
        self.bertConfig = AutoConfig.from_pretrained(config.bert.pretrained_model_name)
        self.bert = AutoModel.from_pretrained(config.bert.pretrained_model_name, config=self.bertConfig)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, len(dataset.label_map))
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)
        self.device = config.device
  

    def get_parameter_optimizer_dict(self):
        params = list()
        no_decay = ["bias", "LayerNorm.weight"]
        params.append({'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01})
        params.append({'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0})
            
        params.append({'params': self.linear.parameters(), 'lr': self.config.bert.linear_lr})
        # params.append({'params': [p for n, p in self.linear.named_parameters() if not any(nd in n for nd in no_decay)], 'lr': self.config.bert.linear_lr, 'weight_decay': 0.01})
        # params.append({'params': [p for n, p in self.linear.named_parameters() if any(nd in n for nd in no_decay)],'lr': self.config.bert.linear_lr, 'weight_decay': 0.0})
        
       
        # params = list(self.bert.named_parameters()) + list(self.linear.named_parameters())
        # params = list()
        # params.append({'params': self.bert.named_parameters()})
        # params.append({'params': self.linear.parameters()})
 
        return params

    def update_lr(self, optimizer, epoch):
        """
        """
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups:
                # if param_group["initial_lr"] == self.config.bert.bert_lr:
                #     param_group["lr"] = self.config.bert.bert_lr * 0.1 / epoch
                # elif param_group["initial_lr"] == self.config.bert.linear_lr:
                #     param_group["lr"] = self.config.bert.linear_lr * 0.1 / epoch
                param_group["lr"] = self.config.bert.bert_lr
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0

    def forward(self, batch):
     
        input_ids = batch["doc_input_ids_bert"].to(self.device)
        attention_mask = batch["doc_token_att_bert"].to(self.device)
        token_type_ids = batch["doc_token_type_ids_bert"].to(self.device)
        
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # pdb.set_trace()
        pooler_output = output[1]
        # pooler_output = output[0]
        # pooler_output = torch.mean(pooler_output,dim=1)
        # pdb.set_trace()
       
        return self.linear(self.dropout(pooler_output))
