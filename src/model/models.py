# -*- coding: utf-8 -*-

import os
import random
import numpy as np

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from src.utils.crf_utils.crf import CRF


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_everything(2022)


class BERT_CRF(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BERT_CRF, self).__init__(config)

        self.tokenize_labels_num = num_labels[0]
        self.entity_labels_num = num_labels[1]
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.entity_classifier = nn.Linear(config.hidden_size, self.entity_labels_num)
        self.crf_classifier = nn.Linear(config.hidden_size, self.tokenize_labels_num + 2)
        self.crf = CRF(self.tokenize_labels_num)

        # self.tokenize_task_weight = nn.Parameter(torch.FloatTensor([1]))
        # self.entity_task_weight = nn.Parameter(torch.FloatTensor([10]))
        self.criterion = nn.CrossEntropyLoss()



    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None,
                tokenize_labels=None,
                entity_labels=None,
                type='Train'):



        # print("input_ids ", input_ids.size())
        # print("token_type_ids ", token_type_ids.size())
        # print("attention_mask ", attention_mask.size())


        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        sequence_output = self.dropout(sequence_output)

        entity_logits = self.entity_classifier(sequence_output)
        crf_logits = self.crf_classifier(sequence_output)


        if tokenize_labels is not None and entity_labels  is not None:


            tokenize_loss = self.crf.neg_log_likelihood_loss(crf_logits, attention_mask, tokenize_labels)
            tokenize_predict = self.crf.decode(crf_logits, attention_mask)


            tokenize_clone = tokenize_labels.clone().detach() if type == 'Train' else tokenize_predict.clone().detach()
            batch_size = entity_logits.size(0)
            seq_len = entity_logits.size(1)
            segment_entity_logits_list = []
            for i in range(batch_size):
                acc_index = 0
                tokenize_clone[i][0] = 0
                for j in range(1, seq_len):
                    if tokenize_clone[i][j] != 2:
                        acc_index += 1
                    tokenize_clone[i][j] = acc_index

                tmp_segment_entity_logits = torch.zeros(entity_logits.shape[1:]).index_add_(0, tokenize_clone[i].clone(), entity_logits[i])
                segment_entity_logits_list.append(tmp_segment_entity_logits)

            segment_entity_logits = torch.concat(segment_entity_logits_list, dim=0).view(batch_size, seq_len, -1)
            entity_logits = torch.gather(segment_entity_logits, 1,
                                tokenize_clone.view(batch_size, seq_len, 1).expand(
                                    batch_size,
                                    seq_len,
                                    self.entity_labels_num))



            entity_loss = self.criterion(entity_logits.view(-1, self.entity_labels_num),
                                         entity_labels.view(-1)) * 100
            _, entity_predict = torch.max(torch.softmax(entity_logits, dim=-1), -1)

            if 1:
                print("\n")
                print("tokenize_loss ", tokenize_loss.item())
                print("tokenize_labels  ", tokenize_labels[0][0:10])
                print("tokenize_predict ", tokenize_predict[0][0:10])
                print("entity_loss ", entity_loss.item())
                print("entity_labels  ", entity_labels[0][0:10])
                print("entity_predict ", entity_predict[0][0:10])
                print("\n")

            # loss = tokenize_loss + entity_loss

            return tokenize_loss, entity_loss, tokenize_predict, entity_predict

        else:
            tokenize_predict = self.crf.decode(crf_logits, attention_mask)
            _, entity_predict = torch.max(torch.softmax(entity_logits, dim=-1), -1)

            return tokenize_predict, entity_predict
