# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from parser.helper.metric import LikelihoodMetric,  UF1, LossMetric, UAS

import time

class CMD(object):
    def __call__(self, args):
        self.args = args

    def train(self, loader):
        self.model.train()
        t = tqdm(loader, total=int(len(loader)),  position=0, leave=True)
        train_arg = self.args.train
        for x, _ in t:

            self.optimizer.zero_grad()
            loss = self.model.loss(x)
            loss.backward()
            if train_arg.clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                     train_arg.clip)
            self.optimizer.step()
            t.set_postfix(loss=loss.item())
        return


    def evaluate(self, loader, eval_dep=False, decode_type='mbr', model=None):
        if model is None:
            model = self.model
        model.eval()
        collect_label_marginal = decode_type == 'label_marginal'
        if not collect_label_marginal:
            metric_f1 = UF1()
            if eval_dep:
                metric_uas = UAS()
            metric_ll = LikelihoodMetric()
        collected = []
        t = tqdm(loader, total=int(len(loader)), position=0, leave=True)
        print('decoding mode:{}'.format(decode_type))
        print('evaluate_dep:{}'.format(eval_dep))
        context = torch.enable_grad() if collect_label_marginal else torch.no_grad()
        with context:
            for x, y in t:
                model.zero_grad(set_to_none=True)
                result = model.evaluate(x, decode_type=decode_type, eval_dep=eval_dep)
                if collect_label_marginal:
                    label_marginal = result.get('label_marginal')
                    if label_marginal is None:
                        continue
                    words = x['word'].detach().cpu()
                    seq_len = x['seq_len'].detach().cpu()
                    label_marginal = label_marginal.detach().cpu()
                    batch_size = seq_len.size(0)
                    for idx in range(batch_size):
                        length = int(seq_len[idx].item())
                        collected.append({
                            'label_marginal': label_marginal[idx, :length, :length].clone(),
                            'seq_len': length,
                            'word': words[idx, :length].clone()
                        })
                else:
                    metric_f1(result['prediction'], y['gold_tree'])
                    metric_ll(result['partition'], x['seq_len'])
                    if eval_dep:
                        metric_uas(result['prediction_arc'], y['head'])
        if collect_label_marginal:
            return collected
        if not eval_dep:
            return metric_f1, metric_ll
        else:
            return metric_f1, metric_uas, metric_ll




