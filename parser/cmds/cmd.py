# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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


    def evaluate(self, loader, eval_dep=False, decode_type='mbr', model=None, label_marginal_out=None):
        if model == None:
            model = self.model
        # Backward through the encoder (e.g., cuDNN RNN) requires training
        # mode; keep eval mode for standard decoding.
        collecting_marginal = decode_type == 'label_marginal' or label_marginal_out is not None
        if collecting_marginal:
            model.train()
        else:
            model.eval()
        metric_f1 = UF1() if not collecting_marginal else None
        metric_ll = LikelihoodMetric() if not collecting_marginal else None
        metric_uas = UAS() if eval_dep and not collecting_marginal else None
        t = tqdm(loader, total=int(len(loader)),  position=0, leave=True)
        print('decoding mode:{}'.format(decode_type))
        print('evaluate_dep:{}'.format(eval_dep))
        collected_marginal = [] if collecting_marginal else None
        collected_seq_len = [] if collecting_marginal else None
        max_seq_len = 0
        context = torch.enable_grad() if collecting_marginal else torch.no_grad()
        with context:
            for x, y in t:
                result = model.evaluate(x, decode_type=decode_type, eval_dep=eval_dep)
                if collecting_marginal:
                    marginal = result['marginal'].detach().cpu()
                    seq_len = x['seq_len'].detach().cpu()
                    max_seq_len = max(max_seq_len, marginal.shape[1])
                    collected_marginal.append(marginal)
                    collected_seq_len.append(seq_len)
                    continue
                metric_f1(result['prediction'], y['gold_tree'])
                metric_ll(result['partition'], x['seq_len'])
                if eval_dep:
                    metric_uas(result['prediction_arc'], y['head'])
        if collecting_marginal:
            padded_marginal = []
            for marginal in collected_marginal:
                if marginal.shape[1] == max_seq_len:
                    padded_marginal.append(marginal)
                    continue
                pad_amount = max_seq_len - marginal.shape[1]
                padded_marginal.append(
                    F.pad(
                        marginal,
                        (0, 0, 0, pad_amount, 0, pad_amount, 0, 0),
                    )
                )

            marginal = torch.cat(padded_marginal, dim=0)
            seq_len = torch.cat(collected_seq_len, dim=0)
            if label_marginal_out:
                dir_name = os.path.dirname(label_marginal_out)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                torch.save({'marginal': marginal, 'seq_len': seq_len}, label_marginal_out)
            return {'marginal': marginal, 'seq_len': seq_len, 'path': label_marginal_out}
        if not eval_dep:
            return metric_f1, metric_ll
        else:
            return metric_f1, metric_uas, metric_ll




