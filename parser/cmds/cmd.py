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
        collect_label_marginal = decode_type == 'label_marginal'
        was_training = model.training
        if collect_label_marginal:
            # Label-marginal decoding backpropagates through the encoder LSTM to
            # accumulate span indicators. cuDNN only supports that backward pass
            # when the module ran in training mode, so keep the original
            # behaviour and restore the previous flag afterwards.
            model.train()
        else:
            model.eval()
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
        vocab = getattr(self, 'word_vocab', None)
        try:
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
                            token_slice = words[idx, :length]
                            surface_tokens = None
                            if vocab is not None:
                                surface_tokens = [vocab.to_word(int(tok)) for tok in token_slice.tolist()]
                            collected.append({
                                'label_marginal': label_marginal[idx, :length, :length].clone(),
                                'seq_len': length,
                                'word': token_slice.clone(),
                                'tokens': surface_tokens
                            })
                    else:
                        metric_f1(result['prediction'], y['gold_tree'])
                        metric_ll(result['partition'], x['seq_len'])
                        if eval_dep:
                            metric_uas(result['prediction_arc'], y['head'])
        finally:
            if was_training:
                model.train()
            else:
                model.eval()
        if collect_label_marginal:
            return collected
        if not eval_dep:
            return metric_f1, metric_ll
        else:
            return metric_f1, metric_uas, metric_ll




