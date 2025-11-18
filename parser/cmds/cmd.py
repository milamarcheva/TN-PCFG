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

        def _iter_label_batches(batch):
            if not collect_label_marginal:
                yield batch
                return
            batch_size = None
            for value in batch.values():
                if isinstance(value, torch.Tensor):
                    batch_size = value.size(0)
                    break
            if batch_size is None:
                yield batch
                return
            for idx in range(batch_size):
                sliced = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        sliced[key] = value[idx:idx + 1]
                    else:
                        sliced[key] = value
                yield sliced

        try:
            with context:
                for x, y in t:
                    if collect_label_marginal:
                        for current_x in _iter_label_batches(x):
                            model.zero_grad(set_to_none=True)
                            result = model.evaluate(current_x, decode_type=decode_type, eval_dep=eval_dep)
                            label_marginal = result.get('label_marginal')
                            if label_marginal is None:
                                continue
                            # Gradients returned by the model are expected counts; convert
                            # them to proper posterior probabilities so every span's
                            # distribution is non-negative and sums to one, avoiding the
                            # near-uniform/negative values observed when reading raw
                            # gradients directly.
                            label_marginal = torch.softmax(label_marginal, dim=-1)
                            words = current_x['word'].detach().cpu()
                            seq_len = current_x['seq_len'].detach().cpu()
                            label_marginal = label_marginal.detach().cpu()
                            batch_size = seq_len.size(0)
                            for idx in range(batch_size):
                                length = int(seq_len[idx].item())
                                token_slice = words[idx, :length]
                                surface_tokens = None
                                if vocab is not None:
                                    surface_tokens = [vocab.to_word(int(tok)) for tok in token_slice.tolist()]
                                span_scores = label_marginal[idx, :length, 1:length + 1].clone()
                                collected.append({
                                    'label_marginal': span_scores,
                                    'seq_len': length,
                                    'word': token_slice.clone(),
                                    'tokens': surface_tokens
                                })
                    else:
                        model.zero_grad(set_to_none=True)
                        result = model.evaluate(x, decode_type=decode_type, eval_dep=eval_dep)
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




