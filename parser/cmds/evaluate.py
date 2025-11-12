# -*- coding: utf-8 -*-


from parser.cmds.cmd import CMD
import torch
import os

from datetime import datetime, timedelta
from parser.helper.loader_wrapper import DataPrefetcher
import numpy as np
from parser.helper.util import *
from parser.helper.data_module import DataModule
import click

class Evaluate(CMD):

    def __call__(self, args, eval_dep=False, decode_type='mbr', split='test', label_marginal_out=None):
        super(Evaluate, self).__call__(args)
        self.device = args.device
        self.args = args
        dataset = DataModule(args)
        self.dataset = dataset
        self.word_vocab = getattr(dataset, 'word_vocab', None)

        def build_model(current_dataset):
            return get_model(args.model, current_dataset)

        try:
            self.model = build_model(dataset)
        except RuntimeError as err:
            if (self.device == 'cuda') and ('out of memory' in str(err).lower()):
                print('Encountered CUDA out of memory while loading the model; retrying on CPU.')
                torch.cuda.empty_cache()
                args.device = 'cpu'
                self.device = 'cpu'
                dataset = DataModule(args)
                self.dataset = dataset
                self.word_vocab = getattr(dataset, 'word_vocab', None)
                self.model = build_model(dataset)
            else:
                raise
        best_model_path = self.args.load_from_dir + "/best.pt"
        map_location = self.device if isinstance(self.device, str) else self.device
        self.model.load_state_dict(torch.load(str(best_model_path), map_location=map_location))
        print('successfully load')

        loader = self._build_prefetcher(dataset, split)

        if decode_type == 'label_marginal':
            self._run_and_maybe_save_label_marginal(loader, split, label_marginal_out)
            return

        if not eval_dep:
            metric_f1, likelihood = self.evaluate(loader, eval_dep=eval_dep, decode_type=decode_type)
            print(metric_f1)
            print(likelihood)
        else:
            metric_f1, metric_uas, likelihood = self.evaluate(loader, eval_dep=eval_dep, decode_type=decode_type)
            print(metric_uas)
            print(metric_f1)
            print(likelihood)

        if label_marginal_out:
            lm_loader = self._build_prefetcher(dataset, split)
            self._run_and_maybe_save_label_marginal(lm_loader, split, label_marginal_out)

    def _build_prefetcher(self, dataset, split):
        if split == 'val':
            loader = dataset.val_dataloader
        elif split == 'test':
            loader = dataset.test_dataloader
        else:
            raise ValueError('split must be "val" or "test"')
        return DataPrefetcher(loader, device=self.device)

    def _run_and_maybe_save_label_marginal(self, loader, split, label_marginal_out):
        records = self.evaluate(loader, eval_dep=False, decode_type='label_marginal')
        original_count, filtered_count = self._sentence_counts(split)
        if label_marginal_out:
            output_dir = os.path.dirname(label_marginal_out)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            payload = {
                'split': split,
                'records': records,
                'nonterminals': records[0]['label_marginal'].shape[-1] if records else None,
                'filtered_sentence_count': filtered_count,
                'original_sentence_count': original_count
            }
            torch.save(payload, label_marginal_out)
            summary = f"Saved {len(records)} label-marginal entries"
            if filtered_count is not None and original_count is not None:
                summary += f" (filtered {filtered_count} / original {original_count} sentences)"
            print(f"{summary} to {label_marginal_out}")
        else:
            summary = f"Collected label marginals for {len(records)} sentences"
            if filtered_count is not None and original_count is not None:
                summary += f" (filtered {filtered_count} / original {original_count})"
            print(f"{summary} (no file written)")

    def _sentence_counts(self, split):
        dataset = getattr(self, 'dataset', None)
        if dataset is None:
            return None, None
        original = getattr(dataset, 'original_sentence_counts', {}).get(split)
        filtered = getattr(dataset, 'filtered_sentence_counts', {}).get(split)
        return original, filtered








