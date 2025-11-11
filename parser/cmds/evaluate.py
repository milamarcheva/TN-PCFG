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
        self.model = get_model(args.model, dataset)
        best_model_path = self.args.load_from_dir + "/best.pt"
        self.model.load_state_dict(torch.load(str(best_model_path)))
        print('successfully load')

        if split == 'val':
            loader = dataset.val_dataloader
        elif split == 'test':
            loader = dataset.test_dataloader
        else:
            raise ValueError('split must be "val" or "test"')

        loader = DataPrefetcher(loader, device=self.device)

        if decode_type == 'label_marginal':
            records = self.evaluate(loader, eval_dep=False, decode_type=decode_type)
            if label_marginal_out:
                output_dir = os.path.dirname(label_marginal_out)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                payload = {
                    'split': split,
                    'records': records,
                    'nonterminals': records[0]['label_marginal'].shape[-1] if records else None
                }
                torch.save(payload, label_marginal_out)
                print(f"Saved {len(records)} label-marginal entries to {label_marginal_out}")
            else:
                print(f"Collected label marginals for {len(records)} sentences (no file written)")
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








