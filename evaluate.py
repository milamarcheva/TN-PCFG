# -*- coding: utf-8 -*-

import os
from parser.cmds import Evaluate
import torch
from easydict import EasyDict as edict
import yaml
import click


@click.command()
@click.option("--eval_dep", default=False, help="evaluate dependency, only for N(B)L-PCFG")
@click.option("--decode_type", default='mbr', help="viterbi, mbr, or label_marginal")
@click.option(
    "--device",
    "-d",
    default='0',
    help="CUDA device id (e.g. 0) or 'cpu' to force CPU execution"
)
@click.option("--load_from_dir", default="")
@click.option("--split", type=click.Choice(['val', 'test']), default='test')
@click.option(
    "--label_marginal_out",
    default="",
    help="If set, export span label posteriors to this path. When used with other decode types,"
         " an additional label-marginal pass will be run after metrics are computed."
)
def main(eval_dep, decode_type, load_from_dir, device, split, label_marginal_out):
    with open(os.path.join(load_from_dir, "config.yaml"), "r") as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    args = edict(yaml_cfg)
    requested_device = device.strip()
    args.load_from_dir = load_from_dir

    if requested_device.lower() == 'cpu':
        print("Running evaluation on CPU")
        args.device = 'cpu'
    else:
        if requested_device:
            print(f"Set the device with ID {requested_device} visible")
            os.environ['CUDA_VISIBLE_DEVICES'] = requested_device
        if torch.cuda.is_available():
            args.device = 'cuda'
        else:
            print("CUDA not available, falling back to CPU")
            args.device = 'cpu'

    command = Evaluate()
    command(
        args,
        decode_type=decode_type,
        eval_dep=eval_dep,
        split=split,
        label_marginal_out=label_marginal_out or None,
    )


if __name__ == '__main__':
    main()