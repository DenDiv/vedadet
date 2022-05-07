import argparse
import os
import os.path as osp
import torch

from vedacore.misc import Config, load_weights, ProgressBar, mkdir_or_exist
from configs.trainval.tinaface.test_widerface import prepare

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--outdir', default='eval_dirs/tmp/tinaface/', help='directory where widerface txt will be saved')

    args = parser.parse_args()
    return args


def test(engine, data_loader):
    engine.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):

        with torch.no_grad():
            result = engine(data)[0]

        results.append(result)
        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)
    mkdir_or_exist(osp.abspath(args.outdir))

    engine, data_loader = prepare(cfg, args.checkpoint)

    results = test(engine, data_loader)

    eval_results = data_loader.dataset.evaluate(results, 'mAP')

if __name__ == '__main__':
    main()
