import argparse
import os
import os.path as osp
import torch

from vedacore.misc import Config, load_weights, ProgressBar, mkdir_or_exist
from configs.trainval.tinaface.test_widerface import prepare
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--outdir', default='eval_dirs/tmp/tinaface/', help='directory where widerface txt will be saved')

    args = parser.parse_args()
    return args


def dump_pr_curves(eval_res, dump_dir):
    fig, axs = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True, sharey=True)
    axs[0].plot(eval_res['eval_res'][0]['recall'], eval_res['eval_res'][0]['precision'], label=f"{eval_res['eval_res'][0]['ap']:.3f}")
    axs[1].plot(eval_res['eval_res'][1]['recall'], eval_res['eval_res'][1]['precision'], label=f"{eval_res['eval_res'][1]['ap']:.3f}")
    axs[2].plot(eval_res['eval_res'][2]['recall'], eval_res['eval_res'][2]['precision'], label=f"{eval_res['eval_res'][2]['ap']:.3f}")
    for i in range(3):
        axs[i].set_xlabel("Recall")
        axs[i].set_ylabel("Precision")
        axs[i].set_title(CLASSES[i])
        axs[i].legend(loc="upper right")
    fig.suptitle(f'PR curves for MAFA test dataset, mAP: {eval_res["mAP"]:.3f}', fontsize=16)
    plt.savefig(os.path.join(dump_dir, "mafa_test_pr.png"))


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


CLASSES = ('unmasked_face', 'masked_face', 'incorrectly_masked_face',)


def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)
    mkdir_or_exist(osp.abspath(args.outdir))

    engine, data_loader = prepare(cfg, args.checkpoint)

    results = test(engine, data_loader)

    eval_results = data_loader.dataset.evaluate(results, 'mAP')
    dump_pr_curves(eval_results, args.outdir)


if __name__ == '__main__':
    main()
