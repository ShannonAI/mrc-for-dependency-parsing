# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: evaluate
@time: 2020/12/18 16:06
@desc: todo(yuxian): try evaluate using this file!!!

"""

import os

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from parser.biaf_trainer import BiafDependency
from parser.data.collate import collate_dependency_data
from parser.data.dependency_reader import DependencyDataset


def evaluate(ckpt, hparams_file):
    """main"""

    trainer = Trainer(gpus=[1,], distributed_backend="ddp")

    model = BiafDependency.load_from_checkpoint(
        checkpoint_path=ckpt,
        hparams_file=hparams_file,
        map_location=None,
        batch_size=32,
        max_length=128,
        workers=0
    )
    args = model.args

    test_dataset = DependencyDataset(
        file_path=os.path.join(args.data_dir, f"test.{args.data_format}"),
        bert=args.bert_dir,
        pos_tags=args.pos_tags,
        dep_tags=args.dep_tags
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_dependency_data
    )

    trainer.test(model=model, test_dataloaders=test_loader)


if __name__ == '__main__':
    # ptb
    HPARAMS = "/data/yuxian/train_logs/dependency/ptb/20210102/input_dropout_adamw_beta0.9_fixignore/lightning_logs/version_1/hparams.yaml"
    CHECKPOINT = "/data/yuxian/train_logs/dependency/ptb/20210102/input_dropout_adamw_beta0.9_fixignore/epoch=10.ckpt"
    evaluate(ckpt=CHECKPOINT, hparams_file=HPARAMS)
