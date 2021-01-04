# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: evaluate
@time: 2020/12/18 16:06
@desc: evaluate biaffine baseline

"""

from pytorch_lightning import Trainer

from parser.biaf_trainer import BiafDependency


def evaluate(ckpt, hparams_file):
    """main"""

    trainer = Trainer(gpus=[1,], distributed_backend="ddp")

    model = BiafDependency.load_from_checkpoint(
        checkpoint_path=ckpt,
        hparams_file=hparams_file,
        map_location=None,
        batch_size=32,
        max_length=128,
        workers=0,
        group_sample=False,
        # if you would like to evaluate on new dataset, defaults to origin test loader configured in training
        # data_dir="/data/nfsdata2/nlp_application/datasets/treebank/LDC99T42/ptb3_parser/",
        # data_format="conllx"
    )

    # test on trainer.test_dataloader
    trainer.test(model)


if __name__ == '__main__':
    # ptb
    HPARAMS = "/data/yuxian/train_logs/dependency/ptb/20210102/lr_1e-3_adam0.999/lightning_logs/version_1/hparams.yaml"
    CHECKPOINT = "/data/yuxian/train_logs/dependency/ptb/20210102/lr_1e-3_adam0.999/epoch=46.ckpt"
    evaluate(ckpt=CHECKPOINT, hparams_file=HPARAMS)
