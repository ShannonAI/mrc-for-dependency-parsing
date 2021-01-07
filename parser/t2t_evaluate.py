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

from parser.t2t_trainer import MrcDependency


def evaluate(ckpt, hparams_file):
    """main"""

    trainer = Trainer(gpus=[0,], distributed_backend="ddp")

    model = MrcDependency.load_from_checkpoint(
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
    HPARAMS = "/userhome/yuxian/train_logs/dependency/ptb/t2t/20210104/finetune_0l_lr1e-5_dropout0.3_bs2560_warmup300/lightning_logs/version_0/hparams.yaml"
    # HPARAMS = "/data/yuxian/train_logs/dependency/ptb/reproduce/lightning_logs/version_0/hparams.yaml"
    # CHECKPOINT = "/data/yuxian/train_logs/dependency/ptb/reproduce/epoch=34.ckpt"
    CHECKPOINT = "/userhome/yuxian/train_logs/dependency/ptb/t2t/20210104/finetune_0l_lr1e-5_dropout0.3_bs2560_warmup300/epoch=6.ckpt"
    evaluate(ckpt=CHECKPOINT, hparams_file=HPARAMS)
