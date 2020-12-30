# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: eval_callback
@time: 2020/12/24 18:15
@desc: 

"""


import pytorch_lightning as pl
from typing import List


class EvalCallback(pl.callbacks.base.Callback):
    """callback to make some modules freeze at training time"""
    def __init__(self, freeze_modules: List[str]):
        self.freeze_modules = freeze_modules

    def on_train_epoch_start(self, trainer, pl_module):
        for name in self.freeze_modules:
            subnames = name.split(".")
            m = getattr(pl_module, subnames[0])
            for subname in subnames[1:]:
                m = getattr(m, subname)
            m.eval()
