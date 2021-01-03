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


class ModelPrintCallback(pl.callbacks.base.Callback):
    """print modules before training"""
    def __init__(self, print_modules: List[str]):
        self.print_modules = print_modules

    def on_fit_start(self, trainer, pl_module):
        # Prints only from process 0
        if not trainer.is_global_zero:
            return
        for name in self.print_modules:
            print(f"==== Model Print of {name}: ")
            if name == ".":
                print(pl_module)
            else:
                subnames = name.split(".")
                m = getattr(pl_module, subnames[0])
                for subname in subnames[1:]:
                    m = getattr(m, subname)
                print(m)
