# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: mrc_biaffine_dependency_config
@time: 2020/12/17 14:50
@desc: 

"""


from typing import List
from transformers import BertConfig


class BertDependencyConfig(BertConfig):
    def __init__(self, pos_tags: List[str], dep_tags: List[str], **kwargs):
        super(BertDependencyConfig, self).__init__(**kwargs)
        self.pos_tags = pos_tags
        self.dep_tags = dep_tags
        self.tag_representation_dim = kwargs.get("tag_representation_dim", self.hidden_size)
        self.arc_representation_dim = kwargs.get("arc_representation_dim", self.hidden_size)
        self.pos_dim = kwargs.get("pos_dim", 0)
        self.biaf_dropout = kwargs.get("biaf_dropout", 0.0)
        self.additional_layer = kwargs.get("additional_layer", 0)
        self.additional_layer_type = kwargs.get("additional_layer_type", "lstm")
        self.additional_layer_dim = kwargs.get("additional_layer_dim", self.hidden_size) or self.hidden_size
