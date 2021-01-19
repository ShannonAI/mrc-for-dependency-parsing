# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2020/12/17 14:50
@desc: 

"""


from typing import List
from transformers import BertConfig


class SpanProposalConfig:
    def __init__(
        self,
        bert_config: BertConfig,
        pos_tags: List[str],
        dep_tags: List[str],
        pos_dim: int = 0,
        mrc_dropout: float = 0.0,
        additional_layer: int = 0,
        additional_layer_type: str = "lstm",
        additional_layer_dim: int = 0,
        arc_representation_dim: int = 0
    ):
        self.bert_config = bert_config
        self.pos_tags = pos_tags
        self.dep_tags = dep_tags
        self.pos_dim = pos_dim
        self.mrc_dropout = mrc_dropout
        self.additional_layer = additional_layer
        self.additional_layer_type = additional_layer_type
        self.additional_layer_dim = additional_layer_dim or bert_config.hidden_size
        self.arc_representation_dim = arc_representation_dim or bert_config.hidden_size
