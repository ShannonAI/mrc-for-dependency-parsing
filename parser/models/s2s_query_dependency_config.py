# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2020/12/17 14:50
@desc: 

"""


from typing import List
from transformers import BertConfig, RobertaConfig, XLMRobertaConfig, PretrainedConfig


class BertMrcS2SDependencyConfig(BertConfig):
    def __init__(self, pos_tags: List[str], dep_tags: List[str], **kwargs):
        super().__init__(**kwargs)
        self.pos_tags = pos_tags
        self.dep_tags = dep_tags
        self.pos_dim = kwargs.get("pos_dim", 0)
        self.mrc_dropout = kwargs.get("mrc_dropout", 0.0)
        self.additional_layer = kwargs.get("additional_layer", 0)
        self.additional_layer_type = kwargs.get("additional_layer_type", "lstm")
        self.additional_layer_dim = kwargs.get("additional_layer_dim", self.hidden_size) or self.hidden_size
        self.predict_child = kwargs.get("predict_child", False)


class RobertaMrcS2SDependencyConfig(RobertaConfig):
    def __init__(self, pos_tags: List[str], dep_tags: List[str], **kwargs):
        super().__init__(**kwargs)
        self.pos_tags = pos_tags
        self.dep_tags = dep_tags
        self.pos_dim = kwargs.get("pos_dim", 0)
        self.mrc_dropout = kwargs.get("mrc_dropout", 0.0)
        self.additional_layer = kwargs.get("additional_layer", 0)
        self.additional_layer_type = kwargs.get("additional_layer_type", "lstm")
        self.additional_layer_dim = kwargs.get("additional_layer_dim", self.hidden_size) or self.hidden_size
        self.predict_child = kwargs.get("predict_child", False)


class XLMRobertaMrcS2SDependencyConfig(XLMRobertaConfig):
    def __init__(self, pos_tags: List[str], dep_tags: List[str], **kwargs):
        super().__init__(**kwargs)
        self.pos_tags = pos_tags
        self.dep_tags = dep_tags
        self.pos_dim = kwargs.get("pos_dim", 0)
        self.mrc_dropout = kwargs.get("mrc_dropout", 0.0)
        self.additional_layer = kwargs.get("additional_layer", 0)
        self.additional_layer_type = kwargs.get("additional_layer_type", "lstm")
        self.additional_layer_dim = kwargs.get("additional_layer_dim", self.hidden_size) or self.hidden_size
        self.predict_child = kwargs.get("predict_child", False)


class S2SConfig:
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
        predict_child: bool = False,
    ):
        self.bert_config = bert_config
        self.pos_tags = pos_tags
        self.dep_tags = dep_tags
        self.pos_dim = pos_dim
        self.mrc_dropout = mrc_dropout
        self.additional_layer = additional_layer
        self.additional_layer_type = additional_layer_type
        self.additional_layer_dim = additional_layer_dim or bert_config.hidden_size
        self.predict_child = predict_child
