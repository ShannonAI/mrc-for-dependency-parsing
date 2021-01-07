import logging
from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from allennlp.modules import InputVariationalDropout
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import StackedBidirectionalLstmSeq2SeqEncoder
from allennlp.nn import util as allennlp_util
from allennlp.nn.util import (
    get_device_of,
)
from allennlp.nn.util import get_range_vector
from overrides import overrides
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_bert import BertEncoder

from parser.models.t2t_dependency_config import BertMrcT2TDependencyConfig

logger = logging.getLogger(__name__)


class BiaffineDependencyT2TParser(BertPreTrainedModel):
    """
    This dependency parser follows the model of
    [Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)]
    (https://arxiv.org/abs/1611.01734) .
    But We use token-to-token MRC to extract parent and labels
    """

    def __init__(self, config: BertMrcT2TDependencyConfig):
        super().__init__(config)

        self.config = config

        num_dep_labels = len(config.dep_tags)
        num_pos_labels = len(config.pos_tags)
        hidden_size = config.additional_layer_dim

        if config.pos_dim > 0:
            self.pos_embedding = nn.Embedding(num_pos_labels, config.pos_dim)
            nn.init.xavier_uniform_(self.pos_embedding.weight)
            if config.additional_layer_type != "lstm" and config.pos_dim+config.hidden_size != hidden_size:
                self.fuse_layer = nn.Linear(config.pos_dim+config.hidden_size, hidden_size)
                nn.init.xavier_uniform_(self.fuse_layer.weight)
                self.fuse_layer.bias.data.zero_()
            else:
                self.fuse_layer = None
        else:
            self.pos_embedding = None

        self.bert = BertModel(config)

        if config.additional_layer > 0:
            if config.additional_layer_type == "transformer":
                new_config = deepcopy(config)
                new_config.hidden_size = hidden_size
                new_config.num_hidden_layers = config.additional_layer
                new_config.hidden_dropout_prob = new_config.attention_probs_dropout_prob = config.mrc_dropout
                # new_config.attention_probs_dropout_prob = config.biaf_dropout  # todo add to hparams and tune
                self.additional_encoder = BertEncoder(new_config)
                self.additional_encoder.apply(self._init_bert_weights)
            else:
                assert hidden_size % 2 == 0, "Bi-LSTM need an even hidden_size"
                self.additional_encoder = StackedBidirectionalLstmSeq2SeqEncoder(
                    input_size=config.pos_dim+config.hidden_size,
                    hidden_size=hidden_size//2, num_layers=config.additional_layer,
                    recurrent_dropout_probability=config.mrc_dropout, use_highway=True
                )

        else:
            self.additional_encoder = None
        # todo try re-init with xavier-uniform and bias=0
        self.parent_feedforward = nn.Linear(hidden_size, 1)
        self.parent_tag_feedforward = nn.Linear(hidden_size, num_dep_labels)

        self.child_feedforward = nn.Linear(hidden_size, 1)
        self.child_tag_feedforward = nn.Linear(hidden_size, num_dep_labels)

        # self.mrc_dropout = nn.Dropout(config.mrc_dropout)
        self._dropout = InputVariationalDropout(config.mrc_dropout)

        # init linear children
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def _init_bert_weights(self, module):
        """ Initialize the weights. copy from transformers.BertPreTrainedModel"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @overrides
    def forward(
        self,  # type: ignore
        token_ids: torch.LongTensor,
        type_ids: torch.LongTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        span_idx: torch.LongTensor,
        span_tag: torch.LongTensor,
        child_arcs: torch.LongTensor,
        child_tags: torch.LongTensor,
        pos_tags: torch.LongTensor,
        word_mask: torch.BoolTensor,
        mrc_mask: torch.BoolTensor,
    ):
        """  todo implement docstring
        Args:
            token_ids: [batch_size, num_word_pieces]
            type_ids: [batch_size, num_word_pieces]
            offsets: [batch_size, num_words, 2]
            wordpiece_mask: [batch_size, num_word_pieces]
            span_idx: [batch_size, 2]
            span_tag: [batch_size, 1]
            child_arcs: [batch_size, num_words]
            child_tags: [batch_size, num_words]
            pos_tags: [batch_size, num_words]
            word_mask: [batch_size, num_words]
            mrc_mask: [batch_size, num_words]
        Returns:
            parent_probs: [batch_size, num_word]
            parent_tag_probs: [batch_size, num_words]
            arc_nll: [1]
            tag_nll: [1]
        """

        embedded_text_input = self.get_word_embedding(
            token_ids=token_ids,
            offsets=offsets,
            wordpiece_mask=wordpiece_mask,
            type_ids=type_ids,
        )
        if self.pos_embedding is not None:
            embedded_pos_tags = self.pos_embedding(pos_tags)
            embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags], -1)
            if self.fuse_layer is not None:
                embedded_text_input = self.fuse_layer(embedded_text_input)
        # todo compare normal dropout with InputVariationalDropout
        embedded_text_input = self._dropout(embedded_text_input)

        if self.additional_encoder is not None:
            if self.config.additional_layer_type == "transformer":
                extended_attention_mask = self.bert.get_extended_attention_mask(word_mask,
                                                                                word_mask.size(),
                                                                                word_mask.device)
                encoded_text = self.additional_encoder(hidden_states=embedded_text_input,
                                                       attention_mask=extended_attention_mask)[0]
            else:
                encoded_text = self.additional_encoder(inputs=embedded_text_input,
                                                       mask=word_mask)
        else:
            encoded_text = embedded_text_input

        batch_size, seq_len, encoding_dim = encoded_text.size()

        # shape (batch_size, sequence_length, tag_classes)
        parent_tag_scores = self.parent_tag_feedforward(encoded_text)
        # shape (batch_size, sequence_length)
        parent_scores = self.parent_feedforward(encoded_text).squeeze(-1)

        # [bsz, seq_len, tag_classes]
        child_tag_scores = self.child_tag_feedforward(encoded_text)
        # [bsz, seq_len]
        child_scores = self.child_feedforward(encoded_text).squeeze(-1)

        # todo support cases that span_idx and span_tag are None
        # [bsz]
        batch_range_vector = get_range_vector(batch_size, get_device_of(encoded_text))
        # [bsz]
        gold_positions = span_idx[:, 0]

        # compute parent arc loss
        minus_inf = -1e8
        mrc_mask = torch.logical_and(mrc_mask, word_mask)
        parent_scores = parent_scores + (~mrc_mask).float() * minus_inf
        child_scores = child_scores + (~mrc_mask).float() * minus_inf

        # [bsz, seq_len]
        parent_logits = F.log_softmax(parent_scores, dim=-1)
        parent_arc_nll = -parent_logits[batch_range_vector, gold_positions].mean()

        # compute parent tag loss
        parent_tag_nll = F.cross_entropy(parent_tag_scores[batch_range_vector, gold_positions], span_tag)

        parent_probs = F.softmax(parent_scores, dim=-1)
        parent_tag_probs = F.softmax(parent_tag_scores, dim=-1)
        child_probs = F.sigmoid(child_scores)
        child_tag_probs = F.softmax(child_tag_scores, dim=-1)

        child_arc_loss = F.binary_cross_entropy_with_logits(child_scores, child_arcs.float(), reduction="none")
        child_arc_loss = (child_arc_loss * mrc_mask.float()).sum() / mrc_mask.float().sum()
        child_tag_loss = F.cross_entropy(child_tag_scores.view(batch_size*seq_len, -1),
                                         child_tags.view(-1), reduction="none")
        child_tag_loss = (child_tag_loss * child_arcs.float().view(-1)).sum() / (child_arcs.float().sum() + 1e-8)

        return parent_probs, parent_tag_probs, child_probs, child_tag_probs, parent_arc_nll, parent_tag_nll, child_arc_loss, child_tag_loss

    def get_word_embedding(
        self,
        token_ids: torch.LongTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """get word-level embedding"""
        # Shape: [batch_size, num_wordpieces, embedding_size].
        embeddings = self.bert(token_ids, token_type_ids=type_ids, attention_mask=wordpiece_mask)[0]

        # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # span_mask: (batch_size, num_orig_tokens, max_span_length)
        span_embeddings, span_mask = allennlp_util.batched_span_select(embeddings,  offsets)
        span_mask = span_mask.unsqueeze(-1)
        span_embeddings *= span_mask  # zero out paddings

        span_embeddings_sum = span_embeddings.sum(2)
        span_embeddings_len = span_mask.sum(2)
        # Shape: (batch_size, num_orig_tokens, embedding_size)
        orig_embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)

        # All the places where the span length is zero, write in zeros.
        orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0

        return orig_embeddings


if __name__ == '__main__':
    from transformers import BertConfig
    bert_path = "/data/nfsdata2/nlp_application/models/bert/bert-large-cased"
    bert_config = BertConfig.from_pretrained(bert_path)
    bert_dep_config = BertMrcT2TDependencyConfig(
        pos_tags=[f"pos_{i}" for i in range(5)],
        dep_tags=[f"dep_{i}" for i in range(5)],
        additional_layer=3,
        additional_layer_dim=800,
        pos_dim=100,
        mrc_dropout=0.3,
        **bert_config.__dict__
    )
    mrc_dep = BiaffineDependencyT2TParser.from_pretrained(
        bert_path,
        config=bert_dep_config,
    )
    bsz = 2
    num_word_pieces = 128
    num_words = 100

    token_ids = type_ids = wordpiece_mask = torch.ones([bsz, num_word_pieces], dtype=torch.long)
    wordpiece_mask = wordpiece_mask.bool()
    offsets = torch.ones([bsz, num_words, 2], dtype=torch.long)
    span_idx = torch.ones([bsz, 2], dtype=torch.long)
    span_tag = torch.ones([bsz], dtype=torch.long)
    child_arcs = child_labels = pos_tags = torch.ones([bsz, num_words], dtype=torch.long)
    mrc_mask = word_mask = pos_tags.bool()
    y = mrc_dep(
        token_ids,
        type_ids,
        offsets,
        wordpiece_mask,
        span_idx,
        span_tag,
        child_arcs,
        child_labels,
        pos_tags,
        mrc_mask,
        word_mask
    )
    print(y)
