import logging
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

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
from transformers import BertModel, BertPreTrainedModel, RobertaModel, AutoModel
from transformers.modeling_bert import BertEncoder

from parser.models.s2s_query_dependency_config import BertMrcS2SDependencyConfig, RobertaMrcS2SDependencyConfig, S2SConfig

logger = logging.getLogger(__name__)


# class BiaffineDependencyS2SQueryParser(BertPreTrainedModel):
class BiaffineDependencyS2SQueryParser(torch.nn.Module):
    """
    This dependency parser follows the model of
    [Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)]
    (https://arxiv.org/abs/1611.01734) .
    But We use token-to-token MRC to extract parent and labels

    Args:
        config: S2SConfig that defines model dim and structure
        bert_dir: pretrained bert directory
    """

    # def __init__(self, config: Union[BertMrcS2SDependencyConfig, RobertaMrcS2SDependencyConfig]):
    def __init__(self, config: S2SConfig, bert_dir: str = ""):
        # super().__init__(config)
        super().__init__()

        self.config = config

        num_dep_labels = len(config.dep_tags)
        num_pos_labels = len(config.pos_tags)
        hidden_size = config.additional_layer_dim

        if config.pos_dim > 0:
            self.pos_embedding = nn.Embedding(num_pos_labels, config.pos_dim)
            nn.init.xavier_uniform_(self.pos_embedding.weight)
            if config.additional_layer_type != "lstm" and config.pos_dim+config.bert_config.hidden_size != hidden_size:
                self.fuse_layer = nn.Linear(config.pos_dim+config.bert_config.hidden_size, hidden_size)
                nn.init.xavier_uniform_(self.fuse_layer.weight)
                self.fuse_layer.bias.data.zero_()
            else:
                self.fuse_layer = None
        else:
            self.pos_embedding = None

        # if isinstance(config, BertMrcS2SDependencyConfig):
        #     self.bert = BertModel(config)
        #     self.arch = "bert"
        # else:
        #     self.roberta = RobertaModel(config)
        #     self.arch = "roberta"
        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path=bert_dir, config=config.bert_config)

        if config.additional_layer > 0:
            if config.additional_layer_type == "transformer":
                new_config = deepcopy(config.bert_config)
                new_config.hidden_size = hidden_size
                new_config.num_hidden_layers = config.additional_layer
                new_config.hidden_dropout_prob = new_config.attention_probs_dropout_prob = config.mrc_dropout
                # new_config.attention_probs_dropout_prob = config.biaf_dropout  # todo add to hparams and tune
                self.additional_encoder = BertEncoder(new_config)
                self.additional_encoder.apply(self._init_bert_weights)
            else:
                assert hidden_size % 2 == 0, "Bi-LSTM need an even hidden_size"
                self.additional_encoder = StackedBidirectionalLstmSeq2SeqEncoder(
                    input_size=config.pos_dim+config.bert_config.hidden_size,
                    hidden_size=hidden_size//2, num_layers=config.additional_layer,
                    recurrent_dropout_probability=config.mrc_dropout, use_highway=True
                )

        else:
            self.additional_encoder = None

        # todo use MLP
        self.parent_feedforward = nn.Linear(hidden_size, 1)
        self.parent_start_feedforward = nn.Linear(hidden_size, 1)
        self.parent_end_feedforward = nn.Linear(hidden_size, 1)
        self.parent_tag_feedforward = nn.Linear(hidden_size, num_dep_labels)

        if config.predict_child:
            self.child_feedforward = nn.Linear(hidden_size, 1)
            self.child_start_feedforward = nn.Linear(hidden_size, 1)
            self.child_end_feedforward = nn.Linear(hidden_size, 1)

        # self._dropout = nn.Dropout(config.mrc_dropout)
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
            module.weight.data.normal_(mean=0.0, std=self.config.bert_config.initializer_range)
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
        pos_tags: torch.LongTensor,
        word_mask: torch.BoolTensor,
        parent_mask: torch.BoolTensor,
        parent_start_mask: torch.BoolTensor,
        parent_end_mask: torch.BoolTensor,
        child_mask: torch.BoolTensor = None,
        parent_idxs: torch.LongTensor = None,
        parent_tags: torch.LongTensor = None,
        parent_starts: torch.BoolTensor = None,
        parent_ends: torch.BoolTensor = None,
        child_idxs: torch.BoolTensor = None,
        child_starts: torch.BoolTensor = None,
        child_ends: torch.BoolTensor = None,
    ):
        """  todo implement docstring
        Args:
            token_ids: [batch_size, num_word_pieces]
            type_ids: [batch_size, num_word_pieces]
            offsets: [batch_size, num_words, 2]
            wordpiece_mask: [batch_size, num_word_pieces]
            pos_tags: [batch_size, num_words]
            word_mask: [batch_size, num_words]
            parent_mask: [batch_size, num_words]
            parent_start_mask: [batch_size, num_words]
            parent_end_mask: [batch_size, num_words]
            child_mask: [batch_size, num_words]
            parent_idxs: [batch_size]
            parent_tags: [batch_size]
            parent_starts: [batch_size]
            parent_ends: [batch_size]
            child_idxs: [batch_size, num_words]
            child_starts: [batch_size, num_words]
            child_ends: [batch_size, num_words]
        Returns:
            parent_probs: [batch_size, num_words]
            parent_tag_probs: [batch_size, num_words, num_tags]
            parent_start_probs: [batch_size, num_words]
            parent_end_probs: [batch_size, num_words]
            child_probs: [batch_size, num_words]
            child_start_probs: [batch_size, num_words]
            child_end_probs: [batch_size, num_words]
            arc_loss (if parent_idx is not None)
            tag_loss (if parent_idxs and parent_tags are not None)
            start_loss (if parent_starts is not None)
            end_loss (if parent_ends is not None)
            child_loss (if child_idxs is not None)
            child_start_loss (if child_starts is not None)
            child_end_loss (if child_ends is not None)
        """

        cls_embedding, embedded_text_input = self.get_word_embedding(
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
                # bert = self.bert if self.arch == "bert" else self.roberta
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
        parent_start_scores = self.parent_start_feedforward(encoded_text).squeeze(-1)
        parent_end_scores = self.parent_end_feedforward(encoded_text).squeeze(-1)

        # mask out impossible positions
        minus_inf = -1e8
        parent_mask = torch.logical_and(parent_mask, word_mask)
        parent_scores = parent_scores + (~parent_mask).float() * minus_inf
        parent_start_mask = torch.logical_and(parent_start_mask, word_mask)
        parent_start_scores = parent_start_scores + (~parent_start_mask).float() * minus_inf
        parent_end_mask = torch.logical_and(parent_end_mask, word_mask)
        parent_end_scores = parent_end_scores + (~parent_end_mask).float() * minus_inf

        parent_probs = F.softmax(parent_scores, dim=-1)
        parent_start_probs = F.softmax(parent_start_scores, dim=-1)
        parent_end_probs = F.softmax(parent_end_scores, dim=-1)
        parent_tag_probs = F.softmax(parent_tag_scores, dim=-1)

        output = (parent_probs, parent_tag_probs, parent_start_probs, parent_end_probs)

        if self.config.predict_child:
            child_scores = self.child_feedforward(encoded_text).squeeze(-1)
            child_start_scores = self.child_start_feedforward(encoded_text).squeeze(-1)
            child_end_scores = self.child_end_feedforward(encoded_text).squeeze(-1)
            # todo add child mask - child should be inside the origin span
            if child_mask is None:
                child_mask = torch.ones_like(word_mask)
            else:
                child_mask = torch.logical_and(child_mask, word_mask)
            child_scores = child_scores + (~child_mask).float() * minus_inf
            child_start_scores = child_start_scores + (~child_mask).float() * minus_inf
            child_end_scores = child_end_scores + (~child_mask).float() * minus_inf
            child_probs = torch.sigmoid(child_scores)
            child_start_probs = torch.sigmoid(child_start_scores)
            child_end_probs = torch.sigmoid(child_end_scores)
            output = output + (child_probs, child_start_probs, child_end_probs)

        # add losses
        batch_range_vector = get_range_vector(batch_size, get_device_of(encoded_text))  # [bsz]
        if parent_idxs is not None:
            # [bsz, seq_len]
            parent_logits = F.log_softmax(parent_scores, dim=-1)
            parent_arc_nll = -parent_logits[batch_range_vector, parent_idxs]
            parent_arc_nll = parent_arc_nll.mean()
            output = output + (parent_arc_nll, )

            if parent_tags is not None:
                parent_tag_nll = F.cross_entropy(parent_tag_scores[batch_range_vector, parent_idxs],
                                                 parent_tags)
                output = output + (parent_tag_nll, )

        if parent_starts is not None:
            # [bsz, seq_len]
            parent_start_logits = F.log_softmax(parent_start_scores, dim=-1)
            parent_start_nll = -parent_start_logits[batch_range_vector, parent_starts].mean()
            output = output + (parent_start_nll,)

        if parent_ends is not None:
            # [bsz, seq_len]
            parent_end_logits = F.log_softmax(parent_end_scores, dim=-1)
            parent_end_nll = -parent_end_logits[batch_range_vector, parent_ends].mean()
            output = output + (parent_end_nll,)

        if self.config.predict_child:
            if child_idxs is not None:
                child_loss = F.binary_cross_entropy_with_logits(child_scores, child_idxs.float(), reduction="none")
                child_loss = (child_loss * child_mask).sum() / (child_mask.sum() + 1e-8)
                output = output + (child_loss, )
            if child_starts is not None:
                child_start_loss = F.binary_cross_entropy_with_logits(child_start_scores, child_starts.float(), reduction="none")
                child_start_loss = (child_start_loss * child_mask).sum() / (child_mask.sum() + 1e-8)
                output = output + (child_start_loss, )
            if child_ends is not None:
                child_end_loss = F.binary_cross_entropy_with_logits(child_end_scores, child_ends.float(), reduction="none")
                child_end_loss = (child_end_loss * child_mask).sum() / (child_mask.sum() + 1e-8)
                output = output + (child_end_loss, )

        return output

    def get_word_embedding(
        self,
        token_ids: torch.LongTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        """get [CLS] embedding and word-level embedding"""
        # Shape: [batch_size, num_wordpieces, embedding_size].
        # embed_model = self.bert if self.arch == "bert" else self.roberta
        # embeddings = embed_model(token_ids, token_type_ids=type_ids, attention_mask=wordpiece_mask)[0]
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

        return embeddings[:, 0, :], orig_embeddings


if __name__ == '__main__':
    from transformers import AutoConfig
    # bert_path = "/data/nfsdata2/nlp_application/models/bert/bert-large-cased"
    bert_path = "/data/nfsdata2/nlp_application/models/bert/roberta-large"
    # bert_config = BertConfig.from_pretrained(bert_path)
    # bert_dep_config = BertMrcS2SDependencyConfig(
    #     pos_tags=[f"pos_{i}" for i in range(5)],
    #     dep_tags=[f"dep_{i}" for i in range(5)],
    #     additional_layer=3,
    #     additional_layer_dim=800,
    #     pos_dim=100,
    #     mrc_dropout=0.3,
    #     predict_child=True,
    #     **bert_config.__dict__
    # )
    # mrc_dep = BiaffineDependencyS2SQueryParser.from_pretrained(
    #     bert_path,
    #     config=bert_dep_config,
    # )

    bert_config = AutoConfig.from_pretrained(bert_path)
    bert_dep_config = S2SConfig(
        bert_config=bert_config,
        pos_tags=[f"pos_{i}" for i in range(5)],
        dep_tags=[f"dep_{i}" for i in range(5)],
        additional_layer=3,
        additional_layer_dim=800,
        pos_dim=100,
        mrc_dropout=0.3,
        predict_child=True,
    )
    mrc_dep = BiaffineDependencyS2SQueryParser(
        bert_dir=bert_path,
        config=bert_dep_config,
    )
    print(mrc_dep)

    bsz = 2
    num_word_pieces = 128
    num_words = 100

    token_ids = wordpiece_mask = torch.ones([bsz, num_word_pieces], dtype=torch.long)
    type_ids = token_ids.clone().fill_(0)
    wordpiece_mask = wordpiece_mask.bool()
    offsets = torch.ones([bsz, num_words, 2], dtype=torch.long)
    pos_tags = torch.ones([bsz, num_words], dtype=torch.long)
    child_starts = child_ends = child_idxs = child_mask = parent_start_mask = parent_end_mask = parent_mask = word_mask = pos_tags.bool()
    parent_start = parent_end = parent_idxs = parent_tags = torch.ones([bsz], dtype=torch.long)

    y = mrc_dep(
        token_ids,
        type_ids,
        offsets,
        wordpiece_mask,
        pos_tags,
        parent_mask,
        parent_start_mask,
        parent_end_mask,
        child_mask,
        word_mask,
        parent_tags,
        parent_idxs,
        parent_start,
        parent_end,
        child_idxs,
        child_starts,
        child_ends
    )
    print(y)
