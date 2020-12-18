from typing import Dict,  Optional
from copy import deepcopy
import logging
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_bert import BertEncoder
from overrides import overrides
import torch
from torch import nn
import torch.nn.functional as F
from allennlp.nn.util import get_range_vector
from allennlp.nn.util import (
    get_device_of,
    masked_log_softmax,
)
from allennlp.nn import util as allennlp_util
from parser.models.mrc_biaffine_dependency_config import BertMrcDependencyConfig

logger = logging.getLogger(__name__)


class BiaffineDependencyT2TParser(BertPreTrainedModel):
    """
    This dependency parser follows the model of
    [Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)]
    (https://arxiv.org/abs/1611.01734) .
    But We use token-to-token MRC to extract parent and labels
    """

    def __init__(self, config: BertMrcDependencyConfig):
        super().__init__(config)

        self.config = config

        num_dep_labels = len(config.dep_tags)
        num_pos_labels = len(config.pos_tags)
        hidden_size = config.hidden_size

        if config.pos_dim > 0:
            self.pos_embedding = nn.Embedding(num_pos_labels, config.pos_dim)
            self.fuse_layer = nn.Linear(config.pos_dim+hidden_size, hidden_size)
        else:
            self.pos_embedding = None

        self.bert = BertModel(config)

        if config.additional_layer > 0:
            new_config = deepcopy(config)
            new_config.hidden_size = hidden_size
            new_config.num_hidden_layers = config.additional_layer
            new_config.hidden_dropout_prob = new_config.attention_probs_dropout_prob = config.mrc_dropout
            self.additional_encoder = BertEncoder(new_config)
        else:
            self.additional_encoder = None

        self.parent_feedforward = nn.Linear(hidden_size, 1)
        self.parent_tag_feedforward = nn.Linear(hidden_size, num_dep_labels)

        self.mrc_dropout = nn.Dropout(config.mrc_dropout)

        self._init_weights(self.parent_feedforward)
        self._init_weights(self.parent_tag_feedforward)
        self._init_weights(self.additional_encoder)

    def _init_weights(self, module):
        """ Initialize the weights. refer to BERT"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
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
        pos_tags: torch.LongTensor,
        word_mask: torch.BoolTensor,
        mrc_mask: torch.BoolTensor,
    ) -> Dict[str, torch.Tensor]:

        embedded_text_input = self.get_word_embedding(
            token_ids=token_ids,
            offsets=offsets,
            wordpiece_mask=wordpiece_mask.bool(),
            type_ids=type_ids,
        )
        if self.pos_embedding is not None:
            embedded_pos_tags = self.pos_embedding(pos_tags)
            embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags], -1)
            embedded_text_input = self.fuse_layer(embedded_text_input)
        embedded_text_input = self.mrc_dropout(embedded_text_input)

        if self.additional_encoder is not None:
            # todo add token-type ids
            extended_attention_mask = self.bert.get_extended_attention_mask(word_mask,
                                                                            word_mask.size(),
                                                                            word_mask.device)
            encoded_text = self.additional_encoder(hidden_states=embedded_text_input,
                                                   attention_mask=extended_attention_mask)[0]
        else:
            encoded_text = embedded_text_input

        batch_size, _, encoding_dim = encoded_text.size()
        # [bsz]
        batch_range_vector = get_range_vector(batch_size, get_device_of(encoded_text))
        # [bsz]
        gold_positions = span_idx[:, 0]

        # shape (batch_size, sequence_length, tag_classes)
        parent_tag_scores = self.parent_tag_feedforward(encoded_text)
        # shape (batch_size, sequence_length)
        parent_scores = self.parent_feedforward(encoded_text).squeeze(-1)

        # compute parent arc loss
        minus_inf = -1e4
        mrc_mask = torch.logical_and(mrc_mask, word_mask)
        parent_scores = parent_scores + (~mrc_mask).float() * minus_inf

        # [bsz, seq_len]
        parent_logits = F.log_softmax(parent_scores, dim=-1)
        arc_nll = -parent_logits[batch_range_vector, gold_positions].mean()

        # compute parent tag loss
        tag_nll = F.cross_entropy(parent_tag_scores[batch_range_vector, gold_positions], span_tag)

        parent_probs = F.softmax(parent_scores, dim=-1)
        parent_tag_probs = F.softmax(parent_tag_scores, dim=-1)

        # loss = arc_nll + tag_nll
        return parent_probs, parent_tag_probs, arc_nll, tag_nll

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
    bert_dep_config = BertMrcDependencyConfig(
        # "/data/nfsdata2/nlp_application/models/bert/bert-large-cased",
        pos_tags=[f"pos_{i}" for i in range(5)],
        dep_tags=[f"dep_{i}" for i in range(5)],
        additional_layer=3,
        pos_dim=100,
        mrc_dropout=0.3,
        **bert_config.__dict__
    )
    mrc_dep = BiaffineDependencyT2TParser.from_pretrained(
        bert_path,
        config=bert_dep_config,
    )
    print(mrc_dep)
    bsz = 2
    num_word_pieces = 128
    num_words = 100

    token_ids = type_ids = wordpiece_mask = torch.ones([bsz, num_word_pieces], dtype=torch.long)
    wordpiece_mask = wordpiece_mask.bool()
    offsets = torch.ones([bsz, num_words, 2], dtype=torch.long)
    span_idx = torch.ones([bsz, 2], dtype=torch.long)
    span_tag = torch.ones([bsz], dtype=torch.long)
    pos_tags = torch.ones([bsz, num_words], dtype=torch.long)
    mrc_mask = word_mask = pos_tags.bool()
    y = mrc_dep(
        token_ids,
        type_ids,
        offsets,
        wordpiece_mask,
        span_idx,
        span_tag,
        pos_tags,
        mrc_mask,
        word_mask
    )
    print(y)
