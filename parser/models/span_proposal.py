import logging
from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F
from allennlp.modules import FeedForward
from allennlp.modules import InputVariationalDropout
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import StackedBidirectionalLstmSeq2SeqEncoder
from allennlp.nn import Activation
from allennlp.nn import util as allennlp_util
from overrides import overrides
from torch import nn
from transformers import AutoModel
from transformers.modeling_bert import BertEncoder

from parser.models.span_proposal_config import SpanProposalConfig

logger = logging.getLogger(__name__)


class SpanProposal(torch.nn.Module):
    """
    This model is used to extract candidate start/end subtree span rooted at each token.

    Args:
        config: SpanProposal Config that defines model dim and structure
        bert_dir: pretrained bert directory

    """

    def __init__(self, config: SpanProposalConfig, bert_dir: str = ""):
        super().__init__()

        self.config = config

        num_pos_labels = len(config.pos_tags)
        hidden_size = config.additional_layer_dim if config.additional_layer > 0 else config.pos_dim + config.bert_config.hidden_size

        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path=bert_dir, config=config.bert_config)

        if config.pos_dim > 0:
            self.pos_embedding = nn.Embedding(num_pos_labels, config.pos_dim)
            nn.init.xavier_uniform_(self.pos_embedding.weight)
            if (
                config.additional_layer
                and config.additional_layer_type != "lstm"
                and config.pos_dim+config.bert_config.hidden_size != hidden_size
            ):
                self.fuse_layer = nn.Linear(config.pos_dim+config.bert_config.hidden_size, hidden_size)
                nn.init.xavier_uniform_(self.fuse_layer.weight)
                self.fuse_layer.bias.data.zero_()
            else:
                self.fuse_layer = None
        else:
            self.pos_embedding = None

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

        self._dropout = InputVariationalDropout(config.mrc_dropout)

        self.subtree_start_feedforward = FeedForward(
            hidden_size, 1, config.arc_representation_dim, Activation.by_name("elu")()
        )
        self.subtree_end_feedforward = deepcopy(self.subtree_start_feedforward)

        # todo: equivalent to self-attention?
        self.subtree_start_attention = BilinearMatrixAttention(
            config.arc_representation_dim, config.arc_representation_dim, use_input_biases=True
        )
        self.subtree_end_attention = deepcopy(self.subtree_start_attention)

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
        subtree_spans: torch.LongTensor = None,
    ):
        """  todo implement docstring
        Args:
            token_ids: [batch_size, num_word_pieces]
            type_ids: [batch_size, num_word_pieces]
            offsets: [batch_size, num_words, 2]
            wordpiece_mask: [batch_size, num_word_pieces]
            pos_tags: [batch_size, num_words]
            word_mask: [batch_size, num_words]
            subtree_spans: [batch_size, num_words, 2]
        Returns:
            span_start_logits: [batch_size, num_words, num_words]
            span_end_logits: [batch_size, num_words, num_words]
            span_loss: if subtree_spans is given.

        """
        # [bsz, seq_len, hidden]
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

        # [bsz, seq_len, dim]
        subtree_start_representation = self._dropout(self.subtree_start_feedforward(encoded_text))
        subtree_end_representation = self._dropout(self.subtree_end_feedforward(encoded_text))
        # [bsz, seq_len, seq_len]
        span_start_scores = self.subtree_start_attention(subtree_start_representation, subtree_start_representation)
        span_end_scores = self.subtree_end_attention(subtree_end_representation, subtree_end_representation)

        # start of word should less equal to it
        start_mask = word_mask.unsqueeze(-1) & (~torch.triu(span_start_scores.bool(), 1))
        # end of word should greater equal to it.
        end_mask = word_mask.unsqueeze(-1) & torch.triu(span_end_scores.bool())

        minus_inf = -1e8
        span_start_scores = span_start_scores + (~start_mask).float() * minus_inf
        span_end_scores = span_end_scores + (~end_mask).float() * minus_inf

        output = (
            F.log_softmax(span_start_scores, dim=-1),
            F.log_softmax(span_end_scores, dim=-1)
        )
        if subtree_spans is not None:

            start_loss = F.cross_entropy(span_start_scores.view(batch_size*seq_len, -1),
                                         subtree_spans[:, :, 0].view(-1))
            end_loss = F.cross_entropy(span_end_scores.view(batch_size*seq_len, -1),
                                       subtree_spans[:, :, 1].view(-1))
            span_loss = start_loss + end_loss
            output = output + (span_loss, )

        return output

    def get_word_embedding(
        self,
        token_ids: torch.LongTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        get word-level embedding
        """
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
    from transformers import AutoConfig
    bert_path = "/data/nfsdata2/nlp_application/models/bert/bert-large-cased"
    bert_config = AutoConfig.from_pretrained(bert_path)
    bert_dep_config = SpanProposalConfig(
        pos_tags=[f"pos_{i}" for i in range(5)],
        dep_tags=[f"dep_{i}" for i in range(5)],
        pos_dim=100,
        mrc_dropout=0.3,
        bert_config=bert_config
    )
    mrc_dep = SpanProposal(
        config=bert_dep_config,
        bert_dir=bert_path
    )
    bsz = 2
    num_word_pieces = 128
    num_words = 100

    token_ids = type_ids = wordpiece_mask = torch.ones([bsz, num_word_pieces], dtype=torch.long)
    wordpiece_mask = wordpiece_mask.bool()
    subtree_spans = offsets = torch.ones([bsz, num_words, 2], dtype=torch.long)
    pos_tags = torch.ones([bsz, num_words], dtype=torch.long)
    word_mask = pos_tags.bool()
    y = mrc_dep(
        token_ids,
        type_ids,
        offsets,
        wordpiece_mask,
        pos_tags,
        word_mask,
        subtree_spans,
    )
    print(y)
