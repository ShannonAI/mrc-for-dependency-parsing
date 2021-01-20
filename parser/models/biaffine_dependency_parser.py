import logging
from copy import deepcopy
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from allennlp.common.params import Params
from allennlp.modules import FeedForward
from allennlp.modules import InputVariationalDropout
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import StackedBidirectionalLstmSeq2SeqEncoder
from allennlp.nn import Activation, InitializerApplicator
from allennlp.nn import util as allennlp_util
from allennlp.nn.chu_liu_edmonds import decode_mst
from allennlp.nn.util import (
    get_device_of,
    masked_log_softmax,
)
from allennlp.nn.util import get_range_vector
from overrides import overrides
from torch import nn
from transformers import BertModel, BertPreTrainedModel, RobertaModel, AutoModel
from transformers.modeling_bert import BertEncoder

from parser.models.biaffine_dependency_config import BertDependencyConfig, RobertaDependencyConfig 

logger = logging.getLogger(__name__)


class BiaffineDependencyParser(nn.Module):
    """
    This dependency parser follows the model of
    [Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)]
    (https://arxiv.org/abs/1611.01734) .
    But we use BERT for embedding
    """

    def __init__(self, bert_dir, config):
        super().__init__()

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
            
        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path=bert_dir)

        if config.additional_layer > 0:
            if config.additional_layer_type == "transformer":
                new_config = deepcopy(config)
                new_config.hidden_size = hidden_size
                new_config.num_hidden_layers = config.additional_layer
                new_config.hidden_dropout_prob = config.biaf_dropout
                new_config.attention_probs_dropout_prob = config.biaf_dropout  # todo add to hparams and tune
                self.additional_encoder = BertEncoder(new_config)
                self.additional_encoder.apply(self._init_bert_weights)

            else:
                assert hidden_size % 2 == 0, "Bi-LSTM need an even hidden_size"
                self.additional_encoder = StackedBidirectionalLstmSeq2SeqEncoder(
                    input_size=config.pos_dim+config.hidden_size,
                    hidden_size=hidden_size//2, num_layers=config.additional_layer,
                    recurrent_dropout_probability=config.biaf_dropout, use_highway=True
                )

        else:
            self.additional_encoder = None

        self.head_arc_feedforward = FeedForward(
            hidden_size, 1, config.arc_representation_dim, Activation.by_name("elu")()
        )
        self.child_arc_feedforward = deepcopy(self.head_arc_feedforward)
        self.arc_attention = BilinearMatrixAttention(
            config.arc_representation_dim, config.arc_representation_dim, use_input_biases=True
        )

        self.head_tag_feedforward = FeedForward(
            hidden_size, 1, config.tag_representation_dim, Activation.by_name("elu")()
        )
        self.child_tag_feedforward = deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = nn.modules.Bilinear(
            config.tag_representation_dim, config.tag_representation_dim, num_dep_labels
        )
        nn.init.xavier_uniform_(self.tag_bilinear.weight)
        self.tag_bilinear.bias.data.zero_()

        self._dropout = InputVariationalDropout(config.biaf_dropout)
        self._input_dropout = nn.Dropout(config.biaf_dropout)

        self._head_sentinel = torch.nn.Parameter(torch.randn([1, 1, hidden_size]))

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
        dep_idxs: torch.LongTensor,
        dep_tags: torch.LongTensor,
        pos_tags: torch.LongTensor,
        word_mask: torch.BoolTensor,
    ):

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
        embedded_text_input = self._input_dropout(embedded_text_input)

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

        batch_size, _, encoding_dim = encoded_text.size()
        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text = torch.cat([head_sentinel, encoded_text], 1)
        word_mask = torch.cat([word_mask.new_ones(batch_size, 1), word_mask], 1)
        dep_idxs = torch.cat([dep_idxs.new_zeros(batch_size, 1), dep_idxs], 1)
        dep_tags = torch.cat([dep_tags.new_zeros(batch_size, 1), dep_tags], 1)

        encoded_text = self._dropout(encoded_text)

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self._dropout(self.head_arc_feedforward(encoded_text))
        child_arc_representation = self._dropout(self.child_arc_feedforward(encoded_text))

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self._dropout(self.head_tag_feedforward(encoded_text))
        child_tag_representation = self._dropout(self.child_tag_feedforward(encoded_text))
        # shape (batch_size, sequence_length, sequence_length)
        attended_arcs = self.arc_attention(head_arc_representation, child_arc_representation)

        minus_inf = -1e8
        minus_mask = ~word_mask * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        if self.training:
            predicted_heads, predicted_head_tags = self._greedy_decode(
                head_tag_representation, child_tag_representation, attended_arcs, word_mask
            )
        else:
            predicted_heads, predicted_head_tags = self._mst_decode(
                head_tag_representation, child_tag_representation, attended_arcs, word_mask
            )

        arc_nll, tag_nll = self._construct_loss(
            head_tag_representation=head_tag_representation,
            child_tag_representation=child_tag_representation,
            attended_arcs=attended_arcs,
            head_indices=dep_idxs,
            head_tags=dep_tags,
            mask=word_mask,
        )

        return predicted_heads, predicted_head_tags, arc_nll, tag_nll

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

    def _construct_loss(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        head_indices: torch.Tensor,
        head_tags: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        # Parameters

        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        head_indices : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for every word.
        head_tags : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length).
            The dependency labels of the heads for every word.
        mask : `torch.BoolTensor`, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        # Returns

        arc_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc loss.
        tag_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc tag loss.
        """
        batch_size, sequence_length, _ = attended_arcs.size()
        # shape (batch_size, 1)
        range_vector = get_range_vector(batch_size, get_device_of(attended_arcs)).unsqueeze(1)
        # shape (batch_size, sequence_length, sequence_length)
        normalised_arc_logits = (
            masked_log_softmax(attended_arcs, mask) * mask.unsqueeze(2) * mask.unsqueeze(1)
        )

        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(
            head_tag_representation, child_tag_representation, head_indices
        )
        normalised_head_tag_logits = masked_log_softmax(
            head_tag_logits, mask.unsqueeze(-1)
        ) * mask.unsqueeze(-1)
        # index matrix with shape (batch, sequence_length)
        timestep_index = get_range_vector(sequence_length, get_device_of(attended_arcs))
        child_index = (
            timestep_index.view(1, sequence_length).expand(batch_size, sequence_length).long()
        )
        # shape (batch_size, sequence_length)
        arc_loss = normalised_arc_logits[range_vector, child_index, head_indices]
        tag_loss = normalised_head_tag_logits[range_vector, child_index, head_tags]
        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        arc_loss = arc_loss[:, 1:]
        tag_loss = tag_loss[:, 1:]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        arc_nll = -arc_loss.sum() / valid_positions.float()
        tag_nll = -tag_loss.sum() / valid_positions.float()
        return arc_nll, tag_nll

    def _greedy_decode(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs independently. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).

        # Parameters

        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.

        # Returns

        heads : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        attended_arcs = attended_arcs + torch.diag(
            attended_arcs.new(mask.size(1)).fill_(-np.inf)
        )
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            minus_mask = ~mask.unsqueeze(2)
            attended_arcs.masked_fill_(minus_mask, -np.inf)

        # Compute the heads greedily.
        # shape (batch_size, sequence_length)
        _, heads = attended_arcs.max(dim=2)

        # Given the greedily predicted heads, decode their dependency tags.
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(
            head_tag_representation, child_tag_representation, heads
        )
        _, head_tags = head_tag_logits.max(dim=2)
        return heads, head_tags

    def _mst_decode(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions using the Edmonds' Algorithm
        for finding minimum spanning trees on directed graphs. Nodes in the
        graph are the words in the sentence, and between each pair of nodes,
        there is an edge in each direction, where the weight of the edge corresponds
        to the most likely dependency label probability for that arc. The MST is
        then generated from this directed graph.

        # Parameters

        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.

        # Returns

        heads : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the optimally decoded heads of each word.
        """
        batch_size, sequence_length, tag_representation_dim = head_tag_representation.size()

        lengths = mask.data.sum(dim=1).long().cpu().numpy()

        expanded_shape = [batch_size, sequence_length, sequence_length, tag_representation_dim]
        head_tag_representation = head_tag_representation.unsqueeze(2)
        head_tag_representation = head_tag_representation.expand(*expanded_shape).contiguous()
        child_tag_representation = child_tag_representation.unsqueeze(1)
        child_tag_representation = child_tag_representation.expand(*expanded_shape).contiguous()
        # Shape (batch_size, sequence_length, sequence_length, num_head_tags)
        pairwise_head_logits = self.tag_bilinear(head_tag_representation, child_tag_representation)

        # Note that this log_softmax is over the tag dimension, and we don't consider pairs
        # of tags which are invalid (e.g are a pair which includes a padded element) anyway below.
        # Shape (batch, num_labels,sequence_length, sequence_length)
        normalized_pairwise_head_logits = F.log_softmax(pairwise_head_logits, dim=3).permute(
            0, 3, 1, 2
        )

        # Mask padded tokens, because we only want to consider actual words as heads.
        minus_inf = -1e8
        minus_mask = ~mask * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # Shape (batch_size, sequence_length, sequence_length)
        normalized_arc_logits = F.log_softmax(attended_arcs, dim=2).transpose(1, 2)

        # Shape (batch_size, num_head_tags, sequence_length, sequence_length)
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that i is the head of j". In this
        # case, we have heads pointing to their children.
        batch_energy = torch.exp(
            normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits
        )
        return self._run_mst_decoding(batch_energy, lengths)

    @staticmethod
    def _run_mst_decoding(
        batch_energy: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        head_tags = []
        for energy, length in zip(batch_energy.detach().cpu(), lengths):
            scores, tag_ids = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want any word to be the parent of the root node.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_tags = []
            for child, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, child].item())
            # We don't care what the head or tag is for the root token, but by default it's
            # not necessarily the same in the batched vs unbatched case, which is annoying.
            # Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_tags[0] = 0
            heads.append(instance_heads)
            head_tags.append(instance_head_tags)
        return (
            torch.from_numpy(np.stack(heads)).to(batch_energy.device),
            torch.from_numpy(np.stack(head_tags)).to(batch_energy.device),
        )

    def _get_head_tags(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        head_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes the head tags given the head and child tag representations
        and a tensor of head indices to compute tags for. Note that these are
        either gold or predicted heads, depending on whether this function is
        being called to compute the loss, or if it's being called during inference.

        # Parameters

        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        head_indices : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length). The indices of the heads
            for every word.

        # Returns

        head_tag_logits : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over tags
            for each arc.
        """
        batch_size = head_tag_representation.size(0)
        # shape (batch_size,)
        range_vector = get_range_vector(
            batch_size, get_device_of(head_tag_representation)
        ).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really
        # need to read the docs to understand. See here:
        # https://docs.scipy.org/doc/np-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word from the
        # sequence length dimension for each element in the batch.

        # shape (batch_size, sequence_length, tag_representation_dim)
        selected_head_tag_representations = head_tag_representation[range_vector, head_indices]
        selected_head_tag_representations = selected_head_tag_representations.contiguous()
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(
            selected_head_tag_representations, child_tag_representation
        )
        return head_tag_logits


if __name__ == '__main__':
    from transformers import BertConfig
    bert_path = "/data/nfsdata2/nlp_application/models/bert/bert-large-uncased-whole-word-masking"
    bert_config = BertConfig.from_pretrained(bert_path)
    bert_dep_config = BertDependencyConfig(
        pos_tags=[f"pos_{i}" for i in range(5)],
        dep_tags=[f"dep_{i}" for i in range(5)],
        additional_layer=3,
        pos_dim=100,
        biaf_dropout=0.3,
        additional_layer_type="transformer",
        additional_layer_dim=1024,
        # additional_layer_type="lstm",
        # additional_layer_dim=800,
        arc_representation_dim=500,
        tag_representation_dim=100,
        **bert_config.__dict__
    )
    parser = BiaffineDependencyParser.from_pretrained(
        bert_path,
        config=bert_dep_config,
    )
    print(parser)
    bsz = 2
    num_word_pieces = 128
    num_words = 100

    token_ids = type_ids = wordpiece_mask = torch.ones([bsz, num_word_pieces], dtype=torch.long)
    wordpiece_mask = wordpiece_mask.bool()
    offsets = torch.ones([bsz, num_words, 2], dtype=torch.long)
    dep_idxs = dep_tags = pos_tags = torch.ones([bsz, num_words], dtype=torch.long)
    mrc_mask = word_mask = pos_tags.bool()
    y = parser(
        token_ids,
        type_ids,
        offsets,
        wordpiece_mask,
        dep_idxs,
        dep_tags,
        pos_tags,
        word_mask
    )
    print(y)
