from itertools import groupby
from typing import Optional, List, Any, Callable
from typing import Tuple

import numpy as np
import torch
from allennlp.nn.chu_liu_edmonds import decode_mst
from pytorch_lightning.metrics import Metric

from parser.metrics.attachment_scores import AttachmentScores

from parser.utils.logger import get_logger

logger = get_logger(__name__)


class S2TAttachmentScores(Metric):
    """
    Computes labeled and unlabeled attachment scores for a
    dependency parse, as well as sentence level exact match
    for both labeled and unlabeled trees. Note that the input
    to this metric is the sampled predictions, not the distribution
    itself. This metrics is used when label and arcs are not computed
    in one single sample, but in multiple continuous samples.

    # Parameters

    ignore_classes : `List[int]`, optional (default = `None`)
        A list of label ids to ignore when computing metrics.
    """

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        ignore_classes: List[int] = None
    ):
        super(S2TAttachmentScores, self).__init__()
        self.normal_attachment_score = AttachmentScores(
            compute_on_step,
            dist_sync_on_step,
            process_group,
            dist_sync_fn,
            ignore_classes
        )
        self.add_state("to_compute_anns", default=[], dist_reduce_fx=None)

    def update(  # type: ignore
        self,
        ann_idxs: List[int],
        word_idxs: List[int],
        sent_lens: List[int],
        parent_probs: torch.Tensor,
        parent_tag_probs: torch.Tensor,
        parent_idxs: torch.Tensor,
        parent_tags: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters
        ann_idxs: `List`, required.
            A list of sample indexes. different querys from same sentence has same ann_idx.
        word_idxs: `List`, required.
            A list of word indexes. which word is queried.
        sent_lens: `List`, required.
            A list of sample length integers.
        parent_probs: `torch.Tensor`, required.
            A tensor of parent predictions of shape (batch_size, timesteps).
        parent_tag_probs: `torch.Tensor`, required.
            A tensor of parent tag predictions of shape (batch_size, timesteps, num_tag_classes)
        parent_idx2 : `torch.LongTensor`, required.
            A torch tensor representing the gold span position target
            in the dependency parse. Has shape `(batch_size)`.
        parent_tag2 : `torch.LongTensor`, required.
            A torch tensor representing the dependency tag. Has shape `(batch_size)`.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of the same shape as `predicted_indices`. if False, mask the corresponding position
        """
        for fields in zip(ann_idxs, word_idxs, sent_lens,
                          parent_probs.detach().cpu().numpy(), parent_tag_probs.detach().cpu().numpy(),
                          parent_idxs.detach().cpu().numpy(), parent_tags.detach().cpu().numpy(),
                          mask.detach().cpu().numpy()):
            self.to_compute_anns.append(fields)

    def compute(self):
        num_tag = self.to_compute_anns[0][4].shape[-1]
        logger.info("grouping and running mst on all samples ...")
        self.to_compute_anns = sorted(self.to_compute_anns, key=lambda x: x[0])
        group_samples = [list(v) for k, v in groupby(self.to_compute_anns, lambda x: x[0])]
        for mrc_samples in group_samples:
            sent_len = mrc_samples[0][2]
            seq_len = sent_len + 1  # add [head]

            if len(mrc_samples) != sent_len:
                logger.warning("compute() should be called only when **all** mrc samples have been updated.")
                continue

            # gather from multiple mrc samples to compute final tree
            gold_indices = torch.ones([sent_len], dtype=torch.long)
            gold_labels = torch.ones([sent_len], dtype=torch.long)
            eval_mask = torch.ones([sent_len], dtype=torch.bool)
            parent_attended_arcs = torch.zeros([seq_len, seq_len])
            parent_pairwise_head_probs = torch.zeros([seq_len, seq_len, num_tag])
            child_attended_arcs = torch.zeros([seq_len, seq_len])
            child_pairwise_head_probs = torch.zeros([seq_len, seq_len, num_tag])
            # note: timesteps==sent_len*2+3, because of mrc format.
            # parent_probs: [seq_len*2+3], parent_tag_probs: [seq_len*2+3, num_tag_classes], span_idx: 2, span_tag: 1
            for (_, word_idx, sent_len, sent_parent_probs, sent_parent_tag_probs,
                 sent_child_probs, sent_child_tag_probs,
                 sent_span_idx, sent_span_tag, mask) in mrc_samples:
                gold_pos = sent_span_idx[0]
                gold_indices[word_idx] = gold_pos-sent_len-2
                gold_labels[word_idx] = sent_span_tag
                eval_mask[word_idx] = bool(mask)
                context_start, context_end = sent_len + 2, 2 * sent_len + 3
                # add parent score
                parent_attended_arcs[word_idx+1] = torch.FloatTensor(sent_parent_probs[context_start: context_end])
                parent_pairwise_head_probs[word_idx+1] = torch.FloatTensor(sent_parent_tag_probs[context_start: context_end])
                # add child score
                child_attended_arcs[:, word_idx+1] = torch.FloatTensor(sent_child_probs[context_start: context_end])
                child_pairwise_head_probs[word_idx + 1] = torch.FloatTensor(sent_parent_tag_probs[context_start: context_end])
            # todo normalize according to row, since every child can only have one parent
            pairwise_head_probs = parent_pairwise_head_probs * child_pairwise_head_probs
            attended_arcs = parent_attended_arcs * child_attended_arcs

            # This energy tensor expresses the following relation:
            # energy[i,j] = "Score that i is the head of j". In this
            # case, we have heads pointing to their children.
            batch_energy = pairwise_head_probs.permute(2, 1, 0).unsqueeze(0) \
                           * attended_arcs.transpose(0, 1).view(1, 1, seq_len, seq_len)

            predicted_indices, predicted_labels = self._run_mst_decoding(batch_energy, lengths=np.array([seq_len]))

            # We calculate attachment scores for the whole sentence
            # but excluding the symbolic ROOT token at the start,
            # which is why we start from the second element in the sequence.
            self.normal_attachment_score.update(
                predicted_indices[:, 1:], predicted_labels[:, 1:], gold_indices, gold_labels, eval_mask
            )

        return self.normal_attachment_score.compute()

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
