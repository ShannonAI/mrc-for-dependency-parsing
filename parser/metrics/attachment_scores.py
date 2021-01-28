# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: attachment_scores
@time: 2020/12/17 19:43
@desc: 

"""


from threading import enumerate
from typing import Optional, List, Any, Callable
from pytorch_lightning.metrics import Metric
import torch


class AttachmentScores(Metric):
    """
    Computes labeled and unlabeled attachment scores for a
    dependency parse, as well as sentence level exact match
    for both labeled and unlabeled trees. Note that the input
    to this metric is the sampled predictions, not the distribution
    itself.

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
    ) -> None:
        super(AttachmentScores, self).__init__(compute_on_step, dist_sync_on_step, process_group, dist_sync_fn)
        self.add_state("labeled_correct", default=torch.tensor(.0), dist_reduce_fx="sum")
        self.add_state("unlabeled_correct", default=torch.tensor(.0), dist_reduce_fx="sum")
        # self.add_state("exact_labeled_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        # self.add_state("exact_unlabeled_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_words", default=torch.tensor(.0), dist_reduce_fx="sum")
        # self.add_state("total_sentences", default=torch.tensor(0), dist_reduce_fx="sum")
        
        #sentence length: [1, 10], [11, 20], [21, 30], [31, 40], [41, 50], [51, ]
        self.add_state("unlabeled_correct_sent_len", default=torch.tensor([.0, .0, .0, .0, .0, .0]), dist_reduce_fx="sum")
        self.add_state("labeled_correct_sent_len", default=torch.tensor([.0, .0, .0, .0, .0, .0]), dist_reduce_fx="sum")
        self.add_state("total_words_sent_len", default=torch.tensor([.0, .0, .0, .0, .0, .0]), dist_reduce_fx="sum")

        self._ignore_classes: List[int] = ignore_classes or []

    def update(  # type: ignore
        self,
        predicted_indices: torch.Tensor,
        predicted_labels: torch.Tensor,
        gold_indices: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters

        predicted_indices : `torch.Tensor`, required.
            A tensor of head index predictions of shape (batch_size, timesteps).
        predicted_labels : `torch.Tensor`, required.
            A tensor of arc label predictions of shape (batch_size, timesteps).
        gold_indices : `torch.Tensor`, required.
            A tensor of the same shape as `predicted_indices`.
        gold_labels : `torch.Tensor`, required.
            A tensor of the same shape as `predicted_labels`.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of the same shape as `predicted_indices`.
        """

        if mask is None:
            mask = torch.ones_like(predicted_indices).bool()

        # Multiply by a mask denoting locations of
        # gold labels which we should ignore.
        for label in self._ignore_classes:
            label_mask = gold_labels.eq(label)
            mask = mask & ~label_mask

        correct_indices = predicted_indices.eq(gold_indices).long() * mask
        # unlabeled_exact_match = (correct_indices + ~mask).prod(dim=-1)
        correct_labels = predicted_labels.eq(gold_labels).long() * mask
        correct_labels_and_indices = correct_indices * correct_labels
        # labeled_exact_match = (correct_labels_and_indices + ~mask).prod(dim=-1)
        # total_sentences = correct_indices.size(0)
        total_words = mask.sum()

        self.unlabeled_correct += correct_indices.sum()
        # self.exact_unlabeled_correct += unlabeled_exact_match.sum()
        self.labeled_correct += correct_labels_and_indices.sum()
        # self.exact_labeled_correct += labeled_exact_match.sum()
        # self.total_sentences += total_sentences
        self.total_words += total_words

    def compute(self):
        epsilon = 1e-4
        metrics = {
            "UAS": self.unlabeled_correct / (self.total_words + epsilon),
            "LAS": self.labeled_correct / (self.total_words + epsilon)
        }
        return metrics

    def update_length_analysis(  # type: ignore
        self,
        predicted_indices: torch.Tensor,
        predicted_labels: torch.Tensor,
        gold_indices: torch.Tensor,
        gold_labels: torch.Tensor,
        nwords: List,
        #subspans: List, # list of [[(word_idx, start, end)]]
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters

        predicted_indices : `torch.Tensor`, required.
            A tensor of head index predictions of shape (batch_size, timesteps).
        predicted_labels : `torch.Tensor`, required.
            A tensor of arc label predictions of shape (batch_size, timesteps).
        gold_indices : `torch.Tensor`, required.
            A tensor of the same shape as `predicted_indices`.
        gold_labels : `torch.Tensor`, required.
            A tensor of the same shape as `predicted_labels`.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of the same shape as `predicted_indices`.
        """
        if mask is None:
            mask = torch.ones_like(predicted_indices).bool()

        # Multiply by a mask denoting locations of
        # gold labels which we should ignore.
        for label in self._ignore_classes:
            label_mask = gold_labels.eq(label)
            mask = mask & ~label_mask

        correct_indices = predicted_indices.eq(gold_indices).long() * mask
        # unlabeled_exact_match = (correct_indices + ~mask).prod(dim=-1)
        correct_labels = predicted_labels.eq(gold_labels).long() * mask
        correct_labels_and_indices = correct_indices * correct_labels
        # labeled_exact_match = (correct_labels_and_indices + ~mask).prod(dim=-1)
        # total_sentences = correct_indices.size(0)
        total_words = mask.sum()

        self.unlabeled_correct += correct_indices.sum()
        # self.exact_unlabeled_correct += unlabeled_exact_match.sum()
        self.labeled_correct += correct_labels_and_indices.sum()
        # self.exact_labeled_correct += labeled_exact_match.sum()
        # self.total_sentences += total_sentences
        self.total_words += total_words

        # sentence length
        cur_sent_len = len(nwords) - 1
        if cur_sent_len >= 1 and cur_sent_len <= 10:
            self.unlabeled_correct_sent_length[0] += correct_indices.sum()
            self.labeled_correct_sent_length[0] += correct_labels_and_indices.sum()
            self.total_words_sent_length[0] += mask.sum()
        elif cur_sent_len >=11 and cur_sent_len <= 20:
            self.unlabeled_correct_sent_length[1] += correct_indices.sum()
            self.labeled_correct_sent_length[1] += correct_labels_and_indices.sum()
            self.total_words_sent_length[1] += mask.sum()
        elif cur_sent_len >= 21 and cur_sent_len <= 30:
            self.unlabeled_correct_sent_length[2] += correct_indices.sum()
            self.labeled_correct_sent_length[2] += correct_labels_and_indices.sum()
            self.total_words_sent_length[2] += mask.sum()
        elif cur_sent_len >= 31 and cur_sent_len <= 40:
            self.unlabeled_correct_sent_length[3] += correct_indices.sum()
            self.labeled_correct_sent_length[3] += correct_labels_and_indices.sum()
            self.total_words_sent_length[3] += mask.sum()
        elif cur_sent_len >= 41 and cur_sent_len <= 50:
            self.unlabeled_correct_sent_length[4] += correct_indices.sum()
            self.labeled_correct_sent_length[4] += correct_labels_and_indices.sum()
            self.total_words_sent_length[4] += mask.sum()
        else:
            self.unlabeled_correct_sent_length[5] += correct_indices.sum()
            self.labeled_correct_sent_length[5] += correct_labels_and_indices.sum()
            self.total_words_sent_length[5] += mask.sum()
        
       
    def compute_length_analysis(self):
        epsilon = 1e-4
        metrics = {}
        for idx, v in {0: 1, 1: 11, 2: 21, 3: 31, 4: 41, 5: 51}.items():
            metrics["UAS_" + str(v)] = self.unlabeled_correct_sent_length[idx] / (self.total_words_sent_length[idx] + epsilon)
            metrics["LAS_" + str(v)] = self.labeled_correct_sent_length[idx] / (self.total_words_sent_length[idx] + epsilon) 
        return metrics