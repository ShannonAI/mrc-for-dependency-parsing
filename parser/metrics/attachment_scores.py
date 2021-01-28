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
        
        self.add_state("unlabeled_correct_1", default=torch.tensor(.0), dist_reduce_fx="sum")
        self.add_state("unlabeled_correct_11", default=torch.tensor(.0), dist_reduce_fx="sum")
        self.add_state("unlabeled_correct_21", default=torch.tensor(.0), dist_reduce_fx="sum")
        self.add_state("unlabeled_correct_31", default=torch.tensor(.0), dist_reduce_fx="sum")
        self.add_state("unlabeled_correct_41", default=torch.tensor(.0), dist_reduce_fx="sum")
        self.add_state("unlabeled_correct_51", default=torch.tensor(.0), dist_reduce_fx="sum")

        self.add_state("labeled_correct_1", default=torch.tensor(.0), dist_reduce_fx="sum")
        self.add_state("labeled_correct_11", default=torch.tensor(.0), dist_reduce_fx="sum")
        self.add_state("labeled_correct_21", default=torch.tensor(.0), dist_reduce_fx="sum")
        self.add_state("labeled_correct_31", default=torch.tensor(.0), dist_reduce_fx="sum")
        self.add_state("labeled_correct_41", default=torch.tensor(.0), dist_reduce_fx="sum")
        self.add_state("labeled_correct_51", default=torch.tensor(.0), dist_reduce_fx="sum")

        self.add_state("total_words_1", default=torch.tensor(.0), dist_reduce_fx="sum")
        self.add_state("total_words_11", default=torch.tensor(.0), dist_reduce_fx="sum")
        self.add_state("total_words_21", default=torch.tensor(.0), dist_reduce_fx="sum")
        self.add_state("total_words_31", default=torch.tensor(.0), dist_reduce_fx="sum")
        self.add_state("total_words_41", default=torch.tensor(.0), dist_reduce_fx="sum")
        self.add_state("total_words_51", default=torch.tensor(.0), dist_reduce_fx="sum")


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
        nwords,
        subspans,
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
        sent_length_bucket = {1: [], 11:[], 21:[], 31:[], 41:[], 51:[]}
        for idx, cur_sent_len in enumerate(nwords):
            if cur_sent_len >= 1 and cur_sent_len <= 10:
                sent_length_bucket[1].append(idx)
            elif cur_sent_len >=11 and cur_sent_len <= 20:
                sent_length_bucket[11].append(idx)
            elif cur_sent_len >= 21 and cur_sent_len <= 30:
                sent_length_bucket[21].append(idx)
            elif cur_sent_len >= 31 and cur_sent_len <= 40:
                sent_length_bucket[31].append(idx)
            elif cur_sent_len >= 41 and cur_sent_len <= 50:
                sent_length_bucket[41].append(idx)
            else:
                sent_length_bucket[51].append(idx)
        
        self.unlabeled_correct_1 += correct_indices[sent_length_bucket[1]].sum()
        self.unlabeled_correct_11 += correct_indices[sent_length_bucket[11]].sum()
        self.unlabeled_correct_21 += correct_indices[sent_length_bucket[21]].sum()
        self.unlabeled_correct_31 += correct_indices[sent_length_bucket[31]].sum()
        self.unlabeled_correct_41 += correct_indices[sent_length_bucket[41]].sum()
        self.unlabeled_correct_51 += correct_indices[sent_length_bucket[51]].sum()
        
        self.labeled_correct_1 += correct_labels_and_indices[sent_length_bucket[1]].sum()
        self.labeled_correct_11 += correct_labels_and_indices[sent_length_bucket[11]].sum()
        self.labeled_correct_21 += correct_labels_and_indices[sent_length_bucket[21]].sum()
        self.labeled_correct_31 += correct_labels_and_indices[sent_length_bucket[31]].sum()
        self.labeled_correct_41 += correct_labels_and_indices[sent_length_bucket[41]].sum()
        self.labeled_correct_51 += correct_labels_and_indices[sent_length_bucket[51]].sum()

        self.total_words_1 += mask[sent_length_bucket[1]].sum()
        self.total_words_11 += mask[sent_length_bucket[11]].sum()
        self.total_words_21 += mask[sent_length_bucket[21]].sum()
        self.total_words_31 += mask[sent_length_bucket[31]].sum()
        self.total_words_41 += mask[sent_length_bucket[41]].sum()
        self.total_words_51 += mask[sent_length_bucket[51]].sum()


    def compute_length_analysis(self):
        epsilon = 1e-4
        metrics = {
            "UAS_1": self.unlabeled_correct_1 / (self.total_words_1 + epsilon),
            "UAS_11": self.unlabeled_correct_11 / (self.total_words_11 + epsilon),
            "UAS_21": self.unlabeled_correct_21 / (self.total_words_21 + epsilon),
            "UAS_31": self.unlabeled_correct_31 / (self.total_words_31 + epsilon),
            "UAS_41": self.unlabeled_correct_41 / (self.total_words_41 + epsilon),
            "UAS_51": self.unlabeled_correct_51 / (self.total_words_51 + epsilon),
            
            "LAS_1": self.labeled_correct_1 / (self.total_words_1 + epsilon),
            "LAS_11": self.labeled_correct_11 / (self.total_words_11 + epsilon),
            "LAS_21": self.labeled_correct_21 / (self.total_words_21 + epsilon),
            "LAS_31": self.labeled_correct_31 / (self.total_words_31 + epsilon),
            "LAS_41": self.labeled_correct_41 / (self.total_words_41 + epsilon),
            "LAS_51": self.labeled_correct_51 / (self.total_words_51 + epsilon),
        } 

        return metrics