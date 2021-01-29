# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: attachment_scores
@time: 2020/12/17 19:43
@desc: 

"""

from typing import Optional, List, Any, Callable
from pytorch_lightning.metrics import Metric
import torch
from parser.data.tree_utils import build_subtree_spans

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
        self.add_state("labeled_correct", default=torch.tensor(.0).cuda(), dist_reduce_fx="sum")
        self.add_state("unlabeled_correct", default=torch.tensor(.0).cuda(), dist_reduce_fx="sum")
        # self.add_state("exact_labeled_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        # self.add_state("exact_unlabeled_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_words", default=torch.tensor(.0).cuda(), dist_reduce_fx="sum")
        # self.add_state("total_sentences", default=torch.tensor(0), dist_reduce_fx="sum")
        
        #sentence length: [1, 10], [11, 20], [21, 30], [31, 40], [41, 50], [51, ]
        self.add_state("unlabeled_correct_sent_len", default=torch.tensor([.0] * 6).cuda(), dist_reduce_fx="sum")
        self.add_state("labeled_correct_sent_len", default=torch.tensor([.0] * 6).cuda(), dist_reduce_fx="sum")
        self.add_state("total_words_sent_len", default=torch.tensor([.0] * 6).cuda(), dist_reduce_fx="sum")

        #dependency distance: 1, 2, 3, 4, 5, 6, 7, >7 
        self.add_state("unlabeled_correct_dep_dis", default=torch.tensor([.0] * 8).cuda(), dist_reduce_fx="sum")
        self.add_state("labeled_correct_dep_dis", default=torch.tensor([.0] * 8).cuda(), dist_reduce_fx="sum")
        self.add_state("total_words_dep_dis", default=torch.tensor([.0] * 8).cuda(), dist_reduce_fx="sum")

        #average subtree span length: [1, 2), [2, 3), [3, 5), [5, 7), [7, 10), [10, 15)	[15, )
        self.add_state("unlabeled_correct_span_len", default=torch.tensor([.0] * 7).cuda(), dist_reduce_fx="sum")
        self.add_state("labeled_correct_span_len", default=torch.tensor([.0] * 7).cuda(), dist_reduce_fx="sum")
        self.add_state("total_words_span_len", default=torch.tensor([.0] * 7).cuda(), dist_reduce_fx="sum")

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
    
    def sent_length_error_analysis(self, words, correct_indices, correct_labels_and_indices, mask):
        # sentence length
        nwords = [len(item) for item in words]
        sent_len_bucket = {0: [], 1:[], 2:[], 3:[], 4:[], 5:[]}

        for idx, cur_sent_len in enumerate(nwords):
            if cur_sent_len >= 1 and cur_sent_len <= 10:
                sent_len_bucket[0].append(idx)
            elif cur_sent_len >=11 and cur_sent_len <= 20:
                sent_len_bucket[1].append(idx)
            elif cur_sent_len >= 21 and cur_sent_len <= 30:
                sent_len_bucket[2].append(idx)
            elif cur_sent_len >= 31 and cur_sent_len <= 40:
                sent_len_bucket[3].append(idx)
            elif cur_sent_len >= 41 and cur_sent_len <= 50:
                sent_len_bucket[4].append(idx)
            else:
                sent_len_bucket[5].append(idx)

        for k, v in sent_len_bucket.items():
            self.unlabeled_correct_sent_len[k] += correct_indices[v].sum()
            self.labeled_correct_sent_len[k] += correct_labels_and_indices[v].sum()
            self.total_words_sent_len[k] += mask[v].sum()

    def dep_distance_error_analysis(self, gold_indices, correct_indices, correct_labels_and_indices, mask):
        bsz, timesteps = gold_indices.size()
        dep_length = torch.abs(gold_indices - torch.range(0, timesteps-1).repeat(bsz, 1).long().cuda()) * mask
        for batch_idx in range(bsz):
            sent_dep_bucket = {0: [], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
            for word_idx in range(timesteps):
                if dep_length[batch_idx][word_idx] == 1:
                    sent_dep_bucket[0].append(word_idx)
                elif dep_length[batch_idx][word_idx] == 2:
                    sent_dep_bucket[1].append(word_idx)
                elif dep_length[batch_idx][word_idx] == 3:
                    sent_dep_bucket[2].append(word_idx)
                elif dep_length[batch_idx][word_idx] == 4:
                    sent_dep_bucket[3].append(word_idx)
                elif dep_length[batch_idx][word_idx] == 5:
                    sent_dep_bucket[4].append(word_idx)
                elif dep_length[batch_idx][word_idx] == 6:
                    sent_dep_bucket[5].append(word_idx)
                elif dep_length[batch_idx][word_idx] == 7:
                    sent_dep_bucket[6].append(word_idx)
                elif dep_length[batch_idx][word_idx] >= 8:
                    sent_dep_bucket[7].append(word_idx)

            for k, v in sent_dep_bucket.items():
                self.unlabeled_correct_dep_dis[k] += correct_indices[batch_idx][v].sum()
                self.labeled_correct_dep_dis[k] += correct_labels_and_indices[batch_idx][v].sum()
                self.total_words_dep_dis[k] += mask[batch_idx][v].sum()   

    def span_length_error_analysis(self, gold_indices, correct_indices, correct_labels_and_indices, mask):
        bsz, timestep = gold_indices.size()
        for batch_idx in range(bsz):
            subtree_span = build_subtree_spans(gold_indices[batch_idx])
            subtree_span_len_bucket = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
            for word_idx in range(timestep): 
                parent_index = gold_indices[batch_idx][word_idx]
                child_subtree_span_len = subtree_span[word_idx][1] - subtree_span[word_idx][0] + 1
                parent_subtree_span_len = subtree_span[parent_index][1] - subtree_span[parent_index][0] + 1
                average_subtree_span_len =  (parent_subtree_span_len + child_subtree_span_len) / 2.0
                if average_subtree_span_len >= 1 and average_subtree_span_len < 2:
                    subtree_span_len_bucket[0].append(word_idx)
                elif average_subtree_span_len >= 2 and average_subtree_span_len < 3:
                    subtree_span_len_bucket[1].append(word_idx)
                elif average_subtree_span_len >= 3 and average_subtree_span_len < 5:
                    subtree_span_len_bucket[2].append(word_idx)
                elif average_subtree_span_len >= 5 and average_subtree_span_len < 7:
                    subtree_span_len_bucket[3].append(word_idx)
                elif average_subtree_span_len >= 7 and average_subtree_span_len < 10:
                    subtree_span_len_bucket[4].append(word_idx)
                elif average_subtree_span_len >= 10 and average_subtree_span_len < 15:
                    subtree_span_len_bucket[5].append(word_idx)
                elif average_subtree_span_len >= 15:
                    subtree_span_len_bucket[6].append(word_idx)

            for k, v in subtree_span_len_bucket.items():
                self.unlabeled_correct_span_len[k] += correct_indices[batch_idx][v].sum()
                self.labeled_correct_span_len[k] += correct_labels_and_indices[batch_idx][v].sum()
                self.total_words_span_len[k] += mask[batch_idx][v].sum()   
         
    
    def update_error_analysis(  # type: ignore
        self,
        predicted_indices: torch.Tensor,
        predicted_labels: torch.Tensor,
        gold_indices: torch.Tensor,
        gold_labels: torch.Tensor,
        words: List,
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

        self.sent_length_error_analysis(words, correct_indices, correct_labels_and_indices, mask)
        self.dep_distance_error_analysis(gold_indices, correct_indices, correct_labels_and_indices, mask)
        self.span_length_error_analysis(gold_indices, correct_indices, correct_labels_and_indices, mask)   

    def compute_error_analysis(self):
        epsilon = 1e-4
        metrics = {}
        for idx, v in {0: 1, 1: 11, 2: 21, 3: 31, 4: 41, 5: 51}.items():
            metrics["UAS_Sent_Len" + str(v)] = self.unlabeled_correct_sent_len[idx] / (self.total_words_sent_len[idx] + epsilon)
            metrics["LAS_Sent_Len" + str(v)] = self.labeled_correct_sent_len[idx] / (self.total_words_sent_len[idx] + epsilon) 

        for idx in range(8):
            metrics["UAS_Dep_Dis" + str(idx + 1)] = self.unlabeled_correct_dep_dis[idx] / (self.total_words_dep_dis[idx] + epsilon)
            metrics["LAS_Dep_Dis" + str(idx + 1)] = self.labeled_correct_dep_dis[idx] / (self.total_words_dep_dis[idx] + epsilon) 
            
        for idx in range(7):
            metrics["UAS_Span_Len" + str(idx + 1)] = self.unlabeled_correct_span_len[idx] / (self.total_words_span_len[idx] + epsilon)
            metrics["LAS_Span_Len" + str(idx + 1)] = self.labeled_correct_span_len[idx] / (self.total_words_span_len[idx] + epsilon) 

        return metrics