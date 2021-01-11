# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/1/9 19:55
@desc: 

"""

import json
import math
import numpy as np
import torch
from typing import List, Tuple, Dict, Union
from collections import defaultdict


def build_possible_splits(start: int, end: int, spans: List[Tuple[int, int]]) -> List[List[int]]:
    """
    find all possible combinations of span indexes that those spans can be merged to [start, end], including end
    Examples:
        >>> spans = [(0, 4), (0, 2), (0, 3), (4, 5), (3, 5), (5, 5), (3, 4)]
        >>> idxs_lst = build_possible_splits(0, 5, spans)
        >>> [[spans[i] for i in lst] for lst in idxs_lst]
        [[(0, 4), (5, 5)], [(0, 2), (3, 5)], [(0, 2), (3, 4), (5, 5)], [(0, 3), (4, 5)]]
    """
    if start > end:
        return [[]]
    # pos2spans[i] contains all valid span idxs that starts with i
    pos2spans: Dict[int, List[int]] = defaultdict(list)
    for span_idx, (i, j) in enumerate(spans):
        if i < start or j > end:
            continue
        pos2spans[i].append(span_idx)

    mem: Dict[int, List[List[int]]] = {}

    def dfs(begin: int) -> List[List[int]]:
        """do memorable dfs, find all combinations that merged as [begin, end]"""
        if begin > end:
            return [[]]
        if begin in mem:
            return mem[begin]
        answers = []
        for candidate_span_idx in pos2spans[begin]:
            candidate_start, candidate_end = spans[candidate_span_idx]
            for candidate_path in dfs(candidate_end + 1):
                answers.append([candidate_span_idx] + candidate_path)
        mem[begin] = answers
        return answers

    dfs(start)
    return mem[start]


class SubTreeStruct:
    """
    SubTree Struct
    Args:
        root: word_idx of root token
        score: score of this tree
        dep_arcs: arcs in this sub-tree (child_idx, parent_idx)
    """
    def __init__(self, root: int, score: float, dep_arcs: List[Tuple[int, int]]):
        self.root = root
        self.score = score
        self.dep_arcs = dep_arcs or []

    def __repr__(self):
        return json.dumps(self.__dict__, ensure_ascii=False, indent=4, sort_keys=True)


class DecodeStruct:
    """
    struct to store dp decoding info
    Args:
        words: origin words
        span_candidates: subtree candidates.
            maps word_idx to list of (span_start, span_end, span_score), sorted from large to small
        span2parent_arc_scores: subtree-parent scores
            maps each span-candidate(word_idx, span_start, span_end) to its parent_scores.
            arc_scores is a numpy array of shape [nwords+1], +1 for [root] score.
        dep_heads: dependency heads
        dep_tags: dependency tags
        todo:
            1.add tag decoding
            2.分数都用idx来存，去掉dict的结构
    """
    def __init__(
        self,
        words: List[str],
        span_candidates: Dict[int, List[Tuple[int, int, float]]],
        span2parent_arc_scores: Dict[Tuple[int, int, int], np.array] = None,
        span2parent_tags_scores: Dict[Tuple[int, int, int], np.array] = None,
        dep_heads: List[int] = None,
        dep_tags: List[str] = None,
        pos_tags: torch.LongTensor = None
    ):
        self.words = words
        self.span_candidates = span_candidates
        self.span2parent_arc_scores = span2parent_arc_scores or dict()
        self.dep_heads = dep_heads
        self.dep_tags = dep_tags
        self.pos_tags = pos_tags
        self.span2root_scores = {}
        for word_idx, subtrees in span_candidates.items():
            for start, end, score in subtrees:
                self.span2root_scores[(word_idx, start, end)] = score

        self.span_lst = self.parent_arc_score_lst = self.span_root_score_lst = None

    def __repr__(self):
        return json.dumps({
            "words": self.words,
            "span_candidates": self.span_candidates

        })

    def get_spans_infos(self, topk=5):
        """
        get span infos that used for deocoding
        Args:
            topk: to accelerate decoding, we only consider topk arcs from every child to its potential parent.
        """
        self.span_lst = list(self.span2parent_arc_scores.keys())
        self.parent_arc_score_lst: List[np.array] = [self.span2parent_arc_scores[s] for s in self.span_lst]
        self.span_root_score_lst: List[float] = [self.span2root_scores[s] for s in self.span_lst]
        # spans that offsets + 1 (for root)
        self.span_lst = [(s[0]+1, s[1]+1, s[2]+1) for s in self.span_lst]

        # add root
        self.words = ["[root]"] + self.words
        # filter span that start > end; out of origin sentence length; or have too low scores
        valid_span_lst = [(0, 0, len(self.words)-1)]
        valid_span_root_scores = [0.0]
        valid_span_arc_lst = [np.zeros_like(self.parent_arc_score_lst[0])]
        for span, root_score, arc_scores in zip(self.span_lst, self.span_root_score_lst, self.parent_arc_score_lst):
            word_idx, start, end = span
            if start > end:
                continue
            if end >= len(self.words):
                continue
            if root_score < -100:
                continue
            valid_span_lst.append(span)
            valid_span_root_scores.append(root_score)
            valid_span_arc_lst.append(arc_scores)

        self.span_lst = valid_span_lst
        self.span_root_score_lst = valid_span_root_scores
        self.parent_arc_score_lst = valid_span_arc_lst
        self.span_parent_arc_topk = []
        for parent_arc_score in self.parent_arc_score_lst:
            topk_parent_idx = np.argpartition(-parent_arc_score, kth=min(topk, len(parent_arc_score)))[: topk]
            self.span_parent_arc_topk.append(topk_parent_idx)
        # print(self.words)
        # print(self.span_lst)
        # print(self.parent_root_score_lst)

    @property
    def nwords(self):
        return len(self.words)

    def top_spans(self, k) -> List[Tuple[int, int]]:
        """get k'th span_start, span_end for every position"""
        output = []
        for word_idx in range(self.nwords):
            output.append((
                self.span_candidates[word_idx][k][0],
                self.span_candidates[word_idx][k][1],
            ))
        return output

    def decode(self) -> Union[None, SubTreeStruct]:
        """decode highest score tree from self"""
        if self.span_lst is None:
            self.get_spans_infos()

        all_trees: List[Tuple[int, int, int]] = self.span_lst
        all_spans = [(s[1], s[2]) for s in all_trees]
        subtree_splits: List[List[List[int]]] = []

        for word_idx, span_start, span_end in all_trees:
            # for [root], it can only has one child that range from 1 to span_end
            if word_idx == 0:
                splits = []
                for span_idx, (candidate_start, candidate_end) in enumerate(all_spans):
                    if candidate_start == 1 and candidate_end == span_end:
                        splits.append([span_idx])
            # for other subtrees, one can be factorized to any number of subtrees
            # as long as every token in its range is covered by all subtrees.
            else:
                # we only consider factorization for topk parent.
                valid_subtree_idxs = [s_idx for s_idx in range(len(all_spans))
                                      if word_idx in self.span_parent_arc_topk[s_idx]]
                left_candidates = build_possible_splits(
                    span_start,
                    word_idx-1,
                    [s for s_idx, s in enumerate(all_spans) if s_idx in valid_subtree_idxs]
                )
                right_candidates = build_possible_splits(
                    word_idx+1,
                    span_end,
                    [s for s_idx, s in enumerate(all_spans) if s_idx in valid_subtree_idxs]
                )
                splits = []
                for l in left_candidates:
                    for r in right_candidates:
                        splits.append([valid_subtree_idxs[x] for x in l+r])
            subtree_splits.append(splits)

        # find all subtree children, because we need to compute all children scores
        # of one tree before we compute its final score
        children_idxs = []
        for splits in subtree_splits:
            s = set()
            for x in splits:
                s.update(x)
            children_idxs.append(s)

        finished = [False] * len(all_trees)
        # We compute tree score from bottom-up, until the root score is computed
        # We firstly finish all spans that start==end, because they are leaves.
        # max_dep contains root_idx, subtree-score and dependency-arcs
        max_dep: List[SubTreeStruct] = [None] * len(all_trees)
        for span_idx, (word_idx, start, end) in enumerate(all_trees):
            if start != end:
                continue
            max_dep[span_idx] = SubTreeStruct(word_idx, self.span_root_score_lst[span_idx], [])
            finished[span_idx] = True

        finished_count = sum(finished)
        while not finished[0]:
            for span_idx, ((word_idx, start, end), finish) in enumerate(zip(all_trees, finished)):
                if finish:
                    continue
                if not all(finished[c] for c in children_idxs[span_idx]):
                    continue

                max_score = -math.inf
                max_arcs = None

                for split_group_idx, split_group in enumerate(subtree_splits[span_idx]):
                    tree_score = self.span_root_score_lst[span_idx]
                    arcs = []
                    for subtree_idx in split_group:
                        tree_score += max_dep[subtree_idx].score
                        tree_score += self.parent_arc_score_lst[subtree_idx][word_idx]
                        arcs += max_dep[subtree_idx].dep_arcs
                        arcs += [(max_dep[subtree_idx].root, word_idx)]
                    if tree_score > max_score:
                        max_score = tree_score
                        max_arcs = arcs

                max_dep[span_idx] = SubTreeStruct(word_idx, max_score, max_arcs)
                finished[span_idx] = True

            new_finished_count = sum(finished)
            if new_finished_count == finished_count:
                raise ValueError(f"can't find valid tree factorization using given spans: {self.span_lst}")
            finished_count = new_finished_count
        return max_dep[0]
