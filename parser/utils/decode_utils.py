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
import sys
from allennlp.nn.chu_liu_edmonds import decode_mst
import warnings

EPS = sys.float_info.min


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


def find_max_splits(start: int, end: int, spans: List[Tuple[int, int]], scores: List[float]) -> Tuple[List[int], float]:
    """
    find the factorization of [start, end] to some sub_spans s_1, ...s_n in spans that have the highest sum scores
    $$S(start, end, s_1, ...,s_n)=\sum_{i=1}^{n} scores_{s_i}$$
    if does not exists, return empty list and -math.inf
    Args:
        start: start of parent span
        end: end of parent span, including iteself
        spans: candidate sub_spans for factorization
        scores: score of each sub_spans

    Returns:
        best_splits: list of sub_span indexes that have highest sum scores
        max_score: float, score of this factorization

    Examples:
        >>> spans = [(0, 4), (0, 2), (0, 3), (4, 5), (3, 5), (5, 5), (3, 4)]
        >>> scores = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        >>> find_max_splits(0, 5, spans, scores)
        ([1, 6, 5], 3.0)
    """
    assert len(scores) == len(spans)

    # end2idxs[i] contains all valid span idxs that endswith i
    end2idxs: Dict[int, List[int]] = defaultdict(list)
    for s_idx, ((s, e), score) in enumerate(zip(spans, scores)):
        if start <= s <= e <= end:
            end2idxs[e].append(s_idx)

    # dp_splits[idx] stores best splits among all possible splits of [start: start+idx], idx not included
    dp_splits = [[] for _ in range((end - start + 2))]
    # dp_scores[idx] stores best sum score among all possible splits of [start: start+idx], idx not included
    dp_scores = [-math.inf for _ in range((end - start + 2))]
    dp_scores[0] = 0.0

    for e in range(start, end+1):
        for last_span_idx in end2idxs[e]:
            last_span_start = spans[last_span_idx][0]
            score = dp_scores[last_span_start-start] + scores[last_span_idx]
            if score > dp_scores[e-start+1]:
                dp_scores[e-start+1] = score
                dp_splits[e-start+1] = dp_splits[last_span_start-start] + [last_span_idx]

    return dp_splits[-1], dp_scores[-1]


class SubTreeStruct:
    """
    SubTree Struct
    Args:
        root: word_idx of root token
        score: score of this tree
        dep_arcs: arcs in this sub-tree (child_idx, parent_idx, tag_idx)
    """
    def __init__(self, root: int, score: float, dep_arcs: List[Tuple[int, int, int]]):
        self.root = root
        self.score = score
        self.dep_arcs = dep_arcs or []

    def __repr__(self):
        return json.dumps(self.__dict__, ensure_ascii=False, indent=4, sort_keys=True)


class DecodeStruct:
    """
    struct to store dp decoding info
    todo refactor and clean code
    Args:
        words: origin words
        span_candidates: subtree candidates.
            maps word_idx to list of (span_start, span_end, span_score), sorted from large to small
        span2parent_arc_scores: subtree-parent scores
            maps each span-candidate(word_idx, span_start, span_end) to its parent probs,
            which is a numpy array of shape [nwords+1], +1 for [root] score.
        span2parent_start_scores: subtree-parent scores
            maps each span-candidate(word_idx, span_start, span_end) to its parent-start probs,
            which is a numpy array of shape [nwords+1], +1 for [root] score.
        span2parent_end_scores: subtree-parent scores
            maps each span-candidate(word_idx, span_start, span_end) to its parent-end probs,
            which is a numpy array of shape [nwords+1], +1 for [root] score.
        span2child_scores: subtree-child scores
            maps each span-candidate(word_idx, span_start, span_end) to its child probs,
            which is a numpy array of shape [nwords+1], +1 for [root] score.
        span2child_start_scores: subtree-child scores
            maps each span-candidate(word_idx, span_start, span_end) to its child start probs,
            which is a numpy array of shape [nwords+1], +1 for [root] score.
        span2child_end_scores: subtree-child scores
            maps each span-candidate(word_idx, span_start, span_end) to its child end probs,
            which is a numpy array of shape [nwords+1], +1 for [root] score.
        span2parent_tags_idxs: subtree-parent tags scores
            maps each span-candidate(word_idx, span_start, span_end) to its parent tag-idx with highest scores,
             a numpy array of shape [nwords+1], +1 for [root] score.
        dep_heads: dependency heads
        dep_tags: dependency tags
        span_start_scores: span_start score by proposal model, shape [seq_len, seq_len]
        span_end_scores: span_end score by proposal model, shape [seq_len, seq_len]
    """
    def __init__(
        self,
        words: List[str],
        span_candidates: Dict[int, List[Tuple[int, int, float]]],
        span2parent_arc_scores: Dict[Tuple[int, int, int], np.array] = None,
        span2parent_start_scores: Dict[Tuple[int, int, int], np.array] = None,
        span2parent_end_scores: Dict[Tuple[int, int, int], np.array] = None,
        span2child_scores: Dict[Tuple[int, int, int], np.array] = None,
        span2child_start_scores: Dict[Tuple[int, int, int], np.array] = None,
        span2child_end_scores: Dict[Tuple[int, int, int], np.array] = None,
        span2parent_tags_idxs: Dict[Tuple[int, int, int], np.array] = None,
        dep_heads: List[int] = None,
        dep_tags: List[str] = None,
        pos_tags: torch.LongTensor = None,
        span_start_scores: np.array = None,
        span_end_scores: np.array = None
    ):
        self.words = words
        self.span_candidates = span_candidates
        self.span2parent_arc_scores = span2parent_arc_scores or dict()
        self.span2parent_start_scores = span2parent_start_scores or dict()
        self.span2parent_end_scores = span2parent_end_scores or dict()
        self.span2child_scores = span2child_scores or dict()
        self.span2child_start_scores = span2child_start_scores or dict()
        self.span2child_end_scores = span2child_end_scores or dict()
        self.span2parent_tags_idxs = span2parent_tags_idxs or dict()
        self.dep_heads = dep_heads
        self.dep_tags = dep_tags
        self.pos_tags = pos_tags
        self.span2root_scores = {}
        for word_idx, subtrees in span_candidates.items():
            for start, end, score in subtrees:
                self.span2root_scores[(word_idx, start, end)] = score

        # values to generate during decode
        self.span_lst = self.parent_arc_score_lst = self.span_root_score_lst = self.parent_tags_idxs_lst = None
        self.parent_start_score_lst = self.parent_end_score_lst = None
        self.child_score_lst = self.child_start_score_lst = self.child_end_score_lst = None

        self.span_start_scores = span_start_scores
        self.span_end_scores = span_end_scores

    def __repr__(self):
        return json.dumps({
            "words": self.words,
            "span_candidates": self.span_candidates

        })

    def get_spans_infos(self):
        """
        get span infos that used for decoding
        """
        self.span_lst = list(self.span2parent_arc_scores.keys())
        self.parent_arc_score_lst: List[np.array] = [self.span2parent_arc_scores[s] for s in self.span_lst]
        self.parent_start_score_lst: List[np.array] = [self.span2parent_start_scores[s] for s in self.span_lst] \
            if self.span2parent_start_scores else [np.ones_like(x) for x in self.parent_arc_score_lst]
        self.parent_end_score_lst: List[np.array] = [self.span2parent_end_scores[s] for s in self.span_lst] \
            if self.span2parent_start_scores else [np.ones_like(x) for x in self.parent_arc_score_lst]
        self.child_score_lst: List[np.array] = [self.span2child_scores[s] for s in self.span_lst] \
            if self.span2child_scores else [np.ones_like(x) for x in self.parent_arc_score_lst]
        self.child_start_score_lst: List[np.array] = [self.span2child_start_scores[s] for s in self.span_lst] \
            if self.span2child_start_scores else [np.ones_like(x) for x in self.parent_arc_score_lst]
        self.child_end_score_lst: List[np.array] = [self.span2child_end_scores[s] for s in self.span_lst] \
            if self.span2child_end_scores else [np.ones_like(x) for x in self.parent_arc_score_lst]
        self.parent_tags_idxs_lst: List[np.array] = [self.span2parent_tags_idxs[s] for s in self.span_lst]
        self.span_root_score_lst: List[float] = [self.span2root_scores[s] for s in self.span_lst]
        # spans that offsets + 1 (for root)
        self.span_lst = [(s[0]+1, s[1]+1, s[2]+1) for s in self.span_lst]

        # add root
        self.words = ["[root]"] + self.words

        # root scores
        valid_span_lst = [(0, 0, len(self.words)-1)]
        valid_span_root_scores = [0.0]
        valid_span_arc_lst = [np.ones_like(self.parent_arc_score_lst[0])]
        valid_span_start_lst = [np.ones_like(self.parent_start_score_lst[0])]
        valid_span_end_lst = [np.ones_like(self.parent_end_score_lst[0])]
        valid_span_tags_lst = [np.ones_like(self.parent_tags_idxs_lst[0])]
        valid_span_child_lst = [np.ones_like(self.child_score_lst[0])]
        valid_span_child_start_lst = [np.ones_like(self.child_start_score_lst[0])]
        valid_span_child_end_lst = [np.ones_like(self.child_end_score_lst[0])]
        # filter invalid spans: start > end or root not in span or out of origin sentence length
        for (span, root_score, arc_scores, start_scores, end_scores,
             child_scores, child_start_scores, child_end_scores, tags) in zip(
            self.span_lst,
            self.span_root_score_lst,
            self.parent_arc_score_lst,
            self.parent_start_score_lst,
            self.parent_end_score_lst,
            self.child_score_lst,
            self.child_start_score_lst,
            self.child_end_score_lst,
            self.parent_tags_idxs_lst
        ):
            word_idx, start, end = span
            if start > end:
                continue
            if end >= len(self.words):
                continue
            if end < word_idx or start > word_idx:
                continue

            valid_span_lst.append(span)
            valid_span_root_scores.append(root_score)
            valid_span_arc_lst.append(arc_scores)
            valid_span_start_lst.append(start_scores)
            valid_span_end_lst.append(end_scores)
            valid_span_child_lst.append(child_scores)
            valid_span_child_start_lst.append(child_start_scores)
            valid_span_child_end_lst.append(child_end_scores)
            valid_span_tags_lst.append(tags)

        self.span_lst = valid_span_lst
        self.span_root_score_lst = valid_span_root_scores
        self.parent_arc_score_lst = valid_span_arc_lst
        self.parent_start_score_lst = valid_span_start_lst
        self.parent_end_score_lst = valid_span_end_lst
        self.child_score_lst = valid_span_child_lst
        self.child_start_score_lst = valid_span_child_start_lst
        self.child_end_score_lst = valid_span_child_end_lst
        self.parent_tags_idxs_lst = valid_span_tags_lst

    @property
    def nwords(self):
        return len(self.words)

    def top_spans(self, k) -> List[Tuple[int, int]]:
        """get k'th span_start, span_end for every position"""
        output = []
        for word_idx in range(self.nwords):
            num_candidates = len(self.span_candidates[word_idx])
            c = k
            if c >= num_candidates:
                warnings.warn(f"argument {k} is larger than candidate num {num_candidates}")
                c = num_candidates-1
            output.append((
                self.span_candidates[word_idx][c][0],
                self.span_candidates[word_idx][c][1],
            ))
        return output

    def greedy_decode(self) -> Tuple[List[int], List[int]]:
        """do greedy decode: for each token, choose the top1 substree span and its parent for final prediction"""
        if self.span_lst is None:
            self.get_spans_infos()

        dep_heads = [0] * (len(self.words) - 1)
        dep_tags = [0] * (len(self.words) - 1)

        for word_idx in range(len(dep_heads)):
            best_start, best_end, score = self.span_candidates[word_idx][0]
            subtree = (word_idx, best_start, best_end)
            best_parent = self.span2parent_arc_scores[subtree].argmax().item()
            best_parent_tag = self.span2parent_tags_idxs[subtree][best_parent].item()
            dep_heads[word_idx] = best_parent
            dep_tags[word_idx] = best_parent_tag

        return dep_heads, dep_tags

    def mst_decode(self, arc_alpha: float = 1.0) -> Tuple[List[int], List[int]]:
        """
        do mst decode:
        S[i][j] = max_{T1, T2}{Score_span(T1) + Score_span(T2) + arc_alpha * Score_link(T1, T2), T1.r==i, T2.r==j}

        """
        if self.span_lst is None:
            self.get_spans_infos()

        root2span_idxs = defaultdict(list)
        for idx, (root, start, end) in enumerate(self.span_lst):
            root2span_idxs[root].append(idx)

        seq_len = len(self.words)

        # scores[i,j] = "Score that i is the head of j"
        # tag_ids[i, j] = "best label of arc i->j"
        scores = torch.zeros([seq_len, seq_len])
        tag_ids = torch.zeros([seq_len, seq_len], dtype=torch.long)
        for i in range(seq_len):
            for j in range(1, seq_len):  # root cannot be child
                # compute energy[i][j] and tag_ids[i][j]
                max_score = -math.inf
                max_tag_id = None
                for parent_idx in root2span_idxs[i]:
                    parent_root, parent_start, parent_end = self.span_lst[parent_idx]
                    for child_idx in root2span_idxs[j]:
                        child_root, child_start, child_end = self.span_lst[child_idx]
                        # for computational stability, we use log prob for comparision
                        s = (
                            self.span_root_score_lst[parent_idx]
                            + self.span_root_score_lst[child_idx]
                            + arc_alpha * (
                                math.log(self.parent_arc_score_lst[child_idx][parent_root]+EPS)
                                + math.log(self.parent_start_score_lst[child_idx][parent_start]+EPS)
                                + math.log(self.parent_end_score_lst[child_idx][parent_end]+EPS)
                                + math.log(self.child_score_lst[parent_idx][child_root]+EPS)
                                + math.log(self.child_start_score_lst[parent_idx][child_start]+EPS)
                                + math.log(self.child_end_score_lst[parent_idx][child_end]+EPS)
                                )
                             )
                        t = self.parent_tags_idxs_lst[child_idx][parent_root]
                        if s > max_score:
                            max_score = s
                            max_tag_id = t

                if max_tag_id is None:
                    warnings.warn(f"no valid arc between {i} and {j} for {self.words} "
                                  f"with spans: {root2span_idxs[i] + root2span_idxs[j]}")
                    max_score = 0.0
                    max_tag_id = 0

                scores[i][j] = math.exp(max_score)
                tag_ids[i][j] = max_tag_id

        # Decode the heads. Because we modify the scores to prevent
        # adding in word -> ROOT edges, we need to find the labels ourselves.
        # print(scores)
        instance_heads, _ = decode_mst(scores.numpy(), seq_len, has_labels=False)

        # Find the labels which correspond to the edges in the max spanning tree.
        instance_head_tags = []
        for child, parent in enumerate(instance_heads):
            instance_head_tags.append(tag_ids[parent, child].item())
        # We don't care what the head or tag is for the root token, but by default it's
        # not necessarily the same in the batched vs unbatched case, which is annoying.
        # Here we'll just set them to zero.
        instance_heads[0] = 0
        instance_head_tags[0] = 0
        return instance_heads.tolist()[1:], instance_head_tags[1:]

    def dp_decode(self, arc_alpha=1.0) -> SubTreeStruct:
        """
        decode highest score tree from self using bottom-up dynamic-programming
        todo write dp function here

        if no valid dependency, score of SubTreeStruct is -math.inf
        Args:
            arc_alpha: float, controls weight of parent-arc score
        """
        if self.span_lst is None:
            self.get_spans_infos()

        all_trees: List[Tuple[int, int, int]] = self.span_lst
        all_spans = [(s[1], s[2]) for s in all_trees]

        # find all candidate subtree children for each span,
        # because before we score one tree,
        # we need to score all of its children.
        left_children_idxs = []
        right_children_idxs = []
        for parent_idx, start, end in all_trees:
            left_children = []
            right_children = []
            for s_idx, (s, e) in enumerate(all_spans):
                if start <= s and e <= parent_idx-1:
                    left_children.append(s_idx)
                # root can only have one right children that startswith 1 and endswith max_length
                elif s >= parent_idx+1 and e <= end and (parent_idx > 0 or (s == 1 and e == end)):
                    right_children.append(s_idx)
            left_children_idxs.append(left_children)
            right_children_idxs.append(right_children)

        finished = [False] * len(all_trees)
        # We compute tree score from bottom-up, until the root score is computed
        # We firstly finish all spans that start==end, because they are leaves.
        max_dep: List[SubTreeStruct] = [None] * len(all_trees)
        for span_idx, (word_idx, start, end) in enumerate(all_trees):
            if start != end:
                continue
            max_dep[span_idx] = SubTreeStruct(word_idx, self.span_root_score_lst[span_idx], [])
            finished[span_idx] = True
        # bottom-up dynamic-programming
        while not finished[0]:
            # print([subtree for subtree, finish in zip(all_trees, finished) if not finish])
            for span_idx, ((word_idx, start, end), finish) in enumerate(zip(all_trees, finished)):
                if finish:
                    continue
                left_children_ids = left_children_idxs[span_idx]
                if not all(finished[c] for c in left_children_ids):
                    continue
                right_children_ids = right_children_idxs[span_idx]
                if not all(finished[c] for c in right_children_ids):
                    continue

                root_score = self.span_root_score_lst[span_idx]

                if word_idx > start:  # it means this span have left children
                    children_spans = [all_spans[i] for i in left_children_ids]
                    children_scores = []
                    for subtree_idx in left_children_ids:
                        subtree_root, subtree_start, subtree_end = all_trees[subtree_idx]
                        children_scores.append(max_dep[subtree_idx].score +
                                               arc_alpha * (
                                                   math.log(self.parent_arc_score_lst[subtree_idx][word_idx]+EPS)
                                                   + math.log(self.parent_start_score_lst[subtree_idx][start]+EPS)
                                                   + math.log(self.parent_end_score_lst[subtree_idx][end]+EPS)
                                                   + math.log(self.child_score_lst[span_idx][subtree_root]+EPS)
                                                   + math.log(self.child_start_score_lst[span_idx][subtree_start]+EPS)
                                                   + math.log(self.child_end_score_lst[span_idx][subtree_end]+EPS)
                                                ))
                    best_splits, best_score = find_max_splits(start=start, end=word_idx-1,
                                                              spans=children_spans,
                                                              scores=children_scores
                                                              )
                    best_arcs = []
                    for subtree_idx in [left_children_ids[i] for i in best_splits]:
                        best_arcs += max_dep[subtree_idx].dep_arcs
                        best_arcs.append((max_dep[subtree_idx].root, word_idx, self.parent_tags_idxs_lst[subtree_idx][word_idx]))
                    left_score = best_score
                    left_arcs = best_arcs
                else:
                    left_score = 0.0
                    left_arcs = []

                if word_idx < end:  # it means this span have right children
                    children_spans = [all_spans[i] for i in right_children_ids]
                    children_scores = []
                    for subtree_idx in right_children_ids:
                        subtree_root, subtree_start, subtree_end = all_trees[subtree_idx]
                        children_scores.append(max_dep[subtree_idx].score +
                                               arc_alpha * (
                                                   math.log(self.parent_arc_score_lst[subtree_idx][word_idx]+EPS)
                                                   + math.log(self.parent_start_score_lst[subtree_idx][start]+EPS)
                                                   + math.log(self.parent_end_score_lst[subtree_idx][end]+EPS)
                                                   + math.log(self.child_score_lst[span_idx][subtree_root]+EPS)
                                                   + math.log(self.child_start_score_lst[span_idx][subtree_start]+EPS)
                                                   + math.log(self.child_end_score_lst[span_idx][subtree_end] + EPS)
                                               ))
                    best_splits, best_score = find_max_splits(start=word_idx+1, end=end,
                                                              spans=children_spans,
                                                              scores=children_scores
                                                              )
                    best_arcs = []
                    for subtree_idx in [right_children_ids[i] for i in best_splits]:
                        best_arcs += max_dep[subtree_idx].dep_arcs
                        best_arcs.append((max_dep[subtree_idx].root, word_idx, self.parent_tags_idxs_lst[subtree_idx][word_idx]))
                    right_score = best_score
                    right_arcs = best_arcs
                else:
                    right_score = 0.0
                    right_arcs = []

                max_dep[span_idx] = SubTreeStruct(word_idx, root_score+left_score+right_score, left_arcs+right_arcs)

                finished[span_idx] = True

        return max_dep[0]
