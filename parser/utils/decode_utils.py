# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/1/9 19:55
@desc: 

"""

from typing import List, Tuple, Dict
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