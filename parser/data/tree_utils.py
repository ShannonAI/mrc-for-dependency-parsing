# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/1/7 10:39
@desc: 

"""
from collections import defaultdict
from typing import List


def build_subtree_spans(dp_heads: List[int]) -> List[List[int]]:
    """
    build subtree-spans according to dpendency heads
    Args:
        dp_heads: dependency heads. dp_heads[i] is the parent idx of words[i].
    Returns:
        subtree_spans: [num_words, 2], subtree start/end(including) offset of each word
    """
    length = len(dp_heads)
    subtree_spans = [[idx, idx] for idx in range(length)]
    finished = [False] * length  # finished computing range of words[idx]
    parent2childs = defaultdict(list)
    for word_idx, parent_idx in enumerate(dp_heads):
        # ignore root
        if parent_idx == 0:
            continue
        parent_idx -= 1
        parent2childs[parent_idx].append(word_idx)
        origin_start, origin_end = subtree_spans[parent_idx]
        new_start = min(origin_start, word_idx)
        new_end = max(origin_end, word_idx)
        subtree_spans[parent_idx] = [new_start, new_end]

    for word_idx, (start, end) in enumerate(subtree_spans):
        if start == end:  # leaf node
            finished[word_idx] = True

    while not all(finished):
        # update tree node whose all children has finished
        for parent_idx, finish in enumerate(finished):
            if finish:
                continue
            if not all(finished[x] for x in parent2childs[parent_idx]):
                continue
            start, end = subtree_spans[parent_idx]
            for child in parent2childs[parent_idx]:
                child_start, child_end = subtree_spans[child]
                start = min(start, child_start)
                end = max(end, child_end)
            subtree_spans[parent_idx] = [start, end]
            finished[parent_idx] = True
    return subtree_spans
