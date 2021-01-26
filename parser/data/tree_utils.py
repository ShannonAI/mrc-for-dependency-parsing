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
    build subtree-spans according to dependency heads
    Args:
        dp_heads: dependency heads. dp_heads[i] is the parent idx of words[i].
    Returns:
        subtree_spans: [num_words, 2], subtree start/end(including) offset of each word
    Notes:
        We ignore root here, so 0 is the actual first word of one sentence!
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


def is_cyclic(dp_heads: List[int]) -> bool:
    """return True if dp_heads is cyclic, else False"""
    dp_heads = [0] + dp_heads
    detected = [False] * len(dp_heads)
    detected[0] = True
    for word_idx, parent_idx in enumerate(dp_heads[1:]):
        if detected[word_idx]:
            continue
        ancestors = set()
        node = word_idx
        while not detected[node]:
            ancestors.add(node)
            node = dp_heads[node]
            if node in ancestors:
                return True
        for node in ancestors:
            detected[node] = True
    return False


def is_projective(dp_heads: List[int]) -> bool:
    """
    Determine whether dp_heads form a projective tree.
    A projective tree is a tree that for each word, its descendants and itself
    form a contiguous substring of the sentence.
    Examples:
        >>> is_projective([0, 1])
        True
        >>> is_projective([0,4,1,3])
        False
    """
    return all(are_projective(dp_heads))


def are_projective(dp_heads: List[int]) -> List[bool]:
    """
    Determine whether dp_heads form a projective tree at each token.
    """
    dp_heads = [0] + dp_heads  # add root
    n = len(dp_heads)
    num_children = [0] * n
    for child, parent in enumerate(dp_heads):
        while parent > 0:
            num_children[parent] += 1
            parent = dp_heads[parent]
    num_children = num_children[1:]
    subtree_spans = build_subtree_spans(dp_heads[1:])
    output = []
    for num, (start, end) in zip(num_children, subtree_spans):
        if end-start != num:
            output.append(False)
        else:
            output.append(True)
    return output


if __name__ == '__main__':
    import codecs
    from conllu import parse_incr
    from tqdm import tqdm

    def compute_projective_percentage(path):
        count = 0
        proj = 0
        with codecs.open(path, "r", "utf-8") as conllu_file:
            for ann_idx, annotation in tqdm(enumerate(parse_incr(conllu_file))):
                annotation = [x for x in annotation if isinstance(x["id"], int)]
                dp_heads = [x["head"] for x in annotation]
                if is_cyclic(dp_heads):
                    continue
                are_projectives = are_projective(dp_heads)
                proj += sum(are_projectives)
                count += len(are_projectives)
        print(f"projective percentage of {path} is {proj/count}")

    for file in [
        "/userhome/yuxian/data/parser/ctb5_parser/test.conllx",
        "/userhome/yuxian/data/parser/ctb5.1_parser/test.conllx",
        "/userhome/yuxian/data/parser/ptb3_parser/test.conllx",
        "/userhome/yuxian/data/parser/ud-treebanks/ud-treebanks-v2.2/UD_Bulgarian-BTB/bg_btb-ud-test.conllu",
        "/userhome/yuxian/data/parser/ud-treebanks/ud-treebanks-v2.2/UD_Catalan-AnCora/ca_ancora-ud-test.conllu",
        "/userhome/yuxian/data/parser/ud-treebanks/ud-treebanks-v2.2/merged_dataset/czech/test.conllu",
        "/userhome/yuxian/data/parser/ud-treebanks/ud-treebanks-v2.2/UD_German-GSD/de_gsd-ud-test.conllu",
        "/userhome/yuxian/data/parser/ud-treebanks/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu",
        "/userhome/yuxian/data/parser/ud-treebanks/ud-treebanks-v2.2/merged_dataset/spanish/test.conllu",
        "/userhome/yuxian/data/parser/ud-treebanks/ud-treebanks-v2.2/UD_French-GSD/fr_gsd-ud-test.conllu",
        "/userhome/yuxian/data/parser/ud-treebanks/ud-treebanks-v2.2/UD_Italian-ISDT/it_isdt-ud-test.conllu",
        "/userhome/yuxian/data/parser/ud-treebanks/ud-treebanks-v2.2/merged_dataset/dutch/test.conllu",
        "/userhome/yuxian/data/parser/ud-treebanks/ud-treebanks-v2.2/merged_dataset/norwegian/test.conllu",
        "/userhome/yuxian/data/parser/ud-treebanks/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-test.conllu",
        "/userhome/yuxian/data/parser/ud-treebanks/ud-treebanks-v2.2/UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu",
    ]:
        compute_projective_percentage(file)
