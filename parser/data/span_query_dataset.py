# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: dependency_t2t_reader
@time: 2020/12/17 10:47
@desc: Dataset that use subtree to query its parent

"""


from copy import deepcopy
from itertools import chain
from typing import List, Dict

import torch

from parser.data.base_bert_dataset import BaseDataset
from parser.data.tree_utils import build_subtree_spans
from parser.utils.logger import get_logger

logger = get_logger(__name__)


class SpanQueryDataset(BaseDataset):
    """
    Depency Dataset used for Span-to-Token MRC
    Build dataset that use subtree-span to query its parent

    Args:
        file_path: conllu/conllx format dataset
        bert: bert directory path
        use_language_specific_pos : `bool`, optional (default = `False`)
            Whether to use UD POS tags, or to use the language specific POS tags
            provided in the conllu format.
        pos_tags: if specified, directly use it instead of counting POS tags from file
        dep_tags: if specified, directly use it instead of counting dependency tags from file
    """

    # we use [unused0] ~ [unused3] in bert vocab to represent bracket around query token and query span
    SPAN_START = 1
    SUBTREE_ROOT_START = 2
    SUBTREE_ROOT_END = 3
    SPAN_END = 4

    SEP_POS = "sep_pos"
    SEP = "[SEP]"

    group_file_suffix = ".mrc"

    def __init__(
        self,
        file_path: str,
        bert: str,
        use_language_specific_pos: bool = False,
        pos_tags: List[str] = None,
        dep_tags: List[str] = None,
    ) -> None:
        super().__init__(file_path, bert, use_language_specific_pos, pos_tags, dep_tags)

        self.subtree_spans = [build_subtree_spans(d[3]) for d in self.data]

        self.offsets = self.build_offsets()
        logger.info(f"build {len(self.offsets)} mrc-samples from {file_path}")

        self.pos_tags = pos_tags or self.build_label_vocab(chain(*[d[1] for d in self.data], [self.SEP_POS]))
        self.pos_tag_2idx = {l: idx for idx, l in enumerate(self.pos_tags)}
        logger.info(f"pos tags: {self.pos_tag_2idx}")
        logger.info(f"dep tags: {self.dep_tag_2idx}")

    def build_offsets(self):
        """offsets[i] = (sent_idx, word_idx) of ith mrc-samples."""
        offsets = []
        # We want to query each subtree(represented by each token as root node) for each sentence
        for ann_idx, (words, _, _, _) in enumerate(self.data):
            for word_idx in range(len(words)):
                offsets.append((ann_idx, word_idx))
        return offsets

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        """
        Returns:
            token_ids: [num_word_pieces]
            type_ids: [num_word_pieces]
            offsets: [num_words, 2]
            wordpiece_mask: [num_word_pieces]
            pos_tags: [num_words]
            word_mask: [num_words]
            mrc_mask: [num_words]
            meta_data: dict of meta_fields
            parent_idxs: [1]
            parent_tags: [1]
        """
        ann_idx, word_idx = self.offsets[idx]
        words, pos_tags, dp_tags, dp_heads = self.data[ann_idx]
        subtree_spans = self.subtree_spans[ann_idx]
        span_start, span_end = subtree_spans[word_idx]
        # mrc sample consists of query and context.
        mrc_length = len(words) * 2 + 5
        # query is a copy of origin sentence, with special token surrounding the query-subtree span and root.
        query_length = len(words) + 4

        type_ids = [0] * (len(words) + 5) + [1] * len(words)  # bert sentence-pair token-ids

        fields = {"type_ids": torch.LongTensor(type_ids)}

        query_tokens = deepcopy(words) + [self.SEP]
        query_tokens.insert(span_start, self.SEP)
        query_tokens.insert(word_idx + 1, self.SEP)
        query_tokens.insert(word_idx + 3, self.SEP)
        query_tokens.insert(span_end + 4, self.SEP)
        mrc_tokens = query_tokens + words

        bert_mismatch_fields = self.get_mismatch_token_idx(mrc_tokens)
        try:
            self.replace_special_token(bert_mismatch_fields, [span_start], self.SPAN_START)
            self.replace_special_token(bert_mismatch_fields, [word_idx+1], self.SUBTREE_ROOT_START)
            self.replace_special_token(bert_mismatch_fields, [word_idx+3], self.SUBTREE_ROOT_END)
            self.replace_special_token(bert_mismatch_fields, [span_end+4], self.SPAN_END)
        except Exception as e:
            logger.error("replace error, be careful that this should not happen unless you're "
                         "doing batch prediction at evaluation", exc_info=True)

        fields.update(bert_mismatch_fields)

        query_pos_tags = pos_tags.copy() + [self.SEP_POS]
        for p in [word_idx, word_idx+1, word_idx+3, word_idx+4]:
            query_pos_tags.insert(p, self.SEP_POS)

        # num_words * 2 + 3
        mrc_pos_tag_idxs = [self.pos_tag_2idx[p] for p in query_pos_tags + pos_tags]
        # valid parent idxs
        mrc_mask = [False] * mrc_length
        # only context(and SEP, which represents root) can be parent/child
        mrc_mask[query_length:] = [True] * (len(words) + 1)
        mrc_mask[query_length + word_idx + 1] = False  # origin word cannot be parent/child of itself

        parent_tag = dp_tags[word_idx]
        parent_idx = dp_heads[word_idx]
        fields["parent_idxs"] = torch.LongTensor([parent_idx + query_length])
        fields["parent_tags"] = torch.LongTensor([self.dep_tag_2idx[parent_tag]])

        fields["pos_tags"] = torch.LongTensor(mrc_pos_tag_idxs)
        fields["mrc_mask"] = torch.BoolTensor(mrc_mask)
        fields["meta_data"] = {
            "words": words,
            "ann_idx": ann_idx,
            "word_idx": word_idx,
            "subtree_span": [span_start, span_end]
        }

        return fields

    def _get_item_lengths(self) -> List[int]:
        """get seqence-length of every item"""
        item_lengths = []
        for words, _, _, _ in self.data:
            item_lengths.extend([len(words)] * len(words))
        return item_lengths


def collate_s2t_data(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of dependency samples, including fields:
            token_ids: [num_word_pieces]
            type_ids: [num_word_pieces]
            offsets: [num_words, 2]
            wordpiece_mask: [num_word_pieces]
            pos_tags: [num_words]
            word_mask: [num_words]
            mrc_mask: [num_words]
            meta_data: dict of meta_fields
            parent_idxs: [1]
            parent_tags: [1]
    Returns:
        output: dict of batched fields
    """

    batch_size = len(batch)
    max_pieces = max(x["token_ids"].size(0) for x in batch)
    max_words = max(x["pos_tags"].size(0) for x in batch)

    output = {
        "parent_idxs": torch.cat([x["parent_idxs"] for x in batch]),
        "parent_tags": torch.cat([x["parent_tags"] for x in batch])
              }

    for field in ["token_ids", "type_ids", "wordpiece_mask"]:
        pad_output = torch.full([batch_size, max_pieces], 0, dtype=batch[0][field].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field]
            pad_output[sample_idx][: data.shape[0]] = data
        output[field] = pad_output

    for field in ["pos_tags", "word_mask", "mrc_mask"]:
        pad_output = torch.full([batch_size, max_words], 0, dtype=batch[0][field].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field]
            pad_output[sample_idx][: data.shape[0]] = data
        output[field] = pad_output

    for field in ["offsets", ]:
        fill_value = 0
        pad_output = torch.full([batch_size, max_words, 2], fill_value, dtype=torch.long)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field]
            pad_output[sample_idx][: data.shape[0]] = data
        output[field] = pad_output

    meta_fields = batch[0]["meta_data"].keys()
    output["meta_data"] = {field: [x["meta_data"][field] for x in batch] for field in meta_fields}

    return output


if __name__ == '__main__':
    from tqdm import tqdm
    from parser.data.samplers.grouped_sampler import GroupedSampler
    from torch.utils.data import SequentialSampler

    dataset = SpanQueryDataset(
        # file_path="/data/nfsdata2/nlp_application/datasets/treebank/LDC99T42/ptb3_parser/dev.conllx",
        file_path="sample.conllu",
        bert="/data/nfsdata2/nlp_application/models/bert/bert-large-uncased-whole-word-masking",
    )

    from torch.utils.data import DataLoader

    group_ids, group_counts = dataset.get_groups()
    loader = DataLoader(dataset, collate_fn=collate_s2t_data,
                        sampler=GroupedSampler(
                            dataset=dataset,
                            sampler=SequentialSampler(dataset),
                            group_ids=group_ids,
                            batch_size=8,
                        ))
    for batch in tqdm(loader):
        print(batch)
