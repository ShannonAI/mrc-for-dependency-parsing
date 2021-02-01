# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/1/13 14:57
@desc: Dataset for Span-to-Span MRC

"""


from copy import deepcopy
from itertools import chain
from typing import List, Dict

import torch
import warnings
from parser.data.base_bert_dataset import BaseDataset
from parser.data.tree_utils import build_subtree_spans
from parser.utils.logger import get_logger
from random import randint

logger = get_logger(__name__)


class S2SDataset(BaseDataset):
    """
    Dependency Dataset used for Span-to-Span MRC
    Build dataset that use subtree-span to query its parent subtree-span
    Args:
        file_path: conllu/conllx format dataset
        bert: bert directory path
        use_language_specific_pos : `bool`, optional (default = `False`)
            Whether to use UD POS tags, or to use the language specific POS tags
            provided in the conllu format.
        pos_tags: if specified, directly use it instead of counting POS tags from file
        dep_tags: if specified, directly use it instead of counting dependency tags from file
        bert_name: "roberta" or "bert". if None, we guess type by finding "roberta" in bert path.
        max_length: max word pieces in a sample, because bert has a maximum length of 512. if one sample's
            length after tokenization is larger than max_length, we randomly choose another sample
        max_words: max words in a sample.
    """

    SEP_POS = "sep_pos"

    group_file_suffix = ".mrc"

    def __init__(
        self,
        file_path: str,
        bert: str,
        use_language_specific_pos: bool = False,
        pos_tags: List[str] = None,
        dep_tags: List[str] = None,
        bert_name: str = None,
        max_length: int = 512,
        max_words: int = 0
    ) -> None:
        super().__init__(file_path, bert, use_language_specific_pos, pos_tags, dep_tags, max_words)
        self.max_length = max_length
        self.subtree_spans = [build_subtree_spans(d[3]) for d in self.data]

        self.offsets = self.build_offsets()
        logger.info(f"build {len(self.offsets)} mrc-samples from {file_path}")

        self.pos_tags = pos_tags or self.build_label_vocab(chain(*[d[1] for d in self.data], [self.SEP_POS]))
        self.pos_tag_2idx = {l: idx for idx, l in enumerate(self.pos_tags)}
        logger.info(f"pos tags: {self.pos_tag_2idx}")
        logger.info(f"dep tags: {self.dep_tag_2idx}")

        self.bert_name = bert_name
        if self.bert_name is None:
            self.bert_name = "roberta" if 'roberta' in bert else "bert"
        assert self.bert_name in ["roberta", "bert"]

        if self.bert_name == "roberta":
            self.SEP = "</s>"
            # roberta 1/2/3/4 are not unused words like bert!
            self.SPAN_START = 2
            self.SUBTREE_ROOT_START = 2
            self.SUBTREE_ROOT_END = 2
            self.SPAN_END = 2
        else:
            self.SEP = "[SEP]"
            # we use [unused0] ~ [unused3] in bert vocab to represent bracket around query token and query span
            self.SPAN_START = 1
            self.SUBTREE_ROOT_START = 2
            self.SUBTREE_ROOT_END = 3
            self.SPAN_END = 4

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
            parent_mask: [num_words]: True if this position can be parent word_idx, otherwise False
            parent_start_mask: [num_words]: True if this position can be parent subtree start, otherwise False
            parent_end_mask: [num_words]: True if this position can be parent subtree end, otherwise False
            child_mask: [num_words]: True if this position can be child subtree root/start/end, otherwise False
            meta_data: dict of meta_fields
            parent_idxs: [1]
            parent_tags: [1]
            parent_starts: [1]
            parent_ends: [1]
            child_idxs: [num_words]: True if this position is a subtree-child of query span, otherwise False
            child_starts: [num_words]
            child_ends: [num_words]
        """
        ann_idx, word_idx = self.offsets[idx]
        words, pos_tags, dp_tags, dp_heads = self.data[ann_idx]
        subtree_spans = self.subtree_spans[ann_idx]
        span_start, span_end = subtree_spans[word_idx]
        # mrc sample consists of query and context.
        mrc_length = len(words) * 2 + 5
        # query is a copy of origin sentence, with special token surrounding the query-subtree span and root.
        query_length = len(words) + 4

        type_ids = [0] * (len(words) + 5) + [1 if self.bert_name == "bert" else 0] * len(words)

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
            warnings.warn("replace error, be careful that this should not happen unless you're "
                         "doing batch prediction at evaluation")
            # logger.error("replace error, be careful that this should not happen unless you're "
            #              "doing batch prediction at evaluation", exc_info=True)

        fields.update(bert_mismatch_fields)

        if len(bert_mismatch_fields["token_ids"]) > self.max_length:
            warnings.warn(f"sample id {idx} exceeds max-length {self.max_length}")
            return self[randint(0, len(self)-1)]

        query_pos_tags = pos_tags.copy() + [self.SEP_POS]
        for p in [word_idx, word_idx+1, word_idx+3, word_idx+4]:
            query_pos_tags.insert(p, self.SEP_POS)

        # num_words * 2 + 3
        mrc_pos_tag_idxs = [self.pos_tag_2idx[p] for p in query_pos_tags + pos_tags]
        # valid parent idxs
        parent_mask = [False] * mrc_length
        # only context(and SEP, which represents root) can be parent
        parent_mask[query_length:] = [True] * (len(words) + 1)

        parent_start_mask = parent_mask.copy()
        # we do not use root span as query, so it cannot be start/end
        parent_start_mask[query_length] = False
        parent_end_mask = parent_start_mask.copy()

        parent_mask[query_length + word_idx + 1] = False  # origin word cannot be parent/child of itself
        # parent should be outside of query_span(only in projective case!)
        # for idx in range(query_length + 1 + span_start, query_length + 1 + span_end + 1):
        #     if idx < len(parent_mask):
        #         parent_mask[idx] = False
        fields["parent_mask"] = torch.BoolTensor(parent_mask)

        # parent_start should be less or equal to span_start
        for idx in range(query_length + 1 + span_start + 1, len(parent_start_mask)):
            parent_start_mask[idx] = False
        fields["parent_start_mask"] = torch.BoolTensor(parent_start_mask)

        # parent_end should be greater or equal to span_end
        for idx in range(min(query_length + 1 + span_end, len(parent_end_mask))):
            parent_end_mask[idx] = False
        fields["parent_end_mask"] = torch.BoolTensor(parent_end_mask)

        parent_tag = dp_tags[word_idx]
        parent_idx = dp_heads[word_idx]  # note: parent_idx start from root
        if parent_idx == 0:
            parent_start, parent_end = 0, len(words)-1
        else:
            parent_start, parent_end = subtree_spans[parent_idx - 1]
        fields["parent_idxs"] = torch.LongTensor([parent_idx + query_length])
        fields["parent_starts"] = torch.LongTensor([parent_start + 1 + query_length])
        fields["parent_ends"] = torch.LongTensor([parent_end + 1 + query_length])
        if parent_tag not in self.dep_tag_2idx:
            warnings.warn(f"#{ann_idx} sample contains unknown dep_tag {parent_tag}, use 0")
            dep_tag_idx = 0
        else:
            dep_tag_idx = self.dep_tag_2idx[parent_tag]
        fields["parent_tags"] = torch.LongTensor([dep_tag_idx])
        try:
            assert parent_mask[parent_idx + query_length]
            assert parent_start_mask[parent_start + 1 + query_length]
            assert parent_end_mask[parent_end + 1 + query_length]
        except Exception as e:
            warnings.warn("assertion error, be careful that this should not happen unless you're "
                         "doing topk prediction, thus predicted span_start may be less than predicted parent_start")

        child_flags = [0] * mrc_length
        child_starts_flags = [0] * mrc_length
        child_ends_flags = [0] * mrc_length
        for child_idx, parent_idx in enumerate(dp_heads):
            # +1 because word_idx start from 0, but in dependency annotation, true word start from 1
            if parent_idx == word_idx + 1:
                child_flags[child_idx + query_length+1] = 1
                child_start, child_end = subtree_spans[child_idx]
                child_starts_flags[min(child_start + query_length + 1, len(child_starts_flags)-1)] = True
                child_ends_flags[min(child_end + query_length + 1, len(child_ends_flags)-1)] = True
        fields["child_idxs"] = torch.BoolTensor(child_flags)
        fields["child_starts"] = torch.BoolTensor(child_starts_flags)
        fields["child_ends"] = torch.BoolTensor(child_ends_flags)

        child_mask = [False] * mrc_length
        # child should be inside parent span
        for idx in range(span_start, span_end+1):
            if idx == word_idx:
                continue
            idx = query_length + 1 + idx
            if idx >= mrc_length:
                continue
            child_mask[idx] = True
        fields["child_mask"] = torch.BoolTensor(child_mask)

        fields["pos_tags"] = torch.LongTensor(mrc_pos_tag_idxs)
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


def collate_s2s_data(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
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
            parent_mask: [num_words]: True if this position can be parent word_idx, otherwise False
            parent_start_mask: [num_words]: True if this position can be parent subtree start, otherwise False
            parent_end_mask: [num_words]: True if this position can be parent subtree end, otherwise False
            child_mask: [num_words]: True if this position can be child subtree root/start/end, otherwise False
            meta_data: dict of meta_fields
            parent_idxs: [1]
            parent_tags: [1]
            parent_starts: [1]
            parent_ends: [1]
            child_idxs: [num_words]: True if this position is a subtree-child of query span, otherwise False
            child_starts: [num_words]
            child_ends: [num_words]
    Returns:
        output: dict of batched fields
    """

    batch_size = len(batch)
    max_pieces = max(x["token_ids"].size(0) for x in batch)
    max_words = max(x["pos_tags"].size(0) for x in batch)

    output = {}

    for field in ["parent_idxs", "parent_tags", "parent_starts", "parent_ends"]:
        output[field] = torch.cat([x[field] for x in batch])

    for field in ["token_ids", "type_ids", "wordpiece_mask"]:
        pad_output = torch.full([batch_size, max_pieces], 0, dtype=batch[0][field].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field]
            pad_output[sample_idx][: data.shape[0]] = data
        output[field] = pad_output

    for field in ["pos_tags", "child_starts", "child_ends", "word_mask", "parent_start_mask",
                  "parent_end_mask", "parent_mask", "child_idxs", "child_mask"]:
        pad_output = torch.full([batch_size, max_words], 0, dtype=batch[0][field].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field]
            pad_output[sample_idx][: data.shape[0]] = data
        output[field] = pad_output

    for field in ["offsets",]:
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

    dataset = S2SDataset(
        # file_path="/data/nfsdata2/nlp_application/datasets/treebank/LDC99T42/ptb3_parser/dev.conllx",
        file_path="/data/nfsdata2/nlp_application/datasets/treebank/LDC2005T01/data/ctb5_parser/test.conllx",
        # file_path="sample.conllu",
        # bert="/data/nfsdata2/nlp_application/models/bert/bert-large-uncased-whole-word-masking",
        bert="/data/nfsdata2/nlp_application/models/bert/chinese_L-12_H-768_A-12",
    )

    from torch.utils.data import DataLoader

    group_ids, group_counts = dataset.get_groups()
    loader = DataLoader(dataset, collate_fn=collate_s2s_data, num_workers=0,
                        sampler=GroupedSampler(
                            dataset=dataset,
                            sampler=SequentialSampler(dataset),
                            group_ids=group_ids,
                            batch_size=100,
                        ))
    for batch in tqdm(loader):
        print(batch)
