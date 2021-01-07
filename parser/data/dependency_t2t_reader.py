# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: dependency_t2t_reader
@time: 2020/12/17 10:47
@desc: 

"""


from copy import deepcopy
from itertools import chain
from typing import List, Dict

import torch

from parser.data.base_bert_dataset import BaseDataset
from parser.utils.logger import get_logger

logger = get_logger(__name__)


class DependencyT2TDataset(BaseDataset):
    """
    Depency Dataset used for Token-to-Token MRC

    Args:
        file_path: conllu/conllx format dataset
        bert: bert directory path
        use_language_specific_pos : `bool`, optional (default = `False`)
            Whether to use UD POS tags, or to use the language specific POS tags
            provided in the conllu format.
        pos_tags: if specified, directly use it instead of counting POS tags from file
        dep_tags: if specified, directly use it instead of counting dependency tags from file
    """

    # we use [unused0] and [unused1] in bert vocab to represent bracket around query token
    SPAN_START = 1
    SPAN_END = 2

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

        self.offsets = self.build_offsets()
        logger.info(f"build {len(self.offsets)} mrc-samples from {file_path}")

        self.pos_tags = pos_tags or self.build_label_vocab(chain(*[d[1] for d in self.data], [self.SEP_POS]))
        self.pos_tag_2idx = {l: idx for idx, l in enumerate(self.pos_tags)}
        logger.info(f"pos tags: {self.pos_tag_2idx}")
        logger.info(f"dep tags: {self.dep_tag_2idx}")

    def build_offsets(self):
        """offsets[i] = (sent_idx, word_idx) of ith mrc-samples """
        offsets = []
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
            token_type_ids: [num_word_pieces]
            offsets: [num_words, 2]
            wordpiece_mask: [num_word_pieces]
            span_idx: [2]  todo change to [1]
            span_tag: [1]
            pos_tags: [num_words]
            mrc_mask: [num_words]
            word_mask: [num_words]
            child_arcs: [num_words]
            child_tags: [num_words]
            meta_data: dict of meta_fields
        """
        ann_idx, word_idx = self.offsets[idx]
        words, pos_tags, dp_tags, dp_heads = self.data[ann_idx]
        mrc_length = len(words) * 2 + 3
        type_ids = [0] * (len(words) + 3) + [1] * len(words)  # bert sentence-pair token-ids
        # todo try type_ids = [0] * (len(words) + 2) + [1] * (len(words) + 1) ?

        fields = {"type_ids": torch.LongTensor(type_ids)}

        query_tokens = deepcopy(words) + [self.SEP]
        query_tokens.insert(word_idx, self.SEP)
        query_tokens.insert(word_idx + 2, self.SEP)
        mrc_tokens = query_tokens + words
        bert_mismatch_fields = self.get_mismatch_token_idx(mrc_tokens)
        self.replace_special_token(bert_mismatch_fields, [word_idx], self.SPAN_START)
        self.replace_special_token(bert_mismatch_fields, [word_idx+2], self.SPAN_END)
        fields.update(bert_mismatch_fields)

        query_pos_tags = pos_tags.copy() + [self.SEP_POS]
        query_pos_tags.insert(word_idx, self.SEP_POS)
        query_pos_tags.insert(word_idx + 2, self.SEP_POS)
        query_length = len(words) + 2

        # num_words * 2 + 3
        mrc_pos_tag_idxs = [self.pos_tag_2idx[p] for p in query_pos_tags + pos_tags]
        # valid parent idxs
        mrc_mask = [False] * mrc_length
        # only context(and SEP, which represents root) can be parent/child
        mrc_mask[query_length:] = [True] * (len(words) + 1)
        mrc_mask[query_length + word_idx + 1] = False  # origin word cannot be parent/child of itself

        # todo root的child怎么算？
        child_flags = [0] * mrc_length
        child_tags = [0] * mrc_length
        for child_idx, parent_idx in enumerate(dp_heads):
            # +1 because word_idx start from 0, but in dependency ann, true word start from 1
            if parent_idx == word_idx + 1:
                child_flags[child_idx + query_length+1] = 1
                child_tags[child_idx + query_length+1] = self.dep_tag_2idx[dp_tags[child_idx]]
        fields["child_arcs"] = torch.LongTensor(child_flags)
        fields["child_tags"] = torch.LongTensor(child_tags)

        parent_tag = dp_tags[word_idx]
        parent_idx = dp_heads[word_idx]
        span_start = span_end = parent_idx + query_length
        fields["span_idx"] = torch.LongTensor([span_start, span_end])
        fields["span_tag"] = torch.LongTensor([self.dep_tag_2idx[parent_tag]])

        fields["pos_tags"] = torch.LongTensor(mrc_pos_tag_idxs)
        fields["mrc_mask"] = torch.BoolTensor(mrc_mask)
        fields["meta_data"] = {
            "words": words,
            "ann_idx": ann_idx,
            "word_idx": word_idx
        }

        return fields

    def _get_item_lengths(self) -> List[int]:
        """get seqence-length of every item"""
        item_lengths = []
        for words, _, _, _ in self.data:
            item_lengths.extend([len(words)] * len(words))
        return item_lengths


def collate_dependency_t2t_data(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of mrc samples, includingfields:
            token_ids: [num_word_pieces]
            type_ids: [num_word_pieces]
            offsets: [num_words, 2]
            wordpiece_mask: [num_word_pieces]
            span_idx: [2]
            span_tag: [1]
            child_arcs: [num_words]
            child_tags: [num_words]
            pos_tags: [num_words]
            mrc_mask: [num_words]
            word_mask: [num_words]
            meta_data: dict
    Returns:
        output: dict of batched fields
    """

    batch_size = len(batch)
    max_pieces = max(x["token_ids"].size(0) for x in batch)
    max_words = max(x["pos_tags"].size(0) for x in batch)

    output = {
        "span_idx": torch.stack([x["span_idx"] for x in batch]),
        "span_tag": torch.cat([x["span_tag"] for x in batch])
              }

    for field in ["token_ids", "type_ids", "wordpiece_mask"]:
        pad_output = torch.full([batch_size, max_pieces], 0, dtype=batch[0][field].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field]
            pad_output[sample_idx][: data.shape[0]] = data
        output[field] = pad_output

    for field in ["pos_tags", "mrc_mask", "word_mask", "child_arcs", "child_tags"]:
        pad_output = torch.full([batch_size, max_words], 0, dtype=batch[0][field].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field]
            pad_output[sample_idx][: data.shape[0]] = data
        output[field] = pad_output

    pad_offsets = torch.full([batch_size, max_words, 2], 0, dtype=torch.long)
    for sample_idx in range(batch_size):
        data = batch[sample_idx]["offsets"]
        pad_offsets[sample_idx][: data.shape[0]] = data
    output["offsets"] = pad_offsets
    meta_fields = batch[0]["meta_data"].keys()
    output["meta_data"] = {field: [x["meta_data"][field] for x in batch] for field in meta_fields}

    return output


if __name__ == '__main__':
    from tqdm import tqdm
    from parser.data.samplers.grouped_sampler import GroupedSampler
    from torch.utils.data import SequentialSampler
    from torch.utils.data import DataLoader

    dataset = DependencyT2TDataset(
        file_path="/data/nfsdata2/nlp_application/datasets/treebank/LDC99T42/ptb3_parser/train.conllx",
        # file_path="sample.conllu",
        bert="/data/nfsdata2/nlp_application/models/bert/bert-large-uncased-whole-word-masking",
    )
    group_ids, group_counts = dataset.get_groups()
    loader = DataLoader(dataset, collate_fn=collate_dependency_t2t_data,
                        sampler=GroupedSampler(
                            dataset=dataset,
                            sampler=SequentialSampler(dataset),
                            group_ids=group_ids,
                            batch_size=32,
                        ))
    for batch in tqdm(loader):
        print(batch)
