# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: dependency_t2t_reader
@time: 2020/12/17 10:47
@desc: 

"""


from typing import List, Dict

import torch

from parser.data.base_bert_dataset import BaseDataset
from parser.utils.logger import get_logger
import warnings
from random import randint

logger = get_logger(__name__)


class DependencyDataset(BaseDataset):
    """
    Reads a file in the conllu/conllx Dependencies format.

    Args:
        file_path: conllu/conllx format dataset
        bert: bert directory path
        use_language_specific_pos : `bool`, optional (default = `False`)
            Whether to use UD POS tags, or to use the language specific POS tags
            provided in the conllu format.
        pos_tags: if specified, directly use it instead of counting POS tags from file
        dep_tags: if specified, directly use it instead of counting dependency tags from file
    """

    def __init__(
        self,
        file_path: str,
        bert: str,
        use_language_specific_pos: bool = False,
        pos_tags: List[str] = None,
        dep_tags: List[str] = None,
        max_length: int = 512
    ) -> None:
        super().__init__(file_path, bert, use_language_specific_pos, pos_tags, dep_tags)
        logger.info(f"pos tags: {self.pos_tag_2idx}")
        logger.info(f"dep tags: {self.dep_tag_2idx}")
        self.max_length = max_length

    def __getitem__(self, idx):
        """
        Returns:
            token_ids: [num_word_pieces]
            type_ids: [num_word_pieces]
            offsets: [num_words, 2]
            wordpiece_mask: [num_word_pieces]
            dp_idxs: [2]
            dp_tags: [1]
            pos_tags: [num_words]
            word_mask: [num_words]
            meta_data: dict of meta_fields
        """
        words, pos_tags, dp_tags, dp_heads = self.data[idx]

        fields = {
            "type_ids": torch.LongTensor([0] * len(words)),
            "word_mask": torch.LongTensor([1] * len(words)),
        }

        bert_mismatch_fields = self.get_mismatch_token_idx(words)
        
        if len(bert_mismatch_fields["token_ids"]) > self.max_length:
            warnings.warn(f"sample id {idx} exceeds max-length {self.max_length}")
            return self[randint(0, len(self)-1)]

        fields.update(bert_mismatch_fields)

        pos_tag_idxs = [self.pos_tag_2idx[p] for p in pos_tags]

        fields["dp_idxs"] = torch.LongTensor(dp_heads)
        fields["dp_tags"] = torch.LongTensor([self.dep_tag_2idx[t] for t in dp_tags])

        fields["pos_tags"] = torch.LongTensor(pos_tag_idxs)
        fields["meta_data"] = {
            "words": words,
            "pos_tags": pos_tags,
            "dp_tags": dp_tags,
            "dp_heads": dp_heads
        }

        return fields


def collate_dependency_data(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of dependency samples, including fields:
            token_ids: [num_word_pieces]
            type_ids: [num_word_pieces]
            offsets: [num_words, 2]
            wordpiece_mask: [num_word_pieces]
            dp_idxs: [num_words]
            dp_tags: [num_words]
            pos_tags: [num_words]
            word_mask: [num_words]
            meta_data: dict of meta_fields
    Returns:
        output: dict of batched fields
    """

    batch_size = len(batch)
    max_pieces = max(x["token_ids"].size(0) for x in batch)
    max_words = max(x["pos_tags"].size(0) for x in batch)

    output = {}

    for field in ["token_ids", "type_ids", "wordpiece_mask"]:
        pad_output = torch.full([batch_size, max_pieces], 0, dtype=batch[0][field].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field]
            pad_output[sample_idx][: data.shape[0]] = data
        output[field] = pad_output

    for field in ["pos_tags", "word_mask", "dp_idxs", "dp_tags"]:
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

    dataset = DependencyDataset(
        # file_path="/data/nfsdata2/nlp_application/datasets/treebank/LDC99T42/ptb3_parser/dev.conllx",
        # file_path="sample.conllu",
        # bert="/data/nfsdata2/nlp_application/models/bert/bert-large-uncased-whole-word-masking",
        file_path="/data/nfsdata2/nlp_application/datasets/treebank/LDC2005T01/data/ctb5_parser/test.conllx",
        bert="/data/nfsdata2/nlp_application/models/bert/chinese_L-12_H-768_A-12",
    )

    from torch.utils.data import DataLoader

    group_ids, group_counts = dataset.get_groups()
    loader = DataLoader(dataset, collate_fn=collate_dependency_data,
                        sampler=GroupedSampler(
                            dataset=dataset,
                            sampler=SequentialSampler(dataset),
                            group_ids=group_ids,
                            batch_size=8,
                        ))
    for batch in tqdm(loader):
        print(batch)
