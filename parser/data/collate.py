# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: collate
@time: 2020/12/18 11:11
@desc: 

"""

import torch
from typing import List, Dict


def collate_dependency_data(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of mrc samples
        fields:
            token_ids: [num_word_pieces]
            type_ids: [num_word_pieces]
            offsets: [num_words, 2]
            wordpiece_mask: [num_word_pieces]
            span_idx: [2]
            span_tag: [1]
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

    for field in ["pos_tags", "mrc_mask", "word_mask"]:
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
