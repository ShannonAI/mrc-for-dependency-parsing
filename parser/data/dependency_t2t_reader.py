# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: dependency_t2t_reader
@time: 2020/12/17 10:47
@desc: 

"""


from itertools import chain
from copy import deepcopy
from typing import List, Iterable

import numpy as np
import torch
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from conllu import parse_incr
from torch.utils.data import Dataset

from parser.data.samplers.grouped_sampler import create_lengths_groups
from parser.utils.logger import get_logger

logger = get_logger(__name__)


class DependencyT2TDataset(Dataset):
    """
    Reads a file in the conllu/conllx Dependencies format.
    Used for Token to token MRC format

    # Parameters
    file_path: conllu/conllx format dataset
    use_language_specific_pos : `bool`, optional (default = `False`)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    tokenizer : `BertTokenizer`, optional (default = `None`)
        A tokenizer to use to split the text. This is useful when the tokens that you pass
        into the model need to have some particular attribute. Typically it is not necessary.
    """

    SPAN_START = "[SEP]"  # todo 用[unused0]，但目前tokenizers的问题会把[unused0]转换为[,unused,##0,]
    SPAN_END = "[SEP]"
    SEP_POS = "sep_pos"
    SEP = "[SEP]"

    POS_TO_IGNORE = {"`", "''", ":", ",", ".", "PU", "PUNCT", "SYM"}

    def __init__(
        self,
        file_path: str,
        bert: str,
        use_language_specific_pos: bool = False,
        pos_tags: List[str] = None,
        dep_tags: List[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.use_language_specific_pos = use_language_specific_pos
        self.file_path = file_path
        self.data = []  # list of (words, pos_tags, dp_tags, dp_heads)
        with open(file_path, "r") as conllu_file:
            logger.info(f"Reading sentences from conll dataset at: {file_path} ...")
            for ann_idx, annotation in enumerate(parse_incr(conllu_file)):
                annotation = [x for x in annotation if isinstance(x["id"], int)]

                dp_heads = [x["head"] for x in annotation]
                dp_tags = [x["deprel"] for x in annotation]
                words = [x["form"] for x in annotation]
                if self.use_language_specific_pos:
                    sample_pos_tags = [x["xpostag"] for x in annotation]
                else:
                    sample_pos_tags = [x["upostag"] for x in annotation]
                self.data.append((words, sample_pos_tags, dp_tags, dp_heads))

            self.offsets = self.build_offsets()
            logger.info(f"Read {len(self.data)} sentences with "
                        f"{len(self.offsets)} mrc-samples from conll dataset at: {file_path}")

        self.pos_tags = pos_tags or self.build_label_vocab(chain(*[d[1] for d in self.data], [self.SEP_POS]))
        self.pos_tag_2idx = {l: idx for idx, l in enumerate(self.pos_tags)}
        self.dep_tags = dep_tags or self.build_label_vocab(chain(*[d[2] for d in self.data]))
        self.dep_tag_2idx = {l: idx for idx, l in enumerate(self.dep_tags)}
        logger.info(f"pos tags: {self.pos_tag_2idx}")
        logger.info(f"dep tags: {self.dep_tag_2idx}")

        self._matched_indexer = PretrainedTransformerIndexer( model_name=bert)
        self._allennlp_tokenizer = self._matched_indexer._allennlp_tokenizer

    def build_offsets(self):
        """offsets[i] = (sent_idx, word_idx) of ith mrc-samples """
        offsets = []
        for ann_idx, (words, _, _, _) in enumerate(self.data):
            for word_idx in range(len(words)):
                offsets.append((ann_idx, word_idx))
        return offsets

    @staticmethod
    def build_label_vocab(labels: Iterable[str]):
        """build label to tag dictionay"""
        labels_set = set()
        for l in labels:
            labels_set.add(l)
        label_list = sorted(list(labels_set))
        return label_list

    @property
    def ignore_pos_tags(self):
        punctuation_tag_indices = {
            tag: index for index, tag in enumerate(self.pos_tags) if tag in self.POS_TO_IGNORE
        }
        return set(punctuation_tag_indices.values())

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        """
        Returns:
            token_ids: [num_word_pieces]
            token_type_ids: [num_word_pieces]
            offsets: [num_words, 2]
            wordpiece_mask: [num_word_pieces]
            span_idx: [2]
            span_tag: [1]
            pos_tags: [num_words]
            mrc_mask: [num_words]
            word_mask: [num_words]
            meta_data: dict of meta_fields
        """
        ann_idx, word_idx = self.offsets[idx]
        words, pos_tags, dp_tags, dp_heads = self.data[ann_idx]
        mrc_length = len(words) * 2 + 3
        type_ids = [0] * (len(words) + 3) + [1] * len(words)  # bert sentence-pair token-ids

        fields = {"type_ids": torch.LongTensor(type_ids), "ann_idx": torch.LongTensor([ann_idx])}

        query_tokens = deepcopy(words) + [self.SEP]
        query_tokens.insert(word_idx, self.SPAN_START)
        query_tokens.insert(word_idx + 2, self.SPAN_END)
        mrc_tokens = query_tokens + words
        bert_mismatch_fields = self.get_mismatch_token_idx(mrc_tokens)
        fields.update(bert_mismatch_fields)

        query_pos_tags = pos_tags.copy() + [self.SEP_POS]
        query_pos_tags.insert(word_idx, self.SEP_POS)
        query_pos_tags.insert(word_idx + 2, self.SEP_POS)
        query_length = len(words) + 2

        # num_words * 2 + 3
        mrc_pos_tag_idxs = [self.pos_tag_2idx[p] for p in query_pos_tags + pos_tags]
        # valid parent idxs
        mrc_mask = [False] * mrc_length
        # only context(and SEP, which represents root) can be parent
        mrc_mask[query_length:] = [True] * (len(words) + 1)
        mrc_mask[query_length + word_idx + 1] = False  # origin word cannot be parent of itself

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

    def get_mismatch_token_idx(self, words: List[str]):
        """
        splits the words into wordpieces. We want to embed these wordpieces and then pull out a single
        vector for each original word.
        For reference: pretrained_transformer_mismatched indexer of AllenNLP
        """
        # todo(yuxian): this line is extremely slow, need optimization
        wordpieces, offsets = self._allennlp_tokenizer.intra_word_tokenize(words)

        # For tokens that don't correspond to any word pieces, we put (-1, -1) into the offsets.
        # That results in the embedding for the token to be all zeros.
        offsets = [x if x is not None else (-1, -1) for x in offsets]

        output = {
            "token_ids": torch.LongTensor([t.text_id for t in wordpieces]),
            "word_mask": torch.BoolTensor([True] * len(words)),  # for original tokens (i.e. word-level)
            "offsets": torch.LongTensor(offsets),
            "wordpiece_mask": torch.BoolTensor([True] * len(wordpieces)),  # for wordpieces (i.e. subword-level)
        }
        return output

    def get_groups(self, max_length=128, cache=True):
        """ge groups, used for GroupSampler"""
        success = False
        if cache:
            group_save_path = self.file_path + ".t2t.groups.npy"
            counts_save_path = self.file_path + ".t2t.groups_counts.npy"
            try:
                logger.info("Loading pre-computed groups")
                counts = np.load(counts_save_path)
                groups = np.load(group_save_path)
                assert len(groups) == len(self), \
                f"number of group_idxs {len(groups)} should have same length as dataset: {len(self)}"
                success = True
            except Exception as e:
                logger.error(f"Loading pre-computed groups from {group_save_path} failed", exc_info=1)
        if not success:
            logger.info("Re-computing groups")
            groups, counts = create_lengths_groups(lengths=self._get_item_lengths(),
                                                   max_length=max_length)
            np.save(group_save_path, groups)
            np.save(counts_save_path, counts)
            logger.info(f"Groups info save to {group_save_path}")

        return groups, counts

    def _get_item_lengths(self) -> List[int]:
        """get seqence-length of every item"""
        item_lengths = []
        for words, _, _, _ in self.data:
            item_lengths.extend([len(words)] * len(words))
        return item_lengths


if __name__ == '__main__':
    from tqdm import tqdm
    from parser.data.samplers.grouped_sampler import GroupedSampler
    from torch.utils.data import SequentialSampler
    from torch.utils.data import DataLoader
    from parser.data.collate import collate_dependency_t2t_data

    dataset = DependencyT2TDataset(
        file_path="/data/nfsdata2/nlp_application/datasets/treebank/LDC99T42/ptb3_parser/train.conllx",
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
