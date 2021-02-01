# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/1/7 17:33
@desc: Base DatasetReader for dependency using bert tokenizer

"""

from itertools import chain
from typing import List, Iterable, Dict

import numpy as np
import torch
import warnings
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from conllu import parse_incr
from torch.utils.data import Dataset
from parser.data.tree_utils import is_cyclic
import codecs

from parser.data.samplers.grouped_sampler import create_lengths_groups
from parser.utils.logger import get_logger

logger = get_logger(__name__)

BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    "\u2013": "--",  # en dash
    "\u2014": "--",  # en dash
}


class BaseDataset(Dataset):
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
        max_words: integer, defaults to 0. if specified, only load samples whose number of words is less than max_words

    """
    # punctuation pos tags that may be ignored when evaluating UAS/LAS
    POS_TO_IGNORE = {
        "``", "''", ":", ",", ".",  # PTB
        "PU",  # CTB
        "PUNCT",  "SYM"  # UD
    }

    # group_file suffix name
    group_file_suffix = ""

    def __init__(
        self,
        file_path: str,
        bert: str,
        use_language_specific_pos: bool = False,
        pos_tags: List[str] = None,
        dep_tags: List[str] = None,
        max_words: int = 0
    ) -> None:
        super().__init__()
        self.use_language_specific_pos = use_language_specific_pos
        self.file_path = file_path
        self.data = []  # list of (words, pos_tags, dp_tags, dp_heads)
        with codecs.open(file_path, "r", "utf-8") as conllu_file:
            logger.info(f"Reading sentences from conll dataset at: {file_path} ...")
            for ann_idx, annotation in enumerate(parse_incr(conllu_file)):
                annotation = [x for x in annotation if isinstance(x["id"], int)]

                dp_heads = [x["head"] for x in annotation]
                dp_tags = [x["deprel"] for x in annotation]
                words = [x["form"] for x in annotation]
                # convert unknown words for bert tokens
                cleaned_words = []
                for word in words:
                    word = BERT_TOKEN_MAPPING.get(word, word)
                    if word == "n't" and cleaned_words:
                        cleaned_words[-1] = cleaned_words[-1] + "n"
                        word = "'t"
                    cleaned_words.append(word)

                if 0 < max_words < len(cleaned_words):
                    continue

                if self.use_language_specific_pos:
                    sample_pos_tags = [x["xpostag"] for x in annotation]
                else:
                    sample_pos_tags = [x["upostag"] for x in annotation]
                self.data.append([cleaned_words, sample_pos_tags, dp_tags, dp_heads])

            logger.info(f"Read {len(self.data)} sentences from conll dataset at: %s", file_path)

        cyclic = [is_cyclic(d[-1]) for d in self.data]
        cyclic_idxs = [idx for idx, c in enumerate(cyclic) if c]
        if len(cyclic_idxs) > 0:
            warnings.warn(f"found {len(cyclic_idxs)} cyclic sample in dataset of id: {cyclic_idxs}")
        # remove cyclic data(found 1 in CTB training dataset)
        self.data = [d for c, d in zip(cyclic, self.data) if not c]

        self.pos_tags = pos_tags or self.build_label_vocab(chain(*[d[1] for d in self.data]))
        self.pos_tag_2idx = {l: idx for idx, l in enumerate(self.pos_tags)}
        self.dep_tags = dep_tags or self.build_label_vocab(chain(*[d[2] for d in self.data]))
        self.dep_tag_2idx = {l: idx for idx, l in enumerate(self.dep_tags)}

        self._matched_indexer = PretrainedTransformerIndexer(model_name=bert)
        self._allennlp_tokenizer = self._matched_indexer._allennlp_tokenizer

    def encode_dep_tags(self, dep_tags: List[str]) -> List[int]:
        return [self.dep_tag_2idx[t] for t in dep_tags]

    @staticmethod
    def build_label_vocab(labels: Iterable[str]):
        """build label to tag dictionay"""
        labels_set = set()
        for l in labels:
            labels_set.add(l)
        label_list = sorted(list(labels_set))
        return label_list

    @property
    def ignore_pos_tags(self) -> List[int]:
        return [
            index for index, tag in enumerate(self.pos_tags) if tag in self.POS_TO_IGNORE
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError

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
        """get groups, used for GroupSampler"""
        success = False
        if cache:
            group_save_path = self.file_path + f"{self.group_file_suffix}.groups.npy"
            counts_save_path = self.file_path + f"{self.group_file_suffix}.groups_counts.npy"
            try:
                logger.info(f"Loading pre-computed groups from {group_save_path}")
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
            assert len(groups) == len(self), \
                f"number of group_idxs {len(groups)} should have same length as dataset: {len(self)}"
            np.save(group_save_path, groups)
            np.save(counts_save_path, counts)
            logger.info(f"Groups info save to {group_save_path}")
        return groups, counts

    def _get_item_lengths(self) -> List[int]:
        """get each sample's length, used for group sampler"""
        return [len(x[0]) for x in self.data]

    def replace_special_token(self, tokenizer_fields: Dict[str, torch.Tensor], positions: List[int], replace_id: int):
        """
        Since AllenNlp will tokenize [unused0] to [ unused ##0 ], but tokenize [SEP] as it is , we passed [SEP] to it,
        then replace it with other special token ids
        """
        token_ids, offsets = tokenizer_fields["token_ids"], tokenizer_fields["offsets"]
        for pos in positions:
            offset = offsets[pos].numpy().tolist()
            if offset[0] != offset[1]:
                warnings.warn(f"replace normally expect token in `positions` has not been split to pieces."
                              f"This warning should NOT happen unless during batch prediction at evaluation")
            token_ids[offset[0]] = replace_id
