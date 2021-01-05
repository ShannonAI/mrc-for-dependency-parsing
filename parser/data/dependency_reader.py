# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: dependency_t2t_reader
@time: 2020/12/17 10:47
@desc: 

"""


from functools import lru_cache
from itertools import chain
from typing import List, Iterable

import numpy as np
import torch
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from conllu import parse_incr
from torch.utils.data import Dataset

from parser.data.samplers.grouped_sampler import create_lengths_groups
from parser.utils.logger import get_logger

logger = get_logger(__name__)


class DependencyDataset(Dataset):
    """
    Reads a file in the conllu/conllx Dependencies format.

    # Parameters
    file_path: conllu/conllx format dataset
    use_language_specific_pos : `bool`, optional (default = `False`)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    tokenizer : `BertTokenizer`, optional (default = `None`)
        A tokenizer to use to split the text. This is useful when the tokens that you pass
        into the model need to have some particular attribute. Typically it is not necessary.
    """
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

            logger.info(f"Read {len(self.data)} sentences from conllu dataset at: %s", file_path)

        self.pos_tags = pos_tags or self.build_label_vocab(chain(*[d[1] for d in self.data]))
        self.pos_tag_2idx = {l: idx for idx, l in enumerate(self.pos_tags)}
        self.dep_tags = dep_tags or self.build_label_vocab(chain(*[d[2] for d in self.data]))
        self.dep_tag_2idx = {l: idx for idx, l in enumerate(self.dep_tags)}
        logger.info(f"pos tags: {self.pos_tag_2idx}")
        logger.info(f"dep tags: {self.dep_tag_2idx}")

        self._matched_indexer = PretrainedTransformerIndexer(model_name=bert)
        self._allennlp_tokenizer = self._matched_indexer._allennlp_tokenizer

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
        return len(self.data)

    @lru_cache(maxsize=100000)
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
        fields.update(bert_mismatch_fields)

        pos_tag_idxs = [self.pos_tag_2idx[p] for p in pos_tags]

        fields["dp_idxs"] = torch.LongTensor(dp_heads)
        fields["dp_tags"] = torch.LongTensor([self.dep_tag_2idx[t] for t in dp_tags])

        fields["pos_tags"] = torch.LongTensor(pos_tag_idxs)
        fields["meta_data"] = {
            "words": words,
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
            group_save_path = self.file_path + ".groups.npy"
            counts_save_path = self.file_path + ".groups_counts.npy"
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
            groups, counts = create_lengths_groups(lengths=[len(x[0]) for x in self.data],
                                                   max_length=max_length)
            np.save(group_save_path, groups)
            np.save(counts_save_path, counts)
            logger.info(f"Groups info save to {group_save_path}")
        return groups, counts


if __name__ == '__main__':
    from tqdm import tqdm
    from parser.data.samplers.grouped_sampler import GroupedSampler
    from torch.utils.data import SequentialSampler

    dataset = DependencyDataset(
        file_path="/data/nfsdata2/nlp_application/datasets/treebank/LDC99T42/ptb3_parser/dev.conllx",
        bert="/data/nfsdata2/nlp_application/models/bert/bert-large-uncased-whole-word-masking",
    )

    from torch.utils.data import DataLoader
    from parser.data.collate import collate_dependency_data

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
