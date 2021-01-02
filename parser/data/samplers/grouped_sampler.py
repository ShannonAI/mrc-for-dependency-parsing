"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: multitas_sampler
@time: 2020/7/8 19:51

    Sampler that gather samples of similar size into one batch
"""

import bisect
import copy
import os
from collections import defaultdict
from typing import List, Union, Tuple, Iterator

import numpy as np
from torch.utils.data.sampler import BatchSampler, Sampler
from torch.utils.data import Dataset

from multiprocessing import Pool
from functools import partial


from parser.utils.logger import get_logger

logger = get_logger(__name__)


def _quantize(x, bins):
    """find bucket index for every n in x"""
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    with Pool(os.cpu_count()) as pool:
        quantized = pool.map(func=partial(bisect.bisect_right, bins), iterable=x)
    return quantized


def create_lengths_groups(lengths, min_length=3, max_length=128, step=4) -> Tuple[np.array, np.array]:
    """
    create_lengths_groups
    Args:
        lengths: List[int], each sample length
        min_length: minimum length, defaults to 3 because of [CLS] and [SEP]
        max_length: maximum length
        step: bucket range
    Returns:
        groups: np.array of length len(lengths). groups[idx] is the idx'th sample group idx
        counts: count of instances per group
    """
    bins = np.arange(start=min_length, stop=max_length, step=step).tolist() if max_length > 0 else [10]
    groups = _quantize(lengths, bins)
    # count number of elements per group
    counts = np.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [np.inf]
    logger.info("Using {} as bins for aspect lengths quantization".format(fbins))
    logger.info("Count of instances per bin: {}".format(counts))
    return np.array(groups, dtype=np.int32), counts


class GroupedSampler(Sampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        dataset: pytorch Dataset
        sampler: Base sampler.
        group_ids: If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size: Size of mini-batch.
        counts: if provided, pre-compute order of each group. Otherwise iterate greedily: output a batch of
        samples when any group is full. This could boost performance in multi-GPU training.
        queue_length: int, max queue length of each group when using counts.
    """
    def __init__(self, dataset: Dataset, sampler: Sampler, group_ids: Union[List[int], np.array], batch_size,
                 counts: np.array = None, queue_length: int = 1000):
        super(GroupedSampler, self).__init__(dataset)
        self.dataset = dataset
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size
        self.counts = counts
        self.group_generator = None
        # if self.counts is not None:
        #     self.group_generator = self.get_group_generator(self.counts)
        self.queue_length = queue_length
        assert self.queue_length >= self.batch_size, "queue_length should greater or equal to batch_size"

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        next_group_id = next(self.group_generator) if self.group_generator else None
        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            if self.group_generator is None:
                if len(buffer_per_group[group_id]) == self.batch_size:
                    for _ in range(self.batch_size):
                        yield buffer_per_group[group_id].pop()
                    num_batches += 1
            elif self.group_generator is not None:
                if group_id == next_group_id and len(buffer_per_group[group_id]) == self.batch_size:
                    for x in buffer_per_group[group_id]:
                        yield x
                    num_batches += 1
                    del buffer_per_group[group_id]
                    next_group_id = next(self.group_generator)
                # prevent oom
                elif len(buffer_per_group[group_id]) >= self.queue_length:
                    for _ in range(self.batch_size):
                        yield buffer_per_group[group_id].pop()
                    num_batches += 1
                    next_group_id = next(self.group_generator)

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            # for the remaining batches, group the batches by similar lengths
            batch_idx = []
            for group_id, idxs in sorted(buffer_per_group.items(), key=lambda x: x[0]):
                batch_idx.extend(idxs)
                while len(batch_idx) >= self.batch_size:
                    for _ in range(self.batch_size):
                        yield batch_idx.pop()
                    num_remaining -= 1
            for x in batch_idx:
                yield x

    def __len__(self):
        """
        Return the number of samples.
        """
        return len(self.sampler)

    # todo(this is for accelerate bucket-sampler speed in ddp training)
    #     because we want all gpus are dealing with samples with similar lengths
    # @staticmethod
    # def get_group_generator(counts: np.array) -> Iterator[int]:
    #     """根据counts生成采样每个group的顺序"""
    #     group_num = len(counts)
    #     largest_group = np.argmax(counts).item()
    #     ratios = counts / counts[largest_group]
    #     accumulates = np.zeros_like(counts, dtype=np.float32)
    #     while True:
    #         accumulates[largest_group] += 1
    #         for group_idx in range(group_num):
    #             delta = ratios[group_idx]
    #             accumulates[group_idx] += delta
    #             if accumulates[group_idx] >= 1.0:
    #                 yield group_idx
    #                 accumulates[group_idx] -= 1


if __name__ == '__main__':
    from tqdm import tqdm
    lengths = np.random.randint(low=3, high=128, size=[100000])
    groups, counts = create_lengths_groups(lengths=lengths, max_length=128)
    generator = GroupedSampler.get_group_generator(counts)
    for group_idx in tqdm(generator):
        print(group_idx)
