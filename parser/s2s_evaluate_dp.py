# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/1/8 10:13
@desc: Evaluate Span-to-Token dependency as Bottom-Up Dynamic Programming
# todo fix potential bug that after pruning, maybe no valid tree-factorization exists.
"""

import os
from typing import Dict

import torch
from tqdm import tqdm
import math

from parser.metrics import AttachmentScores
from parser.span_proposal_trainer import MrcSpanProposal
from parser.s2s_query_trainer import MrcS2SQuery
from parser.utils.decode_utils import DecodeStruct
import warnings
from parser.utils.logger import get_logger

logger = get_logger(__name__)


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()

# proposal_dir = "/userhome/yuxian/train_logs/dependency/ptb/biaf/s2t/proposal"
# proposal_hparams = os.path.join(proposal_dir, "lightning_logs/version_0/hparams.yaml")
# proposal_ckpt = os.path.join(proposal_dir, "epoch=3.ckpt")

proposal_dir = "/userhome/yuxian/train_logs/dependency/ptb/s2s/roberta_proposal_lr1e-5"
proposal_hparams = os.path.join(proposal_dir, "lightning_logs/version_3/hparams.yaml")
proposal_ckpt = os.path.join(proposal_dir, "epoch=5-v0.ckpt")
# proposal_dir = "/data/nfsdata2/yuxian/share/parser/proposal/"
# proposal_hparams = os.path.join(proposal_dir, "version_0/hparams.yaml")
# proposal_ckpt = os.path.join(proposal_dir, "epoch=9.ckpt")

query_dir = "/userhome/yuxian/train_logs/dependency/ptb/s2s/baseline"
query_hparams = os.path.join(query_dir, "lightning_logs/version_0/hparams.yaml")
# query_dir = "/data/yuxian/train_logs/dependency/ptb/s2s"
# query_hparams = os.path.join(query_dir, "version_0/hparams.yaml")
query_ckpt = os.path.join(query_dir, "epoch=3.ckpt")

proposal_model = MrcSpanProposal.load_from_checkpoint(
    checkpoint_path=proposal_ckpt,
    hparams_file=proposal_hparams,
    map_location=None,
    batch_size=128,
    max_length=128,
    workers=0,
    group_sample=False,
    # bert_name="bert",  # todo 改成roberta！！！
    # bert_dir="/data/nfsdata2/nlp_application/models/bert/bert-large-uncased-whole-word-masking",
    # data_dir="/data/nfsdata2/nlp_application/datasets/treebank/LDC99T42/ptb3_parser/",
    gpus="1"  # deactivate distributed sampler
)
proposal_model.cuda()
proposal_model.eval()
dep_tag2idx = {tag: idx for idx, tag in enumerate(proposal_model.args.dep_tags)}

query_model = MrcS2SQuery.load_from_checkpoint(
    checkpoint_path=query_ckpt,
    hparams_file=query_hparams,
    map_location=None,
    batch_size=128,
    max_length=128,
    workers=0,
    group_sample=False,
    bert_dropout=0.2, ## todo remove
    # bert_dir="/data/nfsdata2/nlp_application/models/bert/bert-large-uncased-whole-word-masking",
    # data_dir="/data/nfsdata2/nlp_application/datasets/treebank/LDC99T42/ptb3_parser/",
    gpus="1"
)
query_model.cuda()
query_model.eval()

assert proposal_model.args.pos_tags + ["sep_pos"] == query_model.args.pos_tags
assert proposal_model.args.dep_tags == query_model.args.dep_tags
topk = 5  # use topk starts/ends in inference

with torch.no_grad():

    # ------
    # Stage 1: extract topk subtree spans for every token.
    # ------

    # map ann_idx to all the scores information
    ann_infos: Dict[int, DecodeStruct] = {}

    proposal_loader = proposal_model.test_dataloader()
    logger.info("Extract subtree proposal ...")
    for batch in tqdm(proposal_loader):
        to_cuda(batch)
        token_ids, type_ids, offsets, wordpiece_mask, pos_tags, word_mask, subtree_spans, meta_data = (
            batch["token_ids"], batch["type_ids"], batch["offsets"], batch["wordpiece_mask"],
            batch["pos_tags"], batch["word_mask"], batch["subtree_spans"], batch["meta_data"]
        )
        # [bsz, seq_len, seq_len]
        span_start_scores, span_end_scores = proposal_model(
            token_ids, type_ids, offsets, wordpiece_mask,
            pos_tags, word_mask
        )
        bsz, seq_len, _ = span_start_scores.size()

        # find topk spans for every position, where score(start, end) = score(start) + score(end)
        # [bsz, seq_len, seq_len, seq_len]
        span_scores = span_start_scores.unsqueeze(-1) + span_end_scores.unsqueeze(-2)
        # mask out start > end positions
        minus_inf = -1e-8
        tmp = span_scores[0][0].clone().fill_(minus_inf)  # [seq_len, seq_len]
        triu_mask = torch.triu(tmp, diagonal=1).transpose(0, 1).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        span_scores += triu_mask

        # [bsz, seq_len, seq_len*seq_len]
        span_scores = span_scores.view(bsz, seq_len, -1)
        # [bsz, seq_len, topk]
        topk_span_scores, topk_span_flat_idxs = torch.topk(span_scores, dim=-1, k=topk)
        topk_span_starts = topk_span_flat_idxs // seq_len
        topk_span_ends = topk_span_flat_idxs % seq_len

        topk_span_starts = topk_span_starts.cpu().numpy()
        topk_span_ends = topk_span_ends.cpu().numpy()
        topk_span_scores = topk_span_scores.cpu().numpy()

        ann_idxs = meta_data["ann_idx"]
        for batch_idx in range(len(ann_idxs)):
            ann_idx = ann_idxs[batch_idx]
            words = meta_data["words"][batch_idx]
            word_idx2spans = dict()
            for word_idx in range(len(words)):
                spans = []
                for k_idx in range(topk):
                    spans.append((
                        topk_span_starts[batch_idx][word_idx][k_idx].item(),
                        topk_span_ends[batch_idx][word_idx][k_idx].item(),
                        topk_span_scores[batch_idx][word_idx][k_idx].item()
                    ))
                word_idx2spans[word_idx] = spans

            ann_infos[ann_idx] = DecodeStruct(
                words=words,
                span_candidates=word_idx2spans,
                pos_tags=pos_tags[batch_idx].cpu(),
                dep_heads=meta_data["dp_heads"][batch_idx],
                dep_tags=meta_data["dp_tags"][batch_idx]
            )

    # ------
    # Stage 2: score parents for each subtree span candidates
    # ------
    logger.info("Finding parents according to extracted subtree proposal")
    parent_scores = {}
    query_loader = query_model.test_dataloader()

    for k in range(topk):
        logger.info(f"Finding parents according to top{k+1} subtree proposal ...")
        subtree_spans = []
        for ann_idx in range(len(ann_infos)):
            ann_info = ann_infos[ann_idx]
            subtree_spans.append(ann_info.top_spans(k))
        query_loader.dataset.subtree_spans = subtree_spans

        for batch in tqdm(query_loader):
            to_cuda(batch)
            (token_ids, type_ids, offsets, wordpiece_mask, pos_tags,
             word_mask, parent_mask, parent_start_mask, parent_end_mask,
             meta_data, parent_idxs, parent_tags, parent_starts, parent_ends) = (
                batch["token_ids"], batch["type_ids"], batch["offsets"], batch["wordpiece_mask"],
                batch["pos_tags"], batch["word_mask"], batch["parent_mask"], batch["parent_start_mask"],
                batch["parent_end_mask"], batch["meta_data"],
                batch["parent_idxs"], batch["parent_tags"], batch["parent_starts"], batch["parent_ends"]
            )
            # Note: since parent_starts/ends here are predicted instead of groundtruth,
            # passing it to model may cause out of index error. Therefore we do not pass it.
            (parent_probs, parent_tag_probs, parent_start_probs, parent_end_probs) = query_model(
                token_ids, type_ids, offsets, wordpiece_mask,
                pos_tags, word_mask, parent_mask, parent_start_mask, parent_end_mask
            )
            parent_tags_scores, parent_tags_idxs = torch.max(parent_tag_probs, dim=-1)
            parent_probs = parent_probs * parent_tags_scores

            bsz = parent_probs.size(0)
            for batch_idx in range(bsz):
                ann_idx = meta_data["ann_idx"][batch_idx]
                word_idx = meta_data["word_idx"][batch_idx]
                nwords = len(meta_data["words"][batch_idx])
                span_start, span_end = meta_data["subtree_span"][batch_idx]
                mrc_offset = nwords + 4
                ann_infos[ann_idx].span2parent_arc_scores[(word_idx, span_start, span_end)] = \
                    parent_probs[batch_idx][mrc_offset: mrc_offset + nwords + 1].cpu().numpy()
                ann_infos[ann_idx].span2parent_tags_idxs[(word_idx, span_start, span_end)] = \
                    parent_tags_idxs[batch_idx][mrc_offset: mrc_offset + nwords + 1].cpu().numpy()
                ann_infos[ann_idx].span2parent_start_scores[(word_idx, span_start, span_end)] = \
                    parent_start_probs[batch_idx][mrc_offset: mrc_offset + nwords + 1].cpu().numpy()
                ann_infos[ann_idx].span2parent_end_scores[(word_idx, span_start, span_end)] = \
                    parent_end_probs[batch_idx][mrc_offset: mrc_offset + nwords + 1].cpu().numpy()


import pickle
# save_file = "/userhome/yuxian/data/tmp_scores.pkl"
save_file = os.path.join(query_dir, f"s2s_scores_{topk}.pkl")
# pickle.dump(ann_infos, open(save_file, "wb"))
ann_infos = pickle.load(open(save_file, "rb"))

# for k in [10, 5, 3, 2, 1]:  # used to tune topk value
for k in [topk]:
    for alpha in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        logger.info(f"Decoding final dependency predictions according to top{k} "
                    f"subtree-scores and subtree-parent-scores using alpha {alpha}")
        metric = AttachmentScores()
        for ann_idx, ann_info in tqdm(list(ann_infos.items())):
            # use only top k subtrees for decoding
            topk_subtrees = set()
            for parent_idx, subtrees in ann_info.span_candidates.items():
                subtrees = sorted(subtrees, key=lambda x: x[-1], reverse=True)
                for start, end, _ in subtrees[: k]:
                    topk_subtrees.add((parent_idx, start, end))
            ann_info = DecodeStruct(
                words=ann_info.words,
                span_candidates=ann_info.span_candidates,
                span2parent_arc_scores={k: v for k, v in ann_info.span2parent_arc_scores.items() if k in topk_subtrees},
                span2parent_start_scores={k: v for k, v in ann_info.span2parent_start_scores.items() if k in topk_subtrees},
                span2parent_end_scores={k: v for k, v in ann_info.span2parent_end_scores.items() if k in topk_subtrees},
                span2parent_tags_idxs=ann_info.span2parent_tags_idxs,
                dep_heads=ann_info.dep_heads,
                dep_tags=ann_info.dep_tags,
                pos_tags=ann_info.pos_tags
            )
            # temporarily skip too long sequence
            # if len(ann_info.dep_heads) > 30:
            #     continue
            # print(ann_info)
            decode_tree = ann_info.decode(arc_alpha=alpha)
            # print(decode_tree)
            # query_dataset.encode_dep_tags(
            gold_heads = ann_info.dep_heads
            gold_labels = [dep_tag2idx[t] for t in ann_info.dep_tags]
            pred_heads = [0] + gold_heads  # add root
            pred_labels = [0] + gold_labels
            if decode_tree.score == -math.inf:
                # warnings.warn(f"failed to decode valid projective tree by top {topk} subtrees for sample {ann_idx}")
                raise ValueError(f"failed to decode valid projective tree by top {k} subtrees for sample {ann_idx}")
                # todo use greedy decoding strategy if this happens
            else:
                for child, parent, tag_idx in decode_tree.dep_arcs:
                    pred_heads[child] = parent
                    pred_labels[child] = tag_idx
            pred_heads = pred_heads[1:]  # remove root
            pred_labels = pred_labels[1:]
            metric.update(
                torch.LongTensor(pred_heads),
                torch.LongTensor(pred_labels),
                torch.LongTensor(gold_heads),
                torch.LongTensor(gold_labels),
                query_model._get_mask_for_eval(
                    mask=torch.BoolTensor([True] * len(gold_labels)),
                    pos_tags=ann_info.pos_tags[: len(gold_labels)]
                )
            )

        print(metric.compute())
