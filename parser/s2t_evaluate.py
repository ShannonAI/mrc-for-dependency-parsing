# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/1/8 10:13
@desc: Evaluate Span-to-Token MRC dependency parser as a two-stage pipeline

"""

import os

import torch
from allennlp.nn.util import get_range_vector, get_device_of
from tqdm import tqdm

from parser.metrics import AttachmentScores
from parser.s2t_proposal_trainer import MrcS2TProposal
from parser.s2t_query_trainer import MrcS2TQuery
from parser.utils.logger import get_logger

logger = get_logger(__name__)

# data_path = ""
# bert_dir = "/data/nfsdata2/nlp_application/models/bert/bert-large-uncased-whole-word-masking"

proposal_dir = "/userhome/yuxian/train_logs/dependency/ptb/biaf/s2t/proposal"
proposal_hparams = os.path.join(proposal_dir, "lightning_logs/version_0/hparams.yaml")
# proposal_dir = "/data/nfsdata2/yuxian/share/parser/proposal/"
# proposal_hparams = os.path.join(proposal_dir, "version_0/hparams.yaml")
proposal_ckpt = os.path.join(proposal_dir, "epoch=9.ckpt")

query_dir = "/userhome/yuxian/train_logs/dependency/ptb/s2t/query_newsep"
query_hparams = os.path.join(query_dir, "lightning_logs/version_1/hparams.yaml")
# query_dir = "/data/nfsdata2/yuxian/share/parser/query"
# query_hparams = os.path.join(query_dir, "version_1/hparams.yaml")
query_ckpt = os.path.join(query_dir, "epoch=4.ckpt")

proposal_model = MrcS2TProposal.load_from_checkpoint(
    checkpoint_path=proposal_ckpt,
    hparams_file=proposal_hparams,
    map_location=None,
    batch_size=128,
    max_length=128,
    workers=4,
    group_sample=False,
    # bert_dir="/data/nfsdata2/nlp_application/models/bert/bert-large-uncased-whole-word-masking",
    # data_dir="/data/nfsdata2/nlp_application/datasets/treebank/LDC99T42/ptb3_parser/",
    gpus="1"  # deactivate distributed sampler
)
proposal_model.cuda()
proposal_model.eval()


query_model = MrcS2TQuery.load_from_checkpoint(
    checkpoint_path=query_ckpt,
    hparams_file=query_hparams,
    map_location=None,
    batch_size=128,
    max_length=128,
    workers=4,
    group_sample=False,
    gpus="1",
    # bert_dir="/data/nfsdata2/nlp_application/models/bert/bert-large-uncased-whole-word-masking",
    # data_dir="/data/nfsdata2/nlp_application/datasets/treebank/LDC99T42/ptb3_parser/",
)
query_model.cuda()
query_model.eval()

assert proposal_model.args.pos_tags + ["sep_pos"] == query_model.args.pos_tags
assert proposal_model.args.dep_tags == query_model.args.dep_tags


def tocuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()


with torch.no_grad():

    # ------
    # Stage 1: extract top1 subtree spans for every token.
    # ------
    proposals = {}  # ann_idx to span_candidates
    proposal_loader = proposal_model.test_dataloader()
    logger.info("Extract subtree proposal ...")
    for batch in tqdm(proposal_loader):
        tocuda(batch)
        token_ids, type_ids, offsets, wordpiece_mask, pos_tags, word_mask, subtree_spans, meta_data = (
            batch["token_ids"], batch["type_ids"], batch["offsets"], batch["wordpiece_mask"],
            batch["pos_tags"], batch["word_mask"], batch["subtree_spans"], batch["meta_data"]
        )
        # [bsz, seq_len, seq_len]
        span_start_scores, span_end_scores = proposal_model(
            token_ids, type_ids, offsets, wordpiece_mask,
            pos_tags, word_mask
        )

        # find top1 subtree spans that start <= end, where score(start, end) = score(start) + score(end)
        start_idxs = torch.argmax(span_start_scores, dim=-1).cpu().numpy()
        end_idxs = torch.argmax(span_end_scores, dim=-1).cpu().numpy()

        ann_idxs = meta_data["ann_idx"]
        for idx in range(len(ann_idxs)):
            ann_idx = ann_idxs[idx]
            proposals[ann_idx] = [
                start_idxs[idx],  # [seq_len]
                end_idxs[idx],  # [seq_len]
            ]

    # ------
    # Stage 2: choose best parent for each subtree span
    # ------
    logger.info("Finding parents and evaluating according to subtree proposal")
    parent_scores = {}
    query_loader = query_model.test_dataloader()
    metric = AttachmentScores().cuda()
    for k in [-1, 0]:
        if k >= 0:
            logger.info(f"Finding parents and evaluating according to top1 subtree proposal ...")
            subtree_spans = []
            for ann_idx in range(len(proposals)):
                candidate = proposals[ann_idx]
                starts = candidate[0].tolist()  # seq_len
                ends = candidate[1].tolist()  # seq_len
                subtree_spans.append(list(zip(starts, ends)))
            query_loader.dataset.subtree_spans = subtree_spans
        else:
            logger.info(f"Finding parents and evaluating according to groundtruth subtree proposal ...")
        metric = AttachmentScores().cuda()
        for batch in tqdm(query_loader):
            tocuda(batch)
            token_ids, type_ids, offsets, wordpiece_mask, pos_tags, word_mask, mrc_mask, meta_data, parent_idxs, parent_tags = (
                batch["token_ids"], batch["type_ids"], batch["offsets"], batch["wordpiece_mask"],
                batch["pos_tags"], batch["word_mask"], batch["mrc_mask"], batch["meta_data"],
                batch["parent_idxs"], batch["parent_tags"]
            )
            parent_probs, parent_tag_probs, parent_arc_nll, parent_tag_nll = query_model(
                token_ids, type_ids, offsets, wordpiece_mask,
                pos_tags, word_mask, mrc_mask, parent_idxs, parent_tags
            )

            eval_mask = query_model._get_mask_for_eval(mask=word_mask, pos_tags=pos_tags)
            bsz = parent_probs.size(0)
            # [bsz]
            batch_range_vector = get_range_vector(bsz, get_device_of(parent_tags))
            eval_mask = eval_mask[batch_range_vector, parent_idxs]  # [bsz]
            # [bsz]
            pred_positions = parent_probs.argmax(1)
            metric.update(
                pred_positions.unsqueeze(-1),  # [bsz, 1]
                parent_tag_probs[batch_range_vector, pred_positions].argmax(1).unsqueeze(-1),  # [bsz, 1]
                parent_idxs.unsqueeze(-1),  # [bsz, 1]
                parent_tags.unsqueeze(-1),  # [bsz, 1]
                eval_mask.unsqueeze(-1)  # [bsz, 1]
            )

        print(metric.compute())
