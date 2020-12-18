# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: trainer
@time: 2020/12/17 20:13
@desc:

"""

import os
import argparse
import pytorch_lightning as pl
import torch
from allennlp.nn.util import get_range_vector, get_device_of
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import BertConfig

from parser.data.collate import collate_dependency_data
from parser.data.dependency_t2t_reader import DependencyT2TDataset
from parser.metrics import *
from parser.models import *
from parser.utils.get_parser import get_parser


class MrcDependency(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        bert_config = BertConfig.from_pretrained(args.bert_dir)
        self.model_config = BertMrcDependencyConfig(
            pos_tags=args.pos_tags,
            dep_tags=args.dep_tags,
            pos_dim=args.pos_dim,
            additional_layer=args.additional_layer,
            mrc_dropout=args.mrc_dropout,
            **bert_config.__dict__
        )
        self.model = BiaffineDependencyT2TParser.from_pretrained(args.bert_dir, config=self.model_config)

        if args.freeze_bert:
            for param in self.model.bert.parameters():
                param.requires_grad = False

        self.train_stat = AttachmentScores(ignore_classes=args.ignore_pos_tags)  # todo ignore classes?
        self.val_stat = AttachmentScores(ignore_classes=args.ignore_pos_tags) if not self.args.use_mst else AttachmentScores  # todo implement MST
        self.ignore_pos_tags = list(args.ignore_pos_tags)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--pos_dim", type=int, default=0, help="pos tag dimension")
        parser.add_argument("--mrc_dropout", type=float, default=0.0, help="mrc dropout")
        parser.add_argument("--additional_layer", type=int, default=0, help="additional encoder layer")
        parser.add_argument("--ignore_punct", action="store_true", help="ignore punct pos when evaluating")
        parser.add_argument("--freeze_bert", action="store_true", help="freeze bert embedding")
        return parser

    def forward(self, token_ids, type_ids, offsets, wordpiece_mask, span_idx, span_tag, pos_tags, word_mask, mrc_mask):
        return self.model(
            token_ids, type_ids, offsets, wordpiece_mask, span_idx,
            span_tag, pos_tags, word_mask, mrc_mask
        )

    def common_step(self, batch, phase="train"):
        token_ids, type_ids, offsets, wordpiece_mask, span_idx, span_tag, pos_tags, word_mask, mrc_mask, meta_data = (
            batch["token_ids"], batch["type_ids"], batch["offsets"], batch["wordpiece_mask"], batch["span_idx"],
            batch["span_tag"], batch["pos_tags"], batch["word_mask"], batch["mrc_mask"], batch["meta_data"]
        )
        parent_probs, parent_tag_probs, arc_nll, tag_nll = self(
            token_ids, type_ids, offsets, wordpiece_mask, span_idx,
            span_tag, pos_tags, word_mask, mrc_mask
        )
        loss = arc_nll + tag_nll
        eval_mask = self._get_mask_for_eval(mask=word_mask, pos_tags=pos_tags)

        if phase == "train" or not self.args.use_mst:
            bsz = span_idx.size(0)
            # [bsz]
            batch_range_vector = get_range_vector(bsz, get_device_of(span_idx))
            # [bsz]
            gold_positions = span_idx[:, 0]
            pred_positions = parent_probs.argmax(1)
            metric_name = f"{phase}_stat"
            metric = getattr(self, metric_name)
            metric.update(
                pred_positions.unsqueeze(-1),  # [bsz, 1]
                parent_tag_probs[batch_range_vector, pred_positions].argmax(1).unsqueeze(-1),  # [bsz, 1]
                gold_positions.unsqueeze(-1),  # [bsz, 1]
                span_tag.unsqueeze(-1),  # [bsz, 1]
                eval_mask[batch_range_vector, pred_positions].unsqueeze(-1)
            )
        else:
            metric = self.val_stat
            self.val_stat(
                        [x["ann_idx"] for x in meta_data],
                        [len(x["words"]) for x in meta_data],
                        parent_probs,
                        parent_tag_probs,
                        span_idx,
                        span_tag,
                        # evaluation_mask,
                    )

        self.log(f'{phase}_loss', loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, phase="train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, phase="val")

    def log_on_epoch_end(self, phase="train"):
        metric_name = f"{phase}_stat"
        metric = getattr(self, metric_name)
        metrics = metric.compute()
        for sub_metric, metric_value in metrics.items():
            self.log(f"{phase}_{sub_metric}", metric_value)

    def validation_epoch_end(self, outputs):
        self.log_on_epoch_end("val")

    def training_epoch_end(self, outputs):
        self.log_on_epoch_end("train")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.85,
                                                                  min_lr=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_LAS'
        }

    def _get_mask_for_eval(
        self, mask: torch.BoolTensor, pos_tags: torch.LongTensor
    ) -> torch.LongTensor:
        """
        Dependency evaluation excludes words are punctuation.
        Here, we create a new mask to exclude word indices which
        have a "punctuation-like" part of speech tag.

        # Parameters

        mask : `torch.BoolTensor`, required.
            The original mask.
        pos_tags : `torch.LongTensor`, required.
            The pos tags for the sequence.

        # Returns

        A new mask, where any indices equal to labels
        we should be ignoring are masked.
        """
        new_mask = mask.detach()
        for label in self.ignore_pos_tags:
            label_mask = pos_tags.eq(label)
            new_mask = new_mask & ~label_mask
        return new_mask


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = get_parser()
    parser = MrcDependency.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------

    train_dataset = DependencyT2TDataset(
        file_path=os.path.join(args.data_dir, f"train.{args.data_format}"),
        bert=args.bert_dir
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_dependency_data
    )
    val_dataset = DependencyT2TDataset(
        file_path=os.path.join(args.data_dir, f"dev.{args.data_format}"),
        bert=args.bert_dir
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_dependency_data
    )
    # todo: check if some label in test/valid does not appear in train
    args.pos_tags = train_dataset.pos_tags
    args.dep_tags = train_dataset.dep_tags
    args.ignore_pos_tags = train_dataset.ignore_pos_tags if args.ignore_punct else set()

    # ------------
    # model
    # ------------
    model = MrcDependency(args)
    print(model.model)

    # load pretrained_model
    if args.pretrained:
        model.load_state_dict(
            torch.load(args.pretrained,map_location=torch.device('cpu'))["state_dict"]
        )
    # ------------
    # training
    # ------------
    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='val_UAS',
        dirpath=args.default_root_dir,
        save_top_k=10,
        save_last=True,
        mode='max',
        verbose=True
    )
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    cli_main()
