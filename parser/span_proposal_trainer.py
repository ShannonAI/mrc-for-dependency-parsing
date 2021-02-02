# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@time: 2020/12/17 20:13
@desc:

"""

import argparse
import os
from typing import Dict, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler
from transformers import AdamW, AutoConfig

from parser.callbacks import *
from parser.data.samplers import GroupedSampler
from parser.data.span_proposal_dataset import SubTreeProposalDataset, collate_subtree_data
from parser.metrics import *
from parser.models import *
from parser.utils.get_parser import get_parser

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MrcSpanProposal(pl.LightningModule):

    def __init__(self, args: Union[Dict, argparse.Namespace]):
        super().__init__()

        if not isinstance(args, argparse.Namespace):
            # eval mode
            assert isinstance(args, dict)
            args = argparse.Namespace(**args)

        # compute other fields according to args
        train_dataset = SubTreeProposalDataset(
            file_path=os.path.join(args.data_dir, f"{args.data_prefix}train.{args.data_format}"),
            # file_path=os.path.join(args.data_dir, f"{args.data_prefix}sample.{args.data_format}"),
            bert=args.bert_dir
        )

        # save these information to args to convene evaluation.
        args.pos_tags = train_dataset.pos_tags
        args.dep_tags = train_dataset.dep_tags
        args.ignore_pos_tags = train_dataset.ignore_pos_tags if args.ignore_punct else set()
        args.num_gpus = len([x for x in str(args.gpus).split(",") if x.strip()]) if "," in args.gpus else int(args.gpus)
        args.t_total = (len(train_dataset) // (args.accumulate_grad_batches * args.num_gpus) + 1) * args.max_epochs

        self.save_hyperparameters(args)
        self.args = args

        bert_config = AutoConfig.from_pretrained(args.bert_dir)

        self.model_config = SpanProposalConfig(
            bert_config=bert_config,
            pos_tags=args.pos_tags,
            dep_tags=args.dep_tags,
            pos_dim=args.pos_dim,
            additional_layer=args.additional_layer,
            additional_layer_dim=args.additional_layer_dim,
            additional_layer_type=args.additional_layer_type,
            mrc_dropout=args.mrc_dropout
        )
        self.model = SpanProposal(bert_dir=args.bert_dir, config=self.model_config)

        if args.freeze_bert:
            for param in self.model.bert.parameters():
                param.requires_grad = False

        # evaluate topN acc, for every 1<= N <=k
        self.acc_topk = 5
        self.train_acc = AllTopkAccuracy(self.acc_topk)
        self.val_acc = AllTopkAccuracy(self.acc_topk)
        self.test_acc = AllTopkAccuracy(self.acc_topk)

        self.ignore_pos_tags = list(args.ignore_pos_tags)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--pos_dim", type=int, default=0, help="pos tag dimension")
        parser.add_argument("--mrc_dropout", type=float, default=0.0, help="mrc dropout")
        parser.add_argument("--additional_layer", type=int, default=0, help="additional encoder layer")
        parser.add_argument("--additional_layer_dim", type=int, default=0, help="additional encoder layer dim")
        parser.add_argument("--additional_layer_type", type=str,
                            choices=["lstm", "transformer"], default="lstm",
                            help="additional layer type")
        parser.add_argument("--ignore_punct", action="store_true", help="ignore punct pos when evaluating")
        parser.add_argument("--freeze_bert", action="store_true", help="freeze bert embedding")
        parser.add_argument("--scheduler", default="plateau", choices=["plateau", "linear_decay"],
                            help="scheduler type")
        parser.add_argument("--final_div_factor", type=float, default=1e4,
                            help="final div factor of linear decay scheduler")
        return parser

    def forward(self, token_ids, type_ids, offsets, wordpiece_mask,
                pos_tags, word_mask, subtree_spans=None):
        return self.model(
            token_ids, type_ids, offsets, wordpiece_mask,
            pos_tags, word_mask, subtree_spans
        )

    def common_step(self, batch, phase="train"):
        token_ids, type_ids, offsets, wordpiece_mask, pos_tags, word_mask, subtree_spans, meta_data = (
            batch["token_ids"], batch["type_ids"], batch["offsets"], batch["wordpiece_mask"],
            batch["pos_tags"], batch["word_mask"], batch["subtree_spans"], batch["meta_data"]
        )
        span_start_scores, span_end_scores, loss = self(
            token_ids, type_ids, offsets, wordpiece_mask,
            pos_tags, word_mask, subtree_spans
        )
        bsz, seq_len, _ = span_start_scores.size()
        metric = getattr(self, f"{phase}_acc")
        metric_mask = ~(subtree_spans[:, :, 0].view(-1) == -100)
        metric.update(
            preds=span_start_scores.view(bsz*seq_len, -1)[metric_mask],
            target=subtree_spans[:, :, 0].view(-1)[metric_mask]
        )
        metric.update(
            preds=span_end_scores.view(bsz * seq_len, -1)[metric_mask],
            target=subtree_spans[:, :, 1].view(-1)[metric_mask]
        )

        self.log(f'{phase}_loss', loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, phase="train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, phase="val")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, phase="test")

    def log_on_epoch_end(self, phase="train"):
        metric_name = f"{phase}_acc"
        metric = getattr(self, metric_name)
        metrics = metric.compute()
        for sub_metric, metric_value in metrics.items():
            self.log(f"{phase}_{sub_metric}", metric_value)

    def training_epoch_end(self, outputs):
        self.log_on_epoch_end("train")

    def validation_epoch_end(self, outputs):
        self.log_on_epoch_end("val")

    def test_epoch_end(self, outputs):
        self.log_on_epoch_end("test")

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        no_optimize = ["bert"] if self.args.freeze_bert else []
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
                           and not any(nd in n for nd in no_optimize)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                           and not any(nd in n for nd in no_optimize)],
                "weight_decay": 0.0,
            },
        ]

        # todo add betas to arguments and tune it
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=1e-8,
                          betas=(0.9, 0.999))

        if self.args.scheduler == "plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.85,
                                                                      min_lr=1e-6)

            return {
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
                'monitor': f'val_top{self.acc_topk}_f1'
            }
        # linear decay scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.args.lr, pct_start=float(self.args.warmup_steps / self.args.t_total),
            final_div_factor=self.args.final_div_factor,
            total_steps=self.args.t_total, anneal_strategy='linear'
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

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

    def get_dataloader(self, split="train", shuffle=True):
        dataset = SubTreeProposalDataset(
            file_path=os.path.join(self.args.data_dir, f"{self.args.data_prefix}{split}.{self.args.data_format}"),
            pos_tags=self.args.pos_tags,
            dep_tags=self.args.dep_tags,
            bert=self.args.bert_dir
        )
        if self.args.num_gpus <= 1:
            sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        else:
            sampler = DistributedSampler(dataset, shuffle=shuffle)

        if self.args.group_sample:
            groups, counts = dataset.get_groups()
            sampler = GroupedSampler(
                dataset=dataset,
                sampler=sampler,
                group_ids=groups,
                batch_size=self.args.batch_size,
                counts=counts
            )
        loader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=collate_subtree_data
        )

        return loader

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train", shuffle=True)
        # return self.get_dataloader("sample", shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("dev", shuffle=False)
        # return self.get_dataloader("sample", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", shuffle=False)
        # return self.get_dataloader("sample", shuffle=False)


def main():
    pl.seed_everything(1234)
    # ------------
    # args
    # ------------
    parser = get_parser()
    parser = MrcSpanProposal.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # model
    # ------------
    model = MrcSpanProposal(args)

    # load pretrained_model
    if args.pretrained:
        model.load_state_dict(
            torch.load(args.pretrained, map_location=torch.device('cpu'))["state_dict"]
        )

    # call backs
    checkpoint_callback = ModelCheckpoint(
        monitor=f'val_top1_acc',
        dirpath=args.default_root_dir,
        save_top_k=10,
        save_last=True,
        mode='max',
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    print_model = ModelPrintCallback(print_modules=["model"])
    callbacks = [checkpoint_callback, lr_monitor, print_model]
    if args.freeze_bert:
        callbacks.append(EvalCallback(["model.bert"]))

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        replace_sampler_ddp=False
    )

    trainer.fit(model)


if __name__ == '__main__':
    main()
