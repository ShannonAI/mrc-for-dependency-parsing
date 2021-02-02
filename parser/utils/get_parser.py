# encoding: utf-8


import argparse


def get_parser() -> argparse.ArgumentParser:
    """
    return basic arg parser
    """
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_name", type=str, default="bert",
                        choices=["bert", "roberta"])
    parser.add_argument("--data_dir", type=str, required=True, help="data dir")
    parser.add_argument("--data_prefix", type=str, default="",
                        help="data prefix. For example, in UD treebank, we can use bg_btb-ud- as prefix for bg dataset")
    parser.add_argument("--data_format", type=str, choices=["conllu", "conllx"],
                        default="conllu", help="data format")
    parser.add_argument("--bert_dir", type=str, required=True, help="bert dir")
    parser.add_argument("--use_mst", action="store_true", help="use mst in decoding")
    parser.add_argument("--pretrained", default="", type=str, help="pretrained dependency checkpoint path")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="warmup steps used for scheduler.")
    parser.add_argument("--group_sample", action="store_true",
                        help="use group sampler, which could accelerate training")
    parser.add_argument("--max_words", type=int, default=0,
                        help="max number of words of a sample that used in training.")
    return parser
