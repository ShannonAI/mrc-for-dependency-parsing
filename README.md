# parser
Codes for dependency parsing, including following models:
1. [Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)](https://arxiv.org/abs/1611.01734) .
1. (developing) Mrc Dependency parsing

## Requirements
* python>=3.6
* `pip install -r requirements.txt`

## Reproduction
### 1. Deep Biaffine Attention
#### Train
See `scripts/biaf/biaf_ptb.sh`
Note that you should change `MODEL_DIR` and `BERT_DIR` to your own path.
#### Evaluate
See `biaf_evaluate.py`
Note that you should change `HPARAMS` and `CHECKPOINT` to your own path.

### 2. Token-to-Token MRC (Developing)
#### Train
See `scripts/t2t/train_ptb_freeze.sh`
Note that you should change `MODEL_DIR` and `BERT_DIR` to your own path.
