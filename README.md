# parser
Codes for dependency parsing, including following models:
1. [Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)](https://arxiv.org/abs/1611.01734) .
1. (developing) Token-to-Token Mrc Dependency parsing
    - [x] try better init
    - [ ] fix child score bug
    - [ ] add biaffine struct/multi-layer
    - [ ] change bracket special token from [SEP] to other [unused]
    - [ ] use mst, fix multi-gpu mst
    - [ ] change type-id of [SEP]

1. (developing) Span-to-Token MRC Dependency parsing
    - [ ] allow no answer in second stage?
    - [x] add token-span query
    - [ ] evaluate as pipeline
    - [ ] joint training stage1/stage2
    - [ ] 第二阶段，word的parent一定在span之外。inference的时候可以加上限制
    - [ ] 第一阶段，start/end不应该独立

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
See `scripts/t2t/train.sh`
Note that you should change `MODEL_DIR` and `BERT_DIR` to your own path.
#### Evaluate
See `parser/t2t_evaluate.py`

### 3. Span-to-Token MRC (Developing)
#### Train
* proposal model: `scripts/s2t/pengcheng_ptb_proposal.sh`
* query model: `scripts/s2t/pengcheng_ptb_query.sh`

#### Evaluate
See `parser/s2t_evaluate_dp.py`

## TODO
- [x] refactor dataset reader, add base class and put collate_fn in corresponding file.
- [ ] refactor config/argparser hyper-parameters 
