# parser
Codes for dependency parsing, including following models:
1. [Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)](https://arxiv.org/abs/1611.01734) .
1. (developing) Token-to-Token Mrc Dependency parsing
    - [ ] fix child score bug
    - [ ] add biaffine struct/multi-layer
    - [ ] change bracket special token from [SEP] to other [unused]
    - [ ] use mst, fix multi-gpu mst
    - [ ] change type-id of [SEP]

1. (developing) Span-to-Token MRC Dependency parsing
    - [ ] joint training stage1/stage2
    - [ ] 第一阶段，start/end不应该独立
    - [ ] 扫一波proposal的参数(/userhome/yuxian/train_logs/dependency/ptb/biaf/s2t/)
    - [ ] 加label smoothing?
    - [ ] 加反向的分数
    - [ ] 在dp decoding时加入is_subtree的score
    - [ ] 在argparser里支持是否加入is_subtree/反向计算分数
    - [ ] 支持roberta
    - [ ] 做实验，在dp decode时候把gt推进去是否会变好
    - [ ] add dropout/weight-decay of bert
    - [ ] proposal+query都换用roberta

1. (developing) Span-to-Span MRC Dependency parsing
    - [ ] 支持roberta(proposal+query)
    - [ ] add backward score
    - [ ] 回头检查t2t低的原因，比如改decoder的结构变为biaffine
    - [ ] 后面多加一层transformer(看看别人代码怎么搞的)

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


### 4. Span-to-Span MRC (Developing)
#### Train
* proposal model: `scripts/s2s/x.sh`
* query model: `scripts/s2t/x.sh`

#### Evaluate
See `parser/s2s_evaluate_dp.py`

## TODO
- [ ] refactor config/argparser hyper-parameters 
- [ ] refactor functions that use roberta
- [ ] refactor usage of `from_pretrained` of pretrained bert/roberta, move it into model 
- [ ] ddp sampler may cause training data in same order between different epoch