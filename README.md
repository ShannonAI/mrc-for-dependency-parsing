# parser
Codes for dependency parsing, including following models:
1. [Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)](https://arxiv.org/abs/1611.01734) .
1. (developing) Token-to-Token Mrc Dependency parsing
    - [ ] fix child score bug
    - [ ] add biaffine struct/multi-layer
    - [ ] change bracket special token from [SEP] to other [unused]
    - [ ] use mst, fix multi-gpu mst
    - [ ] change type-id of [SEP]

1. (developing) Span-to-Span MRC Dependency parsing
    - [ ] joint training stage1/stage2
    - [ ] 后面多加1-n层transformer
    - [ ] 加label smoothing
    - [ ] 做实验，在dp decode时候把gt推进去是否会变好

## Requirements
* python>=3.6
* `pip install -r requirements.txt`

## Dataset
We followed [this repo](https://github.com/hankcs/TreebankPreprocessing) for PTB/CTB data preprocessing.


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
* proposal model: `scripts/s2s/*/proposal.sh`
* s2s model: `scripts/s2t/*/s2s.sh`

#### Evaluate
Choose the best proposal model and s2s model independently, and run
```
parser/s2s_evaluate_dp.py \
--proposal_hparams <your best proposal model hparams file> \
--proposal_ckpt <your best proposal model ckpt> \
--s2s_ckpt <your best s2s query model hparams file> \
--s2s_hparams <your best s2s query model ckpt> \
--topk <use topk spans for evaluating>
```


## TODO
- [ ] refactor config/argparser hyper-parameters 
- [ ] refactor functions that use roberta
- [ ] refactor usage of `from_pretrained` of pretrained bert/roberta, move it into model 
- [ ] ddp sampler may cause training data in same order between different epoch
