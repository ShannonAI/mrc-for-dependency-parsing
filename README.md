# parser
Codes for dependency parsing, including following models:
1. [Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)](https://arxiv.org/abs/1611.01734) .
1. Span-to-Span MRC Dependency parsing


## Requirements
* python>=3.6
* `pip install -r requirements.txt`

We build our project on [pytorch-lightning.](https://github.com/PyTorchLightning/pytorch-lightning)
If you want to know more about the arguments used in our training scripts, please 
refer to [pytorch-lightning documentation.](https://pytorch-lightning.readthedocs.io/en/latest/)

## Dataset
We follow [this repo](https://github.com/hankcs/TreebankPreprocessing) for PTB/CTB data preprocessing.

We follow [<Stack-Pointer Networks for Dependency Parsing>](https://arxiv.org/abs/1805.01087) to preprocess data in UD dataset.

## Models
For PTB, we use [RoBerTa-Large](https://huggingface.co/roberta-large)

For CTB, we use [RoBERTa-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext)

For UD, we use [XLM-RoBERTa-large](https://huggingface.co/xlm-roberta-large)

## Reproduction
### 1. Deep Biaffine Attention
#### Train
See `scripts/biaf/biaf_ptb.sh`
Note that you should change `MODEL_DIR`, `BERT_DIR` and `OUTPUT_DIR` to your own path.
#### Evaluate
See `biaf_evaluate.py`
Note that you should change `HPARAMS` and `CHECKPOINT` to your own path.

### 2. Span-to-Span MRC
#### Train
* proposal model: `scripts/s2s/*/proposal.sh`
* s2s model: `scripts/s2t/*/s2s.sh`

Note that you should change `MODEL_DIR`, `BERT_DIR` and `OUTPUT_DIR` to your own path.

#### Evaluate
Choose the best span-proposal model and s2s model according to topk accuracy and UAS respectively, and run
```
parser/s2s_evaluate_dp.py \
--proposal_hparams <your best proposal model hparams file> \
--proposal_ckpt <your best proposal model ckpt> \
--s2s_ckpt <your best s2s query model hparams file> \
--s2s_hparams <your best s2s query model ckpt> \
--topk <use topk spans for evaluating>
```

