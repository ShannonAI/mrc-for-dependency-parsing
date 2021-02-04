# Biaffine baseline reproduction
For comparison, we provide our re-implementation of [Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)](https://arxiv.org/abs/1611.01734) .

#### Train
See `scripts/biaf/biaf_ptb.sh`
Note that you should change `MODEL_DIR`, `BERT_DIR` and `OUTPUT_DIR` to your own path.
#### Evaluate
See `biaf_evaluate.py`
Note that you should change `HPARAMS` and `CHECKPOINT` to your own path.
