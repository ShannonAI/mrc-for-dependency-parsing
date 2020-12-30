export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="/userhome/yuxian/data/parser/ptb3_parser"
BERT_DIR="/userhome/yuxian/data/bert/bert-large-uncased-wwm"


DROPOUT=0.3
LR=1e-3
LAYER=3

OUTPUT_DIR="/userhome/yuxian/train_logs/dependency/ptb/bert_lstm_freeze_yuxianinit_nofuse_noheadinit_withallen"
mkdir -p $OUTPUT_DIR

python parser/biaf_trainer.py \
--default_root_dir $OUTPUT_DIR \
--data_dir $DATA_DIR \
--data_format 'conllx' \
--pos_dim 100 \
--bert_dir $BERT_DIR \
--freeze_bert \
--additional_layer $LAYER \
--additional_layer_dim 800 \
--biaf_dropout $DROPOUT \
--workers 2 \
--gpus="0,1" \
--accelerator "ddp" \
--precision 16 \
--batch_size 64 \
--accumulate_grad_batches 1 \
--lr $LR \
--gradient_clip_val=5.0 \
--ignore_punct \
--max_epochs 100
