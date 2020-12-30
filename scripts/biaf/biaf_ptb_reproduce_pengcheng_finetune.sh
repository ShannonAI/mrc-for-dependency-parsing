export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="/userhome/yuxian/data/parser/ptb3_parser"
BERT_DIR="/userhome/yuxian/data/bert/bert-large-uncased-wwm"


DROPOUT=0.3
LR=2e-5
LAYER=3

OUTPUT_DIR="/userhome/yuxian/train_logs/dependency/ptb/bert_lstm_finetune"
mkdir -p $OUTPUT_DIR

# --freeze_bert \
python parser/biaf_trainer.py \
--default_root_dir $OUTPUT_DIR \
--data_dir $DATA_DIR \
--data_format 'conllx' \
--pos_dim 100 \
--bert_dir $BERT_DIR \
--additional_layer $LAYER \
--biaf_dropout $DROPOUT \
--workers 2 \
--gpus="0,1" \
--accelerator "ddp" \
--precision 16 \
--batch_size 32 \
--accumulate_grad_batches 2 \
--lr $LR \
--gradient_clip_val=1.0 \
--ignore_punct \
--max_epochs 100
