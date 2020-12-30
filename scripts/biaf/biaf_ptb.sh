export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="/data/nfsdata2/nlp_application/datasets/treebank/LDC99T42/ptb3_parser/"
BERT_DIR="/data/nfsdata2/nlp_application/models/bert/bert-base-uncased"


DROPOUT=0.3
LR=2e-5
LAYER=2

OUTPUT_DIR="/data/yuxian/train_logs/dependency/ptb/debug"
mkdir -p $OUTPUT_DIR

# todo 尝试freeze_bert
python parser/biaf_trainer.py \
--default_root_dir $OUTPUT_DIR \
--data_dir $DATA_DIR \
--data_format 'conllx' \
--pos_dim 100 \
--freeze_bert \
--bert_dir $BERT_DIR \
--additional_layer $LAYER \
--biaf_dropout $DROPOUT \
--workers 0 \
--gpus="1," \
--precision 32 \
--batch_size 16 \
--accumulate_grad_batches 8 \
--lr $LR \
--gradient_clip_val=1.0 \
--ignore_punct \
--warmup_steps 0 \
--scheduler linear_decay \
--max_epochs 100
