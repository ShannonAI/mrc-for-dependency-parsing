export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="/data/nfsdata2/nlp_application/datasets/treebank/LDC99T42/ptb3_parser/"
BERT_DIR="/data/nfsdata2/nlp_application/models/bert/bert-large-uncased-whole-word-masking"


DROPOUT=0.3
LR=1e-3
LAYER=3

OUTPUT_DIR="/data/yuxian/train_logs/dependency/ptb/lstm"
mkdir -p $OUTPUT_DIR

python parser/biaf_trainer.py \
--default_root_dir $OUTPUT_DIR \
--data_dir $DATA_DIR \
--data_format 'conllx' \
--pos_dim 100 \
--freeze_bert \
--bert_dir $BERT_DIR \
--additional_layer $LAYER \
--biaf_dropout $DROPOUT \
--workers 1 \
--gpus="1," \
--accelerator "ddp" \
--precision 32 \
--batch_size 128 \
--accumulate_grad_batches 1 \
--lr $LR \
--gradient_clip_val=1.0 \
--ignore_punct \
--max_epochs 100


#--warmup_steps 0 \
#--scheduler linear_decay \