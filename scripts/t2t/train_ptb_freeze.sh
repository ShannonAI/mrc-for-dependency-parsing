export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="/data/nfsdata2/nlp_application/datasets/treebank/LDC99T42/ptb3_parser/"
#BERT_DIR="/data/nfsdata2/nlp_application/models/bert/bert-large-uncased-whole-word-masking"
BERT_DIR="/data/nfsdata2/nlp_application/models/bert/bert-base-uncased"

MRC_DROPOUT=0.3
LR=1e-3
LAYER=3

OUTPUT_DIR="/data/yuxian/train_logs/dependency/ptb/20210103/t2t/freeze_${LAYER}l_bs128_lr${LR}_dropout${MRC_DROPOUT}"


python parser/t2t_trainer.py \
--default_root_dir $OUTPUT_DIR \
--data_dir $DATA_DIR \
--data_format 'conllx' \
--pos_dim 100 \
--bert_dir $BERT_DIR \
--freeze_bert \
--additional_layer $LAYER \
--additional_layer_dim 800 \
--mrc_dropout $MRC_DROPOUT \
--workers 8 \
--gpus="1," \
--accelerator 'ddp' \
--precision 16 \
--batch_size 64 \
--accumulate_grad_batches 2 \
--lr $LR \
--gradient_clip_val=1.0 \
--ignore_punct \
--max_epochs 60 \
--group_sample
