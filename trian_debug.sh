export PYTHONPATH="$PWD"

DATA_DIR="/home/mengyuxian/shannon_parser/parser/data"
BERT_DIR="/data/nfsdata2/nlp_application/models/bert/bert-base-uncased"

python parser/trainer.py \
--data_dir $DATA_DIR \
--data_format 'conllu' \
--pos_dim 100 \
--additional_layer 3 \
--mrc_dropout 0.0 \
--bert_dir $BERT_DIR \
--workers 0 \
--gpus="3," \
--accelerator 'ddp' \
--precision 32 \
--batch_size 32 \
--lr 2e-5 \
--ignore_punct