export PYTHONPATH="$PWD"

DATA_DIR="/userhome/yuxian/data/parser/ptb3_parser"
BERT_DIR="/userhome/yuxian/data/bert/bert-large-uncased-wwm"

python parser/trainer.py \
--data_dir $DATA_DIR \
--data_format 'conllx' \
--pos_dim 100 \
--bert_dir $BERT_DIR \
--workers 4 \
--gpus="0,1" \
--accelerator 'ddp' \
--precision 16 \
--batch_size 16 \
--accumulate_grad_batches 4 \
--lr 2e-5 \
--ignore_punct
