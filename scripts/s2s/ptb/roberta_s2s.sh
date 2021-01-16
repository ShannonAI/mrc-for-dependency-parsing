export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="/userhome/yuxian/data/parser/ptb3_parser"
BERT_DIR="/userhome/yuxian/data/bert/roberta-large"
BERT_TYPE="roberta"

# hyper-params
DROPOUT=0.3
LR=1e-5
bert_dropout=0.1
DECAY=0.0
accumulate=20
WARMUP=300

# save directory
OUTPUT_DIR="/userhome/yuxian/train_logs/dependency/ptb/s2s/roberta_new_bertdrop${bert_dropout}_decay${DECAY}_accumulate${accumulate}"
mkdir -p $OUTPUT_DIR

python parser/s2s_query_trainer.py \
--precision 16 \
--default_root_dir $OUTPUT_DIR \
--data_dir $DATA_DIR \
--data_format 'conllx' \
--pos_dim 100 \
--bert_dir $BERT_DIR --bert_name $BERT_TYPE \
--additional_layer_dim 1124 \
--mrc_dropout $DROPOUT \
--workers 24 \
--gpus="0,1,2,3,4,5,6,7" \
--accelerator "ddp" \
--batch_size 16 \
--accumulate_grad_batches $accumulate \
--lr $LR \
--gradient_clip_val=1.0 \
--ignore_punct \
--max_epochs 10 \
--group_sample \
--bert_dropout $bert_dropout --weight_decay $DECAY \
--scheduler "linear_decay" --warmup_steps $WARMUP --final_div_factor 20
