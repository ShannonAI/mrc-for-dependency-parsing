export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="/userhome/yuxian/data/parser/ptb3_parser"
BERT_DIR="/userhome/yuxian/data/bert/bert-large-uncased-wwm"

# hyper-params
DROPOUT=0.3
LR=1e-5

OUTPUT_DIR="/userhome/yuxian/train_logs/dependency/ptb/s2t/query"
mkdir -p $OUTPUT_DIR

python parser/s2t_query_trainer.py \
--precision 16 \
--default_root_dir $OUTPUT_DIR \
--data_dir $DATA_DIR \
--data_format 'conllx' \
--pos_dim 100 \
--bert_dir $BERT_DIR \
--additional_layer_dim 1124 \
--mrc_dropout $DROPOUT \
--workers 12 \
--gpus="0,1,2,3" \
--accelerator "ddp" \
--batch_size 16 \
--accumulate_grad_batches 40 \
--lr $LR \
--gradient_clip_val=1.0 \
--ignore_punct \
--max_epochs 40 \
--group_sample \
--scheduler "linear_decay" --warmup_steps 300 --final_div_factor 10




# 8card
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="/userhome/yuxian/data/parser/ptb3_parser"
BERT_DIR="/userhome/yuxian/data/bert/bert-large-uncased-wwm"

# hyper-params
DROPOUT=0.3
LR=1e-5

OUTPUT_DIR="/userhome/yuxian/train_logs/dependency/ptb/s2t/query"
mkdir -p $OUTPUT_DIR

python parser/s2t_query_trainer.py \
--precision 16 \
--default_root_dir $OUTPUT_DIR \
--data_dir $DATA_DIR \
--data_format 'conllx' \
--pos_dim 100 \
--bert_dir $BERT_DIR \
--additional_layer_dim 1124 \
--mrc_dropout $DROPOUT \
--workers 24 \
--gpus="0,1,2,3,4,5,6,7" \
--accelerator "ddp" \
--batch_size 16 \
--accumulate_grad_batches 20 \
--lr $LR \
--gradient_clip_val=1.0 \
--ignore_punct \
--max_epochs 40 \
--group_sample \
--scheduler "linear_decay" --warmup_steps 300 --final_div_factor 10
