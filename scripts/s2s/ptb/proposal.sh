cd /userhome/yuxian/shannon_parser
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="/userhome/yuxian/data/parser/ptb3_parser"
BERT_DIR="/userhome/yuxian/data/bert/roberta-large"
BERT_TYPE="roberta"
# hyper-params
DROPOUT=0.3
LR=2e-5
precision=16
add=2

OUTPUT_DIR="/userhome/yuxian/train_logs/dependency/ptb/s2s/new_data_roberta_proposal_lr${LR}_add${add}"
mkdir -p $OUTPUT_DIR

python parser/span_proposal_trainer.py \
--bert_name $BERT_TYPE \
--precision $precision \
--default_root_dir $OUTPUT_DIR \
--data_dir $DATA_DIR \
--data_format 'conllx' \
--pos_dim 100 \
--bert_dir $BERT_DIR \
--additional_layer_dim 1024 \
--additional_layer $add --additional_layer_type "transformer" \
--mrc_dropout $DROPOUT \
--workers 8 \
--gpus="2,3" \
--accelerator "ddp" \
--batch_size 64 \
--accumulate_grad_batches 1 \
--lr $LR \
--gradient_clip_val=1.0 \
--ignore_punct \
--max_epochs 30 \
--group_sample \
--scheduler "linear_decay" --warmup_steps 100 --final_div_factor 10
