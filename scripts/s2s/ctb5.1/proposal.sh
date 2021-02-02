export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="/userhome/yuxian/data/parser/ctb5.1_parser"
BERT_DIR="/userhome/yuxian/data/bert/chinese_roberta_wwm_large_ext_pytorch"
BERT_TYPE="bert"

# hyper-params
DROPOUT=0.3
precision=16
add=2
LR=4e-5

OUTPUT_DIR="/userhome/yuxian/train_logs/dependency/ctb5.1/s2s/proposal_lr${LR}_20210123_add${add}"
mkdir -p $OUTPUT_DIR

python parser/span_proposal_trainer.py \
--precision $precision \
--default_root_dir $OUTPUT_DIR \
--data_dir $DATA_DIR \
--data_format 'conllx' \
--pos_dim 100 \
--bert_dir $BERT_DIR --bert_name $BERT_TYPE \
--additional_layer_dim 1024 \
--additional_layer $add --additional_layer_type "transformer" \
--mrc_dropout $DROPOUT \
--workers 8 \
--gpus="0," \
--accelerator "ddp" \
--batch_size 16 \
--accumulate_grad_batches 8 \
--lr $LR \
--gradient_clip_val=1.0 \
--max_epochs 10 \
--group_sample \
--scheduler "linear_decay" --warmup_steps 100 --final_div_factor 10
