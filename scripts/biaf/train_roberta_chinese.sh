export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="/userhome/yuxian/data/parser/ctb5_parser"
BERT_DIR="/userhome/ganleilei/data/bert/roberta-large"

# hyper-params
DROPOUT=0.3
LR=8e-6
LAYER=0
warmup=300

TIME_DIR="`date +%Y%m%d`"
OUTPUT_DIR="train_logs/dependency/ptb/biaf/20210114/finetune/lr${LR}_drop${DROPOUT}_layer${LAYER}_warmup${warmup}_newinit"
mkdir -p $OUTPUT_DIR

python parser/biaf_trainer.py \
--default_root_dir $OUTPUT_DIR \
--data_dir $DATA_DIR \
--data_format 'conllx' \
--pos_dim 100 \
--bert_dir $BERT_DIR \
--biaf_dropout $DROPOUT \
--additional_layer $LAYER \
--additional_layer_type "transformer" \
--additional_layer_dim 1124 \
--workers 8 \
--gpus="0,1,2,3,4,5,6,7" \
--accelerator "ddp" \
--precision 16 \
--batch_size 16 \
--accumulate_grad_batches 1 \
--lr $LR \
--gradient_clip_val=1.0 \
--ignore_punct \
--max_epochs 30 \
--group_sample \
--scheduler "linear_decay" --warmup_steps $warmup --final_div_factor 20

# todo remove warmup in finetune.
