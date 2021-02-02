cd /userhome/yuxian/shannon_parser
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="/userhome/yuxian/data/parser/ptb3_parser"
BERT_DIR="/userhome/yuxian/data/bert/roberta-large"
BERT_TYPE="roberta"

# hyper-params
DROPOUT=0.3
LR=2e-5
DECAY=0.0
accumulate=20
WARMUP=300
add=1
norm="sigmoid"
tag_weight=2.0
layer_type="lstm"

# save directory
OUTPUT_DIR="/userhome/yuxian/train_logs/dependency/ptb-s2s-new/s2s_lr${LR}_decay${DECAY}_accumulate${accumulate}_warmup${WARMUP}_add${add}_norm${norm}_tweight${tag_weight}_${layer_type}"
mkdir -p $OUTPUT_DIR

python parser/s2s_query_trainer.py \
--precision 16 \
--default_root_dir $OUTPUT_DIR \
--data_dir $DATA_DIR \
--data_format 'conllx' \
--pos_dim 100 --normalize $norm --tag_loss_weight $tag_weight \
--bert_dir $BERT_DIR --bert_name $BERT_TYPE \
--additional_layer_dim 1024 \
--additional_layer $add --additional_layer_type $layer_type \
--mrc_dropout $DROPOUT \
--workers 24 \
--gpus="0,1,2,3,4,5,6,7" \
--accelerator "ddp" \
--batch_size 16 \
--accumulate_grad_batches $accumulate \
--lr $LR \
--gradient_clip_val=1.0 \
--ignore_punct --predict_child \
--max_epochs 20 \
--group_sample \
--weight_decay $DECAY \
--scheduler "linear_decay" --warmup_steps $WARMUP --final_div_factor 20
