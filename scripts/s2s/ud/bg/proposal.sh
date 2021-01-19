export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="/userhome/yuxian/data/parser/ud-treebanks/ud-treebanks-v2.2/UD_Bulgarian-BTB"
DATA_PREFIX="bg_btb-ud-"
BERT_DIR="/userhome/yuxian/data/bert/xlm-roberta-large/"
BERT_TYPE="roberta"

# hyper-params
DROPOUT=0.3
LR=1e-5
precision=16
addition=1

OUTPUT_DIR="/userhome/yuxian/train_logs/dependency/ud-bg/s2s/xlm_proposal_lr${LR}_add${addition}"
mkdir -p $OUTPUT_DIR

python parser/span_proposal_trainer.py \
--bert_name $BERT_TYPE \
--precision $precision \
--default_root_dir $OUTPUT_DIR \
--data_dir $DATA_DIR \
--data_prefix $DATA_PREFIX \
--data_format 'conllu' \
--pos_dim 100 \
--bert_dir $BERT_DIR \
--additional_layer_dim 1024 \
--additional_layer $addition --additional_layer_type "transformer" \
--mrc_dropout $DROPOUT \
--workers 8 \
--gpus="1," \
--accelerator "ddp" \
--batch_size 64 \
--accumulate_grad_batches 2 \
--lr $LR \
--gradient_clip_val=1.0 \
--ignore_punct \
--max_epochs 10 \
--group_sample \
--scheduler "linear_decay" --warmup_steps 100 --final_div_factor 10
