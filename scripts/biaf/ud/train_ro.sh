export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="/userhome/yuxian/data/parser/ud-treebanks/ud-treebanks-v2.2/UD_Romanian_RRT"
DATA_PREFIX="ro_rrt-ud-"
BERT_DIR="/userhome/yuxian/data/bert/xlm-roberta-large/"
BERT_TYPE="roberta"

for LR in 6e-5 8e-5 4e-5 2e-5 1e-6 3e-6 5e-6; do
    # hyper-params
    DROPOUT=0.3
    accumulate=10
    WARMUP=50
    addition=1

    # save directory
    OUTPUT_DIR="/userhome/ganleilei/train_logs/dependency/ud-ro/biaf/finetune/lr${LR}_accumulate${accumulate}_warmup${WARMUP}_add${addition}"
    mkdir -p $OUTPUT_DIR

    python parser/biaf_trainer.py \
    --precision 16 \
    --default_root_dir $OUTPUT_DIR \
    --data_prefix $DATA_PREFIX \
    --data_dir $DATA_DIR \
    --data_format 'conllu' \
    --pos_dim 100 \
    --bert_dir $BERT_DIR --bert_name $BERT_TYPE \
    --additional_layer_dim 768 \
    --additional_layer $addition --additional_layer_type "transformer" \
    --biaf_dropout $DROPOUT \
    --workers 32 \
    --gpus="0,1,2,3,4,5,6,7" \
    --accelerator "ddp" \
    --batch_size 16 \
    --accumulate_grad_batches $accumulate \
    --lr $LR \
    --gradient_clip_val=1.0 \
    --ignore_punct \
    --max_epochs 100 \
    --group_sample \
    --scheduler "linear_decay" --warmup_steps $WARMUP --final_div_factor 20
done