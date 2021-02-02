export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="/userhome/yuxian/data/parser/ctb5.1_parser"
BERT_DIR="/userhome/yuxian/data/bert/chinese_roberta_wwm_large_ext_pytorch"
BERT_TYPE="bert"
# hyper-params
DROPOUT=0.3
LR=6e-4
accumulate=40
add=2
norm="sigmoid"

OUTPUT_DIR="/userhome/yuxian/train_logs/dependency/ctb5.1/s2s/s2s_freeze_lstm_lr${LR}_accumulate${accumulate}_add${add}_norm${norm}_maxwords90_tagw${tag_weight}"
mkdir -p $OUTPUT_DIR

python parser/s2s_query_trainer.py \
--precision 16 \
--default_root_dir $OUTPUT_DIR \
--data_dir $DATA_DIR \
--data_format 'conllx' \
--pos_dim 100 --max_words 90 --normalize $norm \
--bert_dir $BERT_DIR --bert_name $BERT_TYPE \
--freeze_bert --additional_layer_dim 1024 --additional_layer $add --additional_layer_type "lstm" \
--mrc_dropout $DROPOUT \
--workers 24 \
--gpus="0,1,2,3,4,5,6,7" \
--accelerator "ddp" \
--batch_size 16 \
--accumulate_grad_batches $accumulate \
--lr $LR \
--gradient_clip_val=1.0 \
--max_epochs 30 \
--group_sample \
--predict_child --ignore_punct \
--scheduler "linear_decay" --warmup_steps 0 --final_div_factor 10
