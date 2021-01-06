export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="/userhome/yuxian/data/parser/ptb3_parser"
BERT_DIR="/userhome/yuxian/data/bert/bert-large-uncased-wwm"

# hyper-params
DROPOUT=0.3
LR=2e-5

OUTPUT_DIR="/userhome/yuxian/train_logs/dependency/ptb/biaf/s2t/proposal"
mkdir -p $OUTPUT_DIR

python parser/s2t_trainer.py \
--precision 16 \
--default_root_dir $OUTPUT_DIR \
--data_dir $DATA_DIR \
--data_format 'conllx' \
--pos_dim 100 \
--bert_dir $BERT_DIR \
--additional_layer_dim 1124 \
--mrc_dropout $DROPOUT \
--workers 8 \
--gpus="0,1" \
--accelerator "ddp" \
--batch_size 64 \
--accumulate_grad_batches 1 \
--lr $LR \
--gradient_clip_val=1.0 \
--ignore_punct \
--max_epochs 80 \
--group_sample \
--scheduler "linear_decay" --warmup_steps 100 --final_div_factor 10
