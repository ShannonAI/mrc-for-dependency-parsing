export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

proposal_hparams="/home/ganleilei/train_logs/dependency/ptb-s2s-new/proposal_lr3e-5_add{2}/lightning_logs/version_0/hparams.yaml"
proposal_ckpt="/home/ganleilei/train_logs/dependency/ptb-s2s-new/proposal_lr3e-5_add{2}/epoch=8.ckpt"
s2s_hparams="/home/ganleilei/train_logs/dependency/ptb-s2s-new/s2s_lr1e-5_decay0.0_accumulate40_warmup300/lightning_logs/version_0/hparams.yaml"
s2s_ckpt="/home/ganleilei/train_logs/dependency/ptb-s2s-new/s2s_lr1e-5_decay0.0_accumulate40_warmup300/epoch=7.ckpt"
topk=15

python parser/s2s_evaluate_dp.py \
--proposal_hparams $proposal_hparams \
--proposal_ckpt $proposal_ckpt \
--s2s_ckpt $s2s_ckpt \
--s2s_hparams $s2s_hparams \
--topk $topk \
--use_cache