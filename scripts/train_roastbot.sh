#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
bash scripts/download.sh
$PY src/corpus.py
$PY src/tok.py
$PY scripts/emoji_slots.py
$PY src/train.py --size b16 --tokens 1.6e9 --name pre_b16
$PY scripts/unigram.py models
$PY scripts/tiktok_style.py
$PY scripts/roast_bank.py
$PY scripts/gen_dialogs.py 9000 both batch=48
$PY scripts/gen_dialogs.py 2000 mom
$PY scripts/mom_cap.py
$PY scripts/gen_dialogs.py 1000 mom2 batch=24
$PY scripts/gen_dialogs.py 2000 mom3 batch=24
$PY scripts/sense_filter.py 0.15 src=rb,rb_arg,rb_mom
$PY scripts/sense_filter.py 0.2 src=rb_mom2
$PY scripts/sense_filter.py 0.2 src=rb_mom3
for s in rb rb_arg rb_mom rb_mom2 rb_mom3; do $PY scripts/copy_check.py $s drop; done
mkdir -p data/people/pr
$PY scripts/emoji_fit.py 0.4 1.0 src=rb,rb_arg,rb_mom,rb_mom2,rb_mom3 dst=pr
$PY scripts/mom_judge.py dst=pr 0.25
$PY scripts/kd_targets.py pr 20
$PY src/finetune.py pr --init models/pre_b16.pt --out models --steps 9000 --lr 1e-4 --eval-every 50 --patience 30 --until-overfit --mix 0.05 --kd 0.5
$PY scripts/emoji_calibrate.py pr
$PY scripts/kv8_add.py models/pr_board.pkl models/kv8 pr
cp models/kv8/pr_board.pkl models/pr_board.pkl
