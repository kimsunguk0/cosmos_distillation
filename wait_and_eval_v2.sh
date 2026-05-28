#!/bin/bash
LOGFILE=/home/pm97/workspace/sukim/distillation/cosmos_distillation/outputs/kv_distill_pipeline/run_20260527_v2_nohup.log
EVALLOG=/home/pm97/workspace/sukim/distillation/cosmos_distillation/outputs/kv_distill_pipeline/run_20260527_v2_eval_nohup.log
CKPT=/home/pm97/workspace/sukim/distillation/cosmos_distillation/outputs/kv_distill_pipeline/run_20260527_v2/final.pt

echo "[Wed May 27 05:38:08 PM KST 2026] waiting for training to finish..."
until grep -q '"event": "done"' $LOGFILE 2>/dev/null; do
  sleep 60
done

echo "[Wed May 27 05:38:08 PM KST 2026] training done. starting eval."
cd /home/pm97/workspace/sukim/distillation/cosmos_distillation
.venv/bin/python3 scripts/93_eval_kv_distill.py   --checkpoint $CKPT   --baseline   --num-samples 500   --batch-size 4
echo "[Wed May 27 05:38:08 PM KST 2026] eval done."
