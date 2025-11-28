    # --train-lm-head \
    # lmsys-chat-turn-ge-3.jsonl 
export CUDA_VISIBLE_DEVICES=0
python train_beacon.py \
    --data-paths orca-chat-filtered.jsonl \
    --model-path ~/jhaidata/local_model/Qwen2-0.5B-Instruct \
    --output-dir ./runs/beacon-ft-v2 \
    --num-epochs 2 \
    --learning-rate 5e-5 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 2 \
    --warmup-ratio 0.03 \
    --bf16 \
    --train-beacon-only \
    --processed-cache-dir ~/Projects/Beacon_Multiturn/runs/dataset_cache
rungpu