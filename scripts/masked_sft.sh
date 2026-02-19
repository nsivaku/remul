NUM_GPUS=${2:-4}
MODEL_PATH=${1:?  "Usage: $0 <model_path>"}
OUTPUT_DIR=${2:?  "Usage: $0 <output_dir>"}

python -m data.datasets --output_dir data/parquet 2>/dev/null || true

python -u training/masked_sft.py \
    --model_path ${MODEL_PATH} \
    --train_parquet data/parquet/train.parquet \
    --output_dir ${OUTPUT_DIR} \
    --lora_rank 32 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --learning_rate 1e-5 \
    --num_epochs 5 \
    --batch_size 16 \
    --gradient_accumulation_steps 4