#!/bin/sh
#SBATCH --job-name="SWHISPER FT"
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:h100:1

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate sam

# accelerate launch \
#     --num_machines="1" \
#     --num_processes="1" \
# 	--mixed_precision="fp16" \
# 	--num_cpu_threads_per_process="8" \
python finetuning.py \
    --model_name_or_path "openai/whisper-large-v3" \
    --train_dataset "data/hf_dataset" \
    --output_dir="whisper-large_2" \
    --max_train_steps="5000" \
    --num_warmup_steps="500" \
    --per_device_train_batch_size="128" \
    --per_device_eval_batch_size="32" \
    --gradient_accumulation_steps="4" \
    --learning_rate="1e-05" \
    --logging_steps="25" \
    --saving_steps="100" \
    --eval_steps="100" \
    --adam_beta1="0.9" \
    --adam_beta2="0.98" \
    --adam_epsilon="1e-06" \
    --lr_scheduler_type="linear" \
    --gradient_checkpointing \
    --keep_n_percent="1"