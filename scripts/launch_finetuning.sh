"""
 * Software Name: task-inclusion-estimation
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the MIT license,
 * the text of which is available at https://opensource.org/license/MIT/
 * or see the "LICENSE" file for more details.
"""
MODEL=${1:-"mistralai/Mistral-7B-Instruct-v0.2"}
DATA=${2:-"onto_notes_summary"}
CORPUS=${3:-"corpus_onto_compress_1605"}

python3 -u src/finetunings/llm_training.py \
        --output_dir .outputs/train/test \
        --model_name_or_path ${MODEL} \
        --dataset ${DATA} \
        --lr_scheduler_type constant \
        --learning_rate 4e-05 \
        --max_grad_norm 1.0 \
        --batch_size 1 \
        --eval_batch_size 1 \
        --epochs 6 \
        --group_by_length \
        --do_train \
        --save_total_limit 1 \
        --corpus_name ${CORPUS} \
        --gradient_checkpointing \
        --bf16 \
        --peft_finetuning \
        --lora \
        --lora_r 8 \
        --lora_a 16 \
        --target_modules q_proj-v_proj \
        --nb_data -1 \
        --data_seed 42

echo "Training is done !"
