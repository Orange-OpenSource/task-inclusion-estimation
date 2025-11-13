"""
 * Software Name: task-inclusion-estimation
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the MIT license,
 * the text of which is available at https://opensource.org/license/MIT/
 * or see the "LICENSE" file for more details.
"""
DATA=${1:-"onto_notes_summary"}
BASE_MODEL=${2:-"mistralai/Mistral-7B-Instruct-v0.2"}
CORPUS_NAME=${3:-""}
PEFT_CHECKPOINT=${4-:".outputs/train/test/final-checkpoint"}
OUTPUT_DIR=${5:-".outputs/dump/test"}

python3 -u src/dump_embeddings/dump.py \
    --dataset ${DATA} \
    --document_embeddings \
    --base_model ${BASE_MODEL} \
    --peft_checkpoint ${PEFT_CHECKPOINT} \
    --output_dir ${OUTPUT_DIR}

echo "Dump is done !"