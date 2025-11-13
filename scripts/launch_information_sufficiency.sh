"""
 * Software Name: task-inclusion-estimation
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the MIT license,
 * the text of which is available at https://opensource.org/license/MIT/
 * or see the "LICENSE" file for more details.
"""
X=${1}
Y=${2}
OUTPUT_DIR=${3}
ARGUMENTS_DIR=${4}

python3 -u src/information_sufficiency/hierarchical.py \
    --estimator mi \
    --output-dir ${OUTPUT_DIR} \
    --arguments-dir ${ARGUMENTS_DIR} \
    --model-x ${X} \
    --model-y ${y} \
    --normalize-embeddings \
    --label-path None