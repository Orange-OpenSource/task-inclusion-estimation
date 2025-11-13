"""
 * Software Name: task-inclusion-estimation
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the MIT license,
 * the text of which is available at https://opensource.org/license/MIT/
 * or see the "LICENSE" file for more details.
"""
import torch
from safetensors import safe_open
import os

def load_sft(ckpt, device: str = 'cpu', framework: str = 'pt'):
    if not str(ckpt).endswith(".safetensors"):
        raise Exception("Extension must end with safetensor")
    tensors = {}
    with safe_open(ckpt, framework=framework, device=device) as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors