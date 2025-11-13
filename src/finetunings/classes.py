"""
 * Software Name: task-inclusion-estimation
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the MIT license,
 * the text of which is available at https://opensource.org/license/MIT/
 * or see the "LICENSE" file for more details.
"""
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GPTNeoXForCausalLM, 
    GPTNeoXTokenizerFast,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    MistralForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
)

MODEL_CLASSES = {
    "auto": AutoModelForCausalLM,
    "qwen": Qwen2ForCausalLM,
    "mistral": AutoModelForCausalLM,
    "llama": AutoModelForCausalLM,
    "pythia": GPTNeoXForCausalLM,
}
TOKENIZER_CLASSES = {
    "auto": AutoTokenizer,
    "qwen": Qwen2Tokenizer,
    "mistral": AutoTokenizer,
    "llama": AutoTokenizer,
    "pythia": GPTNeoXTokenizerFast,
}