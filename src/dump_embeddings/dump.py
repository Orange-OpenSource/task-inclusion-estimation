"""
 * Software Name: task-inclusion-estimation
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the MIT license,
 * the text of which is available at https://opensource.org/license/MIT/
 * or see the "LICENSE" file for more details.
"""
from typing import List
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    GPTNeoXForCausalLM,
    AutoTokenizer, 
    GPTNeoXTokenizerFast,
    Qwen2MoeForCausalLM,
    Qwen2Tokenizer,
)
import os
from tqdm import tqdm
from datasets import disable_progress_bar
from huggingface_hub import login
from logzero import logger as log
from data import generative_datasets

MISTRAL_CHECKPOINTS: List[str] = [
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
]
LLAMA_3_CHECKPOINTS: List[str] = [
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]
MODEL_CLASSES = {
    "auto": AutoModelForCausalLM,
    "pythia": GPTNeoXForCausalLM,
    "qwen": Qwen2MoeForCausalLM,
}
TOKENIZER_CLASSES = {
    "auto": AutoTokenizer,
    "pythia": GPTNeoXTokenizerFast,
    "qwen": Qwen2Tokenizer,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification_task", action="store_true", default=False)
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        default="test",
    )
    parser.add_argument("--last_token", action='store_true', default=False)
    parser.add_argument("--mean_proceeding", action='store_true', default=False)
    parser.add_argument("--document", action='store_true', default=False)
    parser.add_argument("--get_outputs", action='store_true', default=False)
    parser.add_argument("--corpus_name", type=str, default='onto_json_name_summary_v1',
                        choices=['onto_json_name_summary_v1', 'onto_json_name_summary_10pc', 'onto_json_name_summary_10pc_v2'])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_min", type=int, default=None)
    parser.add_argument("--dataset_max", type=int, default=None)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--full_finetuned_checkpoint", type=str, default=None)
    parser.add_argument("--peft_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--document_embeddings", action="store_true", default=False)
    return parser.parse_args()

@torch.no_grad()
def main():
    
    login(token=os.environ['hf_token'])
    disable_progress_bar()
    args = parse_args()
    log.info(vars(args))
    
    if "pythia" in args.model_name_or_path:
        model_class = MODEL_CLASSES["pythia"]
        tokenizer_class = TOKENIZER_CLASSES["pythia"]
    elif "Qwen" in args.model_name_or_path:
        model_class = MODEL_CLASSES["qwen"]
        tokenizer_class = TOKENIZER_CLASSES["qwen"]
    else:
        model_class = MODEL_CLASSES["auto"]
        tokenizer_class = TOKENIZER_CLASSES["auto"]

    device_map = "auto" # automatically load on the cpu
    if args.full_finetuned_checkpoint:
        log.info("Loading a full finetuned checkpoint")
        model = model_class.from_pretrained(
            args.full_finetuned_checkpoint,
            device_map=device_map,
            trust_remote_code=True,
        )
    elif args.peft_checkpoint:
        log.info("Loading a peft checkpoint")
        base_model = model_class.from_pretrained(
            args.model_name_or_path,
            device_map=device_map,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.peft_checkpoint)
    else:
        log.info("No specific checkpoint is given, loading the pre-trained model")
        model = model_class.from_pretrained(
            args.model_name_or_path,
            device_map=device_map,
            trust_remote_code=True,
        )
    
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if not tokenizer.pad_token:
        log.info("Add pad token to tokenizer")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ############################
    # --- load the dataset --- #
    ############################
    if args.dataset is not None:
        loader = generative_datasets(debug=False, corpus_name=args.corpus_name, get_outputs=args.get_outputs)
        dataset = loader(args.dataset)['test']           
    else:
        raise NotImplementedError('You must select a task')
    
    min_index = None
    max_index = None
    if args.dataset_min is not None and args.dataset_max is not None:
        min_index = max(args.dataset_min, 0)
        max_index = min(args.dataset_max, len(dataset))
        dataset = dataset.select(range(min_index, max_index))

    def _mapping(batch):
        return tokenizer(batch['text'])
    
    dataset = dataset.map(_mapping, batched=True, remove_columns=dataset.column_names)

    if args.model_name_or_path in MISTRAL_CHECKPOINTS:
        document_separator = tokenizer('\n', return_tensors='pt').input_ids[0][-1]
        doc_sep_pos = 2
    elif args.model_name_or_path in LLAMA_3_CHECKPOINTS:
        document_separator = 512
        doc_sep_pos = 0
    elif "pythia" in args.model_name_or_path:
        document_separator = 187
        doc_sep_pos = -1
    elif "Qwen" in args.model_name_or_path:
        document_separator = 510
        doc_sep_pos = -1
    else:
        raise Exception(f"For `{args.model_name_or_path}` the tests had not been proceeded yet !")

    log.info(f"Creating the output dir at {str(args.output_dir)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
        
    ####################################
    # --- get distribution samples --- #
    ####################################
    embedding_dict = {}
    tokens = []
    nb_tokens = 0
    with torch.no_grad():
        model = model.eval() # remove dropouts 
        
        for id_d, d in tqdm(enumerate(dataset), total=len(dataset), desc="Embedding dumper"):
            
            input_ids = torch.tensor([d['input_ids']])
            attention_mask = torch.tensor([d['attention_mask']])
            assert len(input_ids.shape)==2

            start_pos = torch.any(input_ids == document_separator, axis=0).nonzero().flatten()[doc_sep_pos].item()+1
            end_pos = input_ids.shape[-1]

            # --- get the tokens --- #
            t = input_ids[0, start_pos:end_pos].tolist()
            tokens.append(t)
            nb_tokens+=len(t)

            if args.get_outputs:
                start_pos-=1
                end_pos-=1

            output = model(
                input_ids=input_ids.to(args.device), 
                attention_mask=attention_mask.to(args.device),
                output_hidden_states=True,
            )
            hs = output.hidden_states

            for i, buff in enumerate(hs):
                new_e = None
                if args.document_embeddings:
                    new_e = buff[0, :, :][start_pos:end_pos, :].cpu().numpy()
                else:
                    new_e = buff[0, :, :].cpu().numpy()
                if args.mean_proceeding:
                    new_e = buff.mean(dim=0)
                    new_e = new_e.expand_dims(new_e, axis=0)
                if args.last_token:
                    new_e = buff[-1, :]
                    new_e = new_e.expand_dims(new_e, axis=0)

                assert len(new_e.shape) == 2
                assert new_e.shape[-1] == model.config.hidden_size

                if f'layer_{i}' in embedding_dict:
                    embedding_dict[f'layer_{i}'] = np.concatenate((embedding_dict[f'layer_{i}'], new_e), axis=0)
                else:
                    embedding_dict[f'layer_{i}'] = new_e

                    
        if args.mean_proceeding:
            for l in embedding_dict:
                with (args.output_dir / f"{l}_embeddings_mean_{min_index}_{max_index}.npy").open("wb") as f:
                    torch.save(embedding_dict[l], f)

        else:
            for l in embedding_dict:
                with (args.output_dir / f"{l}_embeddings_{min_index}_{max_index}.npy").open("wb") as f:
                    torch.save(embedding_dict[l], f)

    # --- save the different tokens --- #
    with (args.output_dir / 'tokens.json').open('w') as f:
        json.dump(tokens, f)

if __name__ == "__main__":
    main()