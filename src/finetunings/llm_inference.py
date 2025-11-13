"""
 * Software Name: task-inclusion-estimation
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the MIT license,
 * the text of which is available at https://opensource.org/license/MIT/
 * or see the "LICENSE" file for more details.
"""
import os
import sys
import json
sys.path.append(os.getcwd())
from tqdm import tqdm
from pathlib import Path
import argparse
import torch
import evaluate
from data import DataLauncher
from transformers import (
    StoppingCriteria, 
    StoppingCriteriaList,
)
from peft import PeftModel
from logzero import logger as log
from src.finetunings.args import response_templates
from src.finetunings.classes import (
    MODEL_CLASSES, 
    TOKENIZER_CLASSES
)

def get_arguments():
    parser = argparse.ArgumentParser("llm_inference script")
    parser.add_argument("--peft_checkpoint", type=str, default=None)
    parser.add_argument("--source_peft_checkpoint", type=str, default=None)
    parser.add_argument("--target_peft_checkpoint", type=str, default=None)
    parser.add_argument("--full_finetuned_checkpoint", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--corpus_name", type=str, required=True)
    parser.add_argument("--neutral_prompt", action="store_true", default=False)
    return parser.parse_args()

@torch.no_grad()
def main():
    args = get_arguments()
    rouge_score = evaluate.load('rouge')
    bert_score = evaluate.load('bertscore')
    device_map = "auto"

    if "pythia" in args.model_name_or_path:
        key = "pythia"
    elif "MistralAI" in args.model_name_or_path:
        key = "mistral"
    elif "MetaLlama" in args.model_name_or_path:
        key = "llama"
    else:
        key = "auto"

    log.info(f"Get the class related to {key}")
    model_class = MODEL_CLASSES[key]
    tokenizer_class = TOKENIZER_CLASSES[key]

    loader = DataLauncher(
        name=args.dataset, 
        debug=args.debug, 
        corpus_name=args.corpus_name, 
        neutral_prompt=args.neutral_prompt
    )
    data = loader()
    test_data = data["test"]
    response_template = response_templates[args.dataset]

    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path, 
        trust_remote_code=True,
        token=os.environ["hf_token"],
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    stop_words_ids = [tokenizer.encode(stop_word)[-1] for stop_word in ["\n"]]
    class StopOnTokens(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor,**kwargs
        ) -> bool:
            for stop_id in stop_words_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    # --- load the model --- #
    if args.full_finetuned_checkpoint:
        log.info("Loading a full finetuned checkpoint")
        model = model_class.from_pretrained(
            args.full_finetuned_checkpoint,
            device_map=device_map,
            trust_remote_code=True,
            token=os.environ["hf_token"]
        )
    elif args.peft_checkpoint:
        log.info("Loading a peft checkpoint")
        base_model = model_class.from_pretrained(
            args.model_name_or_path,
            device_map=device_map,
            trust_remote_code=True,
            token=os.environ["hf_token"],
        )
        model = PeftModel.from_pretrained(base_model, args.peft_checkpoint)
    elif args.source_peft_checkpoint and args.target_peft_checkpoint:
        log.info("Loading multiple checkpoints")
        log.info(f"Source checkpoint {args.source_peft_checkpoint}")
        log.info(f"Target checkpoint {args.target_peft_checkpoint}")
        base_model = model_class.from_pretrained(
            args.model_name_or_path,
            device_map=device_map,
            trust_remote_code=True,
            token=os.environ["hf_token"],
        )
        model = PeftModel.from_pretrained(base_model, args.source_peft_checkpoint)
        model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, args.target_peft_checkpoint)
    else:
        log.warning("No specific checkpoint is given, loading the pre-trained model")
        model = model_class.from_pretrained(
            args.model_name_or_path,
            device_map=device_map,
            trust_remote_code=True,
            token=os.environ["hf_token"]
        )

    def generate(input_string: str):
        inputs = tokenizer(input_string, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.input_ids.to('cuda'), 
            attention_mask=inputs.attention_mask.to('cuda'), 
            max_new_tokens=args.max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()])
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated.split(response_template)[-1].strip()
        answer = answer.split('\n')[0]
        return answer
    
    
    predicted = []
    ground_truth = []
    prompts = []
    ids = []
    word_metrics = {
        "nb_gen_wd": [], "nb_gt_wd": [], 
    }
    model = model.eval() # remove dropouts
    for d in tqdm(test_data, total=len(test_data), desc="Running inference"):
        ids.append(d['document_ids'])
        buff = d['text'].split(response_template)
        t = buff[0]            # text
        a = buff[-1].strip()   # answer
        word_metrics["nb_gt_wd"].append(len(a.split(" ")))

        prompt = t + response_template
        res = generate(prompt)
        word_metrics["nb_gen_wd"].append(len(res.strip().split(" ")))

        ground_truth.append(a)
        predicted.append(res)
        prompts.append(prompt)

    rs = rouge_score.compute(predictions=predicted, references=ground_truth)
    bs = bert_score.compute(predictions=predicted, references=ground_truth, lang="en")
    new_bs = {}
    for k in bs:
        new_bs[f"bert_score_{k}"] = bs[k]
    res_metrics = {
        **rs,
        **new_bs,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "test-metrics.json").open("w") as f:
        json.dump(res_metrics, f)
    with (args.output_dir / "word-metrics.json").open("w") as f:
        json.dump(word_metrics, f)
    with (args.output_dir / "generation-results.json").open("w") as f:
        json.dump(
            {"predictions": predicted, "references": ground_truth, "prompts": prompts, 'document_ids': ids}, 
            f,
            indent=4,
        )

    return 0

if __name__ == "__main__":
    _ = main()