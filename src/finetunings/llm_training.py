"""
 * Software Name: task-inclusion-estimation
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the MIT license,
 * the text of which is available at https://opensource.org/license/MIT/
 * or see the "LICENSE" file for more details.
"""
import os, json, sys
sys.path.append(os.getcwd())
import torch
import random
import numpy as np
from datasets import disable_progress_bar
from args import (
    get_arguments, 
    response_templates
)
from transformers import (
    BitsAndBytesConfig,
    StoppingCriteria,
)
from unsloth import is_bfloat16_supported
from src.finetunings.utils import EpochStopCB
from peft import (
    LoraConfig,
    PeftModel, 
    TaskType,
)
from trl import (
    SFTTrainer, 
    DataCollatorForCompletionOnlyLM, 
    SFTConfig
)
from data import DataLauncher
from huggingface_hub import login
from logzero import logger as log
from classes import (
    MODEL_CLASSES, 
    TOKENIZER_CLASSES
)

login(token=os.environ['HF_TOKEN'])

disable_progress_bar()
args = get_arguments()
print(json.dumps(vars(args), indent=4))

if "pythia" in args.model_name_or_path:
    key = "pythia"
elif "Mistral" in args.model_name_or_path:
    key = "mistral"
elif "MetaLlama" in args.model_name_or_path:
    key = "llama"
elif "Qwen" in args.model_name_or_path:
    key = "qwen"
else:
    key = "auto"

log.info(f"Get the class related to {key}")
model_class = MODEL_CLASSES[key]
tokenizer_class = TOKENIZER_CLASSES[key]

#############################################
# --- manually set seed for experiments --- #
#############################################
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

###############################
# --- device map for expe --- #
###############################
if args.device_map != "auto":
    device_map = json.loads(args.device_map)
else:
    device_map = "auto"

#############################
# --- load quantization --- #
#############################
if args.use_quantization:
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )
    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
else:
    bnb_config = None

######################
# --- load model --- #
######################
model = model_class.from_pretrained(
    args.model_name_or_path,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

##################################
# --- load an old checkpoint --- #
##################################
if args.base_checkpoint is not None:
    try:
        model = PeftModel.from_pretrained(model, args.base_checkpoint)
        model = model.merge_and_unload()
        log.info("*** loading an old peft checkpoint ***")
    except Exception:
        model = model_class.from_pretrained(
            args.base_checkpoint,
            device_map=device_map,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        log.info("loading an old full-finetuned model")

#####################
# --- tokenizer --- #
#####################
tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, trust_remote_code=True)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

########################################
# --- load adapters (if necessary) --- #
########################################
peft_config = None
if args.peft_finetuning:
    target_modules = args.target_modules.split('-')
    if args.lora: # lora finetuning
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r, 
            lora_alpha=args.lora_a, 
            lora_dropout=args.lora_d,
            bias="none",
            target_modules=target_modules,
        )
    else:
        raise Exception('You forgot to select a peft mode !!!!')

log.info('peft_config', peft_config)

#########################
# --- load the data --- #
######################### 
response_template = response_templates[args.dataset]
log.info(f'response template : {response_template}')
tokens =  tokenizer.tokenize(response_template, add_special_tokens=False)
token_ids = tokenizer.encode(response_template, add_special_tokens=False)
log.info(list(zip(tokens, token_ids)))
collator = DataCollatorForCompletionOnlyLM(token_ids[1:], tokenizer=tokenizer)

loader = DataLauncher(name=args.dataset, debug=args.debug, corpus_name=args.corpus_name, neutral_prompt=args.neutral_prompt)
data = loader()

if "test" not in data and "train" not in data:
    raise Exception("The dataset must have train and test !")

if "validation" in data:
    log.info("A validation split is present !")
    train_data, eval_data, test_data = data['train'], data['validation'], data['test']
else:
    log.info("Creating an artificial validation split, based on the train one !")
    temp, test_data = data['train'], data['test']
    temp = temp.shuffle(seed=args.seed).train_test_split(seed=args.seed, test_size=0.1)
    train_data, eval_data = temp['train'], temp['test']

if args.nb_data > 0:
    log.info('Splitting data !')
    nb_data = min(args.nb_data, len(train_data))
    buff = train_data.shuffle(seed=args.data_seed)
    train_data = buff.select(range(nb_data))
    log.info(f'New train data len : {len(train_data)}')

#############################
# --- stopping criteria --- #
#############################
stop_words_ids = [tokenizer.encode(stop_word)[-1] for stop_word in ["\n"]]
class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor,**kwargs
    ) -> bool:
        for stop_id in stop_words_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

#####################################
# --- definition of the trainer --- #
#####################################
training_args = SFTConfig(
    log_level='info',
    output_dir=args.output_dir,
    logging_dir=args.output_dir,
    eval_strategy='epoch',
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=1,
    gradient_checkpointing=args.gradient_checkpointing,
    per_device_eval_batch_size=args.eval_batch_size,
    eval_accumulation_steps=10,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    lr_scheduler_type=args.lr_scheduler_type,
    report_to='tensorboard',
    max_grad_norm=args.max_grad_norm,
    max_steps=-1,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    group_by_length=args.group_by_length,
    optim='paged_adamw_32bit',
    seed=args.seed,
    save_total_limit=args.save_total_limit,
    save_steps=0,
    save_strategy='epoch',
    warmup_ratio=args.warmup_ratio,
    logging_steps=len(train_data) // (args.batch_size*3),
    disable_tqdm=not args.verbose,
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    max_seq_length=model.config.max_position_embeddings,
    dataset_text_field="text",
    dataset_num_proc=2,
)
callbacks = []
if args.epoch_requeue:
    callbacks.append(EpochStopCB)

trainer = SFTTrainer(
    args=training_args,
    model=model,
    peft_config=peft_config,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=eval_data,
    packing=False,
    data_collator=collator,
    callbacks=callbacks
)

###########################
# --- train the model --- #
###########################
if args.do_train:
    try:
        log.info("*** resume from checkpoint ***")
        trainer.train(resume_from_checkpoint=True)
    except ValueError as e:
        log.info("*** first loop ***")
        log.debug(e)
        trainer.train()

    if trainer.state.epoch < args.epochs:
        log.info("REQUEUE")
        sys.exit(args.relaunch_exit_code)
    else:
        log.info("Training is finished !")
    
    saving_dir = os.path.join(args.output_dir, 'final-checkpoint')
    trainer.model.save_pretrained(saving_dir)
    json.dump(
        vars(args), 
        open(os.path.join(args.output_dir, 'training.hparams.json'), 'w'), 
        indent=4
    )