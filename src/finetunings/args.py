"""
 * Software Name: task-inclusion-estimation
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the MIT license,
 * the text of which is available at https://opensource.org/license/MIT/
 * or see the "LICENSE" file for more details.
"""
import argparse

def get_arguments():
    """ Argument for training a huggingface model """
    parser = argparse.ArgumentParser()

    # --- dataset --- #
    parser.add_argument('--dataset', type=str, default='rte')
    parser.add_argument('--nb_data', type=int, default=-1)
    parser.add_argument('--data_seed', type=int, default=42)
    parser.add_argument('--corpus_name', type=str, default='onto_json_name_summary_v1',
                        help='corpus name for the gitlab filesystem used for the OntoNotes dataset')
    parser.add_argument('--neutral_prompt', action='store_true', default=False)

    # --- model arguments --- #
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base')
    
    # --- optimization argument --- #
    parser.add_argument('--device_map', type=str, default='auto')
    parser.add_argument('--deepspeed', type=str, default=None, help="Path to a deepspeed file")
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--verbose', '-v', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_predict', action='store_true', default=False)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False)
    parser.add_argument('--quantized_training', action='store_true', default=False)
    parser.add_argument('--save_total_limit', type=int, default=1)
    parser.add_argument('--epoch_requeue', action="store_true", default=False)
    parser.add_argument('--relaunch_exit_code', type=int, default=42)

    # --- quantization arguments --- #
    parser.add_argument('--use_quantization', action='store_true', default=False)
    parser.add_argument('--use_4bit', action='store_true', default=False)
    parser.add_argument('--bnb_4bit_compute_dtype', type=str, default='bfloat16')
    parser.add_argument('--bnb_4bit_quant_type', type=str, default='nf4')
    parser.add_argument('--use_nested_quant', action='store_true', default=False)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--bf16', action='store_true', default=False)                
    parser.add_argument('--group_by_length', action='store_true', default=False)     
    parser.add_argument('--packing', action='store_true', default=False)             
    
    # --- peft fine tuning --- #
    parser.add_argument('--peft_finetuning', action='store_true', default=False)
    parser.add_argument('--target_modules', type=str, default='query_value')

    # >> lora << #
    parser.add_argument('--lora', action='store_true', default=False)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_a', type=int, default=16)
    parser.add_argument('--lora_d', type=float, default=0.05)
    parser.add_argument('--freeze_A', action='store_true', default=False)
    parser.add_argument('--freeze_B', action='store_true', default=False)
    parser.add_argument('--shuffle_A', action='store_true', default=False)
    parser.add_argument('--shuffle_B', action='store_true', default=False)
    parser.add_argument('--random_A', action='store_true', default=False)
    parser.add_argument('--random_B', action='store_true', default=False)
    parser.add_argument('--shuffle_seed', type=int, default=42)
    parser.add_argument('--init_fn', type=str, default=None)

    # >> ia3 << #
    parser.add_argument('--ia3', action='store_true', default=False)

    # >> vera << #
    parser.add_argument('--vera', action='store_true', default=False)
    parser.add_argument('--vera_r', type=int, default=8)
    parser.add_argument('--d_initial', type=float, default=0.1)
    parser.add_argument('--prng_key', type=int, default=0)

    parser.add_argument('--fft_combination', action='store_true', default=False)

    # --- save arguments --- #
    parser.add_argument('--output_dir', type=str, default='./outputs/')
    parser.add_argument('--base_checkpoint', type=str, default=None)

    # --- debug mode --- #
    parser.add_argument('--debug', action='store_true', default=False)

    # --- generative option -- #
    parser.add_argument('--max_length', type=int, default=512)

    # --- datcha dataset --- #
    parser.add_argument('--next_act', action='store_true', default=False)
    parser.add_argument('--hist_len', type=int, default=3)

    return parser.parse_args()

labels = {
    'rte': 2,
    'sst2': 2,
    'cola': 2,
    'mrpc': 2,
    'mnli': 3,
    'qnli': 2,
    'qqp': 2,
    'wnli': 2,

    'snli': 3,
    'imdb': 2,
    'yelp_polarity': 2,
}


metric_name = {
    'snli': 'mnli',
    'yelp_polarity': 'sst2', 'yelp': 'sst2',
    'imdb': 'sst2',
}


response_templates = {
    # conversation summarization
    'qmsum' : '### Summary:\n', 'dialogsum': '### Summary:\n', 'samsum': '### Summary:\n', 'qmsum_no_prompt': '### Summary:\n', 'qmsum_default': '### Summary:\n',

    # document summarization
    'xsum' : '### Summary:\n', 'cnn_dailymail': '### Summary:\n', "onto_notes_summary": '### Summary:\n',
    "onto_notes_summary_50_pc": '### Summary:\n', "onto_notes_summary_10_pc": '### Summary:\n',

    # conversation qa
    'friendsqa' : '### Answer:\n', 'molweni': '### Answer:\n', 'qaconv': '### Answer:\n',

    # document qa
    'mrqa': '### Answer:\n', 'hotpotqa': '### Answer:\n', 'squad': '### Answer:\n',

    # ner
    'onto_notes_named_entity': '### Answer:\n',
    'onto_notes_named_entity_compress': '### Answer:\n',

    # coref
    'onto_notes_coref_control_linguistic': '### Answer:\n',
    'onto_notes_coref_compress_linguistic': '### Answer:\n',

    # semantic
    'onto_notes_semantic_control_linguistic': '### Answer:\n',
    'onto_notes_semantic_compress_linguistic': '### Answer:\n',

    # syntax
    'onto_notes_syntax_control_linguistic': '### Answer:\n',
    'onto_notes_syntax_compress_linguistic': '### Answer:\n',

    # format
    'onto_notes_format_first_10pc_wd': '### Answer:\n',
    'onto_notes_format_last_10pc_wd': '### Answer:\n',
    'onto_notes_format_1_every_10_wd': '### Answer:\n',
    'onto_notes_format_1_every_2_wd': '### Answer:\n',
    'onto_notes_format_first_10pc_sent': '### Answer:\n',
    'onto_notes_format_last_10pc_sent': '### Answer:\n',
    'onto_notes_format_1_every_10_sent': '### Answer:\n',
    'onto_notes_format_1_every_2_sent': '### Answer:\n',
}