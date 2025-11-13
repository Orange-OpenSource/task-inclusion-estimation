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
from transformers import StoppingCriteria

def tokenize_dataset(batch, tokenizer, dataset_name: str = 'rte'):
    if dataset_name in ['rte', 'mrpc', 'wnli']:
        text = [p + f' {tokenizer.sep_token} ' + h for (p,h) in zip(batch['sentence1'], batch['sentence2'])]
    elif dataset_name in ['mnli', 'snli']:
        text = [p + f' {tokenizer.sep_token} ' + h for (p,h) in zip(batch['premise'], batch['hypothesis'])]
    elif dataset_name in ['sst2', 'cola']:
        text = batch['sentence']
    elif dataset_name in ['qnli']:
        text = [p + f' {tokenizer.sep_token} ' + h for (p,h) in zip(batch['question'], batch['sentence'])]
    elif dataset_name in ['qqp']:
        text = [p + f' {tokenizer.sep_token} ' + h for (p,h) in zip(batch['question1'], batch['question2'])]
    elif dataset_name in ['yelp_polarity', 'imdb']:
        text = batch['text']
    else:
        raise Exception('Dataset not found !')
    return tokenizer(text, truncation=True)


def prepare_data(dataset, tokenizer, dataset_name: str = 'rte'):
    """ do not forget batched=True """
    dataset = dataset.map(tokenize_dataset, fn_kwargs={'tokenizer': tokenizer, 'dataset_name': dataset_name}, batched=True)
    return dataset

