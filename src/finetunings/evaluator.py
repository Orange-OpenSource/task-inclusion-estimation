"""
 * Software Name: task-inclusion-estimation
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the MIT license,
 * the text of which is available at https://opensource.org/license/MIT/
 * or see the "LICENSE" file for more details.
"""
import numpy as np
from typing import Dict
from typing import List
import evaluate

class Evaluator:
    def __init__(self,):
        self.metrics = {
            "mauve": evaluate.load("mauve"),
            "bleu": evaluate.load("bleu"),
            "rouge": evaluate.load("rouge"),
            "meteor": evaluate.load("meteor"),
            "bertscore": evaluate.load("bertscore"),
        }
        self.metrics_kwargs = {
            "mauve": {},
            "bleu": {},
            "rouge": {"use_aggregator": False},
            "meteor": {},
            "bertscore": {"lang": "en"},
        }
    def __call__(self, predictions: List[str], references: List[str]|List[List[str]]) -> Dict:
        output = {}
        for metric in self.metrics:
            buffer_res = self.metrics[metric].compute(
                predictions=predictions, references=references, **self.metrics_kwargs[metric]
            )
            if not isinstance(buffer_res, dict):
                buffer_res = buffer_res.__dict__
            for k in buffer_res:
                if isinstance(buffer_res[k], np.ndarray):
                    buffer_res[k] = buffer_res[k].tolist()
            output = {
                **output,
                **{
                    f"{metric}_{k}": buffer_res[k] for k in buffer_res
                }
            }
        return output