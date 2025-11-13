"""
 * Software Name: task-inclusion-estimation
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the MIT license,
 * the text of which is available at https://opensource.org/license/MIT/
 * or see the "LICENSE" file for more details.
"""
from .onto_notes_dataset import onto_notes_dataset

class DataLauncher(object):
    def __init__(self, 
                 name: str, 
                 debug: bool = False, 
                 tokenizer = None, 
                 corpus_name: str = None, 
                 neutral_prompt: bool = False):
        self.name: str = name
        self.loader = None
        self.corpus_name = corpus_name

        if self.name.startswith("onto_notes_"):
            self.loader = onto_notes_dataset(
                debug=debug, 
                corpus_name=corpus_name, 
                neutral_prompt=neutral_prompt)
        else:
            raise NotImplementedError('The dataset you are asking for is not in the book !')

    def __call__(self,):
        return self.loader(self.name)
