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
sys.path.append(os.getcwd())
from numpy import ndarray
from torch import Tensor, save
import json
from typing import Dict, List
import hashlib
from pathlib import Path

def create_hash(name: str):
    hash_object = hashlib.sha256(name.encode())
    hash_hex = hash_object.hexdigest()
    return hash_hex

def read_json(path: str):
    path : Path = Path(path)
    path.touch(exist_ok=True)
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None

def write_json(path: str|Path, data: Dict|List):
    path : Path = Path(path)
    path.touch(exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=4)

def write_tensor(path: str, data: Tensor|ndarray):
    path = Path(path)
    if isinstance(data, Tensor): # always save to numpy format
        data = data.numpy()
    with path.open("wb") as f:
        save(data, f)

def is_serializable(obj)->bool:
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False

def serialize_dict(d: Dict)->Dict:
    res = {}
    for key, value in d.items():
        if is_serializable(value):
            res[key] = value
        else:
            res[key] = str(value)
    return res

def args_to_json(args)->Dict:
    res = vars(args)
    res = serialize_dict(res)
    return res

def hash_args(args)->str:
    d = serialize_dict(vars(args))
    return create_hash(json.dumps(d))

def save_args(output_dir: str|Path, args):
    d = args_to_json(args)
    write_json(output_dir, d)
