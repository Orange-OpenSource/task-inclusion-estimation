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
import json
import torch
from pathlib import Path
from logzero import logger as log
import os
import sys
sys.path.append(os.getcwd())
from src.utils import hash_args, save_args
from src.information_sufficiency.utils import (
    load_embeddings,
    load_labels,
    eval_distance,
    ARGUMENTS_CLASSES,
    ESTIMATORS_CLASSES,
)

def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", type=str, default="mi")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arguments-dir", type=Path, required=True)

    parser.add_argument("--model-x", type=Path, required=True)
    parser.add_argument("--model-y", type=Path, required=True)

    # normalization arguments
    parser.add_argument("--normalize-embeddings", action="store_true", default=False)

    # pca arguments
    parser.add_argument("--pca-reduction", action="store_true", default=False)
    parser.add_argument("--pca-nb-dim", type=int, default=2)
    parser.add_argument("--pca-dynamic-reduction", action="store_true", default=False)
    parser.add_argument("--pca-var-th", type=float, default=0.8)

    # clustering arguments
    parser.add_argument("--label-path", type=Path, required=True)
    parser.add_argument("--create-labels", action="store_true", default=False)
    parser.add_argument("--clustering-params-dir", type=Path)
    parser.add_argument("--n-clusters", type=int, default=-1)
    parser.add_argument("--clustering-type", type=str, 
                        default="KMeans", choices=["GaussianMixture", "KMeans"])
    return parser.parse_args()

def main():
    args = _get_arguments()
    expe_hash = hash_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_args(args.output_dir / f"script_arguments_{expe_hash}.json", args)

    log.info("Load model and arguments")
    estimator_args = ARGUMENTS_CLASSES[args.estimator].from_json_path(
        json_path=args.arguments_dir
    )
    estimator_class = ESTIMATORS_CLASSES[args.estimator]
    
    log.info("Load embeddings")
    x : torch.Tensor = load_embeddings(
        model_path=args.model_x,
        normalize=args.normalize_embeddings,
        pca_reduction=args.pca_reduction,
        pca_dynamic_reduction=args.pca_dynamic_reduction,
        pca_nb_dim=args.pca_nb_dim,
        pca_var_th=args.pca_var_th,
    )
    y : torch.Tensor = load_embeddings(
        model_path=args.model_y,
        normalize=args.normalize_embeddings,
        pca_reduction=args.pca_reduction,
        pca_dynamic_reduction=args.pca_dynamic_reduction,
        pca_nb_dim=args.pca_nb_dim,
        pca_var_th=args.pca_var_th,
    )
    assert x.shape[0] == y.shape[0], "The number of embeddings must be the same"

    log.info("Clustering")
    try:
        with args.clustering_params_dir.open("r") as f:
            clustering_params = json.load(f)
        clustering_params["n_clusters"] = args.n_clusters
        labels, unique_labels = load_labels(
            label_path=args.label_path,
            create_labels=args.create_labels,
            clustering_type=args.clustering_type,
            clustering_params=clustering_params,
            K=y,
        )
    except Exception:
        labels = None
    try:
        estimator_args.num_labels = len(unique_labels)
        log.info(unique_labels)
    except Exception:
        log.info("No num labels in estimator_args")

    log.info("Evaluate the distances between the embeddings")
    eval_distance(
        embeddings_x=x,
        embeddings_y=y,
        labels=labels,
        model_x_path=args.model_x,
        model_y_path=args.model_y,
        estimator_args=estimator_args,
        estimator_class=estimator_class,
        output_dir=args.output_dir,
        expe_hash=expe_hash,
    )
    return 0

if __name__ == "__main__":
    _ = main()