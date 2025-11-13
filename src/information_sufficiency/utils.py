"""
 * Software Name: task-inclusion-estimation
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the MIT license,
 * the text of which is available at https://opensource.org/license/MIT/
 * or see the "LICENSE" file for more details.
"""
from logzero import logger as log
from sklearn.decomposition import PCA
import torch, json
from emir import KernelClustering
from typing import Dict, List
from pathlib import Path
import pandas as pd
from emir import (
    KNIFEEstimator, KNIFEArgs,
)

ARGUMENTS_CLASSES = {
    "mi" : KNIFEArgs,
}
ESTIMATORS_CLASSES = {
    "mi" : KNIFEEstimator,
}

def load_labels(
        label_path: Path = None, 
        create_labels: bool = False, 
        clustering_type: str = "GaussianMixture", 
        clustering_params: Dict = None,
        K: torch.tensor = None,
    ):
    if not create_labels: # load the true labels
        log.info("loading the true labels !")
        with (label_path / "labels.json").open("r") as f:
            labels = torch.tensor(json.load(f))
    else: # create labels from a clustering algorithm
        log.info("creating labels from a clustering algorithm !")
        clustering = KernelClustering(
            clustering_type=clustering_type,
            clustering_params=clustering_params,
        )
        labels = clustering.create_concept(K)
    return labels, torch.unique(labels).tolist()

def load_embeddings(
        model_path: Path|List[Path], 
        normalize: bool,
        pca_reduction: bool,
        pca_dynamic_reduction: bool,
        pca_var_th: float,
        pca_nb_dim: int,
) -> torch.Tensor:
    """load_embeddings

    This function allows to load stored embeddings

    Args:
        model_path (Path | List[Path]): Where the embeddings are stored
        normalize (bool): Should we normalize the embeddings ?
        pca_reduction (bool): Should we do pca reduction on the embeddings ?
        pca_dynamic_reduction (bool): Should we proceed dynamic reduction on the embeddings ?
        pca_var_th (float): Variance threshold for the pca reduction
        pca_nb_dim (int): Number of dimension to keep on the pca reduction

    Returns:
        torch.Tensor: Tensor with a shape of `(nb_embeddings, embeddings_dimension)`
    """
    emb : torch.Tensor = None
    if isinstance(model_path, Path):
        emb = torch.tensor(torch.load(model_path, weights_only=False))
    elif isinstance(model_path, List):
        buffer : List[torch.Tensor] = []
        for d in model_path:
            buffer.append(torch.load(d, weights_only=False))
        emb = torch.cat(buffer, 0)
    log.info(f"{emb.shape}")
    if normalize:
        log.info("normalization")
        mean, std = emb.mean(0), emb.std(0)
        embeddings = (emb - mean) / std
    elif pca_reduction:
        log.info("pca reduction")
        pca = PCA(n_components=pca_nb_dim)
        embeddings = torch.tensor(pca.fit_transform(emb)).to(torch.float)
    elif pca_dynamic_reduction:
        log.info("*** Dynamic dimension reduction ***")
        pca = PCA(n_components=emb.shape[-1]) # full dimension (only a rotation of the space)
        embeddings = torch.tensor(pca.fit_transform(emb)).to(torch.float)
        cum_var = pca.explained_variance_ratio_
        nb_useless_dims = (cum_var.cumsum() >= pca_var_th).sum()
        nb_dims = embeddings.shape[-1] - nb_useless_dims
        embeddings = embeddings[:, :(nb_dims+2)] # do the maths to justify the +2
    else:
        log.info("*** No embedding normalization ***")
        embeddings = emb
    return embeddings

def eval_distance(
    embeddings_x: torch.Tensor,
    embeddings_y: torch.Tensor,
    labels: torch.Tensor,
    model_x_path: Path,
    model_y_path: Path,
    estimator_args,
    estimator_class,
    output_dir: Path,
    expe_hash: str,
):
    """eval_distance

    Here this function must be interprated as from x to y 

    Args:
        embeddings_x (torch.Tensor): x embeddings
        embeddings_y (torch.Tensor): y embeddings
        labels (torch.Tensor): the labels
        model_x_path (Path): where the embeddings of x are stored
        model_y_path (Path): where the embeddings of y are stored
        estimator_args (_type_): class of the argument of the estimator
        estimator_class (_type_): class of the chosen estimator
        output_dir (Path): direction to solve the results
        expe_hash (str): unique hash to identify correctly the experiment
    """
    assert embeddings_x.shape[0] == embeddings_y.shape[0], "same number of embeddings"
    date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    torch.cuda.empty_cache()

    d_1 = embeddings_x.shape[1]
    d_2 = embeddings_y.shape[1]

    estimator = estimator_class(
        args=estimator_args,
        x_dim=embeddings_x.shape[1],
        y_dim=embeddings_y.shape[1],
    )

    metrics, preds, loss_metrics, eval_idx = estimator.eval(
        x=embeddings_x, y=embeddings_y, labels=labels,
        pbar=False,
    )

    estimator_args_dict = {k: v for k, v in estimator_args.__dict__.items()}

    res = {
        "date": date,
        "model_1": str(model_x_path),
        "model_2": str(model_y_path),
        "d_1": d_1,
        "d_2": d_2,
        **metrics,          
        **estimator_args_dict,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"metrics_{expe_hash}.json").open("w") as f:
        json.dump(res, f)
    with (output_dir / f"preds_{expe_hash}.json").open("w") as f:
        json.dump(preds, f)
    with (output_dir / f"loss_metrics_{expe_hash}.json").open("w") as f:
        json.dump(loss_metrics, f)
    try:
        with (output_dir / f"eval_idx_{expe_hash}.json").open("w") as f:
            json.dump(eval_idx, f)
    except Exception as e:
        print(e)
        print("Problem with `eval_idx`")


def get_transport_plan():
    pass