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
from torch import Tensor
from logzero import logger as log

def torch_orth(
        X: Tensor,
        rcond = None,
        diff_mode: bool = False,
):
    """ 
    Torch version of the scipy.linalg.orth : https://github.com/scipy/scipy/blob/v1.11.4/scipy/linalg/_decomp_svd.py#L287-L332

    Warning :
    - to avoid gradient problems, make sure that X is a full rank matrix
    - if X is not full rank the pseudo inverse of singular values is not definite
    """
    # --- singular value decomposition --- #
    U, S, Vh = torch.linalg.svd(X)
    M, N = U.shape[0], Vh.shape[1]

    # --- rank calculus --- #
    if rcond is None:
        rcond = torch.finfo(S.dtype).eps * max(M, N)
    tol = torch.amax(S) * rcond
    num = torch.sum(S > tol, dtype=int)
    if diff_mode and num != U.shape[-1]:
        log.warning("Rank is not full: gradient problems")

    # --- get only the right vectors --- #
    Q = U[:, :num]
    return Q

def torch_grassman_distance(
        A: Tensor,
        B: Tensor,
        give_born_sup: bool = False,
):
    """ New version with torch to keep the gradient"""
    
    assert A.shape == B.shape, 'error : the shapes must be the same !'

    # --- get orthonormal basis --- #
    M_A = torch_orth(A)
    M_B = torch_orth(B)

    # --- get the rank --- #
    r_A = M_A.shape[-1]
    r_B = M_B.shape[-1]
    if r_A == 0 or r_B == 0:
        return 0.0
    
    same_dim = r_A == r_B

    M = torch.matmul(torch.transpose(M_A,0,1), M_B)
    s = torch.linalg.svdvals(M)
    
    # --- get argument --- #
    s = torch.clamp(s, min=-0.9999, max=0.9999) # for numerical stability : else gradient can go to infinite
    s = torch.arccos(s)                         # non-diff at edge of space
    
    if not same_dim: # complete the list of angles with pi / 2
        assert len(s) == min(r_A, r_B)
        dif = max(r_A, r_B) - min(r_A, r_B)
        e = torch.tensor([torch.pi / 2.]*dif, dtype=float, device=s.device)
        s = torch.concatenate([s, e], axis=-1)

    if give_born_sup:
        union_dim = torch.linalg.matrix_rank(torch.concatenate([A, B], dim=-1))
        inter_dim = r_A + r_B - union_dim
        sup_bound = (r_A - inter_dim) * torch.pi / 2
        return torch.norm(s), sup_bound
    else:
        return torch.norm(s)

def matrix_cosine_distance(
    A: Tensor,
    B: Tensor,
):
    A = torch.flatten(A)
    B = torch.flatten(B)
    res = torch.dot(A, B) / torch.norm(A) / torch.norm(B)
    return res.item()

def matrix_l2_distance(
    A: Tensor,
    B: Tensor,
):
    A = torch.flatten(A)
    B = torch.flatten(B)
    res = torch.norm(A - B)
    return res.item()

def matrix_frobenius(
    A: Tensor,
    B: Tensor,      
):
    return torch.norm(A - B, p="fro") # compute the Frobenius norm
