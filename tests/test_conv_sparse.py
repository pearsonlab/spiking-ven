"""The three FFT operations must match direct implementations of the same sums.

Everything in convolutional sparse coding rides on getting the alignment right, and an
off-by-one in the correlation lag is invisible in the loss curve -- it just learns a
worse dictionary. These pin each operation against the explicit loop form.
"""

import numpy as np
import pytest

from spiking_ven import conv_sparse


@pytest.fixture
def case():
    rng = np.random.default_rng(0)
    K, C, L, T = 3, 4, 5, 20
    A = rng.standard_normal((K, C, L))
    S = (rng.random((K, T)) < 0.2) * rng.standard_normal((K, T))
    R = rng.standard_normal((C, T))
    return K, C, L, T, A, S, R


def test_reconstruction_matches_explicit_sum(case):
    K, C, L, T, A, S, _ = case
    ref = np.zeros((C, T))
    for k in range(K):
        for t in range(T):
            for tau in range(L):
                if 0 <= t - tau < T:
                    ref[:, t] += S[k, t - tau] * A[k, :, tau]
    np.testing.assert_allclose(conv_sparse.reconstruct(S, A, T), ref, atol=1e-10)


def test_code_gradient_matches_explicit_sum(case):
    K, C, L, T, A, _, R = case
    ref = np.zeros((K, T))
    for k in range(K):
        for t in range(T):
            for tau in range(L):
                if t + tau < T:
                    ref[k, t] += (R[:, t + tau] * A[k, :, tau]).sum()
    np.testing.assert_allclose(conv_sparse.corr_with_atoms(R, A, T), ref, atol=1e-10)


def test_atom_gradient_matches_explicit_sum(case):
    K, C, L, T, _, S, R = case
    ref = np.zeros((K, R.shape[0], L))
    for k in range(K):
        for tau in range(L):
            for t in range(T):
                if t + tau < T:
                    ref[k, :, tau] += R[:, t + tau] * S[k, t]
    np.testing.assert_allclose(conv_sparse.corr_with_code(R, S, L, T), ref, atol=1e-10)


def test_inference_reduces_the_objective(case):
    """FISTA must actually descend; a sign error in the gradient still "runs"."""
    K, C, L, T, A, _, _ = case
    A /= np.linalg.norm(A.reshape(K, -1), axis=1)[:, None, None]
    rng = np.random.default_rng(1)
    X = rng.standard_normal((C, T)) * 0.1
    lam = 0.01

    def obj(S):
        R = X - conv_sparse.reconstruct(S, A, T)
        return 0.5 * float((R ** 2).sum()) + lam * float(np.abs(S).sum())

    S = conv_sparse.infer(X, A, lam, n_iter=60)
    assert obj(S) < obj(np.zeros((K, T))), "inference did not improve on the zero code"
