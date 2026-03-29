"""
Matrix completion via gradient descent on a factored parametrization.

Parametrization:  X = U V^T,  U in R^{d1 x r},  V in R^{d2 x r}

Objective:
    min_{U,V}  (1/|Omega|) * sum_{(i,j) in Omega} ((UV^T)_ij - M_ij)^2
               + lam * ||UV^T||_*

Gradients (chain rule through X = UV^T):
    Let R = mask * (UV^T - M)          # residual on observed entries
    dL/dU = (2/|Omega|) * R @ V  +  lam * P Q^T V
    dL/dV = (2/|Omega|) * R^T @ U  +  lam * Q P^T U
where P diag(s) Q^T is the SVD of X, and P Q^T is the subgradient of ||X||_*.
"""

import numpy as np
import matplotlib.pyplot as plt


# ── core algorithm ────────────────────────────────────────────────────────────

def nuclear_norm(X: np.ndarray) -> float:
    return np.linalg.svd(X, compute_uv=False).sum()


def objective(X: np.ndarray, M: np.ndarray, mask: np.ndarray, lam: float) -> float:
    residual = np.where(mask, X - M, 0.0)
    mse = (residual ** 2).sum() / mask.sum()
    return mse # + lam * nuclear_norm(X)


def matrix_completion(
    M: np.ndarray,
    mask: np.ndarray,
    r: int = 5,
    lam: float = 0.1,
    lr: float = 0.01,
    n_iter: int = 500,
    tol: float = 1e-6,
    verbose: bool = False,
    seed: int = 0,
    M_true: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float], list[float]]:
    """
    Nuclear-norm regularized matrix completion via gradient descent on X = UV^T.

    Args:
        M:      Observed matrix (unobserved entries are ignored).
        mask:   Boolean array, True where entries are observed.
        r:      Rank of the factorization (U is d1 x r, V is d2 x r).
        lam:    Regularization strength for the nuclear norm.
        lr:     Step size (learning rate).
        n_iter: Maximum number of iterations.
        tol:    Convergence tolerance (Frobenius norm of change in X).
        verbose: Print loss every 50 iterations.
        seed:   RNG seed for initialization.
        M_true: If provided, track held-out NRMSE each iteration.

    Returns:
        X:      Completed matrix (= U @ V.T at convergence).
        losses: Objective value after each iteration.
        nrmses: Held-out NRMSE after each iteration (empty if M_true is None).
    """
    d1, d2 = M.shape
    n_obs = mask.sum()
    rng = np.random.default_rng(seed)

    U = rng.standard_normal((d1, r)) * 0.01
    V = rng.standard_normal((d2, r)) * 0.01
    losses = []
    nrmses = []

    held_out = ~mask
    M_true_sq_avg = np.linalg.norm(M_true, "fro")**2 / (d1 * d2)

    for t in range(n_iter):
        X = U @ V.T

        # --- residual on observed entries ---
        R = np.where(mask, X - M, 0.0)

        # --- subgradient of nuclear norm: P Q^T from SVD of X ---
        P, _, Qt = np.linalg.svd(X, full_matrices=False)
        nn_subgrad = P @ Qt          # shape (d1, d2)

        # --- gradients w.r.t. U and V ---
        grad_U = (2.0 / n_obs) * (R @ V) + lam * (nn_subgrad @ V)
        grad_V = (2.0 / n_obs) * (R.T @ U) + lam * (nn_subgrad.T @ U)

        U_new = U - lr * grad_U
        V_new = V - lr * grad_V

        X_new = U_new @ V_new.T
        losses.append(objective(X_new, M, mask, lam))
        mse = ((X_new - M_true) ** 2).mean()
        normalized_mse = mse / M_true_sq_avg
        nrmses.append(normalized_mse)

        if verbose and t % 1 == 0:
            print(f"iter {t:4d}  loss={losses[-1]:.4f}  normalized loss={nrmses[-1]:.4f}")

        if np.linalg.norm(X_new - X, "fro") < tol:
            U, V = U_new, V_new
            if verbose:
                print(f"Converged at iter {t}.")
            break

        U, V = U_new, V_new

    return U @ V.T, losses, nrmses


# ── demo ──────────────────────────────────────────────────────────────────────

def make_low_rank_matrix(m: int, n: int, rank: int, noise: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((m, rank))
    V = rng.standard_normal((n, rank))
    M = U @ V.T
    M += noise * rng.standard_normal((m, n))
    return M


def random_mask(m: int, n: int, obs_fraction: float, seed: int = 1):
    rng = np.random.default_rng(seed)
    return rng.random((m, n)) < obs_fraction


if __name__ == "__main__":
    # ── generate a rank-5 matrix with 10 % of entries observed ──
    m, n, true_rank = 200, 100, 5
    M_true = make_low_rank_matrix(m, n, true_rank, noise=0.00)
    mask = random_mask(m, n, obs_fraction=0.90)

    M_obs = M_true * mask          # zeros at unobserved entries

    # ── run matrix completion ──
    X_hat, losses, nrmses = matrix_completion(
        M_obs, mask,
        r=true_rank,
        lam=0.001,
        lr=10.0,
        n_iter=100,
        tol=1e-7,
        verbose=True,
        M_true=M_true,
    )

    # ── evaluate ──
    # M_true_fro = np.linalg.norm(M_true, "fro")
    # held_out = ~mask
    # train_rmse = np.sqrt(((X_hat - M_true)[mask] ** 2).mean())
    # held_out_rmse = np.sqrt(((X_hat - M_true)[held_out] ** 2).mean())
    # print(f"\nTrain NRMSE:    {train_rmse / M_true_fro:.4f}")
    # print(f"Held-out NRMSE: {held_out_rmse / M_true_fro:.4f}")
    # print(f"Effective rank (singular values > 0.01): "
    #      f"{(np.linalg.svd(X_hat, compute_uv=False) > 0.01).sum()}")

    # ── plot convergence ──
    plt.figure(figsize=(6, 3))
    plt.plot(nrmses)
    plt.xlabel("Iteration")
    plt.ylabel("Normalized Relative MSE")
    plt.title("Nuclear-norm regularized low-rank matrix completion")
    plt.tight_layout()
    plt.savefig("../figures/mc_convergence.png", dpi=150)
    plt.show()
