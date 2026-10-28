import numpy as np
from scipy.linalg import eigh
from scipy.special import roots_hermitenorm
from sklearn.model_selection import KFold

MU_SIGMA = 0.5   # first Hermite coefficient of ReLU, E[relu'(z)]: the prefactor of the rank-one update


# ---------------------------------------------------------
# 1. Closed-form ReLU kernels (Eq. 5.1)
# ---------------------------------------------------------
def compute_analytical_kernel(X, X_prime, v, A=1.4, B=1.0):
    """
    Computes the exact closed-form ReLU kernel
        k(x, x') = E_{w ~ N(0, Gamma/d)}[relu(<w,x>) relu(<w,x'>)],   Gamma = A I_d + B v v^T.
    k_0 is A=1, B=0. Same output as the original implementation, but without forming the d x d matrix
    Gamma (x^T Gamma x' = A <x,x'> + B <x,v><x',v>), which makes d = 3200 cheap.
    """
    d = X.shape[1]

    # 1. Gamma inner products x^T Gamma x' and Gamma norms x^T Gamma x
    inner_prod = A * (X @ X_prime.T)
    diag_X = A * np.einsum("ij,ij->i", X, X)
    diag_Xp = A * np.einsum("ij,ij->i", X_prime, X_prime)
    if B != 0:
        p, p_prime = X @ v, X_prime @ v
        inner_prod = inner_prod + B * np.outer(p, p_prime)
        diag_X = diag_X + B * p ** 2
        diag_Xp = diag_Xp + B * p_prime ** 2

    # Outer product of norms to form the denominator matrix
    norm_matrix = np.sqrt(np.outer(diag_X, diag_Xp))

    # 2. Warped cosine similarity (clipped so that arccos never returns NaN)
    gamma_Gamma = np.clip(inner_prod / norm_matrix, -1.0, 1.0)

    # 3. Bracketed term and scaling factor of the closed form
    bracket_term = gamma_Gamma * (np.pi - np.arccos(gamma_Gamma)) + np.sqrt(1.0 - gamma_Gamma ** 2)
    scaling_factor = norm_matrix / (2 * np.pi * d)
    return scaling_factor * bracket_term


# ---------------------------------------------------------
# 2. Kernel ridge regression with 5-fold CV
# ---------------------------------------------------------
def get_best_lambda(K_full, y_full, lambdas):
    """
    Same 5-fold cross-validation as the original (KFold, random_state=38182, mean fold MSE), but with one
    eigendecomposition per fold instead of one linear solve per (fold, lambda): identical choice of lambda,
    much faster for long lambda grids.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=38182)
    lambdas = np.asarray(lambdas)
    fold_errors = np.zeros(len(lambdas))
    for train_idx, val_idx in kf.split(K_full):
        mu, V = eigh(K_full[np.ix_(train_idx, train_idx)])
        mu = np.maximum(mu, 0.0)
        Vy = V.T @ y_full[train_idx]
        KV = K_full[np.ix_(val_idx, train_idx)] @ V
        for i, l in enumerate(lambdas):
            preds = KV @ (Vy / (mu + l))
            fold_errors[i] += np.mean((preds - y_full[val_idx]) ** 2) / kf.get_n_splits()
    return lambdas[int(np.argmin(fold_errors))]


LAMBDA_REL = np.logspace(-8, 1, 28)   # ridge grid relative to the kernel's scale mean(diag K)


def krr_excess_risk(K_train, K_test, y_train, f_test, Ef2):
    """Fit kernel ridge on (K_train, y_train) with 5-fold CV; return ||f_hat - f*||^2 / E[f*^2] on the test set."""
    lambdas = np.mean(np.diag(K_train)) * LAMBDA_REL
    best_l = get_best_lambda(K_train, y_train, lambdas)
    alpha = np.linalg.solve(K_train + best_l * np.eye(len(y_train)), y_train)
    preds = K_test @ alpha
    return float(np.mean((preds - f_test) ** 2) / Ef2), float(best_l)


# ---------------------------------------------------------
# 3. Constants of the kernels after one step (Eq. 3.3 and the single-dataset kernel k_D)
# ---------------------------------------------------------
def link_statistics(g, n_quad=300):
    """Gaussian statistics of a link g, z ~ N(0,1): mean, E[g^2], c1 = E[g'] = E[z g], s = E[g'^2] + E[g g'']."""
    z, w = roots_hermitenorm(n_quad)
    w = w / w.sum()
    gz = g(z)
    return {"mean": float(w @ gz), "Ef2": float(w @ gz ** 2), "c1": float(w @ (gz * z)),
            "s": float(0.5 * w @ ((z ** 2 - 1) * gz ** 2))}


def idealized_constants(d, n, m, eta, c1, s, Ey2):
    """A, B of the idealized kernel k_1 = k_{A I + B w* w*^T} (Eq. 3.3), with the update prefactor mu_sigma."""
    c = d * MU_SIGMA ** 2 * eta ** 2 / (n * m ** 2)
    return 1.0 + c * Ey2, c * (2 * s + (n - 1) * c1 ** 2)


def single_dataset_b(d, m, eta, h):
    """b_D of the single-dataset kernel k_D = k_{I + b_D hhat hhat^T}, h = X^T y / n of the first-stage sample."""
    return d * MU_SIGMA ** 2 * eta ** 2 * float(h @ h) / m ** 2


def eta_for_B(d, n, m, B, c1, s):
    """Step size for which Eq. (3.3) gives the requested B."""
    c = B / (2 * s + (n - 1) * c1 ** 2)
    return float(np.sqrt(c * n * m ** 2 / (d * MU_SIGMA ** 2)))


# ---------------------------------------------------------
# 4. Alignment with the quadratic harmonic Y_2 (App. D.3)
# ---------------------------------------------------------
def y2_feature(X, w_star, center="paper"):
    """Normalised vector of Y2(omega_i, w*) = <omega_i, w*>^2 - c, omega_i = x_i/||x_i||.
    center='paper': c = 1/d (App. D.3).  center='empirical': c = sample mean (needed for anisotropic inputs)."""
    d = X.shape[1]
    proj2 = (X @ w_star) ** 2 / np.einsum("ij,ij->i", X, X)
    y_2 = proj2 - (1.0 / d if center == "paper" else proj2.mean())
    return y_2 / np.linalg.norm(y_2)


def top_eigenvector(K):
    N = K.shape[0]
    _, vecs = eigh(K, subset_by_index=[N - 1, N - 1])   # same as eigh(K)[1][:, -1]
    return vecs[:, -1]


def alignment(K, y_2):
    """Squared alignment <v_top, Y2(Z)>^2, as in the original code (sign-free)."""
    return float(np.dot(top_eigenvector(K), y_2) ** 2)
