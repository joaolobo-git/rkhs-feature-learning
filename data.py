"""
Targets, input distributions and the stress-test scenarios.

Original scenario (baseline of every stress test):
    link  g(z) = tanh(2z) + 2(e^{-1/2} - cos z)   bounded, Lipschitz, c1 = E[g'] = 0.73 (Assumption 2.1)
    noise Gaussian, sigma = 0.3
    input x ~ N(0, I_d), d = 300
Each stress scenario changes ONE of these:
    bounded link    tanh(2z), sin(2z), cos(2z) [c1 = 0: violates Assumption 2.1]
    unbounded link  3z + 2z^2 + 4 sin(2z) [the original paper target], max(z, 0), e^z
    noise           sigma = 0, sigma = 1, Student-t(3) noise with standard deviation 1
    input law       Rademacher, uniform on the sphere sqrt(d) S^{d-1}, elliptical Student-t(5) with identity
                    covariance, anisotropic Gaussian N(0, Sigma) (log-uniform spectrum in [1/4, 4], random eigenbasis)
    dimension       d = 50, 100, 200 (end-to-end only; the alignment experiment already sweeps d)
Every link is centred and rescaled to the baseline energy E[g^2] = 1.434, so only its shape changes.
"""
import numpy as np
from kernels import link_statistics


def _main_link(z):
    return np.tanh(2 * z) + 2 * (np.exp(-0.5) - np.cos(z))


LINKS = {
    "main":  ("bounded",   _main_link),
    "tanh":  ("bounded",   lambda z: np.tanh(2 * z)),
    "sin":   ("bounded",   lambda z: np.sin(2 * z)),
    "cos":   ("bounded",   lambda z: np.cos(2 * z)),
    "paper": ("unbounded", lambda z: 3 * z + 2 * z ** 2 + 4 * np.sin(2 * z)),
    "relu":  ("unbounded", lambda z: np.maximum(z, 0.0)),
    "exp":   ("unbounded", lambda z: np.exp(z)),
}
EF2_REF = link_statistics(_main_link)["Ef2"]


class Target:
    """y = g(z) + noise, z = <w*, x> (standardised). Holds the Gaussian statistics used by Eq. (3.3)."""

    def __init__(self, link="main", sigma=0.3, noise="gauss"):
        g = LINKS[link][1]
        st = link_statistics(g)
        if link != "main":                      # centre and rescale to the baseline energy
            scale, mean = np.sqrt(EF2_REF / (st["Ef2"] - st["mean"] ** 2)), st["mean"]
            g = (lambda z, g=g, scale=scale, mean=mean: scale * (g(z) - mean))
            st = link_statistics(g)
        self.g, self.sigma, self.noise = g, sigma, noise
        self.c1, self.s, self.Ef2 = st["c1"], st["s"], st["Ef2"]
        self.Ey2 = self.Ef2 + sigma ** 2

    def draw_noise(self, n):
        if self.noise == "gauss":
            return self.sigma * np.random.randn(n)
        return self.sigma * np.random.standard_t(3, size=n) / np.sqrt(3.0)   # unit-variance t_3


class InputLaw:
    """All laws have E[x] = 0 and E[x x^T] = I_d, except 'aniso' (E[x x^T] = Sigma, tr Sigma = d)."""

    def __init__(self, kind, d, w_star):
        self.kind, self.d, self.w_star, self.scale = kind, d, w_star, 1.0
        if kind == "aniso":
            lam = np.exp(np.linspace(np.log(0.25), np.log(4.0), d))
            lam *= d / lam.sum()
            Q, _ = np.linalg.qr(np.random.randn(d, d))
            self.S = (Q * np.sqrt(lam)) @ Q.T                                  # Sigma^{1/2}
            self.scale = float(np.sqrt(w_star @ self.S @ self.S @ w_star))     # sd of <w*, x>

    def sample(self, n):
        d = self.d
        if self.kind == "gauss":
            return np.random.randn(n, d)
        if self.kind == "rademacher":
            return np.random.choice([-1.0, 1.0], size=(n, d))
        if self.kind == "sphere":
            X = np.random.randn(n, d)
            return np.sqrt(d) * X / np.linalg.norm(X, axis=1, keepdims=True)
        if self.kind == "t5":
            return np.random.randn(n, d) * np.sqrt(3.0 / np.random.chisquare(5, size=n))[:, None]
        if self.kind == "aniso":
            return np.random.randn(n, d) @ self.S
        raise ValueError(self.kind)

    def index(self, X):
        return X @ self.w_star / self.scale

    def Ef2(self, T, n_mc=200_000, chunk=20_000):
        """E[f*(x)^2] under this law: exact when <w*, x> is Gaussian, Monte Carlo otherwise
        (on a separate random stream, so the experiment's stream is not disturbed)."""
        if self.kind in ("gauss", "aniso"):
            return T.Ef2
        state = np.random.get_state()
        np.random.seed(777)
        tot = sum(np.sum(T.g(self.index(self.sample(chunk))) ** 2) for _ in range(n_mc // chunk))
        np.random.set_state(state)
        return tot / n_mc


def sample_data(law, T, n):
    X = law.sample(n)
    f = T.g(law.index(X))
    return X, f + T.draw_noise(n), f


# ---------------------------------------------------------
# Scenario list
# ---------------------------------------------------------
BASE = dict(key="base", group="original", label="original scenario",
            link="main", sigma=0.3, noise="gauss", law="gauss", d=300)


def _sc(key, group, label, **kw):
    return {**BASE, "key": key, "group": group, "label": label, **kw}


SCENARIOS = [
    BASE,
    _sc("link_tanh", "bounded link", r"$\tanh(2z)$", link="tanh"),
    _sc("link_sin", "bounded link", r"$\sin(2z)$", link="sin"),
    _sc("link_cos", "bounded link", r"$\cos(2z)$ ($c_1=0$)", link="cos"),
    _sc("link_paper", "unbounded link", r"$3z+2z^2+4\sin(2z)$", link="paper"),
    _sc("link_relu", "unbounded link", r"$\max(z,0)$", link="relu"),
    _sc("link_exp", "unbounded link", r"$e^{z}$", link="exp"),
    _sc("noise_0", "noise", r"$\sigma=0$", sigma=0.0),
    _sc("noise_1", "noise", r"$\sigma=1$", sigma=1.0),
    _sc("noise_t3", "noise", r"Student-$t_3$, sd $1$", sigma=1.0, noise="t3"),
    _sc("in_rad", "input law", r"Rademacher", law="rademacher"),
    _sc("in_sphere", "input law", r"sphere $\sqrt{d}\,\mathbb{S}^{d-1}$", law="sphere"),
    _sc("in_t5", "input law", r"elliptical $t_5$", law="t5"),
    _sc("in_aniso", "input law", r"anisotropic $\Sigma$", law="aniso"),
    _sc("d50", "dimension", r"$d=50$", d=50),
    _sc("d100", "dimension", r"$d=100$", d=100),
    _sc("d200", "dimension", r"$d=200$", d=200),
]
SC = {s["key"]: s for s in SCENARIOS}

# Types of variation, for the grouped figures: (scenario groups, title of the original-scenario panel)
VARIATIONS = {
    "link":      (["bounded link", "unbounded link"], r"original: $\tanh(2z)+2(e^{-1/2}-\cos z)$"),
    "noise":     (["noise"], r"original: Gaussian, $\sigma=0.3$"),
    "input":     (["input law"], r"original: Gaussian inputs"),
    "dimension": (["dimension"], r"$d=300$ (original)"),
}
