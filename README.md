# Experiments: alignment, end-to-end, and stress tests

Same structure as the original code: `kernels.py` (closed-form ReLU kernels, kernel ridge with 5-fold CV),
`two_layer_nn.py` (networks), `experiments.py` (experiments and plots), plus `data.py` (targets, input
distributions, stress scenarios).

```
pip install -r requirements.txt
python experiments.py alignment       # Fig. 1a rerun + the network after one step
python experiments.py e2e             # end-to-end experiment, original scenario (10 trials)
python experiments.py stress_align    # alignment stress test (14 scenarios x 10 trials)
python experiments.py stress_e2e      # end-to-end stress test (17 scenarios x 5 trials)
python experiments.py plots           # figures/*.pdf (curves only) and tables/*.tex from results/*.pkl
python experiments.py all             # all of the above
```

**Plotting options.** `python experiments.py plots` draws every stress scenario in one image
(`figures/stress_end_to_end.pdf`, `figures/stress_alignment.pdf`, one small panel per scenario).
`python experiments.py plots --layout group` draws one image per type of variation, each starting with the original
scenario: `figures/stress_e2e_{link,noise,input,dimension}.pdf` and `figures/stress_align_{link,noise,input}.pdf`
(the dimension figure plots against N/d; the alignment stress test has no dimension scenarios unless run with
`--dims`). `--zeta 0.75` picks the step size shown in these figures (default 0.875).
`python experiments.py plots --per-scenario` draws one image per scenario instead:
`figures/stress_e2e/<scenario>.pdf` (risk against N, laid out like the end-to-end figure) and
`figures/stress_align/<scenario>.pdf` (alignment against B at d = 300). If the alignment stress test is run with
`--dims 50 100 200 400 800 1600 3200`, each scenario is swept over d and its figure is laid out like the alignment
figure (against d, one panel per B); this takes several hours.
A subset of stress scenarios: `python experiments.py stress_e2e --scenarios base link_tanh in_t5`
(keys in `data.py`). Results are saved after every scenario, so a run can be stopped and resumed scenario by
scenario. Expected time on a laptop: about 2 hours, mostly the end-to-end stress test. Works on CPU or GPU.

## Alignment experiment (`alignment`)
Squared alignment <v_top, Y2(Z)>^2 on N = d Gaussian points (seed 54643, same draws as the original code), for
K0 and for the network after one gradient step (width 400, n = 2d samples). For each level B = 5 d^{p/10}, the
step size is solved from Eq. (3.3) so that the network's idealized kernel has this B; K1 uses the A and B of
that step, and K_D the same first-stage sample. The kernel K1 with the hand-set A = 1.2 of the original
experiment is no longer computed. Figure: `alignment_network.pdf` (one panel per B).

**Paper text vs. code.** The code computes the *squared* alignment with N = d, whereas Sec. 5.2 and App. D.3
write <v_top, Y2(Z)> and do not state N. Add the square and "N = d" to the text.

## End-to-end experiment (`e2e`)
The exact two-stage protocol of Sec. 2: one full-batch GD step on the first layer (Eq. 2.3) on n = 2d samples,
eta = 3 d^zeta with zeta in {5/8, 3/4, 7/8}, width m = 4d/3 (400 at d = 300); then ridge regression of the
second layer (Eq. 2.4) on N = d/4 ... 8d fresh samples, lambda by 5-fold CV; excess risk
||f_hat - f*||^2 / E[f*^2] on 2000 test points. Compared on the same second-stage data: the network,
random features at initialization (same width), K0, linear ridge, K_D and K1 (A, B from Eq. 3.3).
Original scenario: d = 300, g(z) = tanh(2z) + 2(e^{-1/2} - cos z) (bounded, Lipschitz, c1 = 0.73), Gaussian
noise with sigma = 0.3, x ~ N(0, I_d). Figure: `end_to_end.pdf`.

**Closer to the theory, same runtime.** `--alpha-e2e 8 --m-per-d 5.333` runs the end-to-end experiment and
its stress test with n = 8d first-stage samples and width m = 16d/3 (n = 2400, m = 1600 at d = 300). The step
size is scaled with the width (eta = 3 d^zeta * m/(4d/3)), so B is unchanged. The runtime is dominated by kernel
ridge on the N second-stage samples, which does not depend on n or m, so this costs about 10% more. Run it in a
copy of this folder to keep the default results:
```
cp -r user_style_code user_style_code_n8d
cd user_style_code_n8d && rm -rf results figures tables
python experiments.py e2e stress_e2e --alpha-e2e 8 --m-per-d 5.333
python experiments.py plots --layout group
```

## Stress tests (`stress_align`, `stress_e2e`)
One factor at a time, everything else at the original scenario, which is rerun as the first row:

| group | scenarios |
|---|---|
| bounded link (the assumption) | tanh(2z); sin(2z); cos(2z) (c1 = 0, violates c1 != 0) |
| unbounded link | 3z + 2z^2 + 4 sin(2z) (the original target); max(z, 0); e^z |
| noise | sigma = 0; sigma = 1; Student-t(3) noise with sd 1 |
| input law | Rademacher; uniform on the sphere sqrt(d) S^{d-1}; elliptical Student-t(5) with identity covariance; N(0, Sigma) with log-uniform spectrum in [1/4, 4] and random eigenbasis |
| dimension (end-to-end only) | d = 50, 100, 200 with n/d = 2, m/d = 4/3 |

Every link is centred and rescaled to the baseline E[g^2] = 1.434, so only its shape changes. A, B always come
from the Gaussian formula of Eq. (3.3); K_D from the realized first-stage data. Step sizes are never retuned:
eta = 3 d^zeta in the end-to-end test; in the alignment test eta is set once, on the original scenario, to give
B = 5 d^{p/10}, so a different link changes B through c1^2 as Eq. (3.3) predicts. In the alignment stress test
Y2 is centred by its sample mean (needed for anisotropic inputs). Figures: `stress_end_to_end.pdf`,
`stress_end_to_end_curves.pdf`, `stress_alignment.pdf`; tables: `tables/stress_e2e.tex`,
`tables/stress_align.tex`.

## Changes to the original files
* `kernels.py`: `compute_analytical_kernel` gives the same output without forming the d x d matrix Gamma;
  `get_best_lambda` runs the same 5-fold CV (same KFold seed) with one eigendecomposition per fold. Added the
  constants of Eq. (3.3), the single-dataset kernel, and the alignment helpers.
* `two_layer_nn.py`: added `OneStepNet` (the network of Sec. 2 and its gradient step). The original
  `TwoLayerNet`, `find_lr_kfold` and `train_model` are unchanged, except that the device falls back to CPU.
