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
