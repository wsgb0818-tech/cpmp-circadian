# CPMP — Static phase-response heuristics over-promise circadian re-entrainment

Reproducible code for the manuscript:

> **Static phase-response heuristics systematically over-promise circadian re-entrainment: a limit-cycle reappraisal for cross-timezone travel guidance**
> Sha Wang. Submitted to *Journal of Theoretical Biology*.

This repository contains the complete, self-contained code that reproduces every
numerical result, figure, and table in the manuscript. It re-implements the
cubic (Van der Pol–type) limit-cycle model of the human circadian pacemaker
(Forger, Jewett & Kronauer, 1999; Kronauer, Forger & Jewett, 1999), calibrates
its single photic-gain parameter to the human light phase-response curve (PRC)
of Khalsa et al. (2003), and uses the calibrated model as ground truth to audit
the fixed-rate "static" jet-lag heuristic across nine long-haul routes and three
chronotypes.

## Requirements

* Python ≥ 3.10
* `numpy`
* `matplotlib`

Install with:

```bash
pip install -r requirements.txt
```

No other dependencies, datasets, or network access are required. All results are
produced by deterministic numerical integration.

## Reproduce all results and figures

Run the scripts in order (each is also runnable independently):

```bash
python step1_validate.py             # Figures 1–3: free-run, entrainment, light PRC
python step2_validate_literature.py  # Figure 4:    PRC calibration + fluence–response
python step3_compare.py              # Figures 5–6: static vs dynamical, 9 scenarios
python verify_independent.py         # independent numerical cross-checks
python graphical_abstract.py         # graphical abstract (optional)
```

Each figure is written as a vector PDF (`Figure_1.pdf` … `Figure_6.pdf`,
`Graphical_Abstract.pdf`) in the working directory.

## What each file does

| File | Purpose |
|------|---------|
| `kjf_engine.py` | Core model: the limit-cycle pacemaker (Process P), the dynamic photic drive (Process L), the ODE integrator, free-running-period and CBTmin extraction, and the single-pulse PRC routine. Defines `calibrated_params()` (photic gain G = 38.7, calibrated to the human PRC) and the published baseline (G = 19.875). |
| `comparison.py` | (a) A faithful port of the static jet-lag heuristic (mode classification, fixed-rate re-entrainment, prescribed light/melatonin windows); (b) simulation of the dynamical clock of a traveller who *follows* that static plan, so the mis-timing imposed on each intervention can be measured. |
| `step1_validate.py` | Generates Figures 1–3 (limit cycle in constant darkness; entrainment to a 24 h light–dark cycle; type-1 light PRC). |
| `step2_validate_literature.py` | Generates Figure 4: calibration of the model PRC against Khalsa et al. (2003) and the saturating fluence–response against Zeitzer et al. (2000). |
| `step3_compare.py` | Generates Figures 5–6 and the per-scenario table: static-assumed versus dynamical-actual CBTmin across the full scenario panel. |
| `calibrate_gain.py` | Root-finds the photic gain G that matches the human peak-to-trough PRC amplitude. |
| `verify_independent.py` | Independent numerical checks of the headline quantities (free-running period, amplitude, PRC extrema, etc.). |
| `graphical_abstract.py` | Builds the manuscript graphical abstract from real model output (Beijing → Los Angeles, −15 zones). |
| `requirements.txt` | Python dependencies. |
| `CITATION.cff` | Machine-readable citation metadata. |
| `LICENSE` | MIT License. |

## Key reproduced quantities

Free-running period 24.16 h; limit-cycle amplitude 1.007; calibrated photic gain
G = 38.7 (published baseline 19.875); calibrated peak-to-trough PRC amplitude
5.04 h (human reference 5.02 h); maximum advance ≈ +2.2 h, maximum delay
≈ −2.8 h; saturating fluence–response from ≈0.35 h at 10 lux to ≈2.5 h at
9500 lux; static heuristic over-promises re-entrainment by 2–4 days; assumed
CBTmin drifts from the true clock by up to 3.7 h; zero wrong-direction-light
days at the calibrated gain.

## Scope and intended use

This is **wellness / lifestyle** modelling code. It generates general guidance on
the timing of light, sleep, and (where the traveller chooses to use it) melatonin,
together with realistic expectations for adjustment. It does **not** diagnose or
treat any condition and makes no claim to reduce the risk of disease.

## Citation

If you use this software, please cite it via the metadata in `CITATION.cff`
(and the accompanying manuscript once published).

## License

Released under the MIT License — see `LICENSE`.
