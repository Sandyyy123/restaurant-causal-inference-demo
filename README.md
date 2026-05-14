# Restaurant Pilot: Causal Inference Demo

**Analyst:** Dr. Sandeep Grover, PhD | Data Scientist & Statistician
**Contact:** [Upwork profile]
**Scholar:** https://scholar.google.com/citations?user=Ge4EC8cAAAAJ

---

## Overview

This repository demonstrates the analytical approach I would apply to the Bespoke AI
restaurant pilot - measuring whether AI coaching software causally lifts average check
size and items per transaction across 6-7 locations.

The code uses simulated POS data (24 months, 7 locations, 3 treated / 4 control) to
walk through the full Phase 1 and Phase 2 workflow described in the job posting.

---

## Repository Structure

```
.
- data/
  - simulate_pos_data.py      # Generate 24-month POS data (7 locations)
  - pos_data.csv              # Simulated dataset (auto-generated)
- analysis/
  - 01_power_analysis.py      # MDE calculation + power curves (small N)
  - 02_did_analysis.py        # Two-way FE DiD + parallel trends test
  - 03_synthetic_control.py   # Synthetic control per treated unit
- reports/                    # Output plots (auto-generated)
- PRE_REGISTRATION.md         # Pre-analysis plan template
- requirements.txt            # Python dependencies
```

---

## Quickstart

```bash
pip install -r requirements.txt
cd data && python simulate_pos_data.py
cd ../analysis && python 01_power_analysis.py
python 02_did_analysis.py
python 03_synthetic_control.py
```

---

## Study Design Recommendation

For a 6-7 location pilot with non-random assignment, I recommend:

### Primary: Difference-in-Differences (Two-Way FE)

**Why DiD here:**
- 24 months of historical data allows a rigorous parallel trends test
- Location + month fixed effects absorb most confounders
- Interpretable ATT estimate with confidence intervals
- Standard in applied econometrics for business interventions

**Key assumption:** Parallel trends - treated and control locations would have moved
together absent the intervention. Tested via event-study in pre-period.

**Limitation:** With only 3-4 treated units, SEs clustered at location level are
conservative but imprecise. We report CIs and flag this explicitly.

### Robustness: Synthetic Control

**Why as robustness check:**
- Does not require parallel trends - constructs a data-driven counterfactual
- Transparent: you see exactly how the synthetic unit is built
- Better suited to small N than cluster-robust DiD alone

**Limitation:** With only 4 control donors, the donor pool is thin. We report
per-unit pre-period fit (RMSE) and flag poor fits.

### When I would prefer Synthetic Control over DiD

If the parallel trends test fails (i.e., pre-period event-study shows diverging
trends), synthetic control becomes the primary estimator. I would pre-specify this
decision rule in the pre-registration.

---

## Power Analysis Summary

At 3 treated / 4 control / 6-month post-period (alpha=0.05, power=0.80):

| Metric | MDE | Relative lift |
|---|---|---|
| Avg check size | ~$0.85 | ~4.4% |
| Items per transaction | ~0.21 | ~7.2% |

The study is adequately powered to detect economically meaningful effects.
Effects below MDE are possible - we will report CIs, not a binary reject/fail.

---

## Why Mendelian Randomization Experience Matters Here

My primary causal inference background is Mendelian Randomization (first-author:
*Neurology* 2019, 2022) - an IV-based design used in human genetics. MR shares the
core challenge of this study: you cannot randomize, so identification depends entirely
on the credibility of your identifying assumption (instrument validity / parallel trends).

The discipline of pre-registering analysis plans, auditing assumption violations, and
reporting honest limitations with confidence intervals comes directly from that background.

---

## My Approach for Phase 1

1. **Data audit** - check for location closures, format changes, outlier months,
   data quality issues before committing to an estimator
2. **Pre-trend visualization** - plot raw monthly trends for all 7 locations
3. **Power analysis** - compute MDE from actual historical variance (not assumed)
4. **Pre-registration** - lock analysis plan before pilot launches (see PRE_REGISTRATION.md)
5. **Recommendation memo** - 2-page writeup: recommended design, assumption audit,
   power table, go/no-go on DiD vs synthetic control

---

## References

- Callaway & Sant'Anna (2021). Difference-in-Differences with multiple time periods.
  *Journal of Econometrics*.
- Abadie, Diamond & Hainmueller (2010). Synthetic Control Methods.
  *Journal of the American Statistical Association*.
- Grover S et al. (2019). Risky behaviors and Parkinson disease: A Mendelian
  randomization study. *Neurology*.
