# Pre-Registration: Bespoke AI Coaching Pilot

**Study:** Causal effect of AI coaching software on server performance
**Registrant:** [Analyst name]
**Date registered:** [Before pilot launch]
**Registration ID:** [OSF / AsPredicted ID]

---

## 1. Research Questions

1. Does Bespoke AI coaching software causally increase average check size?
2. Does it causally increase items per transaction?

---

## 2. Study Design

**Design:** Difference-in-Differences (primary) with Synthetic Control (robustness check)

**Units:** Restaurant locations (N = 6-7)

**Assignment:** Non-random — design partner selects which locations receive coaching first.
This makes causal identification depend entirely on the parallel trends assumption.

**Pre-period:** 18 months of historical POS data (months 1-18)

**Post-period:** 6 months during live pilot (months 19-24)

---

## 3. Outcomes

| Outcome | Operationalization | Primary / Secondary |
|---|---|---|
| Avg check size | Mean transaction value per location per month (USD) | Primary |
| Items per transaction | Mean item count per transaction per location per month | Primary |
| Check size variance | SD of daily check values | Exploratory |

---

## 4. Primary Estimand

Average Treatment Effect on the Treated (ATT): the change in outcome attributable to coaching
software for the locations that received it, relative to their synthetic counterfactual.

---

## 5. Primary Estimator

Two-way fixed effects DiD:

```
Y_it = alpha_i + gamma_t + delta * (Treated_i x Post_t) + epsilon_it
```

Where:
- `alpha_i` = location fixed effects (absorb time-invariant differences)
- `gamma_t` = month fixed effects (absorb shared time trends)
- `delta` = ATT estimate (our coefficient of interest)
- SEs clustered at location level

---

## 6. Identification Assumption

**Parallel trends:** In the absence of treatment, treated and control locations would have
followed parallel outcome trajectories.

**Test:** Event-study plot of pre-period interaction coefficients. We will flag any
pre-trend coefficient with |t| > 2 as a violation.

---

## 7. Secondary Estimator (Robustness)

Synthetic Control Method: construct a weighted combination of control locations that best
matches each treated location's pre-period trajectory. Report unit-level ATTs and pooled
average. Visual inspection of pre-period fit (RMSE < 0.5 SD of outcome).

---

## 8. Exclusion Criteria (Pre-specified)

We will exclude a location from the primary analysis if:
- It was closed for >= 2 consecutive months during the study window
- It underwent a major format change (ownership, full menu change) during post-period
- Fewer than 30 daily transactions recorded in any month

Excluded locations will be analyzed in a sensitivity analysis.

---

## 9. Power Analysis

See `analysis/01_power_analysis.py` for full calculations.

Summary (DiD, 3 treated / 4 control, alpha=0.05, power=0.80):

| Metric | MDE (absolute) | MDE (relative) |
|---|---|---|
| Avg check size | ~$0.85 | ~4.4% |
| Items per transaction | ~0.21 items | ~7.2% |

If true effects are below MDE, the study will be underpowered. We will report
confidence intervals regardless and refrain from claiming "no effect" on this basis.

---

## 10. Planned Deviations Protocol

Any deviation from this pre-registered plan will be:
1. Documented with a reason
2. Run alongside the pre-registered analysis (not replacing it)
3. Labeled clearly as "unplanned" in the final report

---

## 11. Deliverables

- Phase 1: This pre-registration + data audit memo + power analysis report
- Phase 2: Causal analysis writeup with: ATT estimates, 95% CIs, parallel trends
  diagnostic, synthetic control robustness, written limitations section

---

*Registered before pilot launch to prevent specification searching.*
