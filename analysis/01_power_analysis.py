"""
Power analysis for small-N restaurant pilot.
6-7 locations. Computes MDE given N locations and pre-period variance.
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Parameters ────────────────────────────────────────────────────────────────
ALPHA = 0.05
POWER_TARGET = 0.80
N_TREATED = 3
N_CONTROL = 4
PRE_PERIOD_MONTHS = 18
POST_PERIOD_MONTHS = 6

# Estimated from historical POS data (placeholder — replace with real values)
SD_CHECK = 1.20     # within-location monthly SD of avg_check_size
SD_ITEMS = 0.30     # within-location monthly SD of items_per_transaction
BASE_CHECK = 19.5
BASE_ITEMS = 2.9


def mde_did(n_treated, n_control, sd, alpha=0.05, power=0.80, t_pre=18, t_post=6):
    """Minimum detectable effect for a DiD design."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    se = sd * np.sqrt(1/n_treated + 1/n_control) * np.sqrt(1/t_post + 1/t_pre)
    mde = (z_alpha + z_beta) * se
    return mde


def power_curve(n_treated, n_control, sd, true_effect, alpha=0.05, t_pre=18):
    """Power as a function of post-period length."""
    t_posts = range(1, 13)
    powers = []
    for t_post in t_posts:
        se = sd * np.sqrt(1/n_treated + 1/n_control) * np.sqrt(1/t_post + 1/t_pre)
        z = true_effect / se
        pwr = 1 - stats.norm.cdf(stats.norm.ppf(1 - alpha/2) - z)
        powers.append(pwr)
    return list(t_posts), powers


def main():
    print("=" * 60)
    print("POWER ANALYSIS — Restaurant Pilot (DiD Design)")
    print(f"N treated: {N_TREATED}  |  N control: {N_CONTROL}")
    print(f"Pre-period: {PRE_PERIOD_MONTHS}mo  |  Post-period: {POST_PERIOD_MONTHS}mo")
    print("=" * 60)

    for metric, sd, base in [
        ("avg_check_size ($)", SD_CHECK, BASE_CHECK),
        ("items_per_transaction", SD_ITEMS, BASE_ITEMS),
    ]:
        mde = mde_did(N_TREATED, N_CONTROL, sd,
                      t_pre=PRE_PERIOD_MONTHS, t_post=POST_PERIOD_MONTHS)
        mde_pct = 100 * mde / base
        print(f"\nMetric: {metric}")
        print(f"  MDE (absolute): {mde:.3f}")
        print(f"  MDE (relative): {mde_pct:.1f}%")
        print(f"  Interpretation: can detect a {mde_pct:.1f}% lift with 80% power")

    # Power curve plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Power vs Post-Period Length\n(DiD, 3 treated / 4 control locations)", fontsize=13)

    for ax, (metric, sd, true_eff, base) in zip(axes, [
        ("Avg Check Size ($)", SD_CHECK, 1.80, BASE_CHECK),
        ("Items per Transaction", SD_ITEMS, 0.25, BASE_ITEMS),
    ]):
        t_posts, powers = power_curve(N_TREATED, N_CONTROL, sd, true_eff)
        ax.plot(t_posts, powers, "b-o", lw=2)
        ax.axhline(0.80, color="red", linestyle="--", label="80% power threshold")
        ax.axvline(6, color="green", linestyle="--", label="Planned post-period (6mo)")
        ax.fill_between(t_posts, powers, 0.80,
                        where=[p >= 0.80 for p in powers],
                        alpha=0.15, color="green")
        ax.set_xlabel("Post-period length (months)")
        ax.set_ylabel("Statistical power")
        ax.set_title(f"{metric}\nTrue effect = {true_eff} ({100*true_eff/base:.1f}%)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/power_analysis.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to reports/power_analysis.png")

    # Summary table
    print("\n── MDE sensitivity (varying post-period length) ──")
    rows = []
    for t_post in [3, 6, 9, 12]:
        mde_c = mde_did(N_TREATED, N_CONTROL, SD_CHECK, t_pre=PRE_PERIOD_MONTHS, t_post=t_post)
        mde_i = mde_did(N_TREATED, N_CONTROL, SD_ITEMS, t_pre=PRE_PERIOD_MONTHS, t_post=t_post)
        rows.append({
            "post_months": t_post,
            "MDE_check ($)": round(mde_c, 3),
            "MDE_check (%)": f"{100*mde_c/BASE_CHECK:.1f}%",
            "MDE_items": round(mde_i, 3),
            "MDE_items (%)": f"{100*mde_i/BASE_ITEMS:.1f}%",
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
