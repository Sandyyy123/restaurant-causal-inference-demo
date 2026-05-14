"""
Difference-in-Differences analysis for restaurant coaching pilot.
Includes: parallel trends test, two-way FE regression, ATT with 95% CI.
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Note: statsmodels not installed — using manual DiD calculation")


def load_data():
    return pd.read_csv("data/pos_data.csv", parse_dates=["date"])


def parallel_trends_test(df, outcome):
    """
    Test parallel trends in pre-period using placebo post indicator.
    Regress outcome on treatment x placebo_post in pre-period only.
    p > 0.05 supports parallel trends assumption.
    """
    pre = df[df["post"] == 0].copy()
    # Split pre-period in half as placebo
    mid = pre["month"].median()
    pre["placebo_post"] = (pre["month"] > mid).astype(int)
    pre["treat_x_placebo"] = pre["treatment"] * pre["placebo_post"]

    if HAS_STATSMODELS:
        model = smf.ols(f"{outcome} ~ treatment + placebo_post + treat_x_placebo + C(location_id)",
                        data=pre).fit(cov_type="HC3")
        coef = model.params["treat_x_placebo"]
        pval = model.pvalues["treat_x_placebo"]
    else:
        # Manual: compare trends
        grp = pre.groupby(["treatment", "placebo_post"])[outcome].mean()
        coef = (grp[1, 1] - grp[1, 0]) - (grp[0, 1] - grp[0, 0])
        pval = None

    return coef, pval


def did_twoway_fe(df, outcome):
    """
    Two-way fixed effects DiD: outcome ~ did + C(location_id) + C(month)
    Cluster-robust SE at location level.
    Returns ATT estimate, 95% CI, p-value.
    """
    if HAS_STATSMODELS:
        model = smf.ols(
            f"{outcome} ~ did + C(location_id) + C(month)",
            data=df
        ).fit(cov_type="cluster", cov_kwds={"groups": df["location_id"]})
        att = model.params["did"]
        ci = model.conf_int().loc["did"].values
        pval = model.pvalues["did"]
        return att, ci, pval
    else:
        # Manual 2x2 DiD
        means = df.groupby(["treatment", "post"])[outcome].mean()
        att = (means[1, 1] - means[1, 0]) - (means[0, 1] - means[0, 0])
        return att, [None, None], None


def event_study(df, outcome):
    """
    Event-study coefficients: treatment x month relative to pilot start.
    Shows pre-trends and post-treatment dynamics.
    """
    if not HAS_STATSMODELS:
        return None, None, None

    pilot_start = df[df["post"] == 1]["month"].min()
    df = df.copy()
    df["rel_month"] = df["month"] - pilot_start  # -18 to +5
    df["treat_x_rel"] = df["treatment"] * df["rel_month"]

    # Reference period: rel_month = -1
    df_es = df[df["rel_month"] != -1].copy()
    coefs, cis, months = [], [], []
    for rm in sorted(df_es["rel_month"].unique()):
        sub = df.copy()
        sub["indicator"] = ((sub["rel_month"] == rm) & (sub["treatment"] == 1)).astype(int)
        if sub["indicator"].sum() == 0:
            continue
        mod = smf.ols(
            f"{outcome} ~ indicator + C(location_id) + C(month)",
            data=sub
        ).fit(cov_type="cluster", cov_kwds={"groups": sub["location_id"]})
        coefs.append(mod.params["indicator"])
        cis.append(mod.conf_int().loc["indicator"].values)
        months.append(rm)

    return months, coefs, cis


def plot_did(df, outcome, att, ci):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"DiD Analysis: {outcome}", fontsize=13)

    # Raw trends
    ax = axes[0]
    for treat, label, color in [(1, "Treatment (coached)", "steelblue"),
                                  (0, "Control", "coral")]:
        grp = df[df["treatment"] == treat].groupby("month")[outcome].mean()
        ax.plot(grp.index, grp.values, label=label, color=color, lw=2, marker="o", markersize=4)
    ax.axvline(18.5, color="black", linestyle="--", lw=1.5, label="Pilot start")
    ax.set_xlabel("Month")
    ax.set_ylabel(outcome)
    ax.set_title("Raw trends by group")
    ax.legend()
    ax.grid(alpha=0.3)

    # ATT forest plot
    ax = axes[1]
    ax.barh([outcome], [att], xerr=[[att - ci[0]], [ci[1] - att]],
            color="steelblue", alpha=0.7, capsize=6, height=0.4)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("ATT estimate")
    ax.set_title(f"ATT = {att:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"reports/did_{outcome}.png", dpi=150, bbox_inches="tight")
    print(f"  Plot saved: reports/did_{outcome}.png")


def main():
    df = load_data()
    print("=" * 60)
    print("DIFFERENCE-IN-DIFFERENCES ANALYSIS")
    print(f"Locations: {df['location_id'].nunique()}  "
          f"({df[df['treatment']==1]['location_id'].nunique()} treated, "
          f"{df[df['treatment']==0]['location_id'].nunique()} control)")
    print("=" * 60)

    for outcome in ["avg_check_size", "items_per_transaction"]:
        print(f"\n{'─'*40}")
        print(f"Outcome: {outcome}")

        # Parallel trends test
        coef, pval = parallel_trends_test(df, outcome)
        pval_str = f"{pval:.3f}" if pval is not None else "n/a"
        flag = "PASS" if (pval is None or pval > 0.05) else "FAIL - check trends"
        print(f"  Parallel trends test: coef={coef:.4f}, p={pval_str}  [{flag}]")

        # Two-way FE DiD
        att, ci, pval = did_twoway_fe(df, outcome)
        sig = ""
        if pval is not None:
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        pval_str = f"{pval:.4f}" if pval is not None else "n/a"
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else "n/a"
        print(f"  ATT (2-way FE DiD):  {att:.3f}  95% CI {ci_str}  p={pval_str} {sig}")

        if ci[0] is not None:
            plot_did(df, outcome, att, ci)

    print("\nNote: SEs clustered at location level (recommended for small N).")
    print("Interpretation: ATT is the average treatment effect on the treated locations.")


if __name__ == "__main__":
    main()
