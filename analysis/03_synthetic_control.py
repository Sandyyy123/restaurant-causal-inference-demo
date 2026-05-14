"""
Synthetic Control Method for restaurant pilot.
Constructs a weighted combination of control locations to match
each treated location's pre-period trajectory, then estimates
the counterfactual post-period.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def load_data():
    return pd.read_csv("data/pos_data.csv", parse_dates=["date"])


def pivot_metric(df, outcome):
    return df.pivot(index="month", columns="location_id", values=outcome)


def synth_weights(pre_treated, pre_donors):
    """
    Find donor weights w that minimize ||pre_treated - pre_donors @ w||^2
    subject to w >= 0, sum(w) = 1.
    """
    n_donors = pre_donors.shape[1]

    def objective(w):
        synth = pre_donors @ w
        return np.sum((pre_treated - synth) ** 2)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n_donors
    w0 = np.ones(n_donors) / n_donors

    result = minimize(objective, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-10, "maxiter": 1000})
    return result.x


def compute_synthetic_control(df, outcome, pilot_start=19):
    pivot = pivot_metric(df, outcome)
    treated_locs = df[df["treatment"] == 1]["location_id"].unique()
    control_locs = df[df["treatment"] == 0]["location_id"].unique()

    pre_months = pivot.index[pivot.index < pilot_start]
    post_months = pivot.index[pivot.index >= pilot_start]

    results = {}
    for t_loc in treated_locs:
        pre_t = pivot.loc[pre_months, t_loc].values
        pre_d = pivot.loc[pre_months, control_locs].values

        w = synth_weights(pre_t, pre_d)
        synth_full = pivot.loc[:, control_locs].values @ w

        att_post = (
            pivot.loc[post_months, t_loc].values -
            synth_full[pivot.index >= pilot_start]
        ).mean()

        pre_fit = np.sqrt(np.mean((pre_t - synth_full[:len(pre_months)]) ** 2))

        results[t_loc] = {
            "weights": dict(zip(control_locs, w)),
            "synthetic": pd.Series(synth_full, index=pivot.index),
            "actual": pivot.loc[:, t_loc],
            "att_post": att_post,
            "pre_rmse": pre_fit,
        }

    return results


def pooled_att(results):
    atts = [v["att_post"] for v in results.values()]
    return np.mean(atts), np.std(atts, ddof=1) / np.sqrt(len(atts))


def plot_synthetic_control(results, outcome, pilot_start=19):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(f"Synthetic Control: {outcome}", fontsize=13)

    for ax, (loc, res) in zip(axes, results.items()):
        actual = res["actual"]
        synth = res["synthetic"]

        ax.plot(actual.index, actual.values, "b-o", lw=2, markersize=4, label="Actual")
        ax.plot(synth.index, synth.values, "r--", lw=2, label="Synthetic control")
        ax.axvline(pilot_start - 0.5, color="black", linestyle="--", lw=1.5)
        ax.fill_between(actual.index[actual.index >= pilot_start],
                        actual.values[actual.index >= pilot_start],
                        synth.values[synth.index >= pilot_start],
                        alpha=0.25, color="green", label=f"ATT = {res['att_post']:.3f}")
        ax.set_title(f"Location {loc}\n(pre-RMSE={res['pre_rmse']:.3f})")
        ax.set_xlabel("Month")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(outcome)
    plt.tight_layout()
    plt.savefig(f"reports/synth_{outcome}.png", dpi=150, bbox_inches="tight")
    print(f"  Plot saved: reports/synth_{outcome}.png")


def main():
    df = load_data()
    print("=" * 60)
    print("SYNTHETIC CONTROL ANALYSIS")
    print("=" * 60)

    for outcome in ["avg_check_size", "items_per_transaction"]:
        print(f"\n{'─'*40}")
        print(f"Outcome: {outcome}")

        results = compute_synthetic_control(df, outcome)

        for loc, res in results.items():
            top_donors = sorted(res["weights"].items(), key=lambda x: -x[1])
            donor_str = ", ".join(f"{k}:{v:.2f}" for k, v in top_donors if v > 0.01)
            print(f"  {loc}: ATT={res['att_post']:+.3f}  pre-RMSE={res['pre_rmse']:.3f}")
            print(f"        Donor weights: {donor_str}")

        mean_att, se_att = pooled_att(results)
        print(f"\n  Pooled ATT (simple avg): {mean_att:+.3f}  SE={se_att:.3f}")
        print(f"  95% CI (approx):         [{mean_att - 1.96*se_att:.3f}, {mean_att + 1.96*se_att:.3f}]")

        plot_synthetic_control(results, outcome)

    print("\nNote: SE above is across treated units — not a formal inference procedure.")
    print("For small N, supplement with placebo tests (in-time and in-space).")


if __name__ == "__main__":
    main()
