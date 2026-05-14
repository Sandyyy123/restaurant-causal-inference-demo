"""
Simulate 24 months of POS data for 7 restaurant locations.
3 treatment (coaching software), 4 control (no coaching).
Metrics: avg_check_size, items_per_transaction
"""
import numpy as np
import pandas as pd

np.random.seed(42)

LOCATIONS = {
    "T1": {"treatment": True,  "base_check": 18.5, "base_items": 2.8},
    "T2": {"treatment": True,  "base_check": 21.0, "base_items": 3.1},
    "T3": {"treatment": True,  "base_check": 16.8, "base_items": 2.5},
    "C1": {"treatment": False, "base_check": 19.2, "base_items": 2.9},
    "C2": {"treatment": False, "base_check": 20.5, "base_items": 3.0},
    "C3": {"treatment": False, "base_check": 17.5, "base_items": 2.6},
    "C4": {"treatment": False, "base_check": 22.0, "base_items": 3.3},
}

PILOT_START_MONTH = 19  # months 1-18 pre-period, 19-24 post-period
TRUE_ATT_CHECK = 1.80   # $1.80 lift in avg check (true effect)
TRUE_ATT_ITEMS = 0.25   # 0.25 items lift (true effect)


def simulate_location(loc_id, config, n_months=24):
    rows = []
    for month in range(1, n_months + 1):
        date = pd.Timestamp("2023-01-01") + pd.DateOffset(months=month - 1)
        post = month >= PILOT_START_MONTH
        treated = config["treatment"]

        # Shared time trend + seasonality
        time_trend = 0.02 * month
        seasonality = 0.8 * np.sin(2 * np.pi * month / 12)
        noise_check = np.random.normal(0, 0.5)
        noise_items = np.random.normal(0, 0.15)

        check = (
            config["base_check"]
            + time_trend
            + seasonality
            + noise_check
            + (TRUE_ATT_CHECK if treated and post else 0)
        )
        items = (
            config["base_items"]
            + 0.005 * month
            + 0.1 * np.sin(2 * np.pi * month / 12)
            + noise_items
            + (TRUE_ATT_ITEMS if treated and post else 0)
        )

        rows.append({
            "location_id": loc_id,
            "date": date,
            "month": month,
            "treatment": int(treated),
            "post": int(post),
            "did": int(treated and post),
            "avg_check_size": round(check, 2),
            "items_per_transaction": round(items, 2),
            "daily_transactions": np.random.randint(80, 200),
        })
    return rows


def main():
    all_rows = []
    for loc_id, config in LOCATIONS.items():
        all_rows.extend(simulate_location(loc_id, config))
    df = pd.DataFrame(all_rows)
    df.to_csv("data/pos_data.csv", index=False)
    print(f"Saved {len(df)} rows — {df['location_id'].nunique()} locations x {df['month'].nunique()} months")
    print(df.groupby("treatment")[["avg_check_size", "items_per_transaction"]].mean().round(2))


if __name__ == "__main__":
    main()
