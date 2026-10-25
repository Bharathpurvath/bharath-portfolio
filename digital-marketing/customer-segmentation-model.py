"""Reproducible synthetic RFM + campaign-response segmentation example.

Requirements: numpy, pandas, scikit-learn
Example: python customer-segmentation-model.py --output-dir ./segmentation-output

The generated customer records are fictional and contain no personal data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    # NumPy fallback keeps the example runnable in a minimal Python environment.
    HAS_SKLEARN = False


def scale_features(frame):
    values = frame.to_numpy(dtype=float)
    if HAS_SKLEARN:
        scaler = StandardScaler()
        return scaler.fit_transform(values)
    center = values.mean(axis=0)
    spread = values.std(axis=0)
    spread[spread == 0] = 1
    return (values - center) / spread


def fit_labels(values, k, seed):
    if HAS_SKLEARN:
        return KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(values)
    rng = np.random.default_rng(seed + k)
    centers = [values[rng.integers(len(values))]]
    for _ in range(1, k):
        d2 = np.min(((values[:, None, :] - np.asarray(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
        probs = d2 / d2.sum() if d2.sum() else np.full(len(values), 1 / len(values))
        centers.append(values[rng.choice(len(values), p=probs)])
    centers = np.asarray(centers)
    labels = np.zeros(len(values), dtype=int)
    for _ in range(100):
        distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        new_centers = centers.copy()
        for cluster in range(k):
            members = values[new_labels == cluster]
            if len(members):
                new_centers[cluster] = members.mean(axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            break
        labels, centers = new_labels, new_centers
    return labels


def silhouette(values, labels):
    if HAS_SKLEARN:
        return float(silhouette_score(values, labels))
    distances = np.sqrt(((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2))
    unique = np.unique(labels)
    scores = []
    for i, own in enumerate(labels):
        own_mask = labels == own
        own_mask[i] = False
        a = distances[i, own_mask].mean() if own_mask.any() else 0.0
        b = min(distances[i, labels == other].mean() for other in unique if other != own)
        scores.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return float(np.mean(scores))

AS_OF = pd.Timestamp("2026-10-25")
SEED = 42


def clipped_normal(rng, mean, sd, low, high, size):
    return np.clip(rng.normal(mean, sd, size), low, high)


def make_customers(n=1200, seed=SEED):
    rng = np.random.default_rng(seed)
    archetypes = [
        # Generator archetypes only shape the synthetic test data; they are not
        # copied to the model features or exposed as ground-truth segment labels.
        ("recent_high_value", .18, (8, 48), (9, 17), (340, 880), (.12, .30)),
        ("steady_regular", .25, (12, 100), (4, 9), (130, 420), (.10, .25)),
        ("new_low_history", .24, (1, 60), (1, 2), (22, 105), (.08, .28)),
        ("high_value_at_risk", .19, (90, 235), (5, 12), (200, 660), (.04, .15)),
        ("lapsed_low_engagement", .14, (180, 365), (1, 3), (20, 175), (.00, .08)),
    ]
    weights = np.array([a[1] for a in archetypes], dtype=float)
    weights /= weights.sum()
    groups = rng.choice(len(archetypes), size=n, p=weights)

    records = []
    for i, group_idx in enumerate(groups):
        _, _, r_range, f_range, m_range, response_range = archetypes[group_idx]
        recency = int(rng.integers(r_range[0], r_range[1] + 1))
        orders = max(1, int(round(rng.uniform(*f_range))))
        spend = round(float(rng.uniform(*m_range)), 2)
        campaigns_sent = 8
        response_propensity = rng.uniform(*response_range)
        clicks = int(rng.binomial(campaigns_sent, response_propensity))
        campaign_orders = int(rng.binomial(clicks, .11)) if clicks else 0
        records.append({
            "customer_id": f"SYN-{i+1:05d}",
            "last_purchase_date": (AS_OF - pd.Timedelta(days=recency)).date().isoformat(),
            "recency_days": recency,
            "orders_12m": orders,
            "spend_12m_usd": spend,
            "avg_order_value_usd": round(spend / orders, 2),
            "campaigns_delivered_90d": campaigns_sent,
            "campaign_clicks_90d": clicks,
            "campaign_orders_90d": campaign_orders,
            "campaign_response_rate_90d": clicks / campaigns_sent,
        })
    return pd.DataFrame(records)


def label_clusters(profile):
    """Assign readable names with explicit, reproducible profile rules."""
    z = (profile - profile.mean()) / profile.std(ddof=0).replace(0, 1)
    remaining = set(profile.index)
    chosen = {}

    def pick(label, score):
        available = sorted(remaining)
        cluster = score.loc[available].idxmax()
        chosen[cluster] = label
        remaining.remove(cluster)

    pick("Champions", z["orders_12m"] + z["spend_12m_usd"] - z["recency_days"])
    pick("At-risk high value", z["recency_days"] + z["spend_12m_usd"] + z["orders_12m"])
    # Among the remaining clusters, favor recent customers with little history.
    pick("New / low purchase history", -(z["recency_days"] + z["orders_12m"] + z["spend_12m_usd"]))
    pick("Engaged repeat customers", z["campaign_response_rate_90d"] + z["orders_12m"])
    for cluster in remaining:
        chosen[cluster] = "Lapsed / low engagement"
    return chosen


def run(output_dir: Path, n=1200, seed=SEED):
    output_dir.mkdir(parents=True, exist_ok=True)
    customers = make_customers(n=n, seed=seed)
    feature_cols = ["recency_days", "orders_12m", "spend_12m_usd", "campaign_response_rate_90d"]
    x = customers[feature_cols].copy()
    for col in ["recency_days", "orders_12m", "spend_12m_usd"]:
        x[col] = np.log1p(x[col])
    x_scaled = scale_features(x)

    silhouette_by_k = {}
    for k in range(2, 7):
        labels = fit_labels(x_scaled, k, seed)
        silhouette_by_k[k] = round(silhouette(x_scaled, labels), 4)

    # Five segments are kept for actionability and continuity with the brief.
    customers["cluster_id"] = fit_labels(x_scaled, 5, seed)
    raw_profile = customers.groupby("cluster_id")[feature_cols].mean()
    name_map = label_clusters(raw_profile)
    customers["segment"] = customers["cluster_id"].map(name_map)
    profile = customers.groupby("segment").agg(
        customers=("customer_id", "count"),
        avg_recency_days=("recency_days", "mean"),
        avg_orders_12m=("orders_12m", "mean"),
        avg_spend_12m_usd=("spend_12m_usd", "mean"),
        avg_order_value_usd=("avg_order_value_usd", "mean"),
        avg_campaign_response_rate=("campaign_response_rate_90d", "mean"),
        avg_campaign_orders=("campaign_orders_90d", "mean"),
    ).reset_index()
    profile["customer_share"] = profile["customers"] / len(customers)
    profile["avg_recency_days"] = profile["avg_recency_days"].round(1)
    profile["avg_orders_12m"] = profile["avg_orders_12m"].round(2)
    for col in ["avg_spend_12m_usd", "avg_order_value_usd"]:
        profile[col] = profile[col].round(2)
    profile["avg_campaign_response_rate"] = profile["avg_campaign_response_rate"].round(4)
    profile["avg_campaign_orders"] = profile["avg_campaign_orders"].round(3)
    profile["customer_share"] = profile["customer_share"].round(4)

    customers.to_csv(output_dir / "synthetic_customer_segments.csv", index=False)
    profile.to_csv(output_dir / "segment_profiles.csv", index=False)
    diagnostics = {
        "as_of_date": AS_OF.date().isoformat(),
        "seed": seed,
        "customer_count": int(n),
        "selected_k": 5,
        "silhouette_by_k": {str(k): v for k, v in silhouette_by_k.items()},
        "silhouette_k5": silhouette_by_k[5],
        "features": feature_cols,
        "engine": "scikit-learn KMeans + StandardScaler" if HAS_SKLEARN else "NumPy KMeans fallback + equivalent z-score scaling",
        "transforms": "log1p on recency, orders, and spend; response rate unchanged; standardized features; 5-cluster KMeans with fixed seed",
        "warning": "Synthetic, non-personal training example. Segment names are descriptive rules, not ground-truth labels or validated predictions.",
    }
    (output_dir / "model_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps({"diagnostics": diagnostics, "profiles": profile.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--customers", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(args.output_dir, n=args.customers, seed=args.seed)
