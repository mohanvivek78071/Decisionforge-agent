"""Synthetic retail panel with *planted, known* effects.

Because we know the ground truth, the eval harness can score the agent
objectively instead of eyeballing outputs.

Planted effects
  * Promo unit-lift: Snacks 1.35x, Beverages 1.15x, Personal Care 1.02x
    (promo = 15% price cut, so Beverages promos are revenue-NEGATIVE)
  * Supply incident: West / Beverages / stores W02 & W03 - heavy stockouts in
    the last 13 weeks, which depresses revenue
  * Mild seasonality + trend shared by everyone
  * ~1% missing on_hand (data-quality issue for the profiler to catch)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIONS = ["North", "South", "East", "West"]
STORES_PER_REGION = 3
CATEGORIES = ["Snacks", "Beverages", "Personal Care"]
SKUS_PER_CATEGORY = 10
PROMO_LIFT = {"Snacks": 1.35, "Beverages": 1.15, "Personal Care": 1.02}
PROMO_DISCOUNT = 0.15
INCIDENT = {
    "region": "West",
    "category": "Beverages",
    "stores": ["W02", "W03"],
    "weeks": 13,
    "stockout_p": 0.7,
    "stockout_factor": 0.35,
}


def generate(seed: int = 7, weeks: int = 104) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01")
    dates = start + pd.to_timedelta(np.arange(weeks) * 7, unit="D")
    stores = [(f"{r[0]}{i + 1:02d}", r) for r in REGIONS for i in range(STORES_PER_REGION)]
    scale = {sid: rng.uniform(0.8, 1.3) for sid, _ in stores}
    skus = []
    for c in CATEGORIES:
        for i in range(SKUS_PER_CATEGORY):
            skus.append((f"{c[:3].upper()}{i + 1:02d}", c, rng.uniform(2, 12), rng.uniform(15, 60)))
    phase = {"Snacks": 0.0, "Beverages": 1.2, "Personal Care": 2.5}
    W = np.arange(weeks)
    frames = []
    for sid, reg in stores:
        for kid, cat, price, base in skus:
            season = 1 + 0.08 * np.sin(2 * np.pi * W / 52 + phase[cat])
            trend = 1 + 0.001 * W
            promo = rng.random(weeks) < 0.12
            lift = np.where(promo, PROMO_LIFT[cat], 1.0)
            if sid in INCIDENT["stores"] and cat == INCIDENT["category"]:
                recent = W >= weeks - INCIDENT["weeks"]
                stock = recent & (rng.random(weeks) < INCIDENT["stockout_p"])
                factor = np.where(stock, INCIDENT["stockout_factor"], 1.0)
            else:
                stock = rng.random(weeks) < 0.02
                factor = np.where(stock, 0.6, 1.0)
            mu = base * scale[sid] * season * trend * lift * factor
            units = rng.poisson(mu)
            price_eff = np.where(promo, price * (1 - PROMO_DISCOUNT), price)
            on_hand = rng.integers(20, 300, weeks).astype(float)
            on_hand[stock] = 0.0
            on_hand[rng.random(weeks) < 0.01] = np.nan
            frames.append(
                pd.DataFrame(
                    {
                        "week_start": dates,
                        "store_id": sid,
                        "region": reg,
                        "sku_id": kid,
                        "category": cat,
                        "units": units,
                        "price": np.round(price_eff, 2),
                        "revenue": np.round(units * price_eff, 2),
                        "promo_flag": promo.astype(int),
                        "stockout_flag": stock.astype(int),
                        "on_hand": on_hand,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)
