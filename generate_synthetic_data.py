"""
generate_synthetic_data.py
--------------------------
Generates a synthetic Moldova climate dataset with the same schema as the
master_dataset.csv used in the SPEI drought classification pipeline.

The generator reproduces:
  - Realistic seasonal climatology for Moldova (continental climate)
  - North–south aridity gradient across 16 meteorological stations
  - Temporal autocorrelation in precipitation and temperature
  - SPEI indices computed from synthetic water-balance series
  - Consistent lags, rolling means, anomalies, and drought class labels

No real station observations are included — this file is safe to distribute.

Usage
-----
    python generate_synthetic_data.py          # writes master_dataset.csv
    python generate_synthetic_data.py --seed 7 # reproducible alternative seed
"""

import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STATIONS = {
    "Baltata":      (47.25, 28.75),
    "Balti":        (47.76, 27.93),
    "Bravicea":     (47.22, 28.37),
    "Briceni":      (48.37, 27.07),
    "Cahul":        (45.90, 28.19),
    "Camenca":      (48.02, 28.70),
    "Ceadir-Lunga": (46.07, 28.83),
    "Chisinau":     (47.00, 28.86),
    "Comrat":       (46.30, 28.67),
    "Cornesti":     (47.35, 27.98),
    "Dubasari":     (47.27, 29.15),
    "Falesti":      (47.57, 27.72),
    "Leova":        (46.48, 28.25),
    "Rabnita":      (47.75, 29.00),
    "Soroca":       (48.15, 28.30),
    "Tiraspol":     (46.85, 29.60),
}

# Stations south of ~46.5°N — warmer and drier (documented N–S gradient)
SOUTH_STATIONS = {"Cahul", "Comrat", "Ceadir-Lunga", "Leova", "Tiraspol"}

YEARS = list(range(1961, 2021))

# Moldova monthly climatological normals (approximate 1961–1990 reference)
#               J    F    M    A    M    J    J    A    S    O    N    D
PRECIP_CLIM = [35,  30,  35,  40,  55,  70,  65,  55,  45,  35,  40,  35]
TMED_CLIM   = [-3,  -1,   4,  11,  16,  20,  22,  21,  16,  10,   4,  -1]
TMIN_CLIM   = [-7,  -5,  -1,   5,  10,  14,  16,  15,  10,   5,  -1,  -5]
TMAX_CLIM   = [ 1,   3,   9,  17,  22,  26,  28,  27,  22,  15,   9,   3]

# Approximate PET amplitude for water-balance calculation
#   Linear model: PET ≈ max(0, T_med * 5.0) mm/month
#   Calibrated so July (T=22°C) → PET≈110mm, matching Moldova Thornthwaite estimates.
PET_TEMP_COEF = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lag(arr: np.ndarray, k: int) -> np.ndarray:
    result = np.full_like(arr, np.nan)
    result[k:] = arr[:-k]
    return result


def _roll_mean(arr: np.ndarray, k: int, min_periods: int = 1) -> np.ndarray:
    result = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        start = max(0, i - k + 1)
        window = arr[start : i + 1]
        if len(window) >= min_periods:
            result[i] = window.mean()
    return result


def _compute_spei(wb: np.ndarray, k: int) -> np.ndarray:
    """Standardised k-month rolling water-balance sum (proxy for SPEI-k)."""
    n = len(wb)
    rolling = np.full(n, np.nan)
    for i in range(k - 1, n):
        rolling[i] = wb[i - k + 1 : i + 1].sum()
    # Standardise over the full series (normal approximation)
    valid = ~np.isnan(rolling)
    mu = np.nanmean(rolling[valid])
    sd = np.nanstd(rolling[valid])
    if sd > 0:
        rolling = (rolling - mu) / sd
    # Mask early months where rolling window is incomplete (match real data)
    if k >= 6:
        rolling[: k - 1] = np.nan
    return np.round(rolling, 6)


def _classify_spei_3class(s: float) -> str:
    if np.isnan(s):
        return "Normal"  # fallback for first two months
    if s < -1.0:
        return "Dry"
    if s > 1.0:
        return "Wet"
    return "Normal"


# ---------------------------------------------------------------------------
# Per-station generator
# ---------------------------------------------------------------------------

def generate_station(name: str, lat: float, lon: float, rng: np.random.Generator) -> list:
    """Return a list of dicts — one per month — for a single station."""

    n_years  = len(YEARS)
    n_rows   = n_years * 12
    is_south = name in SOUTH_STATIONS

    temp_offset    = 1.5 if is_south else 0.0   # southern stations run warmer
    precip_factor  = 0.88 if is_south else 1.0  # southern stations run drier

    # ── Correlated noise (AR-1, ρ ≈ 0.35) ──────────────────────────────────
    ar = 0.35
    t_innov = rng.normal(0, 1.5, n_rows)
    p_innov = rng.normal(0, 0.22, n_rows)
    for i in range(1, n_rows):
        t_innov[i] += ar * t_innov[i - 1]
        p_innov[i] += ar * p_innov[i - 1]

    # ── Temperature series ────────────────────────────────────────────────
    month_idx = np.tile(np.arange(12), n_years)   # 0-based month index
    tmed_clim_arr  = np.array(TMED_CLIM)[month_idx]
    tmin_clim_arr  = np.array(TMIN_CLIM)[month_idx]
    tmax_clim_arr  = np.array(TMAX_CLIM)[month_idx]

    tmed_arr = tmed_clim_arr + temp_offset + t_innov
    tmin_arr = tmin_clim_arr + temp_offset + t_innov * 0.9 + rng.normal(0, 0.4, n_rows)
    tmax_arr = tmax_clim_arr + temp_offset + t_innov * 0.9 + rng.normal(0, 0.4, n_rows)

    # Enforce physical ordering t_min < t_med < t_max
    tmin_arr = np.minimum(tmin_arr, tmed_arr - 1.5)
    tmax_arr = np.maximum(tmax_arr, tmed_arr + 1.5)

    tmed_arr = np.round(tmed_arr, 1)
    tmin_arr = np.round(tmin_arr, 1)
    tmax_arr = np.round(tmax_arr, 1)

    # ── Precipitation series (Gamma, multiplicative noise) ────────────────
    precip_clim_arr = np.array(PRECIP_CLIM)[month_idx] * precip_factor
    p_scale_base    = precip_clim_arr * (1 + p_innov)
    shape           = 2.5                           # controls dispersion
    p_scale_gamma   = np.maximum(p_scale_base / shape, 0.5)
    precip_arr      = np.round(np.maximum(0.0, rng.gamma(shape, p_scale_gamma)), 1)

    # ── Climatological reference (first 30 years) ─────────────────────────
    ref = 30
    clim_tmed   = np.array([tmed_arr[m::12][:ref].mean()   for m in range(12)])
    clim_tmin   = np.array([tmin_arr[m::12][:ref].mean()   for m in range(12)])
    clim_tmax   = np.array([tmax_arr[m::12][:ref].mean()   for m in range(12)])
    clim_precip = np.array([precip_arr[m::12][:ref].mean() for m in range(12)])

    t_med_anom  = np.round(tmed_arr   - clim_tmed[month_idx],   4)
    t_min_anom  = np.round(tmin_arr   - clim_tmin[month_idx],   4)
    t_max_anom  = np.round(tmax_arr   - clim_tmax[month_idx],   4)
    precip_anom = np.round(precip_arr - clim_precip[month_idx], 4)

    # ── Water balance and SPEI ─────────────────────────────────────────────
    pet_arr = np.maximum(0.0, tmed_arr * PET_TEMP_COEF)
    wb      = precip_arr - pet_arr

    spei_1  = _compute_spei(wb, 1)
    spei_3  = _compute_spei(wb, 3)
    spei_6  = _compute_spei(wb, 6)
    spei_12 = _compute_spei(wb, 12)
    spei_24 = _compute_spei(wb, 24)

    # ── Lag / rolling features ────────────────────────────────────────────
    precip_lag1  = _lag(precip_arr, 1)
    precip_lag3  = _lag(precip_arr, 3)
    t_med_lag1   = _lag(tmed_arr,   1)
    t_med_lag3   = _lag(tmed_arr,   3)
    precip_roll3 = _roll_mean(precip_arr, 3, min_periods=1)

    # ── Seasonality ───────────────────────────────────────────────────────
    months_arr = month_idx + 1   # 1-based
    month_sin  = np.sin(2 * np.pi * months_arr / 12)
    month_cos  = np.cos(2 * np.pi * months_arr / 12)

    # ── Drought class (3-class from SPEI-3) ───────────────────────────────
    drought_class = np.array([_classify_spei_3class(s) for s in spei_3])

    # ── Assemble rows ─────────────────────────────────────────────────────
    years_arr = np.repeat(YEARS, 12)
    records   = []

    def _fmt(v):
        """Return None for NaN, else the value."""
        return None if (isinstance(v, float) and np.isnan(v)) else v

    for i in range(n_rows):
        records.append({
            "station":      name,
            "year":         int(years_arr[i]),
            "month":        int(months_arr[i]),
            "precip":       precip_arr[i],
            "t_med":        tmed_arr[i],
            "t_min":        tmin_arr[i],
            "t_max":        tmax_arr[i],
            "spei_1":       _fmt(spei_1[i]),
            "spei_3":       _fmt(spei_3[i]),
            "spei_6":       _fmt(spei_6[i]),
            "spei_12":      _fmt(spei_12[i]),
            "spei_24":      _fmt(spei_24[i]),
            "month_sin":    round(float(month_sin[i]), 15),
            "month_cos":    round(float(month_cos[i]), 15),
            "t_med_anom":   t_med_anom[i],
            "t_min_anom":   t_min_anom[i],
            "t_max_anom":   t_max_anom[i],
            "precip_anom":  precip_anom[i],
            "precip_lag1":  _fmt(precip_lag1[i]),
            "precip_lag3":  _fmt(precip_lag3[i]),
            "t_med_lag1":   _fmt(t_med_lag1[i]),
            "t_med_lag3":   _fmt(t_med_lag3[i]),
            "precip_roll3": _fmt(precip_roll3[i]),
            "drought_class": drought_class[i],
        })

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int = 42, output: str = "master_dataset.csv") -> None:
    rng = np.random.default_rng(seed)
    print(f"Generating synthetic Moldova climate dataset (seed={seed}) ...")

    all_records = []
    for name, (lat, lon) in STATIONS.items():
        records = generate_station(name, lat, lon, rng)
        all_records.extend(records)
        print(f"  {name:<20}  {len(records):4d} rows  "
              f"lat={lat}  lon={lon}  south={'yes' if name in SOUTH_STATIONS else 'no'}")

    df = pd.DataFrame(all_records)
    df = df.sort_values(["station", "year", "month"]).reset_index(drop=True)

    # ── Quick sanity check ─────────────────────────────────────────────────
    dist = df["drought_class"].value_counts(normalize=True)
    print(f"\n3-class distribution:")
    for cls, pct in dist.items():
        print(f"  {cls:<8}  {100*pct:5.1f}%")

    spei3_valid = df["spei_3"].dropna()
    print(f"\nSPEI-3  mean={spei3_valid.mean():.3f}  "
          f"std={spei3_valid.std():.3f}  "
          f"min={spei3_valid.min():.3f}  max={spei3_valid.max():.3f}")

    df.to_csv(output, index=False)
    print(f"\nSaved: {output}  ({df.shape[0]:,} rows × {df.shape[1]} columns)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Moldova climate data.")
    parser.add_argument("--seed",   type=int, default=42,               help="Random seed")
    parser.add_argument("--output", type=str, default="master_dataset.csv", help="Output CSV path")
    args = parser.parse_args()
    main(seed=args.seed, output=args.output)
