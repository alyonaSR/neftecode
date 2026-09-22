"""Калибровка агента надёжности. Запуск: python -m src.agents.calibrate_reliability"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd
import yaml

from ..data.loaders import load_telemetry
from ..data.tags import CONFIG_DIR

# сетка перцентилей
GRID = np.arange(0, 101, 5)

INSTAB_WINDOW = "1h"

# уставки печи нет, вместо неё медиана за неделю
SETPOINT_WINDOW = "7D"


def running_thresholds(avt: pd.DataFrame, ht: pd.DataFrame) -> Dict[str, float]:
    """Пороги работы установок как доля медианы."""
    return {
        "running_min_t55": round(0.9 * float(avt["AVT:T55"].median()), 1),
        "running_min_t5": round(0.5 * float(ht["242000:T5"].median()), 1),
        "running_min_f9": round(0.2 * float(ht["242000:F9"].median()), 1),
    }


def running_mask(avt: pd.DataFrame, ht: pd.DataFrame, thr: Dict[str, float]) -> pd.Series:
    """Маска: обе установки в работе."""
    return (
        (avt["AVT:T55"] >= thr["running_min_t55"])
        & (ht["242000:T5"] >= thr["running_min_t5"])
        & (ht["242000:F9"] >= thr["running_min_f9"])
    )


def build_factors(avt: pd.DataFrame, ht: pd.DataFrame, thr: Dict[str, float]) -> pd.DataFrame:
    """Пять факторов тяжести режима, остановы исключены."""
    f = pd.DataFrame(index=avt.index)
    ok = running_mask(avt, ht, thr)

    f["reactor_temp"] = ht["242000:T5"].where(ok)

    f["load"] = ht["242000:F9"].where(ok)

    # превышение температуры печи над медианой за неделю
    t55 = avt["AVT:T55"]
    running = t55.where(ok)
    f["thermal_stress"] = running - running.rolling(SETPOINT_WINDOW, min_periods=6).median()

    # разброс за час
    f["pressure_instability"] = (
        avt["AVT:P67"].rolling(INSTAB_WINDOW, min_periods=3).std().where(ok)
    )
    f["temp_instability"] = (
        ht["242000:T5"].rolling(INSTAB_WINDOW, min_periods=3).std().where(ok)
    )

    return f


def calibrate(save: bool = True) -> Dict:
    """Квантили факторов и пороги классов; при save пишет config/reliability.yaml."""
    avt = load_telemetry("AVT")
    ht = load_telemetry("242000")
    thr = running_thresholds(avt, ht)
    f = build_factors(avt, ht, thr)

    quantiles = {}
    for col in f.columns:
        s = f[col].dropna()
        quantiles[col] = [round(float(v), 6) for v in np.percentile(s, GRID)]

    cfg = {
        "_comment": (
            "Сгенерировано src/agents/calibrate_reliability.py, вручную не править."
        ),
        "grid": [int(g) for g in GRID],
        "instab_window": INSTAB_WINDOW,
        "setpoint_window": SETPOINT_WINDOW,
        **thr,
        "quantiles": quantiles,
        "weights": {
            "reactor_temp": 0.35,
            "load": 0.20,
            "thermal_stress": 0.20,
            "pressure_instability": 0.125,
            "temp_instability": 0.125,
        },
        "class_thresholds": {},
    }

    # elevated: верхние 25% истории, high: верхние 5%
    ranks = pd.DataFrame(index=f.index)
    grid01 = GRID / 100.0
    for col in f.columns:
        ranks[col] = np.interp(f[col], quantiles[col], grid01)

    w = cfg["weights"]
    sev = sum(ranks[c] * w[c] for c in f.columns) / sum(w.values())
    sev = sev.dropna()

    cfg["class_thresholds"] = {
        "elevated": round(float(sev.quantile(0.75)), 3),
        "high": round(float(sev.quantile(0.95)), 3),
    }
    cfg["_severity_distribution"] = {
        "p10": round(float(sev.quantile(0.10)), 3),
        "p50": round(float(sev.quantile(0.50)), 3),
        "p90": round(float(sev.quantile(0.90)), 3),
        "p99": round(float(sev.quantile(0.99)), 3),
    }

    if save:
        path = os.path.join(CONFIG_DIR, "reliability.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
        print(f"записано: {path}")

    return cfg


if __name__ == "__main__":
    cfg = calibrate()
    for k, v in cfg["quantiles"].items():
        print(f"{k:>22}: p5 {v[1]:>9.3f}  p50 {v[10]:>9.3f}  p95 {v[19]:>9.3f}")