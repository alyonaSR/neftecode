"""Сборка ProcessState на момент ts и демо-состояния для сценариев ТЗ."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from ..contracts import Measurement, ProcessState
from ..models.features import add_lags, add_rolling
from .loaders import FEED_POINT, TARGET_POINT, detect_stuck
from .tags import load_config, refusal_rules

# Демо-сценарии из раздела 6 ТЗ
SCENARIOS = ("normal", "quality_risk", "degraded_data")


def build_demo_state(scenario: str = "normal", ts: Optional[datetime] = None) -> ProcessState:
    """Демо-состояние для сценария из SCENARIOS."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Сценарий должен быть из {SCENARIOS}")
    ts = ts or datetime(2026, 3, 14, 8, 20)

    # медианы по истории 2023-2026
    tags: Dict[str, float] = {
        "242000:T5": 370.4,     # температура реактора Р-201
        "242000:F26": 258.6,    # расход гидроочищенного ДТ, объёмный
        "242000:T18": 68.7,     # ВА температуры вспышки ГОДТ
        "242000:T5__lag3h": 366.4,
        "242000:T5__lag6h": 367.8,
        "242000:T5__std3h": 1.95,
        "242000:T5__std6h": 1.68,
        # для признаков серы: температура после квенча, ВСГ, сера сырья
        "242000:T6": 362.7,
        "242000:F25": 13091.4,
        "242000:Q20": 7913.8,
        "242000:Q20__lag3h": 7913.8,
        "242000:Q20__lag6h": 7913.8,
        # для формулы ВАК GODT:CFPP
        "242000:T23": 238.2,
        "242000:P8": 0.186,
        "242000:F9": 189.2,
        "242000:W7": 0.188,
        "242000:P24": 0.62,
        "AVT:F30": 128.4,       # отбор фр.290-350
        "AVT:F32": 81.7,        # отбор фр.240-290
        "AVT:F28": 275.6,       # пар в К-9
        "AVT:F14": 253.5,       # 1 ЦО
        "AVT:P22": 1.12,        # давление верха К-2
        "AVT:T33": 338.3,       # низ К-2
        "AVT:T71": 304.3,       # температура отбора ДТ
        "AVT:F65": 922.4,       # производительность К-2
        "AVT:F36": 131.3,
        "AVT:T66": 254.1,
        "AVT:T37": 60.9,
        "AVT:T40": 177.6,
        "AVT:T58": 58.3,
        "AVT:T42": 285.7,
        "AVT:T48": 354.7,
        "AVT:F31": 537.7,
        "AVT:F57": 33.4,
    }
    dq_flags: List[str] = []

    lims = {
        "sulfur_mgkg": Measurement(8.6, ts - timedelta(hours=6), 360.0, "LIMS", "mg/kg"),
        "flash_c": Measurement(68.0, ts - timedelta(hours=6), 360.0, "LIMS", "degC"),
        "cfpp_c": Measurement(-6.0, ts - timedelta(hours=30), 1800.0, "LIMS", "degC"),
        "d15_kgm3": Measurement(836.1, ts - timedelta(hours=6), 360.0, "LIMS", "kg/m3"),
    }
    pak = {
        "sulfur_mgkg": Measurement(8.4, ts, 0.0, "PAK", "mg/kg"),
        "d15_kgm3": Measurement(835.8, ts, 0.0, "PAK", "kg/m3"),
    }

    if scenario == "quality_risk":
        tags["AVT:F30"] = 141.0     # отбор поднят, хвост тяжелее
        tags["AVT:F32"] = 89.0
        tags["242000:T5"] = 366.5   # реактор холоднее, очистка хуже
        pak["sulfur_mgkg"] = Measurement(9.3, ts, 0.0, "PAK", "mg/kg")
        lims["sulfur_mgkg"] = Measurement(9.2, ts - timedelta(hours=9), 540.0, "LIMS", "mg/kg")

    if scenario == "degraded_data":
        lims["sulfur_mgkg"] = Measurement(8.2, ts - timedelta(hours=52), 3120.0, "LIMS", "mg/kg")
        pak["sulfur_mgkg"] = Measurement(8.37, ts, 0.0, "PAK", "mg/kg", healthy=False)
        pak["d15_kgm3"] = Measurement(None, None, None, "PAK", "kg/m3", healthy=False)
        dq_flags += [
            "PAK sulfur залип: 214 точек подряд без изменения (35.7 ч)",
            "LIMS sulfur старше 48 ч",
            "PAK D15 отсутствует",
        ]

    return ProcessState(ts=ts, tags=tags, lims=lims, pak=pak, dq_flags=dq_flags)


TAG_STALE_MIN = 30.0  # мин; тег старше попадает во флаги

# признаки серы по истории, как в scripts/train_go.py
HISTORY_BASE = ["242000:T5", "242000:T6", "242000:Q20", "242000:F25", "242000:F9"]
HISTORY_LAGS = {"242000:T5": (3, 6), "242000:Q20": (3, 6)}
HISTORY_ROLLING = {"242000:T5": (3, 6)}
HISTORY_ROWS = 60      # полных строк достаточно для лага 6 ч
HISTORY_SPAN_H = 24    # где искать эти строки


def build_state(
    ts: datetime,
    telemetry: pd.DataFrame,
    lims_long: pd.DataFrame,
    pak_long: pd.DataFrame,
) -> ProcessState:
    """ProcessState на момент ts: только данные не позже ts, без интерполяции."""
    ts = pd.Timestamp(ts)
    flags: List[str] = []

    tags = _slice_telemetry(ts, telemetry, flags)
    tags.update(_history_features(ts, telemetry, flags))
    lims = _latest_lims(ts, lims_long, flags)
    pak = _latest_pak(ts, pak_long, flags)

    return ProcessState(ts=ts.to_pydatetime(), tags=tags, lims=lims, pak=pak, dq_flags=flags)


def _slice_telemetry(ts, telemetry: pd.DataFrame, flags: List[str]) -> Dict[str, float]:
    """Последнее непустое значение каждого тега не позже ts."""
    tags: Dict[str, float] = {}
    if telemetry is None or telemetry.empty:
        flags.append("телеметрия не передана")
        return tags

    past = telemetry.loc[:ts]
    if past.empty:
        flags.append(f"нет телеметрии до {ts}")
        return tags

    stale = []
    for col in past.columns:
        idx = past[col].last_valid_index()
        if idx is None:
            flags.append(f"{col}: нет ни одного значения до {ts}")
            continue
        tags[col] = float(past.at[idx, col])
        age_min = (ts - idx).total_seconds() / 60.0
        if age_min > TAG_STALE_MIN:
            stale.append(f"{col} ({age_min / 60:.1f} ч)")

    if stale:
        head = ", ".join(stale[:5])
        tail = f" и ещё {len(stale) - 5}" if len(stale) > 5 else ""
        flags.append(f"устаревшие теги: {head}{tail}")
    return tags


def _history_features(ts, telemetry: pd.DataFrame, flags: List[str]) -> Dict[str, float]:
    """Лаги и скользящие признаки серы на момент ts теми же функциями, что при обучении."""
    if telemetry is None or telemetry.empty or not set(HISTORY_BASE) <= set(telemetry.columns):
        return {}
    win = telemetry.loc[ts - pd.Timedelta(hours=HISTORY_SPAN_H):ts, HISTORY_BASE].dropna()
    win = win.tail(HISTORY_ROWS)
    if win.empty:
        flags.append("нет истории для признаков серы")
        return {}

    feats = win
    for col, lags in HISTORY_LAGS.items():
        feats = add_lags(feats, [col], lags_h=lags)
    for col, windows in HISTORY_ROLLING.items():
        feats = add_rolling(feats, [col], windows_h=windows)
    last = feats.iloc[-1]

    age_min = (ts - feats.index[-1]).total_seconds() / 60.0
    if age_min > TAG_STALE_MIN:
        flags.append(f"признаки серы по истории устарели на {age_min / 60:.1f} ч")

    names = [f"{c}__lag{h}h" for c, hs in HISTORY_LAGS.items() for h in hs]
    names += [f"{c}__std{h}h" for c, hs in HISTORY_ROLLING.items() for h in hs]
    return {n: float(last[n]) for n in names if pd.notna(last[n])}


def _latest_lims(ts, lims_long: pd.DataFrame, flags: List[str]) -> Dict[str, Measurement]:
    """Последний анализ ЛИМС не позже ts. Сера сырья под ключом 'feed:sulfur_mgkg'."""
    out: Dict[str, Measurement] = {}
    if lims_long is None or lims_long.empty:
        flags.append("ЛИМС не передан")
        return out

    max_age = float(refusal_rules().get("max_lims_age_min", 1440))

    for point, prefix in ((TARGET_POINT, ""), (FEED_POINT, "feed:")):
        sub = lims_long[(lims_long["sample_point"] == point) & (lims_long["ts"] <= ts)]
        if prefix == "feed:":
            sub = sub[sub["param"] == "sulfur_mgkg"]
        for param, grp in sub.groupby("param"):
            row = grp.loc[grp["ts"].idxmax()]
            age = (ts - row["ts"]).total_seconds() / 60.0
            out[f"{prefix}{param}"] = Measurement(
                value=float(row["value"]),
                ts=row["ts"].to_pydatetime(),
                age_min=round(age, 1),
                source="LIMS",
                units=str(row["units"]),
                healthy=not bool(row.get("outlier", False)),
            )

    key = "sulfur_mgkg"
    if key not in out:
        flags.append(f"ЛИМС: нет ни одного анализа серы до {ts}")
    elif out[key].age_min > max_age:
        flags.append(
            f"ЛИМС sulfur старше {max_age / 60:.0f} ч: {out[key].age_min / 60:.1f} ч"
        )
    return out


def _latest_pak(ts, pak_long: pd.DataFrame, flags: List[str]) -> Dict[str, Measurement]:
    """Последнее показание каждого ПАК не позже ts. Залипшее хранится с healthy=False."""
    out: Dict[str, Measurement] = {}
    if pak_long is None or pak_long.empty:
        flags.append("ПАК не передан")
        return out

    for param, grp in pak_long[pak_long["ts"] <= ts].groupby("param"):
        row = grp.loc[grp["ts"].idxmax()]
        healthy = bool(row.get("healthy", True)) and not bool(row.get("outlier", False))
        age = (ts - row["ts"]).total_seconds() / 60.0
        out[param] = Measurement(
            value=float(row["value"]),
            ts=row["ts"].to_pydatetime(),
            age_min=round(age, 1),
            source="PAK",
            units=str(row["units"]),
            healthy=healthy,
        )
        if not healthy:
            flags.append(f"ПАК {param}: анализатор признан неисправным (залипание)")

    for expected in ("sulfur_mgkg", "d15_kgm3"):
        if expected not in out:
            flags.append(f"ПАК {expected} отсутствует на {ts}")
    return out


def scan_data_quality(df: pd.DataFrame, freq_min: int = 10) -> List[str]:
    """Флаги по колонкам с пропусками больше 5% и залипанием."""
    flags: List[str] = []
    for col in df.columns:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        nan_share = float(s.isna().mean())
        if nan_share > 0.05:
            flags.append(f"{col}: пропусков {nan_share:.0%}")
        stuck = detect_stuck(s.dropna())
        if stuck.any():
            hours = stuck.sum() * freq_min / 60.0
            flags.append(f"{col}: залипание суммарно {hours:.0f} ч")
    return flags