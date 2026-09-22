#!/usr/bin/env python3
"""
Дешёвый аудит: окупится ли сдвиг АВТ-признаков на транспортное
запаздывание (tau1 ~ 1.5-4.0 ч, research.pdf) для предсказания серы ГО.

Ничего не меняет в общих
файлах -- только читает данные и обученный artifacts/models/avt_v1.joblib,
печатает таблицу корреляций. Решение "стоит ли переделывать train_go.py
под лаг-сдвинутые признаки" принимается ПОСЛЕ этого прогона, не раньше.

    python scripts/audit_avt_lag_correlation.py

Логика: для каждого кандидата задержки delay берём состояние АВТ на
момент (ts_lims - delay) для каждой реальной пробы серы ЛИМС (точка 2
гидроочистки), прогоняем через обученный AVTModel.predict(), считаем
корреляцию feed_ebp_c/feed_d15_kgm3 с фактической серой. delay=0 --
это ровно то, что go.py делает сегодня (без сдвига), остальные --
кандидаты на замену.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data.loaders import TARGET_POINT, load_lims, load_telemetry
from src.models.avt import AVTModel

CANDIDATE_DELAYS_H = [0.0, 1.0, 1.5, 2.0, 3.0, 4.0]


def sulfur_with_avt_at_delay(tel_avt: pd.DataFrame, sulfur: pd.DataFrame,
                              avt_model: AVTModel, delay_h: float, tol_h: float = 1.0):
    """
    Для каждой пробы серы ЛИМС -- состояние АВТ на (ts_lims - delay_h),
    через AVTModel.predict(). Возвращает DataFrame с y (сера) и
    feed_ebp_c/feed_d15_kgm3, посчитанными на сдвинутых тегах.
    """
    left = sulfur[["ts", "value"]].dropna().copy()
    left["lookup_ts"] = left["ts"] - pd.Timedelta(hours=delay_h)
    left = left.sort_values("lookup_ts")

    right = tel_avt[avt_model.required_features].dropna().reset_index()
    right = right.rename(columns={right.columns[0]: "avt_ts"}).sort_values("avt_ts")

    merged = pd.merge_asof(
        left, right, left_on="lookup_ts", right_on="avt_ts",
        direction="backward", tolerance=pd.Timedelta(hours=tol_h),
    ).dropna()

    if merged.empty:
        return None

    feats = merged[avt_model.required_features]
    preds = [avt_model.predict(row.to_dict()) for _, row in feats.iterrows()]
    merged["feed_ebp_c"] = [p["feed_ebp_c"].mean for p in preds]
    merged["feed_d15_kgm3"] = [p["feed_d15_kgm3"].mean for p in preds]
    return merged


def main():
    print("Загрузка телеметрии АВТ и ЛИМС...")
    tel_avt = load_telemetry("AVT")
    lims = load_lims()
    avt_model = AVTModel.load(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "artifacts", "models", "avt_v1.joblib")
    )

    sulfur = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == "sulfur_mgkg")]
    sulfur = sulfur[sulfur["value"] <= 20.0]  # тот же порог отсечения выбросов, что в train_go.py
    print(f"Проб серы (после отсечения выбросов >20 мг/кг): {len(sulfur)}")

    print(f"\n{'delay_h':>8}  {'n':>5}  {'corr(feed_ebp_c, S)':>20}  {'corr(feed_d15, S)':>18}")
    print("-" * 60)
    for delay in CANDIDATE_DELAYS_H:
        m = sulfur_with_avt_at_delay(tel_avt, sulfur, avt_model, delay)
        if m is None or len(m) < 20:
            print(f"{delay:8.1f}  {'--':>5}  {'мало сопоставлений':>20}")
            continue
        c_ebp = m["feed_ebp_c"].corr(m["value"])
        c_d15 = m["feed_d15_kgm3"].corr(m["value"])
        print(f"{delay:8.1f}  {len(m):5d}  {c_ebp:20.3f}  {c_d15:18.3f}")

    print("\ndelay_h=0.0 -- это ровно то, что go.py делает сегодня (без сдвига).")
    print("Для сравнения из README: макс. |corr| любой сырой колонки 24-2000 с серой = 0.23.")


if __name__ == "__main__":
    main()
