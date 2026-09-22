#!/usr/bin/env python3
"""
Обучение AVTModel (формула ВАК + LightGBM на остатках).

python scripts/train_avt.py

Для каждого показателя AVTModel.outputs:
  1. собирает признаки -- те же теги, что требует формула (models/avt.py::_SPECS)
  2. присоединяет ЛИМС as-of (строго назад по времени, без утечки)
     к точке отбора, которая ЛУЧШЕ ВСЕГО коррелирует с формулой
     (см. допущение ниже)
  3. FormulaPlusResidual.fit(): хронологический сплит train/calib,
     LightGBM на остатке, эмпирические квантили остатка для интервала
  4. сохраняет в artifacts/models/avt_v1.joblib

ДОПУЩЕНИЕ, требует подтверждения организаторов: какая из 4 точек
отбора АВТ (1 / 2 / 2.1 / 3) физически соответствует дизельному сырью,
уходящему в гидроочистку -- неизвестно (см. Q&A 15.09, ответ только
"1-е на входе... 3-й конечный выход установки", без привязки к кускам
240-350 vs 350). Здесь берётся точка с максимальной |corr| для каждой
формулы отдельно, как и при аудите формул -- это эмпирический, а не
подтверждённый выбор.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data.loaders import load_lims, load_telemetry
from src.models.avt import _SPECS, AVTModel


def best_point(pred: pd.Series, lims: pd.DataFrame, param: str, points: list, tol_h: float = 1.0):
    """Точка отбора с максимальной |corr| формулы против ЛИМС этого показателя."""
    best_pt, best_corr, best_n = None, -1.0, 0
    for pt in points:
        sub = lims[(lims["sample_point"] == pt) & (lims["param"] == param)]
        if sub.empty:
            continue
        left = pred.rename("pred").to_frame().reset_index()
        left.columns = ["ts", "pred"]
        left = left.replace([np.inf, -np.inf], np.nan).dropna().sort_values("ts")
        right = sub[["ts", "value"]].dropna().sort_values("ts").rename(columns={"value": "y"})
        m = pd.merge_asof(left, right, on="ts", direction="backward",
                           tolerance=pd.Timedelta(hours=tol_h)).dropna()
        if len(m) < 30:
            continue
        c = abs(m["pred"].corr(m["y"])) if m["pred"].std() > 0 else 0.0
        if c > best_corr:
            best_pt, best_corr, best_n = pt, c, len(m)
    return best_pt, best_corr, best_n


def build_table(out: str, fn, formula_tags: list, feature_cols: list,
                 tel: pd.DataFrame, lims: pd.DataFrame, points: list):
    """
    Возвращает (X, y) с общим DatetimeIndex, готовые для FormulaPlusResidual.fit.

    formula_tags -- то, что реально ждёт fn() (baseline), feature_cols --
    formula_tags + extra-признаки для остатка (например AVT:F32 у
    feed_ebp_c). Точка отбора ЛИМС ищется по корреляции ФОРМУЛЫ, extra
    в ней не участвует -- это признак для остатка, не для baseline.
    """
    param = {"feed_ebp_c": "ebp_c", "feed_d15_kgm3": "d15_kgm3",
             "feed_cfpp_c": "cfpp_c", "feed_flash_c": "flash_c"}[out]

    X_formula = tel[formula_tags].dropna()
    pred = fn(X_formula) if fn is not None else pd.Series(0.0, index=X_formula.index)

    pt, corr, n = best_point(pred, lims, param, points)
    if pt is None:
        print(f"  {out}: нет точки ЛИМС с параметром '{param}' (>=30 сопоставленных) -- пропуск")
        return None

    X_full = tel[feature_cols].dropna()
    sub = lims[(lims["sample_point"] == pt) & (lims["param"] == param)][["ts", "value"]].dropna()
    left = X_full.reset_index().rename(columns={X_full.index.name or "index": "ts"})
    right = sub.sort_values("ts").rename(columns={"value": "y"})
    merged = pd.merge_asof(left.sort_values("ts"), right, on="ts",
                            direction="backward", tolerance=pd.Timedelta(hours=1.0)).dropna()
    merged = merged.set_index("ts")

    print(f"  {out}: точка='{pt[:60]}...' n={len(merged)} |corr(формула,ЛИМС)|={corr:.3f}")
    return merged[feature_cols], merged["y"]


def main():
    print("Загрузка телеметрии и ЛИМС...")
    tel = load_telemetry("AVT")
    lims = load_lims()
    points = sorted(p for p in lims["sample_point"].unique() if "АВТ" in p)

    print("\nСборка обучающих таблиц (as-of join, без утечки)...")
    tables = {}
    for out, (fn, tags, extra, _fb) in _SPECS.items():
        t = build_table(out, fn, tags, tags + extra, tel, lims, points)
        if t is not None:
            tables[out] = t

    if not tables:
        print("Не собралось ни одной обучающей таблицы, выхожу.")
        return

    print("\nОбучение FormulaPlusResidual по каждому показателю...")
    model = AVTModel()
    for out in list(model._models):
        if out not in tables:
            print(f"  {out}: пропущен (нет обучающей таблицы), остаётся formula-only")
            continue
        X, y = tables[out]
        before = X.apply(lambda r: model._models[out].baseline(r), axis=1)
        rmse_before = float(np.sqrt(((before - y) ** 2).mean()))
        model._models[out].fit(X, y)
        print(f"  {out}: RMSE формулы-baseline={rmse_before:.2f}  "
              f"(после fit: train={model._models[out]._n_train}, calib={model._models[out]._n_calib})")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "artifacts", "models")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "avt_v1.joblib")
    model.save(path)
    print(f"\nсохранено -> {path}")


if __name__ == "__main__":
    main()
