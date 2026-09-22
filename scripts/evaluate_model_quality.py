#!/usr/bin/env python3
"""
Сводная оценка качества всех обученных показателей AVTModel/GOModel:
RMSE полной модели (formula+residual), относительная ошибка, покрытие
конформного интервала -- одна таблица вместо разрозненных чисел по
отдельным скриптам обучения.

Только читает данные и обученные
артефакты, ничего не меняет и не переобучает.

    python scripts/evaluate_model_quality.py

ЧЕСТНО: покрытие считается на калибровочном хвосте (последние 20% по
времени от таблицы, near-LIMS выборка) -- том же, на котором строились
конформные квантили. Это проверка "калибровка применена правильно",
а не независимый held-out тест. README заявляет методологию
"train до 2025-12-31, test 2026 год" (features.time_split), но фактически
FormulaPlusResidual.fit() использует свой внутренний хронологический
80/20 сплит по счётчику строк -- отдельного held-out 2026-теста поверх
него сегодня нет. Это тоже часть честного отчёта, не сокрыто.
"""
from __future__ import annotations

import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from src.data.loaders import TARGET_POINT, load_lims, load_telemetry
from src.models.avt import AVTModel
from src.models.avt import _SPECS as AVT_SPECS
from src.models.go import GOModel
from src.models.go import _SPECS as GO_SPECS

from train_avt import best_point, build_table as build_avt_table  # noqa: E402
from experiments_sulfur import build_sulfur_table  # noqa: E402


def evaluate_output(label: str, model_obj, out: str, X: pd.DataFrame, y: pd.Series):
    sub = model_obj._models[out]
    if sub._model is None:
        print(f"{label}.{out:<16} formula-only, обученного остатка нет -- пропуск")
        return None

    order = X.index.sort_values()
    X, y = X.loc[order], y.loc[order]
    cut = int(len(X) * 0.8)
    X_calib, y_calib = X.iloc[cut:], y.iloc[cut:]
    if len(X_calib) < 5:
        print(f"{label}.{out:<16} мало точек на калибровке ({len(X_calib)}) -- пропуск")
        return None

    means, los, his = [], [], []
    for _, row in X_calib.iterrows():
        iv = sub.predict_one(row.to_dict())
        means.append(iv.mean); los.append(iv.lo); his.append(iv.hi)
    means, los, his = np.array(means), np.array(los), np.array(his)

    rmse = float(np.sqrt(((means - y_calib.values) ** 2).mean()))
    typical = float(y_calib.median())
    rel_rmse = 100 * rmse / typical if typical else float("nan")
    coverage = float(np.mean((y_calib.values >= los) & (y_calib.values <= his)))
    width = float(np.mean(his - los))

    print(f"{label}.{out:<16} n={len(X_calib):5d}  медиана факта={typical:8.2f}  "
          f"RMSE={rmse:7.3f}  RMSE/медиана={rel_rmse:5.1f}%  "
          f"покрытие={coverage:5.1%}  ширина интервала={width:6.2f}")
    return dict(label=label, out=out, n=len(X_calib), typical=typical,
                rmse=rmse, rel_rmse=rel_rmse, coverage=coverage, width=width)


def main():
    print("Загрузка данных и артефактов...")
    tel_avt = load_telemetry("AVT")
    tel_go = load_telemetry("242000")
    lims = load_lims()
    avt = AVTModel.load(os.path.join(ROOT, "artifacts", "models", "avt_v1.joblib"))
    go = GOModel.load(os.path.join(ROOT, "artifacts", "models", "go_v1.joblib"))

    print(f"\n{'показатель':<26}{'n':>7}{'медиана':>12}{'RMSE':>9}{'RMSE/медиана':>14}{'покрытие':>10}{'ширина':>9}")
    print("-" * 90)

    results = []

    # --- AVTModel ---
    avt_points = sorted(p for p in lims["sample_point"].unique() if "АВТ" in p)
    for out, (fn, tags, extra, _fb) in AVT_SPECS.items():
        t = build_avt_table(out, fn, tags, tags + extra, tel_avt, lims, avt_points)
        if t is None:
            print(f"AVTModel.{out:<16} нет обучающей таблицы (formula-only) -- пропуск")
            continue
        X, y = t
        r = evaluate_output("AVTModel", avt, out, X, y)
        if r:
            results.append(r)

    # --- GOModel: sulfur_mgkg ---
    table_s, _ = build_sulfur_table(avt, tel_avt, tel_go, lims, feed_delay_h=0.0)
    r = evaluate_output("GOModel", go, "sulfur_mgkg",
                         table_s[GO_SPECS["sulfur_mgkg"][1]], table_s["y"])
    if r:
        results.append(r)

    # --- GOModel: cfpp_c ---
    cfpp_tags = GO_SPECS["cfpp_c"][1]
    X_cfpp = tel_go[cfpp_tags].dropna()
    left = X_cfpp.reset_index().rename(columns={X_cfpp.index.name or "index": "ts"})
    sub_lims = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == "cfpp_c")][["ts", "value"]].dropna()
    right = sub_lims.sort_values("ts").rename(columns={"value": "y"})
    merged = pd.merge_asof(left.sort_values("ts"), right, on="ts",
                            direction="backward", tolerance=pd.Timedelta(hours=1.0)).dropna().set_index("ts")
    r = evaluate_output("GOModel", go, "cfpp_c", merged[cfpp_tags], merged["y"])
    if r:
        results.append(r)

    # --- GOModel: flash_c (H-C1, 242000:T18) ---
    flash_tags = GO_SPECS["flash_c"][1]
    X_flash = tel_go[flash_tags].dropna()
    left = X_flash.reset_index().rename(columns={X_flash.index.name or "index": "ts"})
    sub_lims = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == "flash_c")][["ts", "value"]].dropna()
    right = sub_lims.sort_values("ts").rename(columns={"value": "y"})
    merged = pd.merge_asof(left.sort_values("ts"), right, on="ts",
                            direction="backward", tolerance=pd.Timedelta(hours=1.0)).dropna().set_index("ts")
    r = evaluate_output("GOModel", go, "flash_c", merged[flash_tags], merged["y"])
    if r:
        results.append(r)

    print("\nfeed_flash_c (AVTModel), d15_kgm3 (GOModel): formula-only / "
          "физическая аппроксимация без обучения -- не в этой таблице, RMSE неприменим.")


if __name__ == "__main__":
    main()
