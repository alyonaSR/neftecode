#!/usr/bin/env python3
"""
Аудит формул ВАК против ЛИМС.

Зачем этот скрипт существует отдельно, а не одноразовый прогон в
блокноте: формулы с листа "ВАК" Теги_хакатон.xlsx до правок
организаторов (Q&A 15.09) давали физическую бессмыслицу -- например,
24-2000:GODT:T90 предсказывал ~203 000 градусов вместо ~330 по ЛИМС
(n=8723). Причина оказалась в масштабе одного из членов формулы
(F15/2000, а не F15), а не в ошибке кода. Это ровно тот класс бага,
который аудит по данным ловит за секунды, а чтение ТЗ -- никогда.
Воспроизводимость проверки -- часть критериев оценки ТЗ, поэтому
результат должен быть скриптом, а не разовым выводом в чат.

    python scripts/audit_vak_formulas.py

Печатает RMSE/медианную ошибку/корреляцию каждой формулы из
models/vak_formulas.py против сопоставленных по времени (as-of,
backward, без утечки) значений ЛИМС.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data.loaders import TARGET_POINT, load_lims, load_telemetry
from src.models import vak_formulas as vak


def evaluate(pred: pd.Series, lims_sub: pd.DataFrame, tol_h: float = 1.0):
    left = pred.rename("pred").to_frame().reset_index()
    left.columns = ["ts", "pred"]
    left = left.replace([np.inf, -np.inf], np.nan).dropna().sort_values("ts")
    right = lims_sub[["ts", "value"]].dropna().sort_values("ts").rename(columns={"value": "actual"})
    if right.empty or left.empty:
        return None
    merged = pd.merge_asof(left, right, on="ts", direction="backward",
                            tolerance=pd.Timedelta(hours=tol_h)).dropna()
    if len(merged) < 5:
        return None
    err = merged["pred"] - merged["actual"]
    return {
        "n": len(merged),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 2),
        "medae": round(float(err.abs().median()), 2),
        "bias": round(float(err.mean()), 2),
        "corr": round(float(merged["pred"].corr(merged["actual"])), 3)
                if merged["pred"].std() > 0 else float("nan"),
    }


def run_registry(registry, tel: pd.DataFrame, lims: pd.DataFrame, points, label: str):
    print(f"\n=== {label} ===")
    for out, (fn, tags) in registry.items():
        try:
            pred = fn(tel)  # векторизовано: fn индексирует tel["AVT:..."] как колонки
        except KeyError as e:
            print(f"  {fn.__name__} (-> {out}): SKIP, нет колонки {e}")
            continue

        best = None
        for pt_name, sub_all in points:
            sub = sub_all[sub_all["param"] == out]
            if sub.empty:
                continue
            res = evaluate(pred, sub)
            if res is None:
                continue
            res["point"] = pt_name
            if best is None or abs(res["corr"]) > abs(best["corr"]):
                best = res
        if best is None:
            print(f"  {fn.__name__} (-> {out}): нет сопоставимых точек ЛИМС")
        else:
            print(f"  {fn.__name__} (-> {out}): point='{best['point']}' n={best['n']} "
                  f"rmse={best['rmse']} medae={best['medae']} bias={best['bias']} corr={best['corr']}")


def main():
    print("Загрузка телеметрии и ЛИМС...")
    tel_avt = load_telemetry("AVT")
    tel_go = load_telemetry("242000")
    lims = load_lims()

    avt_points = [(p, lims[lims["sample_point"] == p])
                  for p in sorted(s for s in lims["sample_point"].unique() if "АВТ" in s)]
    target_points = [(TARGET_POINT, lims[lims["sample_point"] == TARGET_POINT])]

    run_registry(vak.AVT_240_350, tel_avt, lims, avt_points, "АВТ, кусок 240-350 (дизельный диапазон)")
    run_registry(vak.AVT_350, tel_avt, lims, avt_points, "АВТ, кусок 350+ (тяжелее дизеля -- НЕ сырьё ГО)")
    run_registry(vak.GODT, tel_go, lims, target_points, "24-2000 ГОДТ (товарный продукт, точка 2)")

    print("\nПропущены (член формулы содержит LIMS:..., утечка из будущего, "
          "не в контракте GOModel.outputs): godt_d15, godt_t95")


if __name__ == "__main__":
    main()
