"""
Skill score всех целей против ТРИВИАЛЬНОЙ базы -- константы.

ЗАЧЕМ. За всю кампанию модели сравнивались с формулой и сами с собой, но
ни разу с нулевой гипотезой "всегда предсказываем одно и то же число".
Для серы это обнаружилось поздно (H-K3, SULFUR_HYPOTHESES.md): модель
проигрывает константе 1.7-4.4%. Остальные пять целей на константу не
проверялись вообще -- этот скрипт закрывает дыру.

    python scripts/audit_skill_vs_constant.py

МЕТОДИКА (уроки кампании, не менять без причины):
  * walk-forward из 5 блоков, а не один разрез. Разброс RMSE между
    разрезами у серы 2.14-2.69 -- на одном сплите ловится что угодно
    (H-B1: эффект +0.124 на одном разрезе стал -0.054 на пяти).
  * константа = СРЕДНЕЕ обучающей выборки. Оно минимизирует RMSE;
    медиана минимизирует MAE, и сравнивать по RMSE с медианой значит
    давать модели фору (разбор расхождения с Person 2, эксп.19).
  * обучение и константа считаются ТОЛЬКО по прошлому относительно
    тестового блока.

Читает данные и переобучает модели в памяти. Артефакты не трогает.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data.loaders import load_lims, load_telemetry
from src.models.avt import AVTModel, _baseline_fn, _SPECS as AVT_SPECS
from src.models.formula_residual import FormulaPlusResidual
from src.models.go import _SPECS as GO_SPECS
from src.models.base import monotone_vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_avt import build_table as build_avt_table

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "artifacts", "models")
N_BLOCKS = 5


def walk_forward_skill(X: pd.DataFrame, y: pd.Series, spec, label: str,
                        monotone_target: str | None = None):
    """
    Возвращает (rmse_модели, rmse_константы, побед_из_блоков) усреднённо.

    spec = (formula_fn, formula_tags, feature_cols, fallback).
    """
    fn, formula_tags, feature_cols, fb = spec
    order = X.index.sort_values()
    X, y = X.loc[order], y.loc[order]
    n = len(X)
    if n < 300:
        print(f"  {label:<26} мало данных (n={n}) -- пропуск")
        return None

    bounds = np.linspace(int(n * 0.5), n, N_BLOCKS + 1).astype(int)
    r_model, r_const = [], []
    for i in range(N_BLOCKS):
        lo, hi = bounds[i], bounds[i + 1]
        if hi - lo < 20 or lo < 100:
            continue
        Xtr, ytr = X.iloc[:lo], y.iloc[:lo]
        Xte, yte = X.iloc[lo:hi], y.iloc[lo:hi]

        sub = FormulaPlusResidual(
            name=label, formula_fn=fn, formula_tags=formula_tags,
            feature_cols=feature_cols, fallback_mean=fb,
            monotone=(monotone_vector(feature_cols, monotone_target)
                      if monotone_target else None),
        )
        sub.fit(Xtr, ytr)
        pred = np.array([sub.predict_one(r.to_dict()).mean
                         for _, r in Xte.iterrows()])
        const = float(ytr.mean())
        r_model.append(float(np.sqrt(np.mean((pred - yte.values) ** 2))))
        r_const.append(float(np.sqrt(np.mean((const - yte.values) ** 2))))

    if not r_model:
        print(f"  {label:<26} не набралось блоков -- пропуск")
        return None

    m, c = float(np.mean(r_model)), float(np.mean(r_const))
    wins = int(sum(1 for a, b in zip(r_model, r_const) if a < b))
    skill = 100 * (1 - m / c)
    mark = "OK " if skill > 0 else "!! "
    print(f"  {mark}{label:<24}{c:>9.3f}{m:>10.3f}{skill:>+8.1f}%"
          f"{str(wins) + '/' + str(len(r_model)):>10}  n={len(X)}")
    return m, c, wins


def main():
    print("Загрузка данных...")
    tel_avt = load_telemetry("AVT")
    tel_go = load_telemetry("242000")
    lims = load_lims()
    points = sorted(lims["sample_point"].dropna().unique().tolist())

    print(f"\n{'':<27}{'константа':>9}{'модель':>10}{'skill':>9}{'побед':>10}")
    print("  " + "-" * 72)

    print("\n  --- AVTModel ---")
    for out, (fn, tags, extra, fb) in AVT_SPECS.items():
        feature_cols = list(tags) + list(extra)
        t = build_avt_table(out, fn, list(tags), feature_cols, tel_avt, lims, points)
        if t is None:
            continue
        X, y = t
        walk_forward_skill(X, y, (_baseline_fn(out, fn), list(tags), feature_cols, fb), out)

    print("\n  --- GOModel ---")
    avt = AVTModel.load(os.path.join(ARTIFACTS, "avt_v1.joblib"))
    from experiments_sulfur import build_sulfur_table
    table, _ = build_sulfur_table(avt, tel_avt, tel_go, lims, feed_delay_h=0.0)
    table = table.loc[table.index.sort_values()]
    fn, tags, fb = GO_SPECS["sulfur_mgkg"]
    walk_forward_skill(table[tags], table["y"], (fn, tags, tags, fb),
                        "sulfur_mgkg", monotone_target="sulfur_mgkg")

    for out in ("flash_c", "cfpp_c"):
        fn, tags, fb = GO_SPECS[out]
        sub = lims[lims["param"] == out][["ts", "sample_point", "value"]].dropna()
        if sub.empty:
            print(f"  {out:<26} нет меток -- пропуск")
            continue
        pt = sub["sample_point"].value_counts().idxmax()
        sub = sub[sub["sample_point"] == pt][["ts", "value"]]
        Xf = tel_go[tags].dropna()
        left = Xf.reset_index().rename(columns={Xf.index.name or "index": "ts"})
        right = sub.sort_values("ts").rename(columns={"value": "y"})
        merged = pd.merge_asof(left.sort_values("ts"), right, on="ts",
                                direction="backward",
                                tolerance=pd.Timedelta(hours=1)).dropna().set_index("ts")
        walk_forward_skill(merged[tags], merged["y"], (fn, tags, tags, fb), out)

    print("\n  OK = модель бьёт константу, !! = проигрывает ей.")
    print("  Методика: 5 блоков walk-forward, константа = среднее трейна.")


if __name__ == "__main__":
    main()
