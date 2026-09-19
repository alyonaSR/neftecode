#!/usr/bin/env python3
"""
Набор экспериментов по серной модели GOModel -- отвечает на вопрос
"что ещё реально стоит попробовать, прежде чем сдавать эту часть".

Зона ответственности: Person 3. Только читает данные и обученные
артефакты, ничего не переобучает и не перезаписывает go_v1.joblib --
это РАЗВЕДКА, а не изменение продакшен-модели.

    python scripts/experiments_sulfur.py

Эксперименты:
  1. RMSE полной модели (baseline + остаток) против RMSE одной формулы
     на калибровке -- окупает ли себя ML-слой вообще.
  2. Аблация 242000:Q20 (ПАК сера сырья, найдена Person 2, 2026-09-19) --
     честная проверка строже, чем просто gain% в дереве: RMSE на
     калибровке с признаком и без.
  3. corr(feed_d15_kgm3 со сдвигом, ОСТАТОК) -- прошлый аудит лагов
     (scripts/audit_avt_lag_correlation.py) сравнивал с сырой серой,
     а не с тем, что реально видит LightGBM (остаток после baseline).
  4. Ширина конформного интервала при alpha=0.05 (95%, как в
     research.pdf) против текущего alpha=0.10 -- для документации,
     без замены задеплоенной модели.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data.loaders import TARGET_POINT, load_lims, load_telemetry
from src.models.avt import AVTModel
from src.models.conformal import ConformalResidualBounds
from src.models.features import add_lags, add_rolling
from src.models.go import GOModel, sulfur_arrhenius_baseline
from src.models.go import _SPECS as GO_SPECS

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "artifacts", "models")
SULFUR_OUTLIER_THRESHOLD = 20.0
# Источник правды -- go.py._SPECS, не дублируем список руками (дублирование
# и разъехалось в прошлый раз: этот файл не знал о 242000:Q20, добавленной
# в go.py 2026-09-19, что уронило evaluate_model_quality.py KeyError).
SULFUR_FEATURES = GO_SPECS["sulfur_mgkg"][1]


def as_of_target(X: pd.DataFrame, lims: pd.DataFrame, param: str, tol_h: float = 1.0):
    sub = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == param)][["ts", "value"]].dropna()
    left = X.reset_index().rename(columns={X.index.name or "index": "ts"})
    right = sub.sort_values("ts").rename(columns={"value": "y"})
    merged = pd.merge_asof(left.sort_values("ts"), right, on="ts",
                            direction="backward", tolerance=pd.Timedelta(hours=tol_h)).dropna()
    return merged.set_index("ts")


def build_sulfur_table(avt_model: AVTModel, tel_avt_raw, tel_go, lims, feed_delay_h: float = 0.0):
    """
    Та же сборка, что train_go.py, но с опциональным сдвигом feed_ebp_c/
    feed_d15_kgm3 на feed_delay_h назад по времени (эксперимент 3).
    """
    t5 = tel_go[["242000:T5", "242000:Q20"]].dropna()
    t5 = add_lags(t5, ["242000:T5", "242000:Q20"], lags_h=(3, 6))
    t5 = add_rolling(t5, ["242000:T5"], windows_h=(3, 6))
    t5 = t5.dropna()

    common_idx = t5.index.intersection(tel_avt_raw.index)
    avt_aligned = tel_avt_raw.loc[common_idx, avt_model.required_features].dropna()
    common_idx = common_idx.intersection(avt_aligned.index)

    sulfur_lims = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == "sulfur_mgkg")]
    near_mask = pd.Series(False, index=common_idx)
    for ts in sulfur_lims["ts"]:
        near_mask |= (common_idx >= ts - pd.Timedelta(hours=1) - pd.Timedelta(hours=feed_delay_h)) & \
                     (common_idx <= ts + pd.Timedelta(hours=1) - pd.Timedelta(hours=feed_delay_h))
    idx_needed = common_idx[near_mask]

    feed_ebp, feed_d15 = [], []
    for ts in idx_needed:
        out = avt_model.predict(avt_aligned.loc[ts].to_dict())
        feed_ebp.append(out["feed_ebp_c"].mean)
        feed_d15.append(out["feed_d15_kgm3"].mean)

    X_sulfur = t5.loc[idx_needed].copy()
    X_sulfur["feed_ebp_c"] = feed_ebp
    X_sulfur["feed_d15_kgm3"] = feed_d15
    if feed_delay_h:
        X_sulfur.index = X_sulfur.index + pd.Timedelta(hours=feed_delay_h)
    X_sulfur = X_sulfur[SULFUR_FEATURES]

    table = as_of_target(X_sulfur, lims, "sulfur_mgkg")
    before_n = len(table)
    table = table[table["y"] <= SULFUR_OUTLIER_THRESHOLD]
    return table, before_n - len(table)


def experiment_1_ml_value(go: GOModel, X_calib, y_calib):
    print("\n=== Эксперимент 1: окупает ли себя ML-слой поверх формулы ===")
    sub = go._models["sulfur_mgkg"]
    base = X_calib.apply(lambda r: sub.baseline(r), axis=1)
    rmse_baseline = float(np.sqrt(((base - y_calib) ** 2).mean()))

    full_pred = [sub.predict_one(r.to_dict()).mean for _, r in X_calib.iterrows()]
    rmse_full = float(np.sqrt(((np.array(full_pred) - y_calib) ** 2).mean()))

    print(f"  RMSE формулы (baseline)         = {rmse_baseline:.3f}")
    print(f"  RMSE полной модели (+ostatok)   = {rmse_full:.3f}")
    print(f"  Улучшение: {100*(1 - rmse_full/rmse_baseline):.1f}%")


def experiment_2_q20_ablation(X_train, y_train, X_calib, y_calib):
    """
    Та же методология, что раньше проверяла catalyst_age_days (тот
    эксперимент сделал своё дело, признак убран из прода 2026-09-16,
    удалён отсюда). Теперь проверяет 242000:Q20 (ПАК сера сырья,
    найдена Person 2, 2026-09-19) -- честная аблация строже, чем просто
    посмотреть gain% в обученной модели: считает RMSE на калибровке
    с признаком и без, а не только "сколько раз дерево по нему сплитилось".
    """
    print("\n=== Эксперимент 2: аблация 242000:Q20 (ПАК сера сырья) ===")
    import lightgbm as lgb

    def fit_eval(feature_cols, label):
        base_train = X_train[["242000:T5"]].apply(
            lambda r: sulfur_arrhenius_baseline(r), axis=1)
        resid_train = y_train - base_train
        model = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   num_leaves=15, min_child_samples=max(5, len(X_train)//50),
                                   random_state=42, verbose=-1)
        safe_cols = {c: c.replace(":", "__") for c in feature_cols}
        Xt = X_train[feature_cols].rename(columns=safe_cols)
        model.fit(Xt, resid_train)

        base_calib = X_calib[["242000:T5"]].apply(
            lambda r: sulfur_arrhenius_baseline(r), axis=1)
        Xc = X_calib[feature_cols].rename(columns=safe_cols)
        pred_calib = base_calib + model.predict(Xc)
        err = y_calib - pred_calib
        rmse = float(np.sqrt((err ** 2).mean()))
        bias = float(np.median(err))
        gain = model.booster_.feature_importance(importance_type="gain")
        top = sorted(zip(feature_cols, gain), key=lambda r: -r[1])[:3]
        print(f"  {label}: RMSE={rmse:.3f}  bias(median)={bias:.3f}  топ-фичи={top}")
        return rmse, bias

    q20_cols = {"242000:Q20", "242000:Q20__lag3h", "242000:Q20__lag6h"}
    with_q20 = list(SULFUR_FEATURES)
    without_q20 = [c for c in SULFUR_FEATURES if c not in q20_cols]

    r_with = fit_eval(with_q20, "С Q20   ")
    r_without = fit_eval(without_q20, "БЕЗ Q20 ")
    return r_with, r_without


def experiment_3_lagged_feed_d15_vs_residual(avt_model, tel_avt_raw, tel_go, lims):
    print("\n=== Эксперимент 3: corr(feed_d15_kgm3 со сдвигом, ОСТАТОК после baseline) ===")
    print("  (в отличие от audit_avt_lag_correlation.py -- сравниваем с ОСТАТКОМ,")
    print("   не с сырой серой, это то, что реально видит LightGBM)")
    for delay in [0.0, 1.5, 2.0, 3.0, 4.0]:
        table, n_excl = build_sulfur_table(avt_model, tel_avt_raw, tel_go, lims, feed_delay_h=delay)
        if len(table) < 30:
            print(f"  delay={delay:.1f}h: мало точек ({len(table)}), пропуск")
            continue
        base = table[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        residual = table["y"] - base
        c = table["feed_d15_kgm3"].corr(residual)
        print(f"  delay={delay:.1f}h  n={len(table):4d}  corr(feed_d15_kgm3, остаток) = {c:.3f}")


def experiment_4_conformal_at_95pct(go: GOModel):
    print("\n=== Эксперимент 4: ширина интервала при alpha=0.05 (95%) vs 0.10 (90%) ===")
    sub = go._models["sulfur_mgkg"]
    old = sub._conformal
    if old is None:
        print("  нет калибратора -- пропуск")
        return
    offset_lo_90, offset_hi_90 = old.bounds()

    strict = ConformalResidualBounds(alpha=0.05)
    strict._hi_scores = list(old._hi_scores)
    strict._lo_scores = list(old._lo_scores)
    offset_lo_95, offset_hi_95 = strict.bounds()

    print(f"  alpha=0.10 (текущий, задеплоенный): offset=[{offset_lo_90:.2f}, {offset_hi_90:.2f}], ширина={offset_hi_90-offset_lo_90:.2f}")
    print(f"  alpha=0.05 (research.pdf, 95%):     offset=[{offset_lo_95:.2f}, {offset_hi_95:.2f}], ширина={offset_hi_95-offset_lo_95:.2f}")


def main():
    print("Загрузка данных и артефактов...")
    tel_avt_raw = load_telemetry("AVT")
    tel_go = load_telemetry("242000")
    lims = load_lims()
    avt = AVTModel.load(os.path.join(ARTIFACTS, "avt_v1.joblib"))
    go = GOModel.load(os.path.join(ARTIFACTS, "go_v1.joblib"))

    print("Сборка обучающей таблицы серы (как в train_go.py, delay=0)...")
    table, n_excl = build_sulfur_table(avt, tel_avt_raw, tel_go, lims, feed_delay_h=0.0)
    print(f"n={len(table)} (исключено {n_excl} выбросов > {SULFUR_OUTLIER_THRESHOLD} мг/кг)")

    order = table.index.sort_values()
    table = table.loc[order]
    cut = int(len(table) * 0.8)
    X_train, X_calib = table.iloc[:cut][SULFUR_FEATURES], table.iloc[cut:][SULFUR_FEATURES]
    y_train, y_calib = table.iloc[:cut]["y"], table.iloc[cut:]["y"]

    experiment_1_ml_value(go, X_calib, y_calib)
    experiment_2_q20_ablation(X_train, y_train, X_calib, y_calib)
    experiment_3_lagged_feed_d15_vs_residual(avt, tel_avt_raw, tel_go, lims)
    experiment_4_conformal_at_95pct(go)


if __name__ == "__main__":
    main()
