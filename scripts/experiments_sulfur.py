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
  5. H-B3: честный трёхсторонний train/calib/TEST (SULFUR_HYPOTHESES.md).
     За 2026-09-19 калибровочная выборка использовалась 6 раз подряд для
     разных гипотез -- и для conformal-квантилей, и для отчёта RMSE.
     Риск: незаметное переобучение на один и тот же отложенный кусок
     через повторные итерации. Test здесь -- последние по времени 15%
     данных, которых не видел НИ ОДИН прогон сегодня.
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
from src.models.features import add_lags, add_rolling, arrhenius_term
from src.models.go import GOModel, sulfur_arrhenius_baseline
from src.models.go import _SPECS as GO_SPECS
from src.models.formula_residual import FormulaPlusResidual

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


def build_sulfur_table(avt_model: AVTModel, tel_avt_raw, tel_go, lims,
                        feed_delay_h: float = 0.0, drop_outliers: bool = True):
    """
    Та же сборка, что train_go.py, но с опциональным сдвигом feed_ebp_c/
    feed_d15_kgm3 на feed_delay_h назад по времени (эксперимент 3).

    drop_outliers=False оставляет точки с y > SULFUR_OUTLIER_THRESHOLD
    в таблице -- нужно эксперименту 6 (H-B2), который проверяет, не лучше
    ли их взвешивать, а не выбрасывать.
    """
    t5 = tel_go[["242000:T5", "242000:T6", "242000:Q20", "242000:F25", "242000:F9"]].dropna()
    t5["242000:T5_T6_quench_delta"] = t5["242000:T5"] - t5["242000:T6"]
    t5["242000:h2_oil_ratio"] = t5["242000:F25"] / t5["242000:F9"].replace(0, float("nan"))
    t5["242000:arrhenius_t5"] = arrhenius_term(t5["242000:T5"])
    t5["242000:q20_x_arrhenius"] = t5["242000:Q20"] * t5["242000:arrhenius_t5"]
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
    if drop_outliers:
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


def experiment_5_true_holdout_test(table: pd.DataFrame):
    """
    H-B3. table уже отсортирована по времени. Берём последние 15% как
    TEST -- их не видел ни train, ни calib ни в одном прогоне сегодня
    (production fit() всегда использовал последние 20% как calib, эта
    зона теперь разбивается на calib(первые 5 п.п.)/test(последние 15 п.п.),
    так что test -- совершенно новый кусок, не пересекающийся с тем, что
    оценивался в экспериментах 1-4 и во всех сегодняшних retrain).

    Обучает СВЕЖУЮ модель (не трогает artifacts/models/go_v1.joblib) на
    первых 85% (внутри которых FormulaPlusResidual.fit() сам ещё раз
    отрежет свои train/calib 80/20 -- как в продакшене), оценивает на
    последних 15%.
    """
    print("\n=== Эксперимент 5 (H-B3): честный train/calib/TEST, test не видел никто ===")
    n = len(table)
    cut_traincalib = int(n * 0.85)
    dev = table.iloc[:cut_traincalib]
    test = table.iloc[cut_traincalib:]
    print(f"  train+calib(внутр. 80/20)={len(dev)}  TEST(новый, не тронут)={len(test)}")

    model = FormulaPlusResidual(
        name="sulfur_mgkg_holdout_check",
        formula_fn=sulfur_arrhenius_baseline,
        formula_tags=["242000:T5"],
        feature_cols=SULFUR_FEATURES,
        fallback_mean=8.5,
    )
    model.fit(dev[SULFUR_FEATURES], dev["y"])

    test_pred = [model.predict_one(r.to_dict()).mean for _, r in test[SULFUR_FEATURES].iterrows()]
    test_rmse = float(np.sqrt(((np.array(test_pred) - test["y"]) ** 2).mean()))

    dev_calib_cut = int(len(dev) * 0.8)
    calib_part = dev.iloc[dev_calib_cut:]
    calib_pred = [model.predict_one(r.to_dict()).mean for _, r in calib_part[SULFUR_FEATURES].iterrows()]
    calib_rmse = float(np.sqrt(((np.array(calib_pred) - calib_part["y"]) ** 2).mean()))

    print(f"  RMSE на CALIB (внутренний, использовался весь день) = {calib_rmse:.3f}")
    print(f"  RMSE на TEST (новый, никто не видел)                = {test_rmse:.3f}")
    gap = test_rmse - calib_rmse
    print(f"  Разрыв test-calib = {gap:+.3f} ({'подозрительно' if gap > 0.5 else 'в пределах шума'})")


def experiment_6_outlier_weighting(table_full: pd.DataFrame):
    """
    H-B2. Сейчас 59 точек с y > 20 мг/кг просто выбрасываются из обучения.
    Гипотеза: их лучше оставить с пониженным весом -- это реальные
    аварийные режимы, модель могла бы о них что-то узнать, не заражаясь.

    Сравнение честное: оба варианта обучаются на одном и том же train-куске
    (первые 85% по времени) и оцениваются на одном и том же TEST (последние
    15%), причём TEST-метрика считается только по нормальным точкам
    (y <= 20) -- чтобы сравнивать способность предсказывать штатный режим,
    а не то, кто удачнее угадал единичную аварию.
    """
    print("\n=== Эксперимент 6 (H-B2): выбрасывать выбросы или взвешивать ===")
    import lightgbm as lgb

    n = len(table_full)
    cut = int(n * 0.85)
    dev, test = table_full.iloc[:cut], table_full.iloc[cut:]
    test_norm = test[test["y"] <= SULFUR_OUTLIER_THRESHOLD]
    n_out_dev = int((dev["y"] > SULFUR_OUTLIER_THRESHOLD).sum())
    print(f"  dev={len(dev)} (из них выбросов {n_out_dev})  TEST(норм. точки)={len(test_norm)}")

    safe = {c: c.replace(":", "__") for c in SULFUR_FEATURES}

    def fit_eval(dev_part, weights, label):
        base = dev_part[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        resid = dev_part["y"] - base
        model = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   num_leaves=15, min_child_samples=max(5, len(dev_part) // 50),
                                   random_state=42, verbose=-1)
        model.fit(dev_part[SULFUR_FEATURES].rename(columns=safe), resid, sample_weight=weights)

        base_t = test_norm[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        pred = base_t + model.predict(test_norm[SULFUR_FEATURES].rename(columns=safe))
        rmse = float(np.sqrt(((pred - test_norm["y"]) ** 2).mean()))
        print(f"  {label}: RMSE на TEST = {rmse:.3f}")
        return rmse

    dev_dropped = dev[dev["y"] <= SULFUR_OUTLIER_THRESHOLD]
    rmse_drop = fit_eval(dev_dropped, None, "ВЫБРОСИТЬ (как сейчас)   ")

    for w in (0.1, 0.3):
        weights = np.where(dev["y"] > SULFUR_OUTLIER_THRESHOLD, w, 1.0)
        fit_eval(dev, weights, f"ВЗВЕСИТЬ (вес выброса {w})")

    return rmse_drop


def experiment_7_effective_sample_size(table: pd.DataFrame, lims: pd.DataFrame):
    """
    H-H1. В таблице 10056 строк, но различных ЛИМС-проб всего 1444: окно
    near_mask (±1ч при шаге 10 мин) даёт РОВНО 7 телеметрических строк на
    одну лабораторную пробу, и все 7 получают одну метку. Значит:
      - эффективная выборка в 7 раз меньше, чем «видит» модель;
      - min_child_samples = n//50 = 201 -- это ~29 независимых наблюдений
        на лист, а не 201.

    Сравниваем три варианта на одном честном holdout, причём СПЛИТ ДЕЛАЕМ
    ПО ПРОБАМ, а не по строкам -- иначе одна проба может попасть и в train,
    и в test своими разными строками.
    """
    print("\n=== Эксперимент 7 (H-H1): честный эффективный размер выборки ===")
    import lightgbm as lgb

    sul = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == "sulfur_mgkg")][["ts", "value"]]
    sul = sul.dropna().sort_values("ts")
    sul = sul[sul["value"] <= SULFUR_OUTLIER_THRESHOLD]

    left = pd.DataFrame({"ts": table.index}).sort_values("ts")
    link = pd.merge_asof(left, sul.assign(lims_ts=sul["ts"]), on="ts",
                          direction="backward", tolerance=pd.Timedelta(hours=1)).dropna()
    tbl = table.loc[link["ts"].values].copy()
    tbl["_lims_ts"] = link["lims_ts"].values

    samples = np.sort(tbl["_lims_ts"].unique())
    cut_s = samples[int(len(samples) * 0.85)]
    dev = tbl[tbl["_lims_ts"] < cut_s]
    test = tbl[tbl["_lims_ts"] >= cut_s]
    print(f"  проб всего={len(samples)}  dev={dev['_lims_ts'].nunique()} проб / {len(dev)} строк"
          f"  TEST={test['_lims_ts'].nunique()} проб / {len(test)} строк")

    safe = {c: c.replace(":", "__") for c in SULFUR_FEATURES}

    def fit_eval(dev_part, mcs, label):
        base = dev_part[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        model = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   num_leaves=15, min_child_samples=mcs,
                                   random_state=42, verbose=-1)
        model.fit(dev_part[SULFUR_FEATURES].rename(columns=safe), dev_part["y"] - base)
        base_t = test[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        pred = base_t + model.predict(test[SULFUR_FEATURES].rename(columns=safe))
        rmse = float(np.sqrt(((pred - test["y"]) ** 2).mean()))
        print(f"  {label:<46} mcs={mcs:<5} RMSE на TEST = {rmse:.3f}")
        return rmse

    fit_eval(dev, max(5, len(dev) // 50), "A: как сейчас (все строки)")

    dedup = dev.sort_index().groupby("_lims_ts", as_index=False).first()
    dedup.index = range(len(dedup))
    fit_eval(dedup, max(5, len(dedup) // 50), "B: дедупликация (1 строка на пробу)")

    for mult in (3, 7):
        fit_eval(dev, max(5, (len(dev) // 50) * mult),
                 f"C: все строки, mcs x{mult} (поправка на дубли)")


def experiment_8_normalized_conformal(table: pd.DataFrame, lims: pd.DataFrame):
    """
    H-I1. Сейчас конформный квантиль ОДИН глобальный на все режимы, поэтому
    интервал одинаково широкий и в спокойный день, и в турбулентный. Gate
    проверяет верхнюю границу -- значит в спокойном режиме мы отдаём запас
    впустую.

    Нормализованный конформный: обучаем вторую модель на |остаток| ->
    sigma(x), считаем score = остаток / sigma(x), квантиль берём от score,
    а интервал восстанавливаем как pred +- q * sigma(x). Маргинальное
    покрытие сохраняется (стандартный результат), но ширина становится
    условной.

    Сплит по ПРОБАМ (как в эксперименте 7), TEST честный.
    """
    print("\n=== Эксперимент 8 (H-I1): условный (нормализованный) конформный интервал ===")
    import lightgbm as lgb
    from src.models.conformal import finite_sample_quantile

    sul = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == "sulfur_mgkg")][["ts", "value"]]
    sul = sul.dropna().sort_values("ts")
    sul = sul[sul["value"] <= SULFUR_OUTLIER_THRESHOLD]
    left = pd.DataFrame({"ts": table.index}).sort_values("ts")
    link = pd.merge_asof(left, sul.assign(lims_ts=sul["ts"]), on="ts",
                          direction="backward", tolerance=pd.Timedelta(hours=1)).dropna()
    tbl = table.loc[link["ts"].values].copy()
    tbl["_lims_ts"] = link["lims_ts"].values

    samples = np.sort(tbl["_lims_ts"].unique())
    s_train = samples[int(len(samples) * 0.70)]
    s_calib = samples[int(len(samples) * 0.85)]
    tr = tbl[tbl["_lims_ts"] < s_train]
    ca = tbl[(tbl["_lims_ts"] >= s_train) & (tbl["_lims_ts"] < s_calib)]
    te = tbl[tbl["_lims_ts"] >= s_calib]
    print(f"  train={len(tr)} строк / calib={len(ca)} / TEST={len(te)}")

    safe = {c: c.replace(":", "__") for c in SULFUR_FEATURES}

    def base_of(df):
        return df[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)

    mean_model = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    num_leaves=15, min_child_samples=max(5, len(tr) // 50),
                                    random_state=42, verbose=-1)
    mean_model.fit(tr[SULFUR_FEATURES].rename(columns=safe), tr["y"] - base_of(tr))

    def predict_mean(df):
        return base_of(df) + mean_model.predict(df[SULFUR_FEATURES].rename(columns=safe))

    # sigma-модель на |остаток| обучающей части
    resid_tr = (tr["y"] - predict_mean(tr)).abs()
    sigma_model = lgb.LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                     num_leaves=7, min_child_samples=max(5, len(tr) // 30),
                                     random_state=42, verbose=-1)
    sigma_model.fit(tr[SULFUR_FEATURES].rename(columns=safe), resid_tr)

    def sigma_of(df):
        s = sigma_model.predict(df[SULFUR_FEATURES].rename(columns=safe))
        return np.clip(s, 0.2, None)  # пол, чтобы не делить на ~0

    err_ca = (ca["y"] - predict_mean(ca)).values
    alpha = 0.10

    # (1) глобальный конформный -- как сейчас в проде
    q_hi_g = finite_sample_quantile(err_ca, alpha)
    q_lo_g = finite_sample_quantile(-err_ca, alpha)

    # (2) нормализованный
    s_ca = sigma_of(ca)
    q_hi_n = finite_sample_quantile(err_ca / s_ca, alpha)
    q_lo_n = finite_sample_quantile(-err_ca / s_ca, alpha)

    pred_te = predict_mean(te).values
    y_te = te["y"].values
    s_te = sigma_of(te)

    def report(label, lo, hi):
        cov_hi = float(np.mean(y_te <= hi))
        width = float(np.mean(hi - lo))
        # спокойные vs турбулентные -- по волатильности T5 за 6ч
        vol = te["242000:T5__std6h"].values
        calm = vol <= np.quantile(vol, 0.33)
        turb = vol >= np.quantile(vol, 0.67)
        print(f"  {label}")
        print(f"    покрытие по hi={cov_hi:.3f}  средняя ширина={width:.2f}")
        print(f"    ширина: спокойные={np.mean((hi-lo)[calm]):.2f}  турбулентные={np.mean((hi-lo)[turb]):.2f}")
        print(f"    средний ЗАПАС до 10 мг/кг по hi: спокойные={np.mean(10.0-hi[calm]):+.2f}  все={np.mean(10.0-hi):+.2f}")

    report("ГЛОБАЛЬНЫЙ (как в проде)", pred_te - q_lo_g, pred_te + q_hi_g)
    report("НОРМАЛИЗОВАННЫЙ (H-I1)", pred_te - q_lo_n * s_te, pred_te + q_hi_n * s_te)


def experiment_9_log_target(table: pd.DataFrame, lims: pd.DataFrame):
    """
    H-G2. Кинетика HDS интегрируется в степенной/логарифмический закон, а
    не линейный: dCs/dt = -k*Cs^n. Гипотеза: предсказывать ln(сера), а не
    серу. Два ожидаемых эффекта:
      1) остаток в лог-пространстве лучше себя ведёт (ошибка
         мультипликативная, а не аддитивная);
      2) конформный интервал после обратного exp становится
         ПРОПОРЦИОНАЛЬНЫМ -- узким при низкой сере. Gate проверяет
         верхнюю границу у лимита 10 мг/кг, поэтому важна ширина именно
         в зоне 8-10, а не средняя по всему диапазону.

    Обе метрики считаются в ИСХОДНОМ пространстве (после exp), иначе
    сравнение нечестное. Сплит по пробам, TEST честный.
    """
    print("\n=== Эксперимент 9 (H-G2): предсказывать ln(сера) вместо серы ===")
    import lightgbm as lgb
    from src.models.conformal import finite_sample_quantile

    sul = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == "sulfur_mgkg")][["ts", "value"]]
    sul = sul.dropna().sort_values("ts")
    sul = sul[sul["value"] <= SULFUR_OUTLIER_THRESHOLD]
    left = pd.DataFrame({"ts": table.index}).sort_values("ts")
    link = pd.merge_asof(left, sul.assign(lims_ts=sul["ts"]), on="ts",
                          direction="backward", tolerance=pd.Timedelta(hours=1)).dropna()
    tbl = table.loc[link["ts"].values].copy()
    tbl["_lims_ts"] = link["lims_ts"].values
    tbl = tbl[tbl["y"] > 0.05]  # ln нужен положительный аргумент

    samples = np.sort(tbl["_lims_ts"].unique())
    s_train, s_calib = samples[int(len(samples) * 0.70)], samples[int(len(samples) * 0.85)]
    tr = tbl[tbl["_lims_ts"] < s_train]
    ca = tbl[(tbl["_lims_ts"] >= s_train) & (tbl["_lims_ts"] < s_calib)]
    te = tbl[tbl["_lims_ts"] >= s_calib]
    print(f"  train={len(tr)} / calib={len(ca)} / TEST={len(te)}")

    safe = {c: c.replace(":", "__") for c in SULFUR_FEATURES}
    alpha = 0.10

    def base_of(df):
        return df[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)

    def make_model(n):
        return lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  num_leaves=15, min_child_samples=max(5, n // 50),
                                  random_state=42, verbose=-1)

    def evaluate(label, pred_te, lo_te, hi_te):
        y = te["y"].values
        rmse = float(np.sqrt(((pred_te - y) ** 2).mean()))
        cov = float(np.mean(y <= hi_te))
        width = hi_te - lo_te
        near = (pred_te >= 8.0) & (pred_te <= 10.0)   # зона, где Gate решает
        print(f"  {label}")
        print(f"    RMSE={rmse:.3f}  покрытие по hi={cov:.3f}  средняя ширина={width.mean():.2f}")
        if near.sum() >= 20:
            print(f"    В ЗОНЕ 8-10 мг/кг (n={int(near.sum())}): ширина={width[near].mean():.2f}"
                  f"  средний запас до 10 по hi={np.mean(10.0 - hi_te[near]):+.2f}")
        else:
            print(f"    В зоне 8-10 мг/кг точек мало (n={int(near.sum())})")

    # --- A: как сейчас, линейная цель ---
    m_lin = make_model(len(tr))
    m_lin.fit(tr[SULFUR_FEATURES].rename(columns=safe), tr["y"] - base_of(tr))
    pred_ca = (base_of(ca) + m_lin.predict(ca[SULFUR_FEATURES].rename(columns=safe))).values
    err_ca = ca["y"].values - pred_ca
    q_hi = finite_sample_quantile(err_ca, alpha)
    q_lo = finite_sample_quantile(-err_ca, alpha)
    pred_te = (base_of(te) + m_lin.predict(te[SULFUR_FEATURES].rename(columns=safe))).values
    evaluate("A: линейная цель (как сейчас)", pred_te, pred_te - q_lo, pred_te + q_hi)

    # --- B: логарифмическая цель ---
    m_log = make_model(len(tr))
    m_log.fit(tr[SULFUR_FEATURES].rename(columns=safe),
              np.log(tr["y"].values) - np.log(base_of(tr).values))
    lpred_ca = np.log(base_of(ca).values) + m_log.predict(ca[SULFUR_FEATURES].rename(columns=safe))
    lerr_ca = np.log(ca["y"].values) - lpred_ca
    lq_hi = finite_sample_quantile(lerr_ca, alpha)
    lq_lo = finite_sample_quantile(-lerr_ca, alpha)
    lpred_te = np.log(base_of(te).values) + m_log.predict(te[SULFUR_FEATURES].rename(columns=safe))
    evaluate("B: логарифмическая цель (H-G2)",
             np.exp(lpred_te), np.exp(lpred_te - lq_lo), np.exp(lpred_te + lq_hi))


def experiment_10_remaining_features(table: pd.DataFrame, tel_go: pd.DataFrame, lims: pd.DataFrame):
    """
    H-G1/G3/G4/G5 одним прогоном -- все четыре одной формы («добавить
    признак, измерить»), поэтому смысла в четырёх отдельных циклах
    переобучения нет. Каждый кандидат добавляется К ТЕКУЩЕМУ набору,
    оценка на честном holdout со сплитом по пробам.

      G1: arrhenius_t5 x h2_oil_ratio -- кинетика требует ПРОИЗВЕДЕНИЯ
          членов (exp(-Ea/RT) x P_H2^m), а мы дали их по отдельности
      G3: T11 - T6 -- экзотерма второго слоя, пропорциональна тому,
          сколько серы осталось после первого
      G4: F9 -- расход сырья отдельно (сейчас он только знаменатель
          в h2_oil_ratio; сам по себе это время пребывания в реакторе)
      G5: лаги 3/6ч для h2_oil_ratio и quench_delta (T5 и Q20 их имеют)
    """
    print("\n=== Эксперимент 10 (H-G1/G3/G4/G5): оставшиеся признаки ===")
    import lightgbm as lgb

    extra = tel_go[["242000:T11", "242000:T6", "242000:F9"]].copy()
    extra["242000:bed2_exotherm"] = extra["242000:T11"] - extra["242000:T6"]
    cand_cols = {
        "G1 arrhenius x h2_ratio": ["242000:arr_x_h2"],
        "G3 экзотерма 2-го слоя":  ["242000:bed2_exotherm"],
        "G4 расход сырья F9":      ["242000:F9"],
        "G5 лаги h2/quench":       ["242000:h2_oil_ratio__lag3h", "242000:h2_oil_ratio__lag6h",
                                     "242000:T5_T6_quench_delta__lag3h",
                                     "242000:T5_T6_quench_delta__lag6h"],
    }

    tbl = table.join(extra[["242000:bed2_exotherm", "242000:F9"]], how="left")
    tbl["242000:arr_x_h2"] = tbl["242000:arrhenius_t5"] * tbl["242000:h2_oil_ratio"]
    lagged = add_lags(tbl[["242000:h2_oil_ratio", "242000:T5_T6_quench_delta"]],
                      ["242000:h2_oil_ratio", "242000:T5_T6_quench_delta"], lags_h=(3, 6))
    tbl = tbl.join(lagged[[c for c in lagged.columns if "__lag" in c]], how="left")

    sul = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == "sulfur_mgkg")][["ts", "value"]]
    sul = sul.dropna().sort_values("ts")
    sul = sul[sul["value"] <= SULFUR_OUTLIER_THRESHOLD]
    left = pd.DataFrame({"ts": tbl.index}).sort_values("ts")
    link = pd.merge_asof(left, sul.assign(lims_ts=sul["ts"]), on="ts",
                          direction="backward", tolerance=pd.Timedelta(hours=1)).dropna()
    tbl = tbl.loc[link["ts"].values].copy()
    tbl["_lims_ts"] = link["lims_ts"].values

    samples = np.sort(tbl["_lims_ts"].unique())
    s_cut = samples[int(len(samples) * 0.85)]
    dev, te = tbl[tbl["_lims_ts"] < s_cut], tbl[tbl["_lims_ts"] >= s_cut]

    def run(cols, label):
        feats = list(SULFUR_FEATURES) + cols
        d = dev.dropna(subset=feats)
        t = te.dropna(subset=feats)
        if len(t) < 50:
            print(f"  {label:<26} мало точек после dropna ({len(t)}) -- пропуск")
            return
        safe = {c: c.replace(":", "__") for c in feats}
        base_d = d[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                               num_leaves=15, min_child_samples=max(5, len(d) // 50),
                               random_state=42, verbose=-1)
        m.fit(d[feats].rename(columns=safe), d["y"] - base_d)
        base_t = t[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        pred = base_t + m.predict(t[feats].rename(columns=safe))
        rmse = float(np.sqrt(((pred - t["y"]) ** 2).mean()))
        gains = m.booster_.feature_importance(importance_type="gain")
        share = {c: 100 * g / max(gains.sum(), 1) for c, g in zip(feats, gains)}
        add = "  ".join(f"{c.split(':')[-1]}={share[c]:.1f}%" for c in cols)
        print(f"  {label:<26} n_test={len(t):<5} RMSE={rmse:.3f}   {add}")

    run([], "БАЗА (как сейчас)")
    for label, cols in cand_cols.items():
        run(cols, label)


def experiment_11_q21_go_no_go(table: pd.DataFrame, tel_go: pd.DataFrame, lims: pd.DataFrame):
    """
    H-B1, дешёвый go/no-go ПЕРЕД тем как строить предобучение/multi-task.

    Идея H-B1: ЛИМС даёт всего ~1444 меток, а Q21 (ПАК серы продукта) --
    189k точек. Соблазнительно предобучиться на Q21. Риск: Q21 отличается
    от ЛИМС ровно на ту величину, которую мы считаем потолком (корреляция
    0.18-0.31), плюс у ПАК известны залипания до 47 суток.

    Проверка одним замером: обучить модель ТОЛЬКО на метках Q21 и оценить
    её на ЛИМС-holdout. Если RMSE сильно хуже текущих ~2.3 -- переносимого
    сигнала в Q21 нет, предобучение не поможет, дальше не тратимся.
    """
    print("\n=== Эксперимент 11 (H-B1 go/no-go): учить на Q21, проверять на ЛИМС ===")
    import lightgbm as lgb

    sul = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == "sulfur_mgkg")][["ts", "value"]]
    sul = sul.dropna().sort_values("ts")
    sul = sul[sul["value"] <= SULFUR_OUTLIER_THRESHOLD]
    left = pd.DataFrame({"ts": table.index}).sort_values("ts")
    link = pd.merge_asof(left, sul.assign(lims_ts=sul["ts"]), on="ts",
                          direction="backward", tolerance=pd.Timedelta(hours=1)).dropna()
    tbl = table.loc[link["ts"].values].copy()
    tbl["_lims_ts"] = link["lims_ts"].values

    samples = np.sort(tbl["_lims_ts"].unique())
    s_cut = samples[int(len(samples) * 0.85)]
    dev_lims = tbl[tbl["_lims_ts"] < s_cut]
    te = tbl[tbl["_lims_ts"] >= s_cut]           # ЛИМС-holdout, общий для обоих
    te_start = te.index.min()

    # Q21 как метка: берём те же признаки, но y = показание ПАК, и СТРОГО
    # раньше начала ЛИМС-holdout, иначе сравнение нечестное
    q21 = tel_go[["242000:Q21"]].dropna()
    q_tbl = table.join(q21, how="inner")
    q_tbl = q_tbl[(q_tbl.index < te_start) & (q_tbl["242000:Q21"] <= SULFUR_OUTLIER_THRESHOLD)]
    print(f"  меток ЛИМС в dev={dev_lims['_lims_ts'].nunique()} проб / {len(dev_lims)} строк")
    print(f"  меток Q21 в dev={len(q_tbl)} строк  (в {len(q_tbl)//max(len(dev_lims),1)}x больше)")
    print(f"  общий ЛИМС-holdout={len(te)} строк")

    safe = {c: c.replace(":", "__") for c in SULFUR_FEATURES}

    def fit_eval(X, y, label, n_for_mcs):
        base = X[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                               num_leaves=15, min_child_samples=max(5, n_for_mcs // 50),
                               random_state=42, verbose=-1)
        m.fit(X[SULFUR_FEATURES].rename(columns=safe), y - base)
        base_t = te[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        pred = base_t + m.predict(te[SULFUR_FEATURES].rename(columns=safe))
        rmse = float(np.sqrt(((pred - te["y"]) ** 2).mean()))
        print(f"  {label:<38} RMSE на ЛИМС-holdout = {rmse:.3f}")
        return rmse

    r_lims = fit_eval(dev_lims, dev_lims["y"], "обучена на ЛИМС (как сейчас)", len(dev_lims))
    r_q21 = fit_eval(q_tbl, q_tbl["242000:Q21"], "обучена ТОЛЬКО на Q21", len(q_tbl))

    print()
    if r_q21 <= r_lims * 1.15:
        print("  => GO: в Q21 есть переносимый сигнал, предобучение имеет смысл проверять")
    else:
        print(f"  => NO-GO: Q21-модель хуже на {100*(r_q21/r_lims-1):.0f}%, переносимого сигнала нет,")
        print("     предобучение на Q21 не окупится -- закрываем H-B1 без дорогой машинерии")


def experiment_12_q21_multisplit(table: pd.DataFrame, tel_go: pd.DataFrame, lims: pd.DataFrame):
    """
    Проверка находки эксперимента 11 на НЕСКОЛЬКИХ сплитах.

    В эксп.11 модель на метках ПАК (Q21) предсказала ЛИМС лучше, чем
    модель на метках ЛИМС (2.175 против 2.299). Разница 0.124 на ОДНОМ
    сплите -- после урока H-B3 такому не верим. Здесь walk-forward: пять
    точек разреза по времени, в каждой обе модели учатся на одном периоде
    и проверяются на одном тесте. Смотрим не величину, а УСТОЙЧИВОСТЬ
    знака разницы.
    """
    print("\n=== Эксперимент 12: метки ЛИМС против меток Q21 на 5 сплитах ===")
    import lightgbm as lgb

    sul = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == "sulfur_mgkg")][["ts", "value"]]
    sul = sul.dropna().sort_values("ts")
    sul = sul[sul["value"] <= SULFUR_OUTLIER_THRESHOLD]
    left = pd.DataFrame({"ts": table.index}).sort_values("ts")
    link = pd.merge_asof(left, sul.assign(lims_ts=sul["ts"]), on="ts",
                          direction="backward", tolerance=pd.Timedelta(hours=1)).dropna()
    tbl = table.loc[link["ts"].values].copy()
    tbl["_lims_ts"] = link["lims_ts"].values

    q21 = tel_go[["242000:Q21"]].dropna()
    tbl_q = tbl.join(q21, how="inner")
    tbl_q = tbl_q[tbl_q["242000:Q21"] <= SULFUR_OUTLIER_THRESHOLD]

    safe = {c: c.replace(":", "__") for c in SULFUR_FEATURES}
    samples = np.sort(tbl["_lims_ts"].unique())

    def fit_eval(X, y, test):
        base = X[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                               num_leaves=15, min_child_samples=max(5, len(X) // 50),
                               random_state=42, verbose=-1)
        m.fit(X[SULFUR_FEATURES].rename(columns=safe), y - base)
        base_t = test[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        pred = base_t + m.predict(test[SULFUR_FEATURES].rename(columns=safe))
        return float(np.sqrt(((pred - test["y"]) ** 2).mean()))

    print(f"  {'разрез':<10}{'n_test':>8}{'ЛИМС-метки':>13}{'Q21-метки':>12}{'разница':>10}")
    diffs = []
    for frac in (0.60, 0.68, 0.76, 0.84, 0.92):
        cut = samples[int(len(samples) * frac)]
        dev_l = tbl[tbl["_lims_ts"] < cut]
        te = tbl[tbl["_lims_ts"] >= cut]
        dev_q = tbl_q[tbl_q.index < te.index.min()]
        if len(te) < 100 or len(dev_q) < 200:
            continue
        r_l = fit_eval(dev_l, dev_l["y"], te)
        r_q = fit_eval(dev_q, dev_q["242000:Q21"], te)
        diffs.append(r_q - r_l)
        print(f"  {frac:<10.2f}{len(te):>8}{r_l:>13.3f}{r_q:>12.3f}{r_q - r_l:>+10.3f}")

    d = np.array(diffs)
    print()
    print(f"  разница Q21-ЛИМС по сплитам: среднее {d.mean():+.3f}, "
          f"медиана {np.median(d):+.3f}, отрицательна в {int((d < 0).sum())} из {len(d)}")
    if (d < 0).all():
        print("  => эффект УСТОЙЧИВ: метки Q21 лучше на всех сплитах")
    elif (d < 0).sum() >= len(d) - 1:
        print("  => эффект скорее есть, но не на всех сплитах -- нужна осторожность")
    else:
        print("  => эффект НЕ устойчив: находка эксп.11 была особенностью одного сплита")


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
    experiment_5_true_holdout_test(table)

    experiment_7_effective_sample_size(table, lims)
    experiment_8_normalized_conformal(table, lims)
    experiment_9_log_target(table, lims)
    experiment_10_remaining_features(table, tel_go, lims)
    experiment_11_q21_go_no_go(table, tel_go, lims)
    experiment_12_q21_multisplit(table, tel_go, lims)

    print("\nПересборка таблицы БЕЗ отсечения выбросов (для H-B2)...")
    table_full, _ = build_sulfur_table(avt, tel_avt_raw, tel_go, lims,
                                        feed_delay_h=0.0, drop_outliers=False)
    table_full = table_full.loc[table_full.index.sort_values()]
    experiment_6_outlier_weighting(table_full)


if __name__ == "__main__":
    main()
