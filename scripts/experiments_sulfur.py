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

    resid_tr = (tr["y"] - predict_mean(tr)).abs()
    sigma_model = lgb.LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                     num_leaves=7, min_child_samples=max(5, len(tr) // 30),
                                     random_state=42, verbose=-1)
    sigma_model.fit(tr[SULFUR_FEATURES].rename(columns=safe), resid_tr)

    def sigma_of(df):
        s = sigma_model.predict(df[SULFUR_FEATURES].rename(columns=safe))
        return np.clip(s, 0.2, None)

    err_ca = (ca["y"] - predict_mean(ca)).values
    alpha = 0.10

    q_hi_g = finite_sample_quantile(err_ca, alpha)
    q_lo_g = finite_sample_quantile(-err_ca, alpha)

    s_ca = sigma_of(ca)
    q_hi_n = finite_sample_quantile(err_ca / s_ca, alpha)
    q_lo_n = finite_sample_quantile(-err_ca / s_ca, alpha)

    pred_te = predict_mean(te).values
    y_te = te["y"].values
    s_te = sigma_of(te)

    def report(label, lo, hi):
        cov_hi = float(np.mean(y_te <= hi))
        width = float(np.mean(hi - lo))
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
    tbl = tbl[tbl["y"] > 0.05]

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
        near = (pred_te >= 8.0) & (pred_te <= 10.0)
        print(f"  {label}")
        print(f"    RMSE={rmse:.3f}  покрытие по hi={cov:.3f}  средняя ширина={width.mean():.2f}")
        if near.sum() >= 20:
            print(f"    В ЗОНЕ 8-10 мг/кг (n={int(near.sum())}): ширина={width[near].mean():.2f}"
                  f"  средний запас до 10 по hi={np.mean(10.0 - hi_te[near]):+.2f}")
        else:
            print(f"    В зоне 8-10 мг/кг точек мало (n={int(near.sum())})")

    m_lin = make_model(len(tr))
    m_lin.fit(tr[SULFUR_FEATURES].rename(columns=safe), tr["y"] - base_of(tr))
    pred_ca = (base_of(ca) + m_lin.predict(ca[SULFUR_FEATURES].rename(columns=safe))).values
    err_ca = ca["y"].values - pred_ca
    q_hi = finite_sample_quantile(err_ca, alpha)
    q_lo = finite_sample_quantile(-err_ca, alpha)
    pred_te = (base_of(te) + m_lin.predict(te[SULFUR_FEATURES].rename(columns=safe))).values
    evaluate("A: линейная цель (как сейчас)", pred_te, pred_te - q_lo, pred_te + q_hi)

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
    te = tbl[tbl["_lims_ts"] >= s_cut]
    te_start = te.index.min()

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


def experiment_13_three_cornered_hat(table: pd.DataFrame, tel_go: pd.DataFrame, lims: pd.DataFrame):
    """
    H-K1: потолок 2.253 (H-F1) посчитан как расхождение ЛИМС и ПАК. Но это
    расхождение ДВУХ шумных приборов:

        RMSE(ЛИМС, ПАК)^2 ~ sigma_ЛИМС^2 + sigma_ПАК^2

    Модель же сравнивается с ЛИМС, и её измеренная ошибка

        RMSE(ЛИМС, модель)^2 ~ sigma_модель^2 + sigma_ЛИМС^2

    то есть НЕДОСТИЖИМЫЙ минимум для нашей метрики -- это sigma_ЛИМС В ОДИНОЧКУ,
    а вовсе не 2.253. Если sigma_ЛИМС заметно меньше 2.253, зазор для улучшения
    больше, чем мы считали, и ветку RMSE рано закрывать.

    Разложение возможно, потому что у нас ТРИ оценки одной величины: ЛИМС (A),
    ПАК/Q21 (B) и модель (C). Метод трёх углов (three-cornered hat, оценка
    Граббса) даёт каждую дисперсию из трёх попарных расхождений:

        sigma_A^2 = (V_AB + V_AC - V_BC) / 2   и циклически

    Условия: ошибки трёх "приборов" независимы и аддитивны. Встроенная
    проверка -- все три sigma^2 должны выйти НЕОТРИЦАТЕЛЬНЫМИ; отрицательная
    означает нарушение независимости (это и есть замер H-K2).

    Два принципиальных требования к честности замера:
      1. Прогноз модели должен быть ВНЕ ВЫБОРКИ. Иначе sigma_модель занижена и
         всё разложение поедет. Здесь -- расширяющееся окно walk-forward по
         второй половине проб.
      2. Считаем по ПРОБАМ, а не по строкам телеметрии (урок H-H1: на одну
         пробу приходится ~7 строк с одной меткой).
    """
    print("\n=== Эксперимент 13 (H-K1): разложение потолка методом трёх углов ===")
    import lightgbm as lgb

    sul = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == "sulfur_mgkg")][["ts", "value"]]
    sul = sul.dropna().sort_values("ts")
    sul = sul[sul["value"] <= SULFUR_OUTLIER_THRESHOLD]
    left = pd.DataFrame({"ts": table.index}).sort_values("ts")
    link = pd.merge_asof(left, sul.assign(lims_ts=sul["ts"]), on="ts",
                          direction="backward", tolerance=pd.Timedelta(hours=1)).dropna()
    tbl = table.loc[link["ts"].values].copy()
    tbl["_lims_ts"] = link["lims_ts"].values

    safe = {c: c.replace(":", "__") for c in SULFUR_FEATURES}
    samples = np.sort(tbl["_lims_ts"].unique())
    bounds = np.linspace(int(len(samples) * 0.5), len(samples), 6).astype(int)

    blocks = []
    for i in range(5):
        lo_i, hi_i = bounds[i], bounds[i + 1]
        if hi_i <= lo_i:
            continue
        cut_lo, cut_hi = samples[lo_i], samples[hi_i - 1]
        tr = tbl[tbl["_lims_ts"] < cut_lo]
        te = tbl[(tbl["_lims_ts"] >= cut_lo) & (tbl["_lims_ts"] <= cut_hi)]
        if len(tr) < 200 or len(te) < 20:
            continue
        base = tr[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                               num_leaves=15, min_child_samples=max(5, len(tr) // 50),
                               random_state=42, verbose=-1)
        m.fit(tr[SULFUR_FEATURES].rename(columns=safe), tr["y"] - base)
        base_t = te[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        pred = base_t + m.predict(te[SULFUR_FEATURES].rename(columns=safe))
        blocks.append(pd.DataFrame({"_lims_ts": te["_lims_ts"].values,
                                     "A": te["y"].values, "C": pred.values}))

    oos = pd.concat(blocks)
    per = oos.groupby("_lims_ts").agg(A=("A", "first"), C=("C", "mean")).reset_index()
    per = per.rename(columns={"_lims_ts": "ts"}).sort_values("ts")

    q = tel_go[["242000:Q21"]].dropna().sort_index()
    stuck = q["242000:Q21"].rolling("6h").std().fillna(0.0)
    qdf = pd.DataFrame({"ts": q.index, "B": q["242000:Q21"].values,
                        "B_std6h": stuck.values}).sort_values("ts")
    per = pd.merge_asof(per, qdf, on="ts", direction="nearest",
                         tolerance=pd.Timedelta(minutes=15)).dropna()
    per = per[per["B"] <= SULFUR_OUTLIER_THRESHOLD]

    def grubbs(df, label):
        A, B, C = df["A"].values, df["B"].values, df["C"].values
        v_ab = float(np.var(A - B, ddof=1))
        v_ac = float(np.var(A - C, ddof=1))
        v_bc = float(np.var(B - C, ddof=1))
        s2 = {"ЛИМС": (v_ab + v_ac - v_bc) / 2,
              "ПАК Q21": (v_ab + v_bc - v_ac) / 2,
              "модель": (v_ac + v_bc - v_ab) / 2}
        print(f"\n  --- {label} (проб: {len(df)}) ---")
        print(f"  сдвиги (систематика): ЛИМС-ПАК={np.mean(A - B):+.2f}  "
              f"ЛИМС-модель={np.mean(A - C):+.2f}  мг/кг")
        print(f"  попарные RMSE:  ЛИМС/ПАК={np.sqrt(np.mean((A-B)**2)):.3f}  "
              f"ЛИМС/модель={np.sqrt(np.mean((A-C)**2)):.3f}  "
              f"ПАК/модель={np.sqrt(np.mean((B-C)**2)):.3f}")
        print("  разложение Граббса (собственный шум каждого источника):")
        ok = True
        for k, v in s2.items():
            if v < 0:
                ok = False
                print(f"    sigma_{k:<9} = ОТРИЦАТЕЛЬНА ({v:.2f}) -- независимость нарушена")
            else:
                print(f"    sigma_{k:<9} = {np.sqrt(v):.3f} мг/кг")
        if ok:
            print(f"  => НАСТОЯЩИЙ потолок для метрики [модель vs ЛИМС] = "
                  f"sigma_ЛИМС = {np.sqrt(s2['ЛИМС']):.3f} (а не 2.253)")
            print(f"     собственная ошибка модели sigma_модель = {np.sqrt(s2['модель']):.3f}")
        return s2, ok

    grubbs(per, "все пробы")
    calm = per[per["B_std6h"] > 1e-9]
    if len(calm) >= 100:
        grubbs(calm, "без залипаний ПАК (std Q21 за 6ч > 0)")
    else:
        print(f"\n  (проб без залипаний ПАК всего {len(calm)} -- отдельный срез не считаю)")


def experiment_14_skill_vs_constant(table: pd.DataFrame, go: GOModel, tel_go: pd.DataFrame,
                                     lims: pd.DataFrame):
    """
    H-K3 (порождена H-K1): бьёт ли модель ТРИВИАЛЬНУЮ базу?

    За всю кампанию модель серы сравнивалась с формулой Аррениуса и сама с
    собой, но НИ РАЗУ -- с нулевой гипотезой "всегда предсказываем константу".
    Пока потолок считался равным 2.253, а результат 2.28 -- вопрос казался
    закрытым. H-K1 сдвинул потолок до ~1.16, и вопрос открылся заново.

    Замеряем на ДВУХ протоколах, чтобы исключить артефакт разбиения:
      A. хвост по времени (ровно то, по чему репортился RMSE=2.155) --
         с ПРОДАКШЕН-артефактом go_v1.joblib, без переобучения;
      B. walk-forward из 5 блоков по второй половине проб -- переобучение
         в каждом блоке, сравнение с константой, формулой и ПАК Q21.

    Константа берётся как СРЕДНЕЕ обучающей выборки (оно минимизирует RMSE;
    медиана была бы поддавком модели) и, разумеется, только по трейну.
    """
    print("\n=== Эксперимент 14 (H-K3): модель против константы ===")
    import lightgbm as lgb

    print("\n  -- протокол A: хвост по времени, ПРОДАКШЕН-артефакт --")
    for frac in (0.80, 0.85):
        cut = int(len(table) * frac)
        tr, te = table.iloc[:cut], table.iloc[cut:]
        y = te["y"].values
        pred = np.array([go._models["sulfur_mgkg"].predict_one(r.to_dict()).mean
                          for _, r in te[SULFUR_FEATURES].iterrows()])
        rmse = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
        c = float(tr["y"].mean())
        r_c, r_m = rmse(np.full(len(y), c)), rmse(pred)
        print(f"    разрез {frac:.2f} (n_test={len(te)}): константа={c:.2f} -> RMSE {r_c:.3f}   "
              f"модель -> RMSE {r_m:.3f}   skill {100*(1-r_m/r_c):+.1f}%")

    print("\n  -- протокол B: walk-forward, переобучение в каждом блоке --")
    sul = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == "sulfur_mgkg")][["ts", "value"]]
    sul = sul.dropna().sort_values("ts")
    sul = sul[sul["value"] <= SULFUR_OUTLIER_THRESHOLD]
    left = pd.DataFrame({"ts": table.index}).sort_values("ts")
    link = pd.merge_asof(left, sul.assign(lims_ts=sul["ts"]), on="ts",
                          direction="backward", tolerance=pd.Timedelta(hours=1)).dropna()
    tbl = table.loc[link["ts"].values].copy()
    tbl["_lims_ts"] = link["lims_ts"].values

    q = tel_go[["242000:Q21"]].dropna().sort_index()
    qdf = pd.DataFrame({"ts": q.index, "B": q["242000:Q21"].values}).sort_values("ts")
    safe = {c: c.replace(":", "__") for c in SULFUR_FEATURES}
    samples = np.sort(tbl["_lims_ts"].unique())
    bounds = np.linspace(int(len(samples) * 0.5), len(samples), 6).astype(int)

    print(f"    {'блок':<6}{'n':>6}{'константа':>11}{'формула':>10}{'модель':>9}{'ПАК Q21':>10}")
    acc = {"const": [], "base": [], "model": [], "pak": []}
    for i in range(5):
        lo_i, hi_i = bounds[i], bounds[i + 1]
        cut_lo, cut_hi = samples[lo_i], samples[hi_i - 1]
        tr = tbl[tbl["_lims_ts"] < cut_lo]
        te = tbl[(tbl["_lims_ts"] >= cut_lo) & (tbl["_lims_ts"] <= cut_hi)]
        if len(tr) < 200 or len(te) < 20:
            continue
        base = tr[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                               num_leaves=15, min_child_samples=max(5, len(tr) // 50),
                               random_state=42, verbose=-1)
        m.fit(tr[SULFUR_FEATURES].rename(columns=safe), tr["y"] - base)
        base_t = te[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        pred = base_t + m.predict(te[SULFUR_FEATURES].rename(columns=safe))
        y = te["y"]
        rmse = lambda p: float(np.sqrt(((p - y) ** 2).mean()))
        j = pd.merge_asof(pd.DataFrame({"ts": te.index, "y": y.values}).sort_values("ts"),
                           qdf, on="ts", direction="nearest",
                           tolerance=pd.Timedelta(minutes=15)).dropna()
        j = j[j["B"] <= SULFUR_OUTLIER_THRESHOLD]
        vals = (rmse(np.full(len(y), tr["y"].mean())), rmse(base_t), rmse(pred),
                float(np.sqrt(((j["B"] - j["y"]) ** 2).mean())))
        for k, v in zip(acc, vals):
            acc[k].append(v)
        print(f"    {i+1:<6}{len(te):>6}{vals[0]:>11.3f}{vals[1]:>10.3f}{vals[2]:>9.3f}{vals[3]:>10.3f}")

    mc, mm = float(np.mean(acc["const"])), float(np.mean(acc["model"]))
    print(f"    {'сред.':<6}{'':>6}{mc:>11.3f}{np.mean(acc['base']):>10.3f}"
          f"{mm:>9.3f}{np.mean(acc['pak']):>10.3f}")
    worse = sum(1 for a, b in zip(acc["const"], acc["model"]) if b > a)
    print()
    print(f"  skill score модели против константы: {100*(1-mm/mc):+.1f}%  "
          f"(модель хуже константы в {worse} блоках из {len(acc['model'])})")
    if mm >= mc:
        print("  => У МОДЕЛИ НЕТ НАВЫКА как у точечного прогноза: константа не хуже.")
        print("     ML-слой поверх формулы реально помогает (см. эксп.1), но вся связка")
        print("     формула+остаток не дотягивает до тривиальной базы. При этом сигнал")
        print("     в данных ЕСТЬ: ПАК Q21 предсказывает ЛИМС заметно лучше константы.")


def experiment_15_baseline_choice(table: pd.DataFrame, lims: pd.DataFrame):
    """
    H-M1: базовая формула тянет модель вниз?

    Повод: в эксп.14 аррениусовский baseline сам по себе дал RMSE 2.604 --
    заметно ХУЖЕ константы (1.935). Остаток учится поверх систематически
    плохой опоры.

    Возражение, которое надо снять: остаток ведь может сам скомпенсировать
    формулу, ведь T5 -- признак модели, а baseline -- функция от T5.
    Но не может, если мешает monotone-ограничение: baseline убывает по T5,
    и EXPECTED_SIGNS["sulfur_mgkg"]["242000:T5"] = -1 заставляет остаток
    тоже НЕ возрастать по T5. Если формула переотвечает на температуру,
    ограниченный остаток НЕ ИМЕЕТ ПРАВА отыграть это назад. Поэтому схема
    2x2: {baseline: формула / константа} x {monotone: вкл / выкл}.

    Плюс отдельный вариант: линейная перекалибровка формулы на трейне
    (a + b*формула). Если он чинит дело -- у формулы верная ФОРМА, но
    неверные масштаб и сдвиг, и это чинится двумя коэффициентами.

    Протокол -- walk-forward из эксп.14 (5 блоков, переобучение в каждом),
    и в каждом блоке рядом печатается КОНСТАНТА: после H-K3 всё меряется
    против неё, а не против прошлой версии модели.
    """
    print("\n=== Эксперимент 15 (H-M1): выбор baseline и цена monotone ===")
    import lightgbm as lgb
    from src.models.base import monotone_vector

    sul = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == "sulfur_mgkg")][["ts", "value"]]
    sul = sul.dropna().sort_values("ts")
    sul = sul[sul["value"] <= SULFUR_OUTLIER_THRESHOLD]
    left = pd.DataFrame({"ts": table.index}).sort_values("ts")
    link = pd.merge_asof(left, sul.assign(lims_ts=sul["ts"]), on="ts",
                          direction="backward", tolerance=pd.Timedelta(hours=1)).dropna()
    tbl = table.loc[link["ts"].values].copy()
    tbl["_lims_ts"] = link["lims_ts"].values

    safe = {c: c.replace(":", "__") for c in SULFUR_FEATURES}
    mono = monotone_vector(SULFUR_FEATURES, "sulfur_mgkg")
    samples = np.sort(tbl["_lims_ts"].unique())
    bounds = np.linspace(int(len(samples) * 0.5), len(samples), 6).astype(int)

    def formula(df):
        return df[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)

    variants = ["формула+mono", "формула", "константа+mono", "константа", "перекалибр.+mono"]
    acc = {v: [] for v in variants}
    acc["КОНСТАНТА"] = []

    for i in range(5):
        lo_i, hi_i = bounds[i], bounds[i + 1]
        cut_lo, cut_hi = samples[lo_i], samples[hi_i - 1]
        tr = tbl[tbl["_lims_ts"] < cut_lo]
        te = tbl[(tbl["_lims_ts"] >= cut_lo) & (tbl["_lims_ts"] <= cut_hi)]
        if len(tr) < 200 or len(te) < 20:
            continue
        y_tr, y_te = tr["y"], te["y"]
        Xtr = tr[SULFUR_FEATURES].rename(columns=safe)
        Xte = te[SULFUR_FEATURES].rename(columns=safe)
        f_tr, f_te = formula(tr), formula(te)
        const = float(y_tr.mean())
        b, a = np.polyfit(f_tr.values, y_tr.values, 1)
        rc_tr, rc_te = a + b * f_tr, a + b * f_te

        bases = {"формула+mono": (f_tr, f_te, mono), "формула": (f_tr, f_te, None),
                 "константа+mono": (const, const, mono), "константа": (const, const, None),
                 "перекалибр.+mono": (rc_tr, rc_te, mono)}
        for name, (btr, bte, mc) in bases.items():
            kw = {"monotone_constraints": mc} if mc is not None else {}
            m = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   num_leaves=15, min_child_samples=max(5, len(tr) // 50),
                                   random_state=42, verbose=-1, **kw)
            m.fit(Xtr, y_tr - btr)
            pred = bte + m.predict(Xte)
            acc[name].append(float(np.sqrt(((pred - y_te) ** 2).mean())))
        acc["КОНСТАНТА"].append(float(np.sqrt(((const - y_te) ** 2).mean())))

    n_blocks = len(acc["КОНСТАНТА"])
    print(f"  {'вариант':<20}" + "".join(f"{'бл.'+str(i+1):>9}" for i in range(n_blocks))
          + f"{'среднее':>10}{'skill':>8}{'бьёт конст.':>13}")
    base_const = np.array(acc["КОНСТАНТА"])
    for name in ["КОНСТАНТА"] + variants:
        v = np.array(acc[name])
        mean = v.mean()
        skill = 100 * (1 - mean / base_const.mean())
        wins = int((v < base_const).sum())
        tag = f"{wins}/{n_blocks}" if name != "КОНСТАНТА" else "--"
        print(f"  {name:<20}" + "".join(f"{x:>9.3f}" for x in v)
              + f"{mean:>10.3f}{skill:>+7.1f}%{tag:>13}")

    print()
    best = min(variants, key=lambda k: np.mean(acc[k]))
    print(f"  лучший вариант: {best} ({np.mean(acc[best]):.3f}), "
          f"текущий прод: формула+mono ({np.mean(acc['формула+mono']):.3f})")
    d_mono = np.mean(acc["формула"]) - np.mean(acc["формула+mono"])
    d_base = np.mean(acc["константа+mono"]) - np.mean(acc["формула+mono"])
    print(f"  цена monotone при формуле: {d_mono:+.3f} "
          f"(отрицательно = без ограничений лучше)")
    print(f"  эффект замены формулы на константу (при mono): {d_base:+.3f} "
          f"(отрицательно = константа лучше)")


def experiment_16_capacity_and_dedup(table: pd.DataFrame, lims: pd.DataFrame):
    """
    H-M2: модель переглажена -- или сигнала в признаках просто нет?

    Эксп.15 показал, что ML-слой поверх КОНСТАНТНОГО baseline даёт 1.962
    против 1.935 у самой константы, то есть слой не просто бесполезен, а
    слегка вреден. Два принципиально разных объяснения:

      (а) НЕДООБУЧЕНИЕ. min_child_samples = n//50 = 201, но эффективная
          выборка (H-H1) -- 1444 пробы, не 10 056 строк, значит на лист
          приходится ~29 независимых наблюдений. Дерево вырождается почти
          в константу. Лечится ёмкостью.
      (б) СИГНАЛА НЕТ. Признаки-условия процесса не предсказывают лабораторную
          серу вне периода обучения. Ёмкость не поможет, лечится только
          другими входами.

    Различаются они ошибкой НА ТРЕЙНЕ: при (а) train RMSE тоже высок, при
    (б) train RMSE низок, а test -- нет. Поэтому печатаем обе.

    Дополнительно -- дедупликация до одной строки на пробу (веса вместо
    семикратного дублирования метки), чего H-H1 касалась лишь на одном сплите.
    """
    print("\n=== Эксперимент 16 (H-M2): ёмкость модели и дедупликация ===")
    import lightgbm as lgb
    from src.models.base import monotone_vector

    sul = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == "sulfur_mgkg")][["ts", "value"]]
    sul = sul.dropna().sort_values("ts")
    sul = sul[sul["value"] <= SULFUR_OUTLIER_THRESHOLD]
    left = pd.DataFrame({"ts": table.index}).sort_values("ts")
    link = pd.merge_asof(left, sul.assign(lims_ts=sul["ts"]), on="ts",
                          direction="backward", tolerance=pd.Timedelta(hours=1)).dropna()
    tbl = table.loc[link["ts"].values].copy()
    tbl["_lims_ts"] = link["lims_ts"].values

    safe = {c: c.replace(":", "__") for c in SULFUR_FEATURES}
    mono = monotone_vector(SULFUR_FEATURES, "sulfur_mgkg")
    samples = np.sort(tbl["_lims_ts"].unique())
    bounds = np.linspace(int(len(samples) * 0.5), len(samples), 6).astype(int)

    cfgs = [("прод: mcs=n/50, d4", 50, 4, 15, False),
            ("mcs=n/150, d4",     150, 4, 15, False),
            ("mcs=n/300, d6",     300, 6, 31, False),
            ("mcs=20, d8 (много)", None, 8, 63, False),
            ("дедуп + mcs=n/50",   50, 4, 15, True),
            ("дедуп + mcs=20, d8", None, 8, 63, True)]

    res = {c[0]: {"tr": [], "te": []} for c in cfgs}
    const_te = []

    for i in range(5):
        lo_i, hi_i = bounds[i], bounds[i + 1]
        cut_lo, cut_hi = samples[lo_i], samples[hi_i - 1]
        tr_all = tbl[tbl["_lims_ts"] < cut_lo]
        te = tbl[(tbl["_lims_ts"] >= cut_lo) & (tbl["_lims_ts"] <= cut_hi)]
        if len(tr_all) < 200 or len(te) < 20:
            continue
        const = float(tr_all["y"].mean())
        const_te.append(float(np.sqrt(((const - te["y"]) ** 2).mean())))
        Xte = te[SULFUR_FEATURES].rename(columns=safe)

        for name, div, depth, leaves, dedup in cfgs:
            tr = (tr_all.groupby("_lims_ts").first().reset_index() if dedup else tr_all)
            mcs = 20 if div is None else max(5, len(tr) // div)
            Xtr = tr[SULFUR_FEATURES].rename(columns=safe)
            m = lgb.LGBMRegressor(n_estimators=200, max_depth=depth, learning_rate=0.05,
                                   num_leaves=leaves, min_child_samples=mcs,
                                   monotone_constraints=mono, random_state=42, verbose=-1)
            m.fit(Xtr, tr["y"] - const)
            p_tr = const + m.predict(Xtr)
            p_te = const + m.predict(Xte)
            res[name]["tr"].append(float(np.sqrt(((p_tr - tr["y"]) ** 2).mean())))
            res[name]["te"].append(float(np.sqrt(((p_te - te["y"]) ** 2).mean())))

    c_mean = float(np.mean(const_te))
    print(f"  константа (эталон): test RMSE = {c_mean:.3f}\n")
    print(f"  {'конфигурация':<22}{'train':>8}{'test':>8}{'зазор':>8}{'skill':>8}{'бьёт конст.':>13}")
    for name, *_ in cfgs:
        tr_m, te_m = float(np.mean(res[name]["tr"])), float(np.mean(res[name]["te"]))
        wins = int((np.array(res[name]["te"]) < np.array(const_te)).sum())
        print(f"  {name:<22}{tr_m:>8.3f}{te_m:>8.3f}{te_m-tr_m:>8.3f}"
              f"{100*(1-te_m/c_mean):>+7.1f}%{str(wins)+'/'+str(len(const_te)):>13}")

    print()
    prod = float(np.mean(res["прод: mcs=n/50, d4"]["tr"]))
    big = float(np.mean(res["mcs=20, d8 (много)"]["tr"]))
    print(f"  train RMSE: прод {prod:.3f} -> самая ёмкая {big:.3f}")
    if big < prod - 0.2:
        print("  => ёмкости ХВАТАЕТ: ёмкая модель уверенно запоминает трейн,")
        print("     но это не переносится на тест. Значит дело НЕ в переглаженности,")
        print("     а в том, что признаки-условия процесса не переносятся во времени.")
    else:
        print("  => модель НЕ МОЖЕТ подогнать даже трейн -- недообучение/нет сигнала.")


def experiment_17_gate_decision_matrix(table_full: pd.DataFrame, lims: pd.DataFrame):
    """
    H-J1: метрика ПРОДУКТА, а не прогноза.

    За всю кампанию всё меряли в RMSE, но Gate принимает БИНАРНОЕ решение:
    по config/constraints.yaml для sulfur_mgkg limit=10.0, direction=max,
    check_on=hi -- то есть партия отклоняется, если ВЕРХНЯЯ граница
    конформного интервала превышает 10 мг/кг. Среднее Gate вообще не смотрит.

    Отсюда две принципиально разные ошибки:
      ЛОЖНЫЙ ПРОПУСК  -- Gate сказал ОК, а ЛИМС показал > 10. Внеспековое
                         топливо ушло потребителю. Катастрофа, цена высокая.
      ЛОЖНЫЙ ОТКАЗ    -- Gate забраковал нормальную партию. Упущенная
                         выручка / лишняя жёсткость режима. Цена умеренная.

    Это единственная метрика, по которой отсутствие навыка в среднем
    (H-K3) может оказаться неважным: консервативный hi держится конформным
    интервалом, а не точностью среднего.

    Для сравнения считается тот же Gate поверх КОНСТАНТНОГО прогноза
    (среднее трейна + конформный интервал по той же калибровке). Если
    матрицы совпадут -- модель не добавляет ценности и на уровне решений.

    ВАЖНО: обучение фильтрует выбросы > 20 мг/кг (как в проде), но
    ОЦЕНКА идёт по НЕфильтрованным меткам: точки y > 20 -- это и есть
    настоящие внеспековые события, ради которых Gate существует.
    """
    print("\n=== Эксперимент 17 (H-J1): матрица решений Gate ===")
    LIMIT = 10.0

    sul = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == "sulfur_mgkg")][["ts", "value"]]
    sul = sul.dropna().sort_values("ts")
    left = pd.DataFrame({"ts": table_full.index}).sort_values("ts")
    link = pd.merge_asof(left, sul.assign(lims_ts=sul["ts"]), on="ts",
                          direction="backward", tolerance=pd.Timedelta(hours=1)).dropna()
    tbl = table_full.loc[link["ts"].values].copy()
    tbl["_lims_ts"] = link["lims_ts"].values

    samples = np.sort(tbl["_lims_ts"].unique())
    bounds = np.linspace(int(len(samples) * 0.5), len(samples), 6).astype(int)
    spec = GO_SPECS["sulfur_mgkg"]
    from src.models.base import monotone_vector

    cells = {k: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for k in ("модель", "константа")}
    n_off = n_tot = 0

    for i in range(5):
        lo_i, hi_i = bounds[i], bounds[i + 1]
        cut_lo, cut_hi = samples[lo_i], samples[hi_i - 1]
        tr = tbl[tbl["_lims_ts"] < cut_lo]
        te = tbl[(tbl["_lims_ts"] >= cut_lo) & (tbl["_lims_ts"] <= cut_hi)]
        if len(tr) < 200 or len(te) < 20:
            continue
        tr_fit = tr[tr["y"] <= SULFUR_OUTLIER_THRESHOLD]

        sub = FormulaPlusResidual(name="sulfur_mgkg", formula_fn=spec[0],
                                   formula_tags=spec[1], feature_cols=spec[1],
                                   fallback_mean=spec[2],
                                   monotone=monotone_vector(spec[1], "sulfur_mgkg"))
        sub.fit(tr_fit[SULFUR_FEATURES], tr_fit["y"])

        n_tr = len(tr_fit)
        c_cut = int(n_tr * 0.8)
        const = float(tr_fit["y"].iloc[:c_cut].mean())
        calib_err = tr_fit["y"].iloc[c_cut:].values - const
        cb = ConformalResidualBounds().fit(calib_err - np.median(calib_err))
        off_lo, off_hi = cb.bounds()
        const_hi = const + float(np.median(calib_err)) + off_hi

        for _, r in te.iterrows():
            y = float(r["y"])
            off = y > LIMIT
            n_off += off
            n_tot += 1
            hi_model = sub.predict_one(r[SULFUR_FEATURES].to_dict()).hi
            for key, hi in (("модель", hi_model), ("константа", const_hi)):
                reject = hi > LIMIT
                if reject and off:      cells[key]["TP"] += 1
                elif reject and not off: cells[key]["FP"] += 1
                elif not reject and off: cells[key]["FN"] += 1
                else:                    cells[key]["TN"] += 1

    print(f"  всего проверок={n_tot}, из них реально внеспековых (ЛИМС>10)={n_off} "
          f"({100*n_off/max(n_tot,1):.1f}%)\n")
    for key, c in cells.items():
        tot, off = c["TP"] + c["FP"] + c["FN"] + c["TN"], c["TP"] + c["FN"]
        ok = c["FP"] + c["TN"]
        print(f"  --- Gate поверх: {key} ---")
        print(f"    ЛОЖНЫХ ПРОПУСКОВ (брак ушёл):   {c['FN']:>5}  "
              f"= {100*c['FN']/max(off,1):.1f}% от всех внеспековых")
        print(f"    пойманных внеспековых:          {c['TP']:>5}  "
              f"= {100*c['TP']/max(off,1):.1f}%")
        print(f"    ложных отказов (зря забракован):{c['FP']:>5}  "
              f"= {100*c['FP']/max(ok,1):.1f}% от нормальных партий")
        print(f"    доля отказов всего:             {100*(c['TP']+c['FP'])/max(tot,1):.1f}%")
    d_fn = cells["модель"]["FN"] - cells["константа"]["FN"]
    d_fp = cells["модель"]["FP"] - cells["константа"]["FP"]
    print(f"\n  модель против константы: ложных пропусков {d_fn:+d}, ложных отказов {d_fp:+d}")

    c = cells["модель"]
    tpr = c["TP"] / max(c["TP"] + c["FN"], 1)
    fpr = c["FP"] / max(c["FP"] + c["TN"], 1)
    print(f"  различающая способность Gate: TPR={100*tpr:.1f}% против FPR={100*fpr:.1f}%, "
          f"Youden J={100*(tpr-fpr):+.1f} п.п.")
    print("  (у бесполезного классификатора TPR=FPR, то есть J=0)")


def experiment_18_discrimination_auc(table_full: pd.DataFrame, tel_go: pd.DataFrame,
                                      lims: pd.DataFrame):
    """
    Достройка H-J1. Матрица решений смешивает два разных вопроса:
      (1) удачно ли выбран ПОРОГ (ширина конформного интервала),
      (2) есть ли у прогноза РАЗЛИЧАЮЩАЯ СПОСОБНОСТЬ в принципе.

    AUC отвечает только на (2): это вероятность, что случайно взятая
    внеспековая партия получит прогноз выше, чем случайно взятая нормальная.
    AUC=0.5 -- монета, различения нет ни при каком пороге; AUC=1.0 -- идеал.

    Если у модели AUC близок к 0.5, а у ПАК Q21 заметно выше -- история та
    же, что с RMSE (H-K3), и порог тут ни при чём.

    Отдельно считается ПОТОЛОК различения: метка "ЛИМС > 10" сама шумная,
    sigma_ЛИМС ~ 1.16 (H-K1), поэтому даже идеальный прогноз истинной серы
    не даст AUC=1. Оцениваем симуляцией: берём ЛИМС как истину, добавляем
    независимый шум sigma_ЛИМС и смотрим AUC такого "идеального прибора".
    """
    print("\n=== Эксперимент 18 (H-J1, достройка): различающая способность ===")
    import lightgbm as lgb
    from src.models.base import monotone_vector
    LIMIT = 10.0

    def auc(score, label):
        score, label = np.asarray(score, float), np.asarray(label, bool)
        n_pos, n_neg = label.sum(), (~label).sum()
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        r = pd.Series(score).rank().values
        return float((r[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    sul = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == "sulfur_mgkg")][["ts", "value"]]
    sul = sul.dropna().sort_values("ts")
    left = pd.DataFrame({"ts": table_full.index}).sort_values("ts")
    link = pd.merge_asof(left, sul.assign(lims_ts=sul["ts"]), on="ts",
                          direction="backward", tolerance=pd.Timedelta(hours=1)).dropna()
    tbl = table_full.loc[link["ts"].values].copy()
    tbl["_lims_ts"] = link["lims_ts"].values

    safe = {c: c.replace(":", "__") for c in SULFUR_FEATURES}
    mono = monotone_vector(SULFUR_FEATURES, "sulfur_mgkg")
    samples = np.sort(tbl["_lims_ts"].unique())
    bounds = np.linspace(int(len(samples) * 0.5), len(samples), 6).astype(int)
    q = tel_go[["242000:Q21"]].dropna().sort_index()
    qdf = pd.DataFrame({"ts": q.index, "B": q["242000:Q21"].values}).sort_values("ts")

    parts = []
    for i in range(5):
        lo_i, hi_i = bounds[i], bounds[i + 1]
        cut_lo, cut_hi = samples[lo_i], samples[hi_i - 1]
        tr = tbl[tbl["_lims_ts"] < cut_lo]
        te = tbl[(tbl["_lims_ts"] >= cut_lo) & (tbl["_lims_ts"] <= cut_hi)]
        if len(tr) < 200 or len(te) < 20:
            continue
        tr_fit = tr[tr["y"] <= SULFUR_OUTLIER_THRESHOLD]
        base = tr_fit[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                               num_leaves=15, min_child_samples=max(5, len(tr_fit) // 50),
                               monotone_constraints=mono, random_state=42, verbose=-1)
        m.fit(tr_fit[SULFUR_FEATURES].rename(columns=safe), tr_fit["y"] - base)
        base_t = te[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        parts.append(pd.DataFrame({
            "ts": te.index, "y": te["y"].values,
            "model": (base_t + m.predict(te[SULFUR_FEATURES].rename(columns=safe))).values,
            "formula": base_t.values,
            "q20": te["242000:Q20"].values}))

    d = pd.concat(parts).sort_values("ts")
    d = pd.merge_asof(d, qdf, on="ts", direction="nearest",
                       tolerance=pd.Timedelta(minutes=15))
    lab = d["y"].values > LIMIT
    print(f"  n={len(d)}, внеспековых={int(lab.sum())} ({100*lab.mean():.1f}%)\n")
    print(f"  {'источник прогноза':<28}{'AUC':>8}")
    for name, col in (("наша модель", "model"), ("формула Аррениуса", "formula"),
                       ("Q20 (сера сырья)", "q20"), ("ПАК Q21 (сера продукта)", "B")):
        sub = d.dropna(subset=[col])
        print(f"  {name:<28}{auc(sub[col].values, sub['y'].values > LIMIT):>8.3f}")

    rng = np.random.default_rng(0)
    y = d["y"].values
    ideal = [auc(y + rng.normal(0, 1.16, len(y)), lab) for _ in range(200)]
    print(f"  {'идеальный прибор (потолок)':<28}{np.mean(ideal):>8.3f}")

    mild = d[d["y"] <= SULFUR_OUTLIER_THRESHOLD]
    n_hard = int((d["y"] > SULFUR_OUTLIER_THRESHOLD).sum())
    print(f"\n  контроль: из {int(lab.sum())} внеспековых {n_hard} -- это y > 20 "
          f"(модель их при обучении не видела)")
    print(f"  AUC только на «мягких» случаях 10 < y <= 20 (n={len(mild)}):")
    for name, col in (("наша модель", "model"), ("Q20 (сера сырья)", "q20"),
                       ("ПАК Q21 (сера продукта)", "B")):
        sub = mild.dropna(subset=[col])
        print(f"    {name:<26}{auc(sub[col].values, sub['y'].values > LIMIT):>8.3f}")
    print("\n  потолок < 1.0 потому, что сама метка 'ЛИМС > 10' шумная:")
    print("  sigma_ЛИМС ~ 1.16 (H-K1), а медиана серы 8.6 при лимите 10 --")
    print("  часть 'внеспековых' событий это шум лаборатории, а не режим.")


def _sulfur_samples(table: pd.DataFrame, lims: pd.DataFrame, causal: bool = False):
    """
    Привязывает строки таблицы к пробам ЛИМС.

    causal=True оставляет только телеметрию, снятую ДО отбора пробы.
    Замечание Person 2 (2026-09-22): near_mask в build_sulfur_table берёт
    окно +-1 час ВОКРУГ пробы, то есть в обучение попадают показания уже
    ПОСЛЕ момента отбора. Для задачи "оценить серу сейчас" это заглядывание
    в будущее, и особенно грубое для Q21 -- прибор к этому моменту уже
    частично измерил то, что лаборатория только повезла анализировать.
    """
    sul = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == "sulfur_mgkg")][["ts", "value"]]
    sul = sul.dropna().sort_values("ts")
    left = pd.DataFrame({"ts": table.index}).sort_values("ts")
    link = pd.merge_asof(left, sul.assign(lims_ts=sul["ts"]), on="ts",
                          direction="backward", tolerance=pd.Timedelta(hours=1)).dropna()
    tbl = table.loc[link["ts"].values].copy()
    tbl["_lims_ts"] = link["lims_ts"].values
    if causal:
        tbl = tbl[tbl.index <= tbl["_lims_ts"]]
    return tbl


def experiment_19_baseline_dispute(table: pd.DataFrame, lims: pd.DataFrame):
    """
    Расхождение с замером Person 2 (2026-09-22).

    Она: "константа (медиана до 2025) 2.36, модель без Q21 2.28" -- то есть
    модель ВЫИГРЫВАЕТ у константы около 4%.
    Я (эксп.14): модель ПРОИГРЫВАЕТ константе 1.4-4.4%.

    Оба замера могут быть арифметически верны и означать разное. Два
    подозрения на источник расхождения:
      1. ОПРЕДЕЛЕНИЕ КОНСТАНТЫ. Медиана минимизирует MAE, среднее -- RMSE.
         Сравнивать по RMSE с медианой -- давать модели фору.
      2. ОДИН СПЛИТ ПРОТИВ ПЯТИ. Эксп.12 уже ловил этот капкан: эффект
         +0.124 на одном разрезе превратился в -0.054 на пяти.
    Развожу оба фактора явно, на её же протоколе (тест = 2026 год).
    """
    print("\n=== Эксперимент 19: чем мерить базу -- медианой или средним ===")
    import lightgbm as lgb
    from src.models.base import monotone_vector

    tbl = _sulfur_samples(table, lims)
    safe = {c: c.replace(":", "__") for c in SULFUR_FEATURES}
    mono = monotone_vector(SULFUR_FEATURES, "sulfur_mgkg")

    def fit_predict(tr, te):
        tr = tr[tr["y"] <= SULFUR_OUTLIER_THRESHOLD]
        base = tr[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                               num_leaves=15, min_child_samples=max(5, len(tr) // 50),
                               monotone_constraints=mono, random_state=42, verbose=-1)
        m.fit(tr[SULFUR_FEATURES].rename(columns=safe), tr["y"] - base)
        base_t = te[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        return (base_t + m.predict(te[SULFUR_FEATURES].rename(columns=safe))).values

    print("\n  -- протокол Person 2: тест = 2026 год, обучение = всё до него --")
    tr = tbl[tbl["_lims_ts"] < pd.Timestamp("2026-01-01")]
    te = tbl[tbl["_lims_ts"] >= pd.Timestamp("2026-01-01")]
    y = te["y"].values
    trf = tr[tr["y"] <= SULFUR_OUTLIER_THRESHOLD]
    rmse = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
    med, avg = float(trf["y"].median()), float(trf["y"].mean())
    r_model = rmse(fit_predict(tr, te))
    r_med, r_avg = rmse(np.full(len(y), med)), rmse(np.full(len(y), avg))
    print(f"    n_train={len(trf)}, n_test={len(te)}")
    print(f"    константа = МЕДИАНА трейна ({med:.2f}):  RMSE {r_med:.3f}")
    print(f"    константа = СРЕДНЕЕ трейна ({avg:.2f}):  RMSE {r_avg:.3f}")
    print(f"    модель без Q21:                      RMSE {r_model:.3f}")
    print(f"    skill против медианы: {100*(1-r_model/r_med):+.1f}%   "
          f"против среднего: {100*(1-r_model/r_avg):+.1f}%")

    print("\n  -- тот же вопрос на 5 разрезах (устойчивость знака) --")
    samples = np.sort(tbl["_lims_ts"].unique())
    print(f"    {'разрез':<9}{'n_test':>8}{'медиана':>10}{'среднее':>10}{'модель':>9}"
          f"{'vs мед.':>9}{'vs сред.':>10}")
    sk_med, sk_avg = [], []
    for frac in (0.60, 0.68, 0.76, 0.84, 0.92):
        cut = samples[int(len(samples) * frac)]
        tr = tbl[tbl["_lims_ts"] < cut]
        te = tbl[tbl["_lims_ts"] >= cut]
        if len(te) < 100:
            continue
        y = te["y"].values
        trf = tr[tr["y"] <= SULFUR_OUTLIER_THRESHOLD]
        rmse = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
        rm, ra, r_mod = (rmse(np.full(len(y), trf["y"].median())),
                          rmse(np.full(len(y), trf["y"].mean())),
                          rmse(fit_predict(tr, te)))
        sk_med.append(100 * (1 - r_mod / rm))
        sk_avg.append(100 * (1 - r_mod / ra))
        print(f"    {frac:<9.2f}{len(te):>8}{rm:>10.3f}{ra:>10.3f}{r_mod:>9.3f}"
              f"{sk_med[-1]:>+8.1f}%{sk_avg[-1]:>+9.1f}%")
    print(f"\n    средний skill против МЕДИАНЫ: {np.mean(sk_med):+.1f}% "
          f"(положителен в {sum(1 for x in sk_med if x > 0)}/{len(sk_med)})")
    print(f"    средний skill против СРЕДНЕГО: {np.mean(sk_avg):+.1f}% "
          f"(положителен в {sum(1 for x in sk_avg if x > 0)}/{len(sk_avg)})")


def experiment_20_q21_causal(table: pd.DataFrame, tel_go: pd.DataFrame, lims: pd.DataFrame):
    """
    Проверка, которую запросила Person 2: результат Q21=1.589 получен на
    окне +-1 час вокруг пробы, то есть с заглядыванием вперёд. Перемерить
    строго причинно -- только показания ДО момента отбора пробы.

    Дополнительно -- распад информативности по горизонту: Q21, взятый за H
    часов до пробы. Person 2 замерила корреляцию (0.47 -> -0.03 за 4 часа);
    здесь то же в единицах RMSE, то есть в том, что реально важно.
    """
    print("\n=== Эксперимент 20: Q21 строго до момента отбора пробы ===")
    q = tel_go[["242000:Q21"]].dropna().sort_index()
    qdf = pd.DataFrame({"ts": q.index, "B": q["242000:Q21"].values}).sort_values("ts")
    tbl = _sulfur_samples(table, lims)
    per = tbl.groupby("_lims_ts").agg(y=("y", "first")).reset_index()
    per = per.rename(columns={"_lims_ts": "lims_ts"}).sort_values("lims_ts")
    per = per[per["y"] <= SULFUR_OUTLIER_THRESHOLD]

    print(f"  {'показание Q21':<34}{'n':>6}{'RMSE':>9}{'corr':>8}")
    for lag_h, label in ((None, "ближайшее в +-15 мин (как было)"),
                          (0.0, "строго ДО пробы (причинно)"),
                          (1.0, "за 1 ч до пробы"),
                          (2.0, "за 2 ч до пробы"),
                          (4.0, "за 4 ч до пробы"),
                          (8.0, "за 8 ч до пробы")):
        left = per.rename(columns={"lims_ts": "ts"}).copy()
        if lag_h is None:
            m = pd.merge_asof(left, qdf, on="ts", direction="nearest",
                               tolerance=pd.Timedelta(minutes=15))
        else:
            left["ts"] = left["ts"] - pd.Timedelta(hours=lag_h)
            m = pd.merge_asof(left, qdf, on="ts", direction="backward",
                               tolerance=pd.Timedelta(hours=2))
        m = m.dropna(subset=["B"])
        m = m[m["B"] <= SULFUR_OUTLIER_THRESHOLD]
        r = float(np.sqrt(np.mean((m["B"] - m["y"]) ** 2)))
        print(f"  {label:<34}{len(m):>6}{r:>9.3f}{m['B'].corr(m['y']):>8.3f}")
    print("\n  (RMSE здесь -- это сам прибор как прогноз, без всякой модели)")


def experiment_21_sensitivity_t5(table: pd.DataFrame, tel_go: pd.DataFrame,
                                  lims: pd.DataFrame):
    """
    H-L1 + проверка, которую запросила Person 2.

    Optimizer использует не сам прогноз, а ОТКЛИК на изменение уставки.
    monotone_constraints гарантируют ЗНАК (не возрастает по T5), но не
    величину: отклик может быть нулевым, и тогда рычаг мёртвый -- в
    сценарии "риск качества" агент не найдёт корректирующего действия.

    Person 2 предупредила, что модель С Q21 выучит "сера ~ Q21" и потеряет
    чувствительность к T5. Проверяем ровно это.

    Возмущение прокидывается по всем производным признакам от T5
    (arrhenius_t5, q20_x_arrhenius, quench_delta), кроме лагов и скользящих --
    мгновенное изменение уставки не меняет историю за 3 и 6 часов назад.
    """
    print("\n=== Эксперимент 21 (H-L1): отклик прогноза на +2 C по T5 ===")
    import lightgbm as lgb
    from src.models.base import monotone_vector
    from src.models.features import arrhenius_term

    tbl = _sulfur_samples(table, lims)
    q = tel_go[["242000:Q21"]].dropna().sort_index()
    qdf = pd.DataFrame({"ts": q.index, "B": q["242000:Q21"].values}).sort_values("ts")
    tbl = pd.merge_asof(tbl.rename_axis("ts").reset_index().sort_values("ts"),
                         qdf, on="ts", direction="backward",
                         tolerance=pd.Timedelta(hours=1)).dropna(subset=["B"]).set_index("ts")

    samples = np.sort(tbl["_lims_ts"].unique())
    cut = samples[int(len(samples) * 0.8)]
    tr = tbl[(tbl["_lims_ts"] < cut) & (tbl["y"] <= SULFUR_OUTLIER_THRESHOLD)]
    te = tbl[tbl["_lims_ts"] >= cut]

    def perturb(X, delta):
        Z = X.copy()
        t6 = X["242000:T5"] - X["242000:T5_T6_quench_delta"]
        Z["242000:T5"] = X["242000:T5"] + delta
        Z["242000:arrhenius_t5"] = arrhenius_term(Z["242000:T5"])
        Z["242000:q20_x_arrhenius"] = Z["242000:Q20"] * Z["242000:arrhenius_t5"]
        Z["242000:T5_T6_quench_delta"] = Z["242000:T5"] - t6
        return Z

    for label, feats in (("без Q21 (модель эффекта)", list(SULFUR_FEATURES)),
                          ("с Q21 (модель уровня)", list(SULFUR_FEATURES) + ["B"])):
        safe = {c: c.replace(":", "__") for c in feats}
        mono = monotone_vector(SULFUR_FEATURES, "sulfur_mgkg") + ([0] if "B" in feats else [])
        base = tr[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                               num_leaves=15, min_child_samples=max(5, len(tr) // 50),
                               monotone_constraints=mono, random_state=42, verbose=-1)
        m.fit(tr[feats].rename(columns=safe), tr["y"] - base)

        def predict(X):
            b = X[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
            return b.values + m.predict(X[feats].rename(columns=safe))

        te2 = perturb(te, +2.0)
        p0, p2 = predict(te), predict(te2)
        d = p2 - p0
        rmse = float(np.sqrt(np.mean((p0 - te["y"].values) ** 2)))
        b0 = te[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1).values
        b2 = te2[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1).values
        print(f"\n  --- {label} ---   RMSE={rmse:.3f}")
        print(f"    отклик на +2 C: медиана {np.median(d):+.3f} мг/кг, "
              f"среднее {d.mean():+.3f}, "
              f"разброс [{np.percentile(d,5):+.3f}, {np.percentile(d,95):+.3f}]")
        print(f"      из них от формулы: {np.median(b2-b0):+.3f}, "
              f"от ML-остатка: {np.median(d - (b2-b0)):+.3f}")
        print(f"    доля точек с нулевым откликом (|d| < 0.01): "
              f"{100*np.mean(np.abs(d) < 0.01):.1f}%")
    print("\n  Ожидание по кинетике (Ea=55 кДж/моль): единицы десятых мг/кг на 2 C.")
    print("  Нулевой отклик = мёртвый рычаг: Optimizer не найдёт корректирующего действия.")


def experiment_22_sign_leak(table: pd.DataFrame, lims: pd.DataFrame):
    """
    Эксп.21 показал, что отклик прогноза на +2 C по T5 ПОЛОЖИТЕЛЬНЫЙ
    (+0.197 мг/кг), хотя физика требует отрицательного: выше температура ->
    глубже HDS -> меньше серы. И это при включённом monotone-ограничении
    EXPECTED_SIGNS["sulfur_mgkg"]["242000:T5"] = -1.

    Гипотеза о механизме: ограничение обходится через ПРОИЗВОДНЫЕ от T5
    признаки, у которых знака не задано вовсе --
        242000:arrhenius_t5     = exp(-Ea/RT), растёт с T
        242000:q20_x_arrhenius  = Q20 * arrhenius_t5
        242000:T5_T6_quench_delta = T5 - T6
    LightGBM ограничивает монотонность ПОКОЛОНОЧНО. Запретив рост по
    колонке T5, мы ничего не запретили по колонке arrhenius_t5, которая
    является строго возрастающей функцией той же T5. Дверь заперта,
    окно открыто.

    Это ровно тот риск, о котором предупреждает комментарий в go.py
    ("модель может выучить контур регулирования и перепутать знак у
    температуры реактора") -- он реализовался, просто в обход.

    Проверяем механизм и лечение. Знак для arrhenius_t5 физически
    однозначен: больше arrhenius -> глубже обессеривание -> МЕНЬШЕ серы,
    то есть -1. Для q20_x_arrhenius знак неоднозначен (произведение
    растущего и убывающего вкладов), поэтому вариант с его удалением.
    """
    print("\n=== Эксперимент 22: обход monotone через производные от T5 ===")
    import lightgbm as lgb
    from src.models.base import EXPECTED_SIGNS
    from src.models.features import arrhenius_term

    tbl = _sulfur_samples(table, lims)
    samples = np.sort(tbl["_lims_ts"].unique())
    base_signs = dict(EXPECTED_SIGNS["sulfur_mgkg"])

    def perturb(X, delta):
        Z = X.copy()
        t6 = X["242000:T5"] - X["242000:T5_T6_quench_delta"]
        Z["242000:T5"] = X["242000:T5"] + delta
        Z["242000:arrhenius_t5"] = arrhenius_term(Z["242000:T5"])
        Z["242000:q20_x_arrhenius"] = Z["242000:Q20"] * Z["242000:arrhenius_t5"]
        Z["242000:T5_T6_quench_delta"] = Z["242000:T5"] - t6
        return Z

    variants = [
        ("как сейчас", list(SULFUR_FEATURES), {}),
        ("+ arrhenius_t5 = -1", list(SULFUR_FEATURES), {"242000:arrhenius_t5": -1}),
        ("+ arrh=-1, без q20_x_arrh",
         [c for c in SULFUR_FEATURES if c != "242000:q20_x_arrhenius"],
         {"242000:arrhenius_t5": -1}),
        ("без обоих arrhenius-фич",
         [c for c in SULFUR_FEATURES
          if c not in ("242000:arrhenius_t5", "242000:q20_x_arrhenius")], {}),
    ]

    print(f"  {'вариант':<28}{'RMSE':>8}{'отклик +2C':>13}{'доля >0':>10}{'блоков':>8}")
    for label, feats, extra in variants:
        signs = {**base_signs, **extra}
        mono = [signs.get(c, 0) for c in feats]
        safe = {c: c.replace(":", "__") for c in feats}
        rmses, meds, wrong = [], [], []
        for frac in (0.60, 0.72, 0.84):
            cut = samples[int(len(samples) * frac)]
            tr = tbl[(tbl["_lims_ts"] < cut) & (tbl["y"] <= SULFUR_OUTLIER_THRESHOLD)]
            te = tbl[tbl["_lims_ts"] >= cut]
            if len(te) < 100:
                continue
            b = tr[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
            m = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   num_leaves=15, min_child_samples=max(5, len(tr) // 50),
                                   monotone_constraints=mono, random_state=42, verbose=-1)
            m.fit(tr[feats].rename(columns=safe), tr["y"] - b)

            def pred(X):
                bb = X[["242000:T5"]].apply(lambda r: sulfur_arrhenius_baseline(r), axis=1)
                return bb.values + m.predict(X[feats].rename(columns=safe))

            p0, p2 = pred(te), pred(perturb(te, +2.0))
            rmses.append(float(np.sqrt(np.mean((p0 - te["y"].values) ** 2))))
            meds.append(float(np.median(p2 - p0)))
            wrong.append(float(np.mean((p2 - p0) > 0)))
        print(f"  {label:<28}{np.mean(rmses):>8.3f}{np.mean(meds):>+13.3f}"
              f"{100*np.mean(wrong):>9.1f}%{len(rmses):>8}")

    print("\n  физика: выше T -> глубже HDS -> МЕНЬШЕ серы, отклик должен быть < 0.")
    print("  'доля >0' -- сколько точек модель считает, что нагрев ПОВЫШАЕТ серу.")


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
    experiment_13_three_cornered_hat(table, tel_go, lims)
    experiment_14_skill_vs_constant(table, go, tel_go, lims)

    print("\nПересборка таблицы БЕЗ отсечения выбросов (для H-B2)...")
    table_full, _ = build_sulfur_table(avt, tel_avt_raw, tel_go, lims,
                                        feed_delay_h=0.0, drop_outliers=False)
    table_full = table_full.loc[table_full.index.sort_values()]
    experiment_6_outlier_weighting(table_full)


if __name__ == "__main__":
    main()
