#!/usr/bin/env python3
"""
Stage 2: обучение GOModel (сера -- чистый ML с лагами, cfpp_c -- формула+остаток).

Зона ответственности: Person 3 (ML Engineer).

    python scripts/train_go.py

Требует уже обученный artifacts/models/avt_v1.joblib -- признаки серы
включают feed_ebp_c/feed_d15_kgm3, выход AVTModel. Без него подставит
formula-only AVTModel (менее точные признаки, но не упадёт).

Сборка: телеметрия АВТ и 24-2000 синхронны по 'date' (README, оба файла
189 217 строк с одним и тем же 10-минутным шагом) -- джойн по индексу,
без merge_asof. ЛИМС серы (точка 2 гидроочистки) присоединяется as-of
назад по времени, как и везде.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data.loaders import TARGET_POINT, load_lims, load_telemetry
from src.models.avt import AVTModel
from src.models.features import add_lags, add_rolling, arrhenius_term
from src.models.go import _SPECS, GOModel


def as_of_target(X: pd.DataFrame, lims: pd.DataFrame, param: str, tol_h: float = 1.0):
    sub = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == param)][["ts", "value"]].dropna()
    left = X.reset_index().rename(columns={X.index.name or "index": "ts"})
    right = sub.sort_values("ts").rename(columns={"value": "y"})
    merged = pd.merge_asof(left.sort_values("ts"), right, on="ts",
                            direction="backward", tolerance=pd.Timedelta(hours=tol_h)).dropna()
    return merged.set_index("ts")


def main():
    print("Загрузка телеметрии и ЛИМС...")
    tel_avt_raw = load_telemetry("AVT")
    tel_go = load_telemetry("242000")
    lims = load_lims()

    avt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "artifacts", "models", "avt_v1.joblib")
    if os.path.exists(avt_path):
        print(f"AVTModel загружен из {avt_path}")
        avt_model = AVTModel.load(avt_path)
    else:
        print("artifacts/models/avt_v1.joblib не найден -- запусти scripts/train_avt.py "
              "сначала. Использую formula-only AVTModel (менее точно).")
        avt_model = AVTModel()

    # --- признаки серы: T5 + лаги/волатильность + Q20 (ПАК сера сырья) + AVT-выход ---
    # Q20 найдена Person 2 (242000:Q20, 189217 точек, 10 мин шаг) -- ПАК сера
    # СЫРЬЯ гидроочистки. Раньше сера сырья была видна только из ЛИМС
    # (132 значения, Mass.Sulfur), теперь непрерывно. Добавлена с лагами по
    # той же схеме, что T5 -- дать модели шанс найти правильную динамику,
    # а не гадать заранее (см. models/features.py, тот же паттерн для T5).
    print("\nСборка признаков серы (лаги T5, Q20 -- сера сырья, T5-T6 квенч-дельта, выход AVTModel)...")
    t5 = tel_go[["242000:T5", "242000:T6", "242000:Q20", "242000:F25", "242000:F9"]].dropna()
    # H-A1b (SULFUR_HYPOTHESES.md): T5-T6 правдоподобно величина межступенчатого
    # квенч-охлаждения (T5 после слоя 1, T6 после квенча/перед слоем 2) --
    # T5 и T6 сами коллинеарны на 0.998, но их РАЗНОСТЬ не избыточна с T5.
    t5["242000:T5_T6_quench_delta"] = t5["242000:T5"] - t5["242000:T6"]
    # H-A4 (P8, перепад давления) проверена и ОТКЛОНЕНА -- см. go.py._SPECS,
    # переобучение вместо реального сигнала (SULFUR_HYPOTHESES.md)
    # H-A3: соотношение H2/сырьё (свежий ВСГ / расход сырья, нм3/т) --
    # влияет на парциальное давление H2, глубину HDS (research.pdf)
    t5["242000:h2_oil_ratio"] = t5["242000:F25"] / t5["242000:F9"].replace(0, float("nan"))
    # H-A2: явный аррениусовский член exp(-Ea/RT) на сыром T5 (WABT из
    # H-A1 отклонена -- физически необоснована для этой пары аппаратов),
    # и его произведение с Q20 (кинетика первого порядка: скорость ~
    # [S_сырья] x exp(-Ea/RT)) -- явное взаимодействие вместо надежды,
    # что дерево само его найдёт.
    t5["242000:arrhenius_t5"] = arrhenius_term(t5["242000:T5"])
    t5["242000:q20_x_arrhenius"] = t5["242000:Q20"] * t5["242000:arrhenius_t5"]
    t5 = add_lags(t5, ["242000:T5", "242000:Q20"], lags_h=(3, 6))
    t5 = add_rolling(t5, ["242000:T5"], windows_h=(3, 6))
    t5 = t5.dropna()

    # AVT и 24-2000 синхронны по индексу (README: оба 189217 строк, шаг 10 мин)
    common_idx = t5.index.intersection(tel_avt_raw.index)
    avt_aligned = tel_avt_raw.loc[common_idx, avt_model.required_features].dropna()
    common_idx = common_idx.intersection(avt_aligned.index)

    print(f"  считаю AVTModel.predict() построчно на {len(common_idx)} точках "
          f"(нужно только там, где дальше есть сопоставимый ЛИМС серы)...")
    # ограничиваем до окрестностей реальных проб серы, иначе 189k вызовов predict_one
    sulfur_lims = lims[(lims["sample_point"] == TARGET_POINT) & (lims["param"] == "sulfur_mgkg")]
    near_mask = pd.Series(False, index=common_idx)
    for ts in sulfur_lims["ts"]:
        near_mask |= (common_idx >= ts - pd.Timedelta(hours=1)) & (common_idx <= ts + pd.Timedelta(hours=1))
    idx_needed = common_idx[near_mask]
    print(f"  сузили до {len(idx_needed)} точек рядом с реальными пробами серы")

    feed_ebp, feed_d15 = [], []
    for ts in idx_needed:
        out = avt_model.predict(avt_aligned.loc[ts].to_dict())
        feed_ebp.append(out["feed_ebp_c"].mean)
        feed_d15.append(out["feed_d15_kgm3"].mean)

    X_sulfur = t5.loc[idx_needed].copy()
    X_sulfur["feed_ebp_c"] = feed_ebp
    X_sulfur["feed_d15_kgm3"] = feed_d15
    X_sulfur = X_sulfur[_SPECS["sulfur_mgkg"][1]]

    table_sulfur = as_of_target(X_sulfur, lims, "sulfur_mgkg")
    print(f"  n={len(table_sulfur)} после as-of join с ЛИМС серы")

    # ЛИМС серы содержит редкие явные выбросы (до 2120 мг/кг при
    # медиане 8.6 -- см. Stage 2 находку), PLAUSIBLE-фильтр в loaders.py
    # для них слишком широкий (0..5000). Не путать с 10.0 мг/кг --
    # рабочим пределом спеки: это те самые единицы точки за 15, которые
    # сами по себе валидный демо-сценарий "аномалия", а не типичный
    # режим для регрессии на лагах температуры. Исключаем из обучения
    # остатка явно, не молча -- порог документирован, не тонкая настройка.
    SULFUR_OUTLIER_THRESHOLD = 20.0
    before_n = len(table_sulfur)
    table_sulfur = table_sulfur[table_sulfur["y"] <= SULFUR_OUTLIER_THRESHOLD]
    print(f"  исключено {before_n - len(table_sulfur)} точек с y > {SULFUR_OUTLIER_THRESHOLD} "
          f"мг/кг (явные выбросы/аномалии, не типичный режим для лаговой регрессии)")

    # --- признаки cfpp_c: формула 24-2000:GODT:CFPP ---
    print("\nСборка признаков cfpp_c...")
    cfpp_tags = _SPECS["cfpp_c"][1]
    X_cfpp = tel_go[cfpp_tags].dropna()
    table_cfpp = as_of_target(X_cfpp, lims, "cfpp_c")
    print(f"  n={len(table_cfpp)} после as-of join с ЛИМС cfpp")

    # --- признаки flash_c: 242000:T18, готовый ВАК вспышки (H-C1) ---
    print("\nСборка признаков flash_c (242000:T18)...")
    flash_tags = _SPECS["flash_c"][1]
    X_flash = tel_go[flash_tags].dropna()
    table_flash = as_of_target(X_flash, lims, "flash_c")
    print(f"  n={len(table_flash)} после as-of join с ЛИМС flash_c")

    print("\nОбучение...")
    model = GOModel()
    tables = {}
    if len(table_sulfur) >= 30:
        tables["sulfur_mgkg"] = (table_sulfur[_SPECS["sulfur_mgkg"][1]], table_sulfur["y"])
    else:
        print("  sulfur_mgkg: мало точек, пропуск")
    if len(table_cfpp) >= 30:
        tables["cfpp_c"] = (table_cfpp[cfpp_tags], table_cfpp["y"])
    else:
        print("  cfpp_c: мало точек, пропуск")
    if len(table_flash) >= 30:
        tables["flash_c"] = (table_flash[flash_tags], table_flash["y"])
    else:
        print("  flash_c: мало точек, пропуск")

    for out, (X, y) in tables.items():
        before = X.apply(lambda r: model._models[out].baseline(r), axis=1)
        rmse_before = float(np.sqrt(((before - y) ** 2).mean()))
        model._models[out].fit(X, y)
        sub = model._models[out]
        bounds_str = ""
        if sub._conformal is not None:
            offset_lo, offset_hi = sub._conformal.bounds()
            # H-I1: после условного конформа offsets заданы в единицах sigma(x),
            # а не в мг/кг -- умножаются на sigma строки в predict_one. Печатаем
            # это явно, иначе число выглядит как «интервал вдруг стал вдвое уже».
            units = "x sigma(x)" if sub._sigma_model is not None else "мг/кг (глобальный)"
            bounds_str = (f"  bias={sub._resid_bias:.2f}  "
                          f"conformal offset=[{offset_lo:.2f}, {offset_hi:.2f}] {units}")
        print(f"  {out}: RMSE baseline={rmse_before:.3f}  train={sub._n_train} "
              f"calib={sub._n_calib}{bounds_str}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "artifacts", "models")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "go_v1.joblib")
    model.save(path)
    print(f"\nсохранено -> {path}")


if __name__ == "__main__":
    main()
