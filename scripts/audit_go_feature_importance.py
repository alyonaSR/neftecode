#!/usr/bin/env python3
"""
Feature importance уже обученной GOModel -- что реально несёт
предсказательную силу, а что нет, прежде чем ставить новые ML-эксперименты
вслепую (АВТ-лаги и
Аррениус-калибровка уже проверены и опровергнуты этим же методом мышления).

Ничего не меняет, только читает
artifacts/models/go_v1.joblib (и avt_v1.joblib для полноты картины) и
печатает встроенную важность признаков LightGBM (gain -- суммарный вклад
в снижение ошибки; split -- сколько раз признак использован для сплита).
Если установлен пакет shap, дополнительно печатает mean(|SHAP|) на
калибровочной выборке -- более честная метрика, учитывает направление и
взаимодействия признаков, а не только частоту/выигрыш в дереве.

    python scripts/audit_go_feature_importance.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "artifacts", "models")


def print_importance(label: str, model_bundle, feature_source: pd.DataFrame = None):
    for out, sub in model_bundle._models.items():
        print(f"\n=== {label}.{out} ===")
        if sub._model is None:
            print("  нет обученного остатка (formula-only) -- важность недоступна")
            continue

        booster = sub._model.booster_
        gain = booster.feature_importance(importance_type="gain")
        split = booster.feature_importance(importance_type="split")
        # feature_cols передавались в fit() в этом же порядке (см. formula_residual.py),
        # booster.feature_name() -- их санитизированная версия того же порядка,
        # поэтому используем ОРИГИНАЛЬНЫЕ имена из sub.feature_cols напрямую
        names = sub.feature_cols

        total_gain = gain.sum() or 1.0
        rows = sorted(zip(names, gain, split), key=lambda r: -r[1])
        print(f"  {'признак':<28}{'gain %':>10}{'split':>8}")
        for name, g, s in rows:
            print(f"  {name:<28}{100*g/total_gain:>9.1f}%{int(s):>8}")

        if feature_source is not None:
            try:
                import shap
                explainer = shap.TreeExplainer(sub._model)
                safe = feature_source[names].rename(
                    columns=lambda c: c.replace(":", "__"))
                sv = explainer.shap_values(safe)
                mean_abs = np.abs(sv).mean(axis=0)
                print(f"  -- mean(|SHAP|) на {len(safe)} точках --")
                for name, v in sorted(zip(names, mean_abs), key=lambda r: -r[1]):
                    print(f"  {name:<28}{v:>10.3f}")
            except ImportError:
                print("  (пакет shap не установлен -- пропуск SHAP, ограничились gain/split)")


def main():
    from src.models.avt import AVTModel
    from src.models.go import GOModel

    avt = AVTModel.load(os.path.join(ARTIFACTS, "avt_v1.joblib"))
    go = GOModel.load(os.path.join(ARTIFACTS, "go_v1.joblib"))

    print_importance("AVTModel", avt)
    print_importance("GOModel", go)


if __name__ == "__main__":
    main()
