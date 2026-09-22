"""
Обёртка "формула ВАК как baseline + LightGBM на остатках".

Общий паттерн soft-сенсора для всех показателей AVTModel и не-серных
показателей GOModel: если для показателя есть формула ВАК, из неё
берётся физически осмысленная опорная линия, и модель учится только на
том, что формула не объясняет. Если формулы нет (или она сломана --
см. vak_formulas.avt_240_350_cfpp) -- formula_fn=None, тогда baseline=0
и это вырождается в обычный ML на признаках, без специального кода.

Почему так, а не отдельный predict() на каждый показатель:
  - формула уже несёт знак и физику (Аррениус, материальный баланс),
    LightGBM учит только то, чего формула не знает -- меньше данных
    нужно для той же точности, чем на голом ML
  - один класс тестируется один раз, а не N раз на N показателей

Интервал -- split conformal prediction с онлайн-адаптацией (Adaptive
Conformal Inference), см. conformal.py. Реализация из research.pdf,
заменяет прежнюю эвристику "эмпирические 10/90 перцентили остатка без
поправки на конечную выборку и без доказанной гарантии покрытия".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from ..contracts import Interval
from .conformal import ConformalResidualBounds

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class NotFittedError(RuntimeError):
    pass


def _safe_name(col: str) -> str:
    """LightGBM запрещает спецсимволы в именах колонок, у нас 'AVT:F30'."""
    return col.replace(":", "__")


def _fit_shrink(pred, y, const: float) -> float:
    """
    Вес доверия к прогнозу: итог = w*прогноз + (1-w)*константа.

    Подбирается на калибровке, поэтому механизм самоограничивающийся --
    у хорошей модели w уходит в 1.0 и ничего не меняется. Нужен там, где
    признаки не переносятся во времени и остаток переобучается: замер
    2026-09-22 (walk-forward, 5 блоков, skill против константы)

        AVT.feed_cfpp_c   -18.9%  ->  +1.1%   (w=0.20)
        GO.cfpp_c         -12.9%  ->  -1.4%   (w=0.09)

    Малый вес -- не поражение, а честное признание: столько сигнала в
    признаках и есть.
    """
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = np.linspace(0.0, 1.0, 21)
    errs = [np.mean(((w * pred + (1.0 - w) * const) - y) ** 2) for w in grid]
    return float(grid[int(np.argmin(errs))])


def _as_float_or_nan(v):
    """
    Найдено при прогоне полного цикла, воспроизведено и подтверждено:
    отсутствующий тег -> features.get(c) -> None -> колонка DataFrame
    получает dtype=object -> LightGBM.predict() падает с ValueError вместо
    штатной обработки пропуска. В реальной эксплуатации дырка в теге --
    рутина (поверка датчика, обрыв связи, пропуск в архиве), а не повод
    ронять весь цикл принятия решения. np.nan -- плавающая точка, у
    LightGBM для неё есть нативный механизм missing values (выбор ветки
    по default direction), это не костыль поверх модели.
    """
    return float(v) if v is not None else np.nan


@dataclass
class FormulaPlusResidual:
    name: str
    formula_fn: Optional[Callable[[Mapping], float]]
    formula_tags: List[str]
    feature_cols: List[str]
    monotone: Optional[List[int]] = None
    fallback_mean: float = 0.0

    def __post_init__(self):
        self._model = None
        self._conformal: Optional[ConformalResidualBounds] = None
        self._resid_bias = 0.0
        self._const = None
        self._shrink = 1.0
        self._sigma_model = None
        self._sigma_floor = 1.0
        self._n_train = 0
        self._n_calib = 0

    @property
    def is_fitted(self) -> bool:
        return self._model is not None or self.formula_fn is not None

    def baseline(self, row: Mapping) -> float:
        if self.formula_fn is None:
            return 0.0
        try:
            return float(self.formula_fn(row))
        except (KeyError, TypeError, ZeroDivisionError):
            return 0.0

    def _fit_sigma(self, X_train: pd.DataFrame, resid_train, params: dict):
        """
        H-I1. Модель величины ошибки sigma(x) = E[|остаток|]. Намеренно
        МЕНЬШЕ основной (глубина 3, 100 деревьев): её задача -- грубо
        отличить спокойный режим от турбулентного, а не выучить шум.
        Переобученная sigma сделала бы интервал случайным, а не условным.
        """
        p = dict(params)
        p.update(n_estimators=100, max_depth=3, num_leaves=7,
                 min_child_samples=max(5, len(X_train) // 30))
        p.pop("monotone_constraints", None)
        m = lgb.LGBMRegressor(**p)
        m.fit(X_train[self.feature_cols].rename(columns=_safe_name), np.abs(resid_train))
        pred = m.predict(X_train[self.feature_cols].rename(columns=_safe_name))
        self._sigma_floor = max(1e-6, 0.2 * float(np.median(np.abs(pred))))
        return m

    def _sigma_of(self, X: pd.DataFrame):
        """sigma(x) для набора строк. Без sigma-модели -- единицы (глобальный режим)."""
        if self._sigma_model is None:
            return np.ones(len(X))
        s = self._sigma_model.predict(X[self.feature_cols].rename(columns=_safe_name))
        return np.clip(s, self._sigma_floor, None)

    def _sigma_one(self, row: pd.DataFrame) -> float:
        if self._sigma_model is None:
            return 1.0
        s = float(self._sigma_model.predict(row.rename(columns=_safe_name))[0])
        return max(s, self._sigma_floor)

    def fit(self, X: pd.DataFrame, y: pd.Series, train_frac: float = 0.8,
            **lgbm_kwargs) -> "FormulaPlusResidual":
        """
        X: строки времени (сортировка по ts гарантируется здесь), колонки
           -- объединение formula_tags и feature_cols.
        y: целевая величина, тот же индекс.

        Сплит хронологический (train_frac по времени, не случайный),
        ТЗ запрещает shuffle для временных рядов.
        """
        if lgb is None:
            raise ImportError("pip install lightgbm")

        order = X.index.sort_values()
        X = X.loc[order]
        y = y.loc[order]

        n = len(X)
        cut = int(n * train_frac)
        X_train, X_calib = X.iloc[:cut], X.iloc[cut:]
        y_train, y_calib = y.iloc[:cut], y.iloc[cut:]

        base_train = X_train.apply(lambda r: self.baseline(r), axis=1)
        resid_train = y_train - base_train

        params = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                       num_leaves=15, min_child_samples=max(5, n // 50),
                       random_state=42, verbose=-1)
        params.update(lgbm_kwargs)
        if self.monotone is not None:
            params["monotone_constraints"] = self.monotone

        model = lgb.LGBMRegressor(**params)
        safe_X = X_train[self.feature_cols].rename(columns=_safe_name)
        model.fit(safe_X, resid_train)
        self._model = model
        self._n_train = len(X_train)

        if len(X_calib) >= 5:
            base_calib = X_calib.apply(lambda r: self.baseline(r), axis=1)
            safe_calib = X_calib[self.feature_cols].rename(columns=_safe_name)
            pred_calib = base_calib + model.predict(safe_calib)
            err = (y_calib - pred_calib)

            self._const = float(y_train.mean())
            self._shrink = _fit_shrink(pred_calib.values, y_calib.values, self._const)
            blended = self._shrink * pred_calib + (1.0 - self._shrink) * self._const
            err = y_calib - blended
            self._resid_bias = float(np.median(err))
            err_debiased = err - self._resid_bias

            self._sigma_model = self._fit_sigma(X_train, resid_train, params)
            sigma_calib = self._sigma_of(X_calib)

            self._conformal = ConformalResidualBounds().fit(err_debiased / sigma_calib)
            self._n_calib = len(X_calib)
        else:
            spread = float(resid_train.std()) if len(resid_train) > 1 else 1.0
            self._resid_bias = 0.0
            self._sigma_model = None
            self._sigma_floor = 1.0
            self._conformal = ConformalResidualBounds().fit(
                np.array([-1.5 * spread, 1.5 * spread])
            )

        return self

    def predict_one(self, features: Dict[str, float]) -> Interval:
        base = self.baseline(features)

        if self._model is None:
            if self.formula_fn is None and base == 0.0:
                base = self.fallback_mean
            half = 8.0
            return Interval(base, base - half, base + half)

        row = pd.DataFrame([{c: _as_float_or_nan(features.get(c)) for c in self.feature_cols}])
        safe_row = row[self.feature_cols].rename(columns=_safe_name)
        resid = float(self._model.predict(safe_row)[0])
        point = base + resid
        if self._const is not None and self._shrink < 1.0:
            point = self._shrink * point + (1.0 - self._shrink) * self._const
        mean = point + (self._resid_bias or 0.0)
        if self._conformal is not None:
            offset_lo, offset_hi = self._conformal.bounds()
        else:
            offset_lo, offset_hi = -1.0, 1.0
        sigma = self._sigma_one(row)
        lo, hi = mean + offset_lo * sigma, mean + offset_hi * sigma
        if lo > hi:
            lo, hi = hi, lo
        return Interval(mean, lo, hi)

    def observe(self, features: Dict[str, float], y_true: float) -> None:
        """
        Adaptive Conformal Inference: онлайн-шаг, когда пришёл РЕАЛЬНЫЙ
        факт (новый анализ ЛИМС) для ранее сделанного прогноза.

        Не вызывается автоматически ни из какого продакшен-цикла --
        для этого нужен живой поток решений с обратной связью, это
        зона Orchestrator. Метод готов быть подключённым,
        покрыт tests/test_conformal.py.
        """
        if self._model is None or self._conformal is None:
            return
        row = pd.DataFrame([{c: _as_float_or_nan(features.get(c)) for c in self.feature_cols}])
        safe_row = row[self.feature_cols].rename(columns=_safe_name)
        resid = float(self._model.predict(safe_row)[0])
        pred = self.baseline(features) + resid + self._resid_bias
        self._conformal.update((y_true - pred) / self._sigma_one(row))

    def to_state(self) -> dict:
        return {
            "name": self.name, "model": self._model,
            "conformal": self._conformal.to_state() if self._conformal else None,
            "resid_bias": self._resid_bias,
            "sigma_model": self._sigma_model,
            "sigma_floor": self._sigma_floor,
            "const": self._const,
            "shrink": self._shrink,
            "n_train": self._n_train, "n_calib": self._n_calib,
            "fallback_mean": self.fallback_mean,
        }

    @classmethod
    def from_state(cls, state: dict, formula_fn, formula_tags, feature_cols,
                    monotone=None, fallback_mean: float = 0.0):
        """
        БАГ, НАЙДЕН И ИСПРАВЛЕН: cls(...) без fallback_mean
        тихо обнулял его (дефолт дата-класса 0.0), хотя вызывающая
        сторона (AVTModel.load / GOModel.load) прекрасно знает правильное
        значение из _SPECS. Для показателя без формулы и без обученного
        остатка (feed_flash_c до появления LIMS-точки с flash_c на АВТ)
        predict_one() тогда возвращал Interval(0.0, -8.0, 8.0) вместо
        Interval(68.0, 60.0, 76.0) -- физически бессмысленный ноль вместо
        честного "формула отсутствует, используем опорную константу".
        Это и роняло spec_risk_prob до 1.0 на demo-сценарии normal.

        Артефакты ранних версий хранят resid_lo/resid_hi
        (плоские числа) вместо conformal (state калибратора) -- строим
        ConformalResidualBounds из них как разовый откат, дальше
        используется честная калибровка при следующем переобучении.
        """
        obj = cls(state["name"], formula_fn, formula_tags, feature_cols, monotone)
        obj._model = state["model"]
        if state.get("conformal") is not None:
            obj._conformal = ConformalResidualBounds.from_state(state["conformal"])
        elif state.get("resid_lo") is not None and state.get("resid_hi") is not None:
            legacy = ConformalResidualBounds()
            legacy._hi_scores = [state["resid_hi"]]
            legacy._lo_scores = [-state["resid_lo"]]
            obj._conformal = legacy
        obj._resid_bias = state.get("resid_bias", 0.0)
        obj._sigma_model = state.get("sigma_model")
        obj._sigma_floor = state.get("sigma_floor", 1.0)
        obj._const = state.get("const")
        obj._shrink = state.get("shrink", 1.0)
        obj._n_train = state.get("n_train", 0)
        obj._n_calib = state.get("n_calib", 0)
        obj.fallback_mean = state.get("fallback_mean", fallback_mean)
        return obj
