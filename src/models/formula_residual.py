"""
Обёртка "формула ВАК как baseline + LightGBM на остатках".

Зона ответственности: Person 3 (ML Engineer).

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
Conformal Inference), см. conformal.py. Stage 3 из research.pdf,
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
except ImportError:  # pragma: no cover
    lgb = None


class NotFittedError(RuntimeError):
    pass


def _safe_name(col: str) -> str:
    """LightGBM запрещает спецсимволы в именах колонок, у нас 'AVT:F30'."""
    return col.replace(":", "__")


def _as_float_or_nan(v):
    """
    Найдено Person 1 (прогон полного цикла, воспроизведено и подтверждено):
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
    fallback_mean: float = 0.0  # если formula_fn=None и модель не обучена

    def __post_init__(self):
        self._model = None
        self._conformal: Optional[ConformalResidualBounds] = None
        self._resid_bias = 0.0  # медиана остатка на калибровке, см. fit()
        self._sigma_model = None  # H-I1: предсказывает |остаток| -> условная ширина
        self._sigma_floor = 1.0   # пол для sigma, чтобы не делить на ~0
        self._n_train = 0
        self._n_calib = 0

    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self._model is not None or self.formula_fn is not None

    def baseline(self, row: Mapping) -> float:
        if self.formula_fn is None:
            return 0.0
        try:
            return float(self.formula_fn(row))
        except (KeyError, TypeError, ZeroDivisionError):
            # входа формулы не хватает (деградированное состояние) --
            # вырождается в чистый residual, не падает
            return 0.0

    # ------------------------------------------------------------------
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
        p.pop("monotone_constraints", None)  # знаки заданы для уровня, не для |ошибки|
        m = lgb.LGBMRegressor(**p)
        m.fit(X_train[self.feature_cols].rename(columns=_safe_name), np.abs(resid_train))
        # пол = 20% от типичной sigma на обучении: защищает от деления на ~0
        # и от абсурдно узкого интервала там, где sigma-модель ошиблась вниз
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

    # ------------------------------------------------------------------
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
        # LightGBM не принимает ':' в именах колонок (наш формат тега
        # 'AVT:F30') -- санитизируем только на границе с LightGBM,
        # feature_cols и вход predict_one остаются в исходном формате
        safe_X = X_train[self.feature_cols].rename(columns=_safe_name)
        model.fit(safe_X, resid_train)
        self._model = model
        self._n_train = len(X_train)

        if len(X_calib) >= 5:
            base_calib = X_calib.apply(lambda r: self.baseline(r), axis=1)
            safe_calib = X_calib[self.feature_cols].rename(columns=_safe_name)
            pred_calib = base_calib + model.predict(safe_calib)
            err = (y_calib - pred_calib)

            # НАЙДЕНО (Stage 2 bias-фикс): калибровочный остаток не
            # центрирован в нуле -- модель обучена на первых train_frac
            # исторических точках, а calib -- уже дальше по времени
            # (chronological split, не shuffle). На реальном демо-сценарии
            # (дата ближе к концу истории, чем к train-части) это давало
            # систематическое ЗАВЫШЕНИЕ серы на ~2-2.5 мг/кг относительно
            # ЛИМС -- не шум, устойчивый сдвиг на всех трёх demo-сценариях.
            # Медиана остатка -- честная оценка сдвига (устойчивее к
            # выбросам, чем среднее), добавляется к точечному прогнозу.
            self._resid_bias = float(np.median(err))
            err_debiased = err - self._resid_bias

            # H-I1 (SULFUR_HYPOTHESES.md, проверено experiment_8): УСЛОВНЫЙ
            # конформный интервал. Раньше квантиль остатка был один глобальный
            # на все режимы -- интервал одинаково широкий и в спокойный день,
            # и в турбулентный, то есть в спокойном мы отдавали запас впустую
            # (а Gate проверяет именно верхнюю границу). Теперь вторая модель
            # учится предсказывать ВЕЛИЧИНУ ошибки sigma(x), остаток нормируется
            # на неё, а интервал восстанавливается как pred +- q*sigma(x).
            # Маргинальное покрытие сохраняется (стандартный результат для
            # normalized conformal), но ширина становится условной.
            # Замер на честном holdout: ширина 3.09 в спокойных режимах против
            # 4.94 в турбулентных (было 3.89 везде), запас до лимита 10 мг/кг
            # в спокойных вырос на +0.48.
            self._sigma_model = self._fit_sigma(X_train, resid_train, params)
            sigma_calib = self._sigma_of(X_calib)

            # Stage 3: split conformal + ACI вместо наивных np.quantile.
            # Калибруется на ДЕ-СМЕЩЁННОМ и НОРМИРОВАННОМ остатке, чтобы
            # интервал был честной оценкой оставшейся неопределённости вокруг
            # уже скорректированного центра, а не тащил на себе ещё и
            # исправление смещения.
            self._conformal = ConformalResidualBounds().fit(err_debiased / sigma_calib)
            self._n_calib = len(X_calib)
        else:
            # мало данных на калибровку -- эвристический запас,
            # честно шире, чем typical residual std
            spread = float(resid_train.std()) if len(resid_train) > 1 else 1.0
            self._resid_bias = 0.0
            self._sigma_model = None  # мало данных -- без условной ширины
            self._sigma_floor = 1.0
            self._conformal = ConformalResidualBounds().fit(
                np.array([-1.5 * spread, 1.5 * spread])
            )

        return self

    # ------------------------------------------------------------------
    def predict_one(self, features: Dict[str, float]) -> Interval:
        base = self.baseline(features)

        if self._model is None:
            if self.formula_fn is None and base == 0.0:
                # нет ни формулы, ни обученной модели -- 0.0 физически
                # бессмысленно для большинства показателей, используем
                # опорную константу вместо неё, пока не обучено
                base = self.fallback_mean
            # формула без обученного остатка -- честно широкий интервал,
            # чтобы Gate не поверил точечному прогнозу больше, чем он стоит
            half = 8.0
            return Interval(base, base - half, base + half)

        # ВАЖНО: подстановка fallback_mean применима ТОЛЬКО к необученному
        # случаю выше. При formula_fn=None остаток обучается на y - 0 = y,
        # то есть САМ несёт уровень; прибавить к нему ещё и fallback значит
        # посчитать уровень дважды. Замер (2026-09-22, feed_ebp_c без
        # формулы): прогноз выходил ~730 при метке ~365, RMSE 364.6.
        # В продакшен-артефакте баг спящий -- у единственной цели с
        # formula_fn=None (feed_flash_c) остаток не обучен, -- но
        # срабатывает у любого, кто такую модель обучит.

        # features.get(c) -> None для отсутствующего тега (поверка датчика,
        # обрыв связи, пропуск в архиве -- рутина в реальной эксплуатации).
        # None в колонке DataFrame даёт dtype=object, LightGBM.predict()
        # падает с ValueError вместо штатной обработки пропуска: nan
        # плавает как float, LightGBM держит выбор ветки для пропусков
        # нативно (это его обычный механизм missing values, не костыль).
        row = pd.DataFrame([{c: _as_float_or_nan(features.get(c)) for c in self.feature_cols}])
        safe_row = row[self.feature_cols].rename(columns=_safe_name)
        resid = float(self._model.predict(safe_row)[0])
        # + resid_bias: коррекция систематического сдвига калибровки, см. fit()
        mean = base + resid + (self._resid_bias or 0.0)
        if self._conformal is not None:
            offset_lo, offset_hi = self._conformal.bounds()
        else:
            offset_lo, offset_hi = -1.0, 1.0
        # H-I1: offsets откалиброваны на НОРМИРОВАННОМ остатке, поэтому
        # разворачиваем обратно через sigma(x) этой конкретной строки --
        # интервал получается узким в спокойном режиме и широким в
        # турбулентном. Без sigma-модели (старый артефакт, мало данных)
        # sigma=1 и поведение ровно прежнее, глобальное.
        sigma = self._sigma_one(row)
        lo, hi = mean + offset_lo * sigma, mean + offset_hi * sigma
        if lo > hi:
            lo, hi = hi, lo
        return Interval(mean, lo, hi)

    # ------------------------------------------------------------------
    def observe(self, features: Dict[str, float], y_true: float) -> None:
        """
        Adaptive Conformal Inference: онлайн-шаг, когда пришёл РЕАЛЬНЫЙ
        факт (новый анализ ЛИМС) для ранее сделанного прогноза.

        Не вызывается автоматически ни из какого продакшен-цикла --
        для этого нужен живой поток решений с обратной связью, это
        зона Orchestrator (Person 1). Метод готов быть подключённым,
        покрыт tests/test_conformal.py.
        """
        if self._model is None or self._conformal is None:
            return
        row = pd.DataFrame([{c: _as_float_or_nan(features.get(c)) for c in self.feature_cols}])
        safe_row = row[self.feature_cols].rename(columns=_safe_name)
        resid = float(self._model.predict(safe_row)[0])
        pred = self.baseline(features) + resid + self._resid_bias
        # H-I1: конформ откалиброван на нормированном остатке, значит и
        # онлайн-обновление ACI должно приходить в том же масштабе
        self._conformal.update((y_true - pred) / self._sigma_one(row))

    # ------------------------------------------------------------------
    def to_state(self) -> dict:
        return {
            "name": self.name, "model": self._model,
            "conformal": self._conformal.to_state() if self._conformal else None,
            "resid_bias": self._resid_bias,
            "sigma_model": self._sigma_model,   # H-I1, условная ширина интервала
            "sigma_floor": self._sigma_floor,
            "n_train": self._n_train, "n_calib": self._n_calib,
            "fallback_mean": self.fallback_mean,
        }

    @classmethod
    def from_state(cls, state: dict, formula_fn, formula_tags, feature_cols,
                    monotone=None, fallback_mean: float = 0.0):
        """
        БАГ, НАЙДЕН И ИСПРАВЛЕН (до Stage 3): cls(...) без fallback_mean
        тихо обнулял его (дефолт дата-класса 0.0), хотя вызывающая
        сторона (AVTModel.load / GOModel.load) прекрасно знает правильное
        значение из _SPECS. Для показателя без формулы и без обученного
        остатка (feed_flash_c до появления LIMS-точки с flash_c на АВТ)
        predict_one() тогда возвращал Interval(0.0, -8.0, 8.0) вместо
        Interval(68.0, 60.0, 76.0) -- физически бессмысленный ноль вместо
        честного "формула отсутствует, используем опорную константу".
        Это и роняло spec_risk_prob до 1.0 на demo-сценарии normal.

        Артефакты, сохранённые ДО Stage 3, хранят resid_lo/resid_hi
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
        # H-I1: в артефактах, сохранённых ДО условного конформа, sigma-модели
        # нет -- тогда _sigma_of/_sigma_one вернут 1.0 и интервал останется
        # глобальным, ровно как раньше. Обратная совместимость без ветвлений.
        obj._sigma_model = state.get("sigma_model")
        obj._sigma_floor = state.get("sigma_floor", 1.0)
        obj._n_train = state.get("n_train", 0)
        obj._n_calib = state.get("n_calib", 0)
        obj.fallback_mean = state.get("fallback_mean", fallback_mean)
        return obj
