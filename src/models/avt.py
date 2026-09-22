"""
Модель установки АВТ.

ЦЕЛЕВАЯ ВЕЛИЧИНА ЦЕПОЧКИ: feed_ebp_c, конец кипения. T95 выбран НЕ был:
формулы ВАК для AVT6:240-350 есть только на D15, T50, EBP и CFPP,
а EBP и T95 сырья коррелируют на 0.91, то есть несут одно и то же.

ЧТО ПРЕДСКАЗЫВАЕТ: качество дизельной фракции, которая уходит с АВТ
в гидроочистку (feed_ebp_c, feed_d15_kgm3, feed_cfpp_c, feed_flash_c).

АРХИТЕКТУРА (см. models/vak_formulas.py и
models/formula_residual.py):

  Каждый показатель = формула ВАК (физически осмысленная опорная линия)
  + LightGBM на остатке. Если рабочей формулы нет или она сломана --
  baseline=0, это просто ML.

  feed_ebp_c    <- AVT6:240-350:EBP
  feed_d15_kgm3 <- AVT6:240-350:D15  (НЕ AVT6:350:D15! У AVT6:350:D15 был
                                       меньше RMSE при аудите формул (6.3 против 13.7),
                                       но это неправильный поток: "350" --
                                       это фракция ТЯЖЕЛЕЕ дизеля, не сырьё
                                       гидроочистки. Проверено на реальной
                                       строке телеметрии: AVT6:350:D15 даёт
                                       ~884 кг/м3, что не бьётся со спекой
                                       товарного продукта 845 -- потому что
                                       это буквально не тот материал.
                                       AVT6:240-350:D15 (тот же кусок, что
                                       и EBP) даёт ~871, ближе к реальному
                                       ЛИМС точки 1 (медиана 878.8))
  feed_cfpp_c   <- формулы нет: AVT6:240-350:CFPP сломана даже после
                                 правки организаторов (corr -0.32, обратный
                                 знак), а AVT6:350:CFPP -- снова не тот
                                 поток (350+ вместо 240-350). Чистый ML.
  feed_flash_c  <- формулы нет вообще ни на одном листе, чистый ML

Признаки -- ТОЛЬКО текущий снимок (без лагов): ProcessState даёт только
срез тегов на момент t (contracts.py), истории у agents/quality.py нет.
Лаги нужны для серы (GOModel) и потребуют расширения ProcessState --
это требует расширения ProcessState и решается не здесь.

ЧТО МОЖНО ВЗЯТЬ ГОТОВЫМ vs ЧЕГО В ВАК НЕТ -- см. vak_formulas.py,
там же таблица со статусом каждой формулы после аудита по ЛИМС.
"""
from __future__ import annotations

from typing import Dict, Optional

from ..contracts import Interval
from . import vak_formulas as vak
from .base import BaseQualityModel, monotone_vector
from .formula_residual import FormulaPlusResidual

_SPECS = {
    "feed_ebp_c": (vak.avt_240_350_ebp, vak.AVT_240_350["ebp_c"][1], ["AVT:F32"], 365.0),
    "feed_d15_kgm3": (vak.avt_240_350_d15, vak.AVT_240_350["d15_kgm3"][1], [], 871.0),
    "feed_cfpp_c": (vak.avt_240_350_cfpp, vak.AVT_240_350["cfpp_c"][1], [], -5.0),
    "feed_flash_c": (None, vak.AVT_240_350["ebp_c"][1], [], 68.0),
}


_NO_FORMULA_BASELINE = {"feed_ebp_c"}


def _baseline_fn(out: str, fn):
    """
    Формула как опорная линия -- или None, если она вредит.

    feed_ebp_c: формула транскрибирована верно и теги сопоставлены верно,
    но она численно неустойчива. Слагаемые -14.08*T37 и +14.60*T58 дают
    вклады -857.67 и +851.72, в сумме +3.38: гасят друг друга только в
    среднем, потому что corr(T37,T58)=0.62, а не ~1. Расхождение этих
    температур на 1 C стоит 14 C на выходе, и std формулы выходит 186.88
    при std самой метки 10.62 -- она в 18 раз шумнее величины, которую
    описывает, а остаток тратит ёмкость на её гашение.

    Замер (walk-forward, 5 блоков, skill против константы):
        baseline = формула              RMSE 32.335   -328.0%
        baseline = формула, обрезанная  RMSE 24.442   -223.5%
        БЕЗ формулы, чистый ML          RMSE  8.203     -8.6%

    Формула при этом не выбрасывается: её теги остаются признаками
    остатка, и train_avt.py по-прежнему выбирает ею точку отбора ЛИМС.
    """
    return None if out in _NO_FORMULA_BASELINE else fn


class AVTModel(BaseQualityModel):
    """
    Композиция из четырёх FormulaPlusResidual, по одному на показатель.

    Без обученного артефакта (fit()/load() не вызывались) каждый
    показатель работает в режиме "только формула": предсказание точное
    там, где формула точна, интервал намеренно широкий (+-8), чтобы
    Gate не поверил непроверенному числу больше, чем оно того стоит.
    Ровно так система работает и без обучения -- это осознанное
    свойство архитектуры, см. README ("никто никого не ждёт").
    """
    outputs = ["feed_ebp_c", "feed_d15_kgm3", "feed_cfpp_c", "feed_flash_c"]
    required_features = sorted({t for _, tags, extra, _ in _SPECS.values() for t in tags + extra})
    model_id = "avt_formula_residual_v3"

    def __init__(self, models: Optional[Dict[str, FormulaPlusResidual]] = None):
        self._models = models or {
            out: FormulaPlusResidual(
                name=out, formula_fn=_baseline_fn(out, fn), formula_tags=tags,
                feature_cols=tags + extra, fallback_mean=fb,
                monotone=monotone_vector(tags + extra, out),
            )
            for out, (fn, tags, extra, fb) in _SPECS.items()
        }

    def predict(self, features: Dict[str, float]) -> Dict[str, Interval]:
        return {out: model.predict_one(features) for out, model in self._models.items()}

    def fit(self, tables: Dict[str, "tuple"]) -> "AVTModel":
        """
        Вызывается из scripts/train_avt.py, не напрямую.
         tables[output] = (X, y),
        где X -- DataFrame с DatetimeIndex и колонками required_features,
        y -- Series той же длины (значение ЛИМС, as-of присоединённое).
        """
        for out, model in self._models.items():
            X, y = tables[out]
            model.fit(X, y)
        return self

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({out: m.to_state() for out, m in self._models.items()}, path)

    @classmethod
    def load(cls, path: str) -> "AVTModel":
        import joblib
        states = joblib.load(path)
        models = {
            out: FormulaPlusResidual.from_state(
                states[out], _baseline_fn(out, fn), tags, tags + extra,
                monotone=monotone_vector(tags + extra, out), fallback_mean=fb,
            )
            for out, (fn, tags, extra, fb) in _SPECS.items()
        }
        return cls(models=models)
