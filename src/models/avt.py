"""
Модель установки АВТ.

Зона ответственности: Person 3 (ML Engineer).

ЦЕЛЕВАЯ ВЕЛИЧИНА ЦЕПОЧКИ: feed_ebp_c, конец кипения. T95 выбран НЕ был:
формулы ВАК для AVT6:240-350 есть только на D15, T50, EBP и CFPP,
а EBP и T95 сырья коррелируют на 0.91, то есть несут одно и то же
(решение Person 1, подтверждено org-схемой: avt_diesel_t95_c на P&ID
это тот же физический поток, что мы аппроксимируем через EBP).

ЧТО ПРЕДСКАЗЫВАЕТ: качество дизельной фракции, которая уходит с АВТ
в гидроочистку (feed_ebp_c, feed_d15_kgm3, feed_cfpp_c, feed_flash_c).

АРХИТЕКТУРА (после Stage 0/1, см. models/vak_formulas.py и
models/formula_residual.py):

  Каждый показатель = формула ВАК (физически осмысленная опорная линия)
  + LightGBM на остатке. Если рабочей формулы нет или она сломана --
  baseline=0, это просто ML.

  feed_ebp_c    <- AVT6:240-350:EBP
  feed_d15_kgm3 <- AVT6:240-350:D15  (НЕ AVT6:350:D15! У AVT6:350:D15 был
                                       меньше RMSE в Stage 0 (6.3 против 13.7),
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
это отдельный разговор с Person 1, не в рамках AVTModel.

ЧТО МОЖНО ВЗЯТЬ ГОТОВЫМ vs ЧЕГО В ВАК НЕТ -- см. vak_formulas.py,
там же таблица со статусом каждой формулы после аудита по ЛИМС.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..contracts import Interval
from . import vak_formulas as vak
from .base import BaseQualityModel, monotone_vector
from .formula_residual import FormulaPlusResidual

# показатель -> (формула ВАК или None, теги формулы (физический baseline),
# доп. признаки ТОЛЬКО для остатка -- не входят в формулу, но модель
# вольна найти в них сигнал, fallback-константа на случай formula_fn=None
# и необученной модели -- иначе Interval(0,...) физически бессмысленен)
#
# AVT:F32 добавлена в feed_ebp_c как extra (Person 1, прогон полного
# цикла): раньше изменение F32 не влияло на EBP вообще (формулы ВАК её
# не содержат) при том что F32 -- активный рычаг оптимизатора, то есть
# треть управляющих переменных не делала ничего. F32 физически связана
# с тем же куском колонны (используется в формулах D15 и CFPP через
# F65/(F32+F30)), поэтому сигнал правдоподобен -- пусть остаток решает
# сам, а не гарантированно игнорирует.
_SPECS = {
    "feed_ebp_c": (vak.avt_240_350_ebp, vak.AVT_240_350["ebp_c"][1], ["AVT:F32"], 365.0),
    "feed_d15_kgm3": (vak.avt_240_350_d15, vak.AVT_240_350["d15_kgm3"][1], [], 871.0),
    # ИСПРАВЛЕНО (формулы_ВАК.xlsx, 2026-09-17): формула не была сломана,
    # была неверно транскрибирована (скобки), см. vak_formulas.py. Больше
    # не formula_fn=None -- своя формула и свои теги, не заимствованные у EBP.
    "feed_cfpp_c": (vak.avt_240_350_cfpp, vak.AVT_240_350["cfpp_c"][1], [], -5.0),
    "feed_flash_c": (None, vak.AVT_240_350["ebp_c"][1], [], 68.0),
}


# Показатели, у которых формула ВАК НЕ используется как baseline.
#
# feed_ebp_c (H-O2, SULFUR_HYPOTHESES.md, 2026-09-22). Формула
# транскрибирована верно и теги сопоставлены верно -- проверено против
# формулы_ВАК.xlsx построчно. Дефект в её ОБУСЛОВЛЕННОСТИ: она содержит
# -14.08235*T37 и +14.60206*T58, вклады -857.67 и +851.72, в сумме +3.38.
# Слагаемые по 860 гасят друг друга, но только В СРЕДНЕМ: corr(T37,T58)
# = 0.62, а не ~1, и расхождение этих температур на 1 C стоит ~14 C на
# выходе. Итог: std формулы 186.88 при std самой метки 10.62 -- формула
# в 18 раз шумнее величины, которую описывает (catastrophic cancellation).
# Остаток тратил всю ёмкость на то, чтобы её гасить.
#
# Замер (walk-forward, 5 блоков, против константы = среднее трейна):
#     baseline = формула              RMSE 32.335   skill -328.0%
#     baseline = формула, обрезанная  RMSE 24.442   skill -223.5%
#     БЕЗ формулы, чистый ML          RMSE  8.203   skill   -8.6%
#
# Формула при этом НЕ выбрасывается: её входные теги остаются признаками
# остатка, и scripts/train_avt.py по-прежнему выбирает ею точку отбора
# ЛИМС (corr 0.206 слабая, но точку определяет верно -- у чистого нуля
# корреляции нет вовсе, и выбор точки становится случайным, как это
# видно у feed_flash_c).
_NO_FORMULA_BASELINE = {"feed_ebp_c"}


def _baseline_fn(out: str, fn):
    """Формула как baseline -- или None, если она вредит (см. выше)."""
    return None if out in _NO_FORMULA_BASELINE else fn


class AVTModel(BaseQualityModel):
    """
    Композиция из четырёх FormulaPlusResidual, по одному на показатель.

    Без обученного артефакта (fit()/load() не вызывались) каждый
    показатель работает в режиме "только формула": предсказание точное
    там, где формула точна, интервал намеренно широкий (+-8), чтобы
    Gate не поверил непроверенному числу больше, чем оно того стоит.
    Ровно так демо работает уже сегодня, без обучения -- это осознанное
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
                # monotone constraints: см. base.EXPECTED_SIGNS и находку
                # Person 1 (feed_ebp_c немонотонен по активному рычагу
                # AVT:F30 -- переобучение остатка у края локальной
                # плотности данных). Считается для feature_cols (tags+extra),
                # не только formula_tags, чтобы extra-признаки без знака
                # (F32) корректно получили 0, а не выпали из вектора.
                monotone=monotone_vector(tags + extra, out),
            )
            for out, (fn, tags, extra, fb) in _SPECS.items()
        }

    # ------------------------------------------------------------------
    def predict(self, features: Dict[str, float]) -> Dict[str, Interval]:
        return {out: model.predict_one(features) for out, model in self._models.items()}

    # ------------------------------------------------------------------
    def fit(self, tables: Dict[str, "tuple"]) -> "AVTModel":
        """
        TODO(Person 3, следующий шаг после первого обучения): вызывается
        из scripts/train_avt.py, не напрямую. tables[output] = (X, y),
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
