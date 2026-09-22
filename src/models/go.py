"""
Модель установки гидроочистки 24-2000.

ЧТО ПРЕДСКАЗЫВАЕТ: качество ТОВАРНОГО дизельного топлива.
Главный показатель -- сера, по ней жёсткое ограничение 10 мг/кг.

СМЕНА АРХИТЕКТУРЫ: раньше модель работала в приращениях
(sulfur_anchor + d_go_temp_c + d_feed_tail_c). Это ломало саму
возможность использовать лаги/волатильность -- обучающих примеров
вида "что было бы, если бы я изменил T5 на +2" в истории нет, есть
только фактические траектории. Теперь GOModel предсказывает
АБСОЛЮТНОЕ значение по полному вектору признаков, тем же паттерном,
что уже AVTModel: agents/quality.py вызывает predict() дважды --
на текущем состоянии и на состоянии с предложенным изменением -- и
берёт разницу сам.

ПОЧЕМУ ЛАГИ КРИТИЧНЫ (аудит по данным):
  Сырой T5 (температура реактора) коррелирует с серой ЛИМС на -0.09 --
  почти ничего. Но T5 со сдвигом на 3-6ч -- уже -0.33, а ВОЛАТИЛЬНОСТЬ
  T5 за 3 часа (std3h) -- 0.45, сильнейший признак из всех проверенных
  (сильнее, чем любой сырой тег). Расход сырья (F26) и подача
  ВСГ (P24) не коррелируют почти никак ни в каком виде (~-0.02..-0.06)
  -- вероятно, оператор постоянно компенсирует их через T5, и сырой
  расход из-за этого не виден. Это ровно тот сценарий "контур
  регулирования маскирует физику", от которого предостерегает
  base.py.EXPECTED_SIGNS.

  Три "тега качества" 24-2000 (T6, W7, P13), подписанные в справочнике
  как связанные с серой, ПРОВЕРЕНЫ против настоящей ЛИМС/ПАК серы и НЕ
  являются рабочими прокси (corr -0.01..-0.18) -- не использовать как
  признак серы напрямую, справочник для них так же ненадёжен, как для
  T6 отдельно.

ЧТО ИЗВЕСТНО ИЗ РАЗВЕДКИ ДАННЫХ:
  Готовой ВАК-формулы на серу нет. Для cfpp_c формула есть и рабочая
  (24-2000:GODT:CFPP, исправлена организаторами на Q&A 15.09) --
  используется как baseline. Для flash_c и d15_kgm3 формул нет --
  осталась простая физическая аппроксимация от качества сырья АВТ
  (см. _APPROX ниже), калибровка на LightGBM для них не делалась.

Спецификации показателей ГОДТ: показатель -> (формула ВАК или None,
требуемые теги, fallback).

ПРО ПРИЗНАКИ СЕРЫ (история решений, не переоткрывать):

  catalyst_age_days УБРАН -- буквально функция календарного времени,
  переносил temporal drift между train и calib. Без него врождённое
  смещение упало с -0.564 до +0.076, в 7 раз.

  242000:Q20 ДОБАВЛЕН -- ПАК сера СЫРЬЯ, 189217 точек. Сырая корреляция
  с товарной серой всего 0.038, но у feed_d15_kgm3 она тоже была слабой,
  а в остатке дала 40%+ gain.

  242000:Q21 НЕ ДОБАВЛЯТЬ НИКОГДА -- это ПАК серы ТОВАРНОГО ДТ, то же
  физическое измерение, что целевая переменная. Прямая утечка ответа.

  242000:T5_T6_quench_delta ДОБАВЛЕН -- T5 и T6 коррелируют на 0.998 и
  почти взаимозаменяемы, но их РАЗНОСТЬ отражает межступенчатый
  водородный квенч и не избыточна с T5.

  242000:h2_oil_ratio ДОБАВЛЕН -- F25/F9, парциальное давление H2.

  242000:P8 ПРОВЕРЕН И ОТКЛОНЁН -- gain 42.7%, но RMSE вырос
  2.631->2.658 и интервал на 13%. Высокий gain на трейне сам по себе не
  доказательство пользы признака.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Optional

from ..contracts import Interval
from . import vak_formulas as vak
from .base import BaseQualityModel, monotone_vector
from .features import arrhenius_term
from .formula_residual import FormulaPlusResidual
_R_KJ_MOL_K = 8.314e-3
_SULFUR_EA_KJ_MOL = 55.0
_SULFUR_ARRHENIUS_A = 0.000298


def sulfur_arrhenius_baseline(m: Mapping) -> float:
    """
    Опорная линия серы: Аррениус с ЛИТЕРАТУРНЫМ Ea, не подогнанным.

    Обучение без формулы дало обратный знак (горячее -> больше серы):
    LightGBM выучил контур регулирования, а не физику -- температуру
    поднимали, когда сырьё было плохим. Поэтому знак и Ea берутся из
    research.pdf (47.2-66.1 кДж/моль, взята середина), по данным
    калибруется ТОЛЬКО предэкспоненциальный множитель A -- это не может
    воспроизвести confound.
    """
    t_k = m["242000:T5"] + 273.15
    return _SULFUR_ARRHENIUS_A * math.exp(_SULFUR_EA_KJ_MOL / (_R_KJ_MOL_K * t_k))


def flash_from_t18(m: Mapping) -> float:
    """
    ГИПОТЕЗА H-C1 (SULFUR_HYPOTHESES.md), ПОДТВЕРЖДЕНА 2026-09-19:
    242000:T18 подписан в новом файле тегов организаторов как "ВАК.
    Температура вспышки ГОДТ, аналитический показатель" -- уже готовый
    виртуальный анализатор, а не догадка. Проверено против ЛИМС flash_c
    (n=1515): corr=0.620, RMSE как прямой прогноз без остатка = 6.21.
    Прежняя заглушка (feed_flash_c, ~68.0 константа) давала RMSE=7.08.
    Identity-формула, остаток доучивает то, что T18 не объясняет.
    """
    return float(m["242000:T18"])


_SPECS = {
    "sulfur_mgkg": (sulfur_arrhenius_baseline, [
        "242000:T5", "242000:T5__lag3h", "242000:T5__lag6h",
        "242000:T5__std3h", "242000:T5__std6h",
        "242000:Q20", "242000:Q20__lag3h", "242000:Q20__lag6h",
        "242000:T5_T6_quench_delta", "242000:h2_oil_ratio",
        "242000:arrhenius_t5", "242000:q20_x_arrhenius",
        "feed_ebp_c", "feed_d15_kgm3",
    ], 8.5),
    "cfpp_c": (vak.godt_cfpp, vak.GODT["cfpp_c"][1], -5.0),
    "flash_c": (flash_from_t18, ["242000:T18"], 68.0),
}
_APPROX = {
    "d15_kgm3": lambda f: f.get("feed_d15_kgm3", 840.0) - 34.0,
}
_APPROX_HALF = {"d15_kgm3": 3.0}


class GOModel(BaseQualityModel):
    """
    sulfur_mgkg и cfpp_c -- FormulaPlusResidual (сера чистый ML, cfpp_c
    формула+остаток). flash_c/d15_kgm3 -- простая физическая
    аппроксимация от качества сырья АВТ, без обучения (см. докстринг модуля).

    required_features -- реальные ключи, которые agents/quality.py
    обязан положить в словарь перед вызовом predict(): часть -- сырые
    теги state.tag(...), часть (feed_*) -- уже посчитанные им самим
    из выхода AVTModel. Никакого anchor/delta больше нет -- вход и
    выход абсолютные.
    """
    outputs = ["sulfur_mgkg", "flash_c", "cfpp_c", "d15_kgm3"]
    _DERIVED_INPUTS = ["242000:T6", "242000:F25"]
    required_features = sorted(
        {t for _, tags, _ in _SPECS.values() for t in tags} | set(_DERIVED_INPUTS)
    )
    model_id = "go_formula_residual_v3"

    def __init__(self, models: Optional[Dict[str, FormulaPlusResidual]] = None):
        self._models = models or {
            out: FormulaPlusResidual(
                name=out, formula_fn=fn, formula_tags=tags, feature_cols=tags,
                fallback_mean=fb,
                monotone=monotone_vector(tags, "sulfur_mgkg") if out == "sulfur_mgkg" else None,
            )
            for out, (fn, tags, fb) in _SPECS.items()
        }

    @staticmethod
    def derive_features(features: Dict[str, float]) -> Dict[str, float]:
        """
        Досчитывает производные признаки из сырых тегов.

        ЗАЧЕМ. Фичеинжиниринг серы жил ТОЛЬКО в обучающих скриптах
        (build_sulfur_table в train_go.py и experiments_sulfur.py), а путь
        инференса -- state_builder -> quality.py -> GOModel.predict --
        клал в модель одни сырые теги. Производные признаки приходили как
        NaN, то есть модель работала без них. Классический training/serving
        skew: обучали на одном наборе, предсказывали на другом.

        Замер до исправления (2026-09-22): из 12 телеметрийных признаков
        серы на реальном пути ОТСУТСТВОВАЛО 10, доходили только T5 и Q20.
        В демо-сценарии прогноз выходил 3.65 мг/кг при реальной медиане
        8.6 -- Gate видел несуществующий запас до лимита 10.

        Лаги и скользящие (T5__lag3h, Q20__lag6h, T5__std3h, ...) отсюда
        НЕ восстановить -- они требуют истории, а на вход приходит срез в
        одной точке времени. Их должен подавать state_builder; здесь они
        остаются как есть.

        Функция чистая: не трогает то, что уже передано (если вызывающий
        посчитал признак сам -- его значение в приоритете), и не ломается
        на отсутствующем сырье.
        """
        f = dict(features)

        def val(key):
            v = f.get(key)
            return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)

        def put(key, value):
            if val(key) is None and value is not None:
                f[key] = value

        t5, t6 = val("242000:T5"), val("242000:T6")
        q20 = val("242000:Q20")
        f25, f9 = val("242000:F25"), val("242000:F9")

        if t5 is not None:
            put("242000:arrhenius_t5", float(arrhenius_term(t5)))
        if t5 is not None and t6 is not None:
            put("242000:T5_T6_quench_delta", t5 - t6)
        if f25 is not None and f9:
            put("242000:h2_oil_ratio", f25 / f9)
        arr = val("242000:arrhenius_t5")
        if q20 is not None and arr is not None:
            put("242000:q20_x_arrhenius", q20 * arr)
        return f

    def predict_effect(self, features_now: Dict[str, float],
                       features_new: Dict[str, float]) -> Dict[str, float]:
        """
        Насколько изменится качество, если перейти из режима `now` в `new`.
        Считается ТОЛЬКО по физической формуле, БЕЗ ML-остатка.

        ЗАЧЕМ ИМЕННО ТАК (H-L1, SULFUR_HYPOTHESES.md, 2026-09-22). Замер
        отклика на +2 C по T5 на отложенных данных:

            чистая формула Аррениуса:  -0.254 мг/кг, неверный знак у 0.0%
            формула + ML-остаток:      +0.197 мг/кг, неверный знак у 59.4%

        То есть ML-остаток не просто шумит, а УВЕРЕННО переворачивает знак:
        модель считает, что нагрев реактора ПОВЫШАЕТ серу. Физика требует
        обратного (выше T -> глубже HDS -> меньше серы).

        Почему это не ловится monotone_constraints: LightGBM ограничивает
        монотонность ПОКОЛОНОЧНО. Знак задан для 242000:T5, но производные
        от неё -- arrhenius_t5 (строго возрастает по T), q20_x_arrhenius,
        T5_T6_quench_delta -- знака не имеют, и ограничение обходится через
        них. Попытка зажать и их делает отклик нулевым (+0.011), а RMSE
        хуже (2.232 -> 2.501): получаем мёртвый рычаг вместо перевёрнутого.

        Поэтому разделение по схеме:
            УРОВЕНЬ  -- откуда мы стартуем, берётся из свежего замера
                        (ЛИМС/ПАК) в quality.py._sulfur_anchored;
            ЭФФЕКТ   -- насколько сдвинет действие, берётся ОТСЮДА.
        Прогноз кандидата = уровень + эффект. Точность уровня даёт прибор,
        физически верную чувствительность -- формула.
        """
        now = self.derive_features(features_now)
        new = self.derive_features(features_new)
        return {name: m.baseline(new) - m.baseline(now)
                for name, m in self._models.items()}

    def predict(self, features: Dict[str, float]) -> Dict[str, Interval]:
        features = self.derive_features(features)
        out = {name: model.predict_one(features) for name, model in self._models.items()}
        for name, fn in _APPROX.items():
            mean = float(fn(features))
            half = _APPROX_HALF[name]
            out[name] = Interval(mean, mean - half, mean + half)
        return out

    def fit(self, tables: Dict[str, "tuple"]) -> "GOModel":
        """Вызывается из scripts/train_go.py. tables[out] = (X, y)."""
        for out, model in self._models.items():
            if out in tables:
                X, y = tables[out]
                model.fit(X, y)
        return self

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({out: m.to_state() for out, m in self._models.items()}, path)

    @classmethod
    def load(cls, path: str) -> "GOModel":
        import joblib
        states = joblib.load(path)
        models = {
            out: FormulaPlusResidual.from_state(
                states[out], fn, tags, tags,
                monotone=monotone_vector(tags, "sulfur_mgkg") if out == "sulfur_mgkg" else None,
                fallback_mean=_fb,
            )
            for out, (fn, tags, _fb) in _SPECS.items()
        }
        return cls(models=models)
