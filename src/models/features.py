"""
Признаки для моделей качества.

Здесь живёт всё, что превращает сырую телеметрию 10-минутного шага
в признаки, на которых модель реально может учиться.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Стандартные окна в часах. Шаг телеметрии 10 минут, то есть
# 1 час = 6 точек, 12 часов = 72 точки.
DEFAULT_LAGS_H: Sequence[float] = (1, 3, 6, 12)
POINTS_PER_HOUR = 6


def add_lags(
    df: pd.DataFrame,
    cols: List[str],
    lags_h: Sequence[float] = DEFAULT_LAGS_H,
) -> pd.DataFrame:
    """Лаговые признаки. Только назад по времени, никогда вперёд."""
    out = df.copy()
    for c in cols:
        for h in lags_h:
            out[f"{c}__lag{h}h"] = df[c].shift(int(h * POINTS_PER_HOUR))
    return out


def add_rolling(
    df: pd.DataFrame,
    cols: List[str],
    windows_h: Sequence[float] = DEFAULT_LAGS_H,
) -> pd.DataFrame:
    """
    Скользящие средние и стандартные отклонения.

    Среднее сглаживает шум прибора. Отклонение говорит, насколько
    режим был устойчив - это хороший признак для оценки уверенности.
    """
    out = df.copy()
    for c in cols:
        for h in windows_h:
            w = int(h * POINTS_PER_HOUR)
            out[f"{c}__mean{h}h"] = df[c].rolling(w, min_periods=w // 2).mean()
            out[f"{c}__std{h}h"] = df[c].rolling(w, min_periods=w // 2).std()
    return out


def wabt(t_in: pd.Series, t_out: pd.Series) -> pd.Series:
    """
    Средневзвешенная температура слоя катализатора.

        WABT = T_вход + 2/3 * (T_выход - T_вход)

    Стандартная отраслевая формула для адиабатического реактора
    гидроочистки. Реакция экзотермическая, температура растёт по слою,
    поэтому среднее арифметическое входа и выхода занижает реальную
    температуру катализатора.

    Практическое значение: реактор стабилизируется, если WABT держат
    постоянной при меняющихся условиях.
    """
    return t_in + (2.0 / 3.0) * (t_out - t_in)


def catalyst_age_days_scalar(ts, cycle_start: str = "2023-01-01") -> float:
    """Версия catalyst_age_days для одного снимка времени (runtime, не обучение)."""
    return float((pd.Timestamp(ts) - pd.Timestamp(cycle_start)).days)


def arrhenius_term(temp_celsius, ea_kj_mol: float = 55.0):
    """
    exp(-Ea / (R*T)), T в Кельвинах. Из research.pdf: Ea 47.2-66.1 кДж/моль
    для HDS, 55 -- середина диапазона. Признак для LightGBM: должен
    линеаризовать то, что для сырой температуры нелинейно (Аррениус).

    Работает и со скаляром, и с pd.Series/np.array (векторизованно) --
    НЕ приводить к float() принудительно, иначе падает на Series
    (H-A2, SULFUR_HYPOTHESES.md, найдено при первом использовании на
    векторе телеметрии).
    """
    R = 8.314e-3  # кДж/(моль*К)
    T_k = temp_celsius + 273.15
    return np.exp(-ea_kj_mol / (R * T_k))


def catalyst_age_days(index: pd.DatetimeIndex, cycle_start: str = "2023-01-01") -> pd.Series:
    """
    Возраст катализатора в сутках.

    ДОПУЩЕНИЕ: реальная дата начала цикла в пакете не выдана.
    Принимается 2023-01-01. Признак всё равно полезен как тренд:
    катализатор садится, требуемая температура растёт.
    """
    start = pd.Timestamp(cycle_start)
    return pd.Series((index - start).days, index=index, name="catalyst_age_days")


def normalized_temperature(
    measured_temp: pd.Series,
    predicted_temp_for_target: pd.Series,
) -> pd.Series:
    """
    Нормализованная температура - прокси деактивации катализатора.

    Идея: сколько градусов нужно, чтобы получить фиксированную серу
    (например 8 мг/кг) при текущем расходе и качестве сырья.
    Её рост во времени и есть потеря активности катализатора.
    Наклон этого ряда - скорость деактивации, градусов в месяц.

    Это то, что агент надёжности вернёт как severity_index.
    """
    return measured_temp - predicted_temp_for_target


def lag_by_feed_rate(feed_rate: float, base_lag_h: float = 4.0,
                     ref_feed: float = 256.0) -> float:
    """
    Оценка запаздывания отклика, зависящая от нагрузки.

    Чем выше расход сырья, тем меньше время пребывания в реакторе
    и тем быстрее изменение температуры доходит до продукта.

    base_lag_h подбирается по кросс-корреляции ПАК-серы
    с температурой отдельно по квартилям расхода.
    Пока это допущение, а не измеренная величина.
    """
    if feed_rate is None or feed_rate <= 0:
        return base_lag_h
    return float(base_lag_h * ref_feed / feed_rate)


def time_split(df: pd.DataFrame, cutoff: str = "2025-12-31"):
    """
    Сплит строго по времени.

    Случайное перемешивание строк временного ряда запрещено ТЗ:
    оно даёт утечку информации из будущего в обучение.
    Функция существует, чтобы никто случайно не вызвал train_test_split.
    """
    c = pd.Timestamp(cutoff)
    return df[df.index <= c], df[df.index > c]


# мгновенные признаки серы: одна функция для обучения и для работы системы
GO_INSTANT_BASE = ["242000:T5", "242000:T6", "242000:Q20", "242000:F25", "242000:F9"]
GO_INSTANT_NAMES = ["242000:T5_T6_quench_delta", "242000:h2_oil_ratio",
                    "242000:arrhenius_t5", "242000:q20_x_arrhenius"]


def add_go_instant(df: pd.DataFrame) -> pd.DataFrame:
    """Квенч-дельта, водород/сырьё, аррениусовский член и его произведение с серой сырья."""
    out = df.copy()
    out["242000:T5_T6_quench_delta"] = df["242000:T5"] - df["242000:T6"]
    out["242000:h2_oil_ratio"] = df["242000:F25"] / df["242000:F9"].replace(0, float("nan"))
    out["242000:arrhenius_t5"] = arrhenius_term(df["242000:T5"])
    out["242000:q20_x_arrhenius"] = df["242000:Q20"] * out["242000:arrhenius_t5"]
    return out


def go_instant_features(raw: Dict[str, Optional[float]]) -> Dict[str, float]:
    """То же для одной точки: словарь тегов в словарь признаков без пропусков."""
    row = pd.DataFrame([{t: raw.get(t) for t in GO_INSTANT_BASE}], dtype=float)
    last = add_go_instant(row).iloc[0]
    return {n: float(last[n]) for n in GO_INSTANT_NAMES if pd.notna(last[n])}
