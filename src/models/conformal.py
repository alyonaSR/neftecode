"""
Split conformal prediction + Adaptive Conformal Inference.

Реализует то, что описано
в research.pdf ("Soft-сенсоры и гарантии безопасности через конформное
прогнозирование") и было заявлено в README: "conformal
prediction вместо эвристического интервала в QualityAgent".

ЗАМЕНЯЕТ в FormulaPlusResidual эмпирические 10/90 перцентили остатка
(наивный np.quantile, БЕЗ поправки на конечный размер выборки, значит
без доказанной гарантии покрытия) на:

  1. Split conformal с точной формулой квантиля для конечной выборки n
     (research.pdf, шаг 4 процедуры):
         q_hat = Quantile(scores, ceil((n+1)(1-alpha)) / n)

  2. Adaptive Conformal Inference (Gibbs & Candes 2021, research.pdf
     ссылки 11/14) -- онлайн-подстройка порога под дрейф: если факт
     регулярно пробивает интервал, alpha_t уменьшается (интервал
     расширяется), если система стабильно перекрывает факт с запасом --
     alpha_t растёт (интервал сужается). Отдельно для нижней и верхней
     стороны, потому что Gate проверяет их независимо и по разным
     показателям (сера -- max по hi, вспышка -- min по lo).

ЧЕСТНО, важно понимать: конформный интервал при ТОМ ЖЕ alpha не может
быть уже, чем корректная оценка на тех же данных -- это гарантия
покрытия, не трюк для сужения. Если раньше эвристика (10/90 перцентиль)
давала интервал уже, это не значило, что модель точнее -- значило, что
заявленное покрытие не было доказано. По умолчанию здесь alpha=0.10
(90% одностороннее) -- ДОПУЩЕНИЕ, компромисс между строгостью
research.pdf (95%) и тем, чтобы не взорвать ширину интервала вдвое на
старте; поднять до alpha=0.05 -- once-строчная правка при вызове fit().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

ALPHA_DEFAULT = 0.10          # 90% одностороннее покрытие, ДОПУЩЕНИЕ
ACI_GAMMA_DEFAULT = 0.02      # скорость онлайн-адаптации alpha_t, ДОПУЩЕНИЕ
ACI_ALPHA_MIN = 0.01
ACI_ALPHA_MAX = 0.5


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Верхний (1 - alpha) квантиль с поправкой на конечный размер выборки.

        q_hat = Quantile(scores, ceil((n+1)(1-alpha)) / n)

    research.pdf, "Процедура построения конформного soft-сенсора", шаг 4.
    При n -> inf уровень стремится к (1-alpha); без этой поправки
    формальная гарантия покрытия не доказана для конечной выборки.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        raise ValueError("Пустая калибровочная выборка")
    level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha))) / n)
    return float(np.quantile(scores, level))


@dataclass
class ConformalResidualBounds:
    """
    Двусторонний split conformal интервал остатка (y - pred) с
    независимой онлайн-адаптацией (ACI) верхнего и нижнего порога.

    Score -- НЕ |y - pred| (дало бы симметричный интервал), а разнесённый
    по сторонам остаток: для верхней границы -- (y - pred) напрямую
    (насколько факт выше прогноза), для нижней -- -(y - pred) (насколько
    факт ниже). Каждая сторона получает собственную калибровку вместо
    одного симметричного бюджета на двоих -- ровно то, что нужно Gate:
    сера проверяется только по hi, вспышка только по lo.
    """
    alpha: float = ALPHA_DEFAULT
    gamma: float = ACI_GAMMA_DEFAULT

    def __post_init__(self):
        self._alpha_hi = self.alpha
        self._alpha_lo = self.alpha
        self._hi_scores: List[float] = []
        self._lo_scores: List[float] = []

    # ------------------------------------------------------------------
    def fit(self, residuals: np.ndarray) -> "ConformalResidualBounds":
        """residuals = y_calib - pred_calib, хронологический calib-хвост."""
        r = np.asarray(residuals, dtype=float)
        self._hi_scores = list(r)          # насколько факт ВЫШЕ прогноза
        self._lo_scores = list(-r)         # насколько факт НИЖЕ прогноза
        self._alpha_hi = self.alpha
        self._alpha_lo = self.alpha
        return self

    # ------------------------------------------------------------------
    def bounds(self) -> Tuple[float, float]:
        """(offset_lo, offset_hi) -- прибавляются к точечному прогнозу."""
        q_hi = finite_sample_quantile(np.array(self._hi_scores), self._alpha_hi)
        q_lo = finite_sample_quantile(np.array(self._lo_scores), self._alpha_lo)
        return -q_lo, q_hi

    # ------------------------------------------------------------------
    def update(self, residual: float) -> None:
        """
        Шаг Adaptive Conformal Inference (Gibbs & Candes 2021).

        Вызывается, когда пришёл РЕАЛЬНЫЙ факт (новый анализ ЛИМС) и
        известен residual = y_true - прогноз_на_тот_момент. Раздельно
        штрафует/поощряет верхний и нижний порог: пробой сверху сужает
        alpha_hi (значит следующая верхняя граница станет шире), пробой
        снизу -- симметрично для нижней сторона. В спокойные периоды без
        пробоев обе стороны постепенно сужаются -- "сужать интервал в
        периоды стабильной работы и расширять его при росте
        неопределённости" (research.pdf).

        НЕ вызывается автоматически ни из какого продакшен-цикла
        -- для этого нужен живой поток решений с обратной связью по
        новым анализам ЛИМС, а это авторизация Orchestrator.
        Метод протестирован отдельно (tests/test_conformal.py) и готов
        быть подключённым, когда появится такой цикл.
        """
        q_lo, q_hi = self.bounds()
        miss_hi = 1.0 if residual > q_hi else 0.0
        miss_lo = 1.0 if -residual > q_lo else 0.0
        self._alpha_hi = float(np.clip(
            self._alpha_hi + self.gamma * (self.alpha - miss_hi), ACI_ALPHA_MIN, ACI_ALPHA_MAX))
        self._alpha_lo = float(np.clip(
            self._alpha_lo + self.gamma * (self.alpha - miss_lo), ACI_ALPHA_MIN, ACI_ALPHA_MAX))
        self._hi_scores.append(float(residual))
        self._lo_scores.append(float(-residual))

    # ------------------------------------------------------------------
    def to_state(self) -> dict:
        return {
            "alpha": self.alpha, "gamma": self.gamma,
            "alpha_hi": self._alpha_hi, "alpha_lo": self._alpha_lo,
            "hi_scores": self._hi_scores, "lo_scores": self._lo_scores,
        }

    @classmethod
    def from_state(cls, state: dict) -> "ConformalResidualBounds":
        obj = cls(alpha=state.get("alpha", ALPHA_DEFAULT),
                   gamma=state.get("gamma", ACI_GAMMA_DEFAULT))
        obj._alpha_hi = state.get("alpha_hi", obj.alpha)
        obj._alpha_lo = state.get("alpha_lo", obj.alpha)
        obj._hi_scores = list(state.get("hi_scores", []))
        obj._lo_scores = list(state.get("lo_scores", []))
        return obj
