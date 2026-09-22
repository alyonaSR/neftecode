"""Агент самоконтроля: сверяет интервал серы с лабораторией и подстраивает его ширину."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, List, Optional, Tuple

import numpy as np

from ..contracts import Interval, ProcessState, QualityAssess
from ..data.tags import load_config

PARAM = "sulfur_mgkg"


@dataclass
class ModelHealthAssess:
    """Здоровье прогноза серы."""
    status: str                       # ok | degraded | failed
    coverage: Optional[float]         # доля попаданий в окне
    n_obs: int
    scale: float                      # множитель ширины интервала серы
    message: str
    last_checks: List[dict] = field(default_factory=list)


class ModelMonitorAgent:
    """Хранит прогнозы серы и сверяет их с анализами ЛИМС."""
    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or load_config("model_monitor")
        self.scale = 1.0
        self._forecasts: Deque[Tuple[datetime, float, float]] = deque(maxlen=100)
        self._hits: Deque[bool] = deque(maxlen=int(self.cfg["window"]))
        self._checks: Deque[dict] = deque(maxlen=int(self.cfg["window"]))
        self._last_lims_ts: Optional[datetime] = None

    def observe(self, state: ProcessState) -> None:
        """Сверяет новый анализ серы с прогнозом и обновляет масштаб."""
        m = state.lims.get(PARAM)
        if m is None or m.ts is None or m.value is None:
            return
        if self._last_lims_ts is not None and m.ts <= self._last_lims_ts:
            return
        self._last_lims_ts = m.ts

        fc = self._forecast_before(m.ts)
        if fc is None:
            return
        f_ts, mean, hi = fc
        hit = m.value <= hi
        self._hits.append(hit)
        self._checks.append({"lims_ts": m.ts, "forecast_ts": f_ts, "lims": m.value,
                             "mean": round(mean, 3), "hi": round(hi, 3), "hit": hit})
        self._update_scale(hit)

    def apply(self, iv: Interval) -> Interval:
        """Интервал, растянутый вокруг центра на текущий масштаб."""
        return Interval(iv.mean, iv.mean - self.scale * (iv.mean - iv.lo),
                        iv.mean + self.scale * (iv.hi - iv.mean))

    def record(self, state: ProcessState, q: QualityAssess) -> None:
        """Запоминает прогноз серы для будущей сверки."""
        iv = q.predictions.get(PARAM)
        if iv is None:
            return
        self._forecasts.append((state.ts, float(iv.mean), float(iv.hi)))

    def warm_up(self, quality, telemetry, lims, pak, until: datetime, n_labs: int = 90) -> int:
        """Прокручивает последние анализы серы до until. Возвращает число сверок."""
        from ..data.loaders import TARGET_POINT
        from ..data.state_builder import build_state

        labs = lims[(lims.sample_point == TARGET_POINT) & (lims.param == PARAM)
                    & (~lims.outlier) & (lims.ts < until)].sort_values("ts").tail(n_labs)
        saved = getattr(quality, "sulfur_scale", 1.0)
        quality.sulfur_scale = 1.0
        try:
            for ts in labs.ts:
                state = build_state(ts - timedelta(minutes=10), telemetry, lims, pak)
                self.observe(state)
                iv = quality.assess(state).predictions.get(PARAM)
                if iv is not None:
                    adj = self.apply(iv)
                    self._forecasts.append((state.ts, float(adj.mean), float(adj.hi)))
            self.observe(build_state(until, telemetry, lims, pak))
        finally:
            quality.sulfur_scale = saved
        return len(self._hits)

    def assess(self) -> ModelHealthAssess:
        """Статус по скользящему окну последних сверок."""
        n = len(self._hits)
        if n < int(self.cfg["min_obs"]):
            return ModelHealthAssess("ok", None, n, round(self.scale, 3),
                                     f"мало сверок с лабораторией ({n}), статус по умолчанию",
                                     list(self._checks))
        cov = float(np.mean(self._hits))
        target = float(self.cfg["target_coverage"])
        if cov < float(self.cfg["failed_below"]):
            status = "failed"
            msg = (f"интервал серы накрыл лабораторию в {cov:.0%} из {n} последних анализов "
                   f"при цели {target:.0%}: прогнозу доверять нельзя")
        elif cov < float(self.cfg["degraded_below"]):
            status = "degraded"
            msg = (f"покрытие {cov:.0%} из {n} анализов ниже цели {target:.0%}: "
                   f"интервал расширен в {self.scale:.1f} раза")
        elif self.scale >= float(self.cfg["degraded_scale"]):
            status = "degraded"
            msg = (f"прогноз серы ошибается сильнее обычного: интервал расширен "
                   f"в {self.scale:.1f} раза, покрытие {cov:.0%} из {n} анализов")
        else:
            status = "ok"
            msg = f"покрытие {cov:.0%} из {n} анализов, масштаб интервала {self.scale:.2f}"
        return ModelHealthAssess(status, round(cov, 3), n, round(self.scale, 3), msg,
                                 list(self._checks))

    def _forecast_before(self, lims_ts: datetime) -> Optional[Tuple[datetime, float, float]]:
        """Последний прогноз не позже пробы и не старше match_max_lag_min."""
        earliest = lims_ts - timedelta(minutes=float(self.cfg["match_max_lag_min"]))
        for f_ts, mean, hi in reversed(self._forecasts):
            if f_ts > lims_ts:
                continue
            return (f_ts, mean, hi) if f_ts >= earliest else None
        return None

    def _update_scale(self, hit: bool) -> None:
        """Промах расширяет интервал, попадание сужает."""
        miss = 0.0 if hit else 1.0
        alpha = 1.0 - float(self.cfg["target_coverage"])
        k = self.scale * np.exp(float(self.cfg["step"]) * (miss - alpha))
        self.scale = float(np.clip(k, self.cfg["scale_min"], self.cfg["scale_max"]))