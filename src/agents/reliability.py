"""Агент надёжности: тяжесть режима и allowed_ranges для оптимизатора. Пороги в config/reliability.yaml."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..contracts import ProcessState, ReliabilityAssess
from ..data.tags import load_config, manipulated_vars

# доля диапазона температур, отрезаемая сверху при high / elevated
NARROW_AT_HIGH = 0.25
NARROW_AT_ELEVATED = 0.10

# фактор попадает в объяснение, если его ранг не ниже
FACTOR_REPORT_PCTL = 0.85


class ReliabilityAgent:
    """telemetry нужен для факторов со скользящим окном; срез всегда .loc[:ts]."""
    def __init__(
        self,
        telemetry: Optional[pd.DataFrame] = None,
        quality_model=None,
        cycle_start_iso: str = "2023-01-01",
    ):
        self.telemetry = telemetry
        self.quality_model = quality_model
        self.cycle_start_iso = cycle_start_iso
        self.cfg = load_config("reliability")
        self.grid = np.array(self.cfg["grid"], dtype=float) / 100.0

    def assess(self, state: ProcessState) -> ReliabilityAssess:
        """Оценка тяжести режима. При останове класс shutdown."""
        raw = self._raw_factors(state)
        shutdown = raw.pop("shutdown", []) or []
        ranks = {
            k: self._pctl_rank(k, v)
            for k, v in raw.items()
            if v is not None and k in self.cfg["quantiles"]
        }

        severity = self._combine(ranks)
        cls = "shutdown" if shutdown else self._classify(severity)
        if shutdown:
            factors = [
                f"установка остановлена ({', '.join(shutdown)}): "
                "оценка тяжести режима неприменима, управляющие воздействия бессмысленны"
            ]
        else:
            factors = self._describe(raw, ranks)

        assumptions = [
            "severity = взвешенный перцентильный ранг режима в истории, "
            "пороги в config/reliability.yaml",
            "уставка DCS по температуре печи недоступна, прокси: скользящая "
            "медиана за 7 суток",
            f"начало цикла катализатора принято {self.cycle_start_iso}, допущение",
            "прямой разметки деактивации нет, используются прокси-метрики",
            "WABT не считается: вход реактора не опознан среди тегов 24-2000, "
            "используется температура выхода 242000:T5 как прокси",
            "модельные диапазоны из config/constraints.yaml помечены source: assumption",
        ]
        if self.quality_model is None:
            assumptions.append(
                "метрика деактивации катализатора не считается: нужна модель серы"
            )
        if self.telemetry is None:
            assumptions.append(
                "телеметрия не передана: факторы со скользящим окном пропущены"
            )

        return ReliabilityAssess(
            severity_index=round(severity, 3),
            severity_class=cls,
            factors=factors,
            allowed_ranges=self._allowed_ranges(ranks, severity),
            assumptions=assumptions,
        )

    def _raw_factors(self, state: ProcessState) -> Dict[str, Optional[float]]:
        """Сырые значения факторов на момент state.ts; None, если не рассчитан."""
        out: Dict[str, Optional[float]] = {
            # выход реактора: вход не опознан, WABT не считаем
            "reactor_temp": state.tag("242000:T5"),
            # расход сырья, т/ч
            "load": state.tag("242000:F9"),
            "thermal_stress": None,
            "pressure_instability": None,
            "temp_instability": None,
            "catalyst_degradation": self._catalyst_degradation(state),
            "shutdown": self._shutdown(state),
        }

        if self.telemetry is None or self.telemetry.empty:
            return out

        past = self.telemetry.loc[: pd.Timestamp(state.ts)]
        if past.empty:
            return out

        w_instab = self.cfg.get("instab_window", "1h")
        w_set = self.cfg.get("setpoint_window", "7D")

        # только по работающей печи
        t55 = past.get("AVT:T55")
        t_min = float(self.cfg.get("running_min_t55", 343.0))
        if t55 is not None and t55.notna().any():
            running = t55.where(t55 >= t_min)
            now = running.iloc[-1]
            target = running.rolling(w_set, min_periods=6).median().iloc[-1]
            if pd.notna(now) and pd.notna(target):
                out["thermal_stress"] = float(now - target)

        for key, col in (
            ("pressure_instability", "AVT:P67"),
            ("temp_instability", "242000:T5"),
        ):
            s = past.get(col)
            if s is not None and s.notna().any():
                v = s.rolling(w_instab, min_periods=3).std().iloc[-1]
                if pd.notna(v):
                    out[key] = float(v)

        return out

    def _shutdown(self, state: ProcessState) -> List[str]:
        """Остановленные установки по порогам running_min_*."""
        down: List[str] = []
        checks = (
            ("АВТ", "AVT:T55", "running_min_t55"),
            ("гидроочистка", "242000:T5", "running_min_t5"),
            ("гидроочистка", "242000:F9", "running_min_f9"),
        )
        for unit, tag, key in checks:
            v = state.tag(tag)
            thr = self.cfg.get(key)
            if v is not None and thr is not None and v < float(thr):
                if unit not in down:
                    down.append(unit)
        return down

    def _catalyst_degradation(self, state: ProcessState) -> Optional[float]:
        """Заглушка до появления модели серы."""
        if self.quality_model is None:
            return None
        raise NotImplementedError(
            "подключить обратную задачу, когда появится модель серы"
        )

    def _pctl_rank(self, name: str, value: float) -> float:
        """Перцентильный ранг значения в истории фактора, 0..1."""
        q = self.cfg["quantiles"].get(name)
        if not q:
            return 0.5
        return float(np.interp(value, np.array(q, dtype=float), self.grid))

    def _combine(self, ranks: Dict[str, float]) -> float:
        """Взвешенное среднее рангов."""
        weights = self.cfg["weights"]
        used = {k: w for k, w in weights.items() if k in ranks}
        if not used:
            return 0.5
        total = sum(used.values())
        return sum(ranks[k] * w for k, w in used.items()) / total

    def _classify(self, sev: float) -> str:
        """Класс по class_thresholds: normal, elevated или high."""
        th = self.cfg["class_thresholds"]
        if sev >= th["high"]:
            return "high"
        if sev >= th["elevated"]:
            return "elevated"
        return "normal"

    def _describe(self, raw: Dict, ranks: Dict) -> List[str]:
        """Текст по значимым и нерассчитанным факторам."""
        labels = {
            "reactor_temp": ("температура реактора", "{:.1f} degC"),
            "load": ("нагрузка по сырью", "{:.0f} т/ч"),
            "thermal_stress": ("тепловое напряжение печи", "{:+.1f} degC к целевой"),
            "pressure_instability": ("разброс давления за час", "{:.3f} кгс/см²"),
            "temp_instability": ("разброс температуры за час", "{:.2f} degC"),
        }
        out: List[str] = []
        for key, rank in sorted(ranks.items(), key=lambda x: -x[1]):
            if rank < FACTOR_REPORT_PCTL or key not in labels:
                continue
            name, fmt = labels[key]
            out.append(
                f"{name} {fmt.format(raw[key])}: выше {rank:.0%} исторических значений"
            )
        if not out:
            out.append("режим в пределах обычного исторического диапазона")

        missing = [k for k, v in raw.items() if v is None and k != "catalyst_degradation"]
        if missing:
            out.append(f"не рассчитаны факторы: {', '.join(missing)}")
        return out

    def _allowed_ranges(self, ranks: Dict[str, float], severity: float) -> Dict[str, List[float]]:
        """Диапазоны управляемых переменных, суженные по тяжести режима."""
        th = self.cfg["class_thresholds"]
        if severity >= th["high"]:
            cut = NARROW_AT_HIGH
        elif severity >= th["elevated"]:
            cut = NARROW_AT_ELEVATED
        else:
            cut = 0.0

        instab = max(
            ranks.get("pressure_instability", 0.0),
            ranks.get("temp_instability", 0.0),
        )
        symmetric = 0.10 if instab >= 0.90 else 0.0  # при нестабильности сужаем с обеих сторон

        out: Dict[str, List[float]] = {}
        for tag, spec in manipulated_vars().items():
            lo, hi = float(spec["range"][0]), float(spec["range"][1])
            span = hi - lo

            if cut and spec.get("units") == "degC":
                hi = lo + (1.0 - cut) * span

            if symmetric:
                lo += symmetric * span
                hi -= symmetric * span

            out[tag] = [round(lo, 2), round(hi, 2)]
        return out