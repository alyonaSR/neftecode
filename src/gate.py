"""
L3. Жёсткий фильтр.
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Union

from .contracts import Candidate, GateVerdict, Interval, ProcessState
from .data.tags import manipulated_vars, quality_specs, load_config


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


class ConstraintGate:
    def __init__(self):
        self.specs = quality_specs()
        self.mvars = manipulated_vars()
        self.blending = load_config("constraints")["blending"]

    def check(
        self,
        candidate: Candidate,
        state: ProcessState,
        allowed_ranges: Optional[Dict[str, List[float]]] = None,
        blend_fractions: Optional[Union[Mapping[str, float], Sequence[float]]] = None,
    ) -> GateVerdict:
        violated: List[str] = []
        warnings: List[str] = []
        margins: Dict[str, float] = {}
        checked: List[str] = []

        self._check_quality(candidate, violated, warnings, margins, checked)
        self._check_ranges(candidate, state, allowed_ranges, violated, margins, checked)
        self._check_steps(candidate, violated, margins, checked)
        if blend_fractions is not None:
            self._check_blending(blend_fractions, violated, margins, checked)

        return GateVerdict(
            candidate_id=candidate.candidate_id,
            passed=len(violated) == 0,
            violated=violated,
            warnings=warnings,
            margins=margins,
            checked=checked,
        )

    # ------------------------------------------------------------------
    def _check_quality(self, cand, violated, warnings, margins, checked):
        """
        Спецификация проверяется по КОНСЕРВАТИВНОЙ границе интервала.

        Для серы: hi, а не mean. Прогноз mean=9.1 при лимите 10 выглядит
        безопасно, но hi=10.4 означает реальный риск нарушения.
        Проверка по mean - это ошибка, которая стоит всей задачи.
        """
        for param, spec in self.specs.items():
            hard = spec.get("source") == "spec"
            bucket = violated if hard else warnings

            iv: Interval = cand.predicted.get(param)
            if iv is None:
                bucket.append(f"{param}: прогноз отсутствует, проверка невозможна")
                continue

            limit = float(spec["limit"])
            value = iv.hi if spec["check_on"] == "hi" else iv.lo
            if not _is_number(value):
                bucket.append(
                    f"{param}: прогноз не число ({value}), проверка невозможна"
                )
                continue
            checked.append(
                f"{param} {spec['direction']} {limit} по {spec['check_on']} [{spec['source']}]"
            )

            if spec["direction"] == "max":
                margin = limit - value
            else:
                margin = value - limit

            margins[param] = round(margin, 3)
            if margin < 0:
                bucket.append(
                    f"{param}: {value:.2f} против лимита {limit:.2f} "
                    f"(проверка по {spec['check_on']}), "
                    + ("нарушение " if hard else "отклонение от допущения ")
                    + f"{abs(margin):.2f}"
                )

    # ------------------------------------------------------------------
    def _check_ranges(self, cand, state, allowed_ranges, violated, margins, checked):
        allowed_ranges = allowed_ranges or {}
        for tag, delta in cand.deltas.items():
            spec = self.mvars.get(tag)
            if spec is None:
                violated.append(
                    f"{tag}: не входит в список управляемых переменных, "
                    f"допустимый диапазон не задан"
                )
                continue

            cur = state.tag(tag)
            if cur is None:
                violated.append(f"{tag}: текущее значение неизвестно")
                continue
            if not _is_number(delta):
                violated.append(f"{tag}: изменение не число ({delta})")
                continue
            new = cur + delta

            lo, hi = spec["range"]
            src = spec["source"]
            if tag in allowed_ranges:
                alo, ahi = allowed_ranges[tag]
                lo, hi = max(lo, alo), min(hi, ahi)

            checked.append(f"{tag} в [{lo}, {hi}] [{src}]")
            margin = min(new - lo, hi - new)
            margins[f"{tag}__range"] = round(margin, 3)
            if margin < 0:
                violated.append(
                    f"{tag}: {new:.2f} вне допустимого диапазона [{lo}, {hi}]"
                )

    def _check_steps(self, cand, violated, margins, checked):
        for tag, delta in cand.deltas.items():
            spec = self.mvars.get(tag)
            if spec is None or not _is_number(delta):
                continue
            max_step = float(spec["max_step"])
            checked.append(f"|delta {tag}| <= {max_step}")
            margin = max_step - abs(delta)
            margins[f"{tag}__step"] = round(margin, 3)
            if margin < 0:
                violated.append(
                    f"{tag}: шаг {delta:+.2f} больше допустимого {max_step}"
                )

    def _check_blending(self, fractions, violated, margins, checked):
        """
        Доли компонентов блендинга обязаны давать 100 процентов
        """
        values = list(fractions.values()) if isinstance(fractions, Mapping) else list(fractions)
        if not all(_is_number(f) for f in values):
            violated.append("доли компонентов блендинга не числа")
            return

        total = float(sum(values))
        target = float(self.blending["components_sum"])
        tol = float(self.blending["tolerance"])
        checked.append(f"сумма долей блендинга = {target}")
        margins["blend_sum"] = round(tol - abs(total - target), 9)
        if abs(total - target) > tol:
            violated.append(f"сумма долей блендинга {total:.6f} вместо {target}")
        if any(f < 0 for f in values):
            violated.append("отрицательная доля компонента блендинга")

        if not isinstance(fractions, Mapping):
            return

        src = self.blending.get("share_source", "assumption")
        for tag, share in fractions.items():
            window = self.blending.get("share_range", {}).get(tag)
            if window is None:
                continue
            lo, hi = float(window[0]), float(window[1])
            checked.append(f"доля {tag} в [{lo}, {hi}] [{src}]")
            margin = min(share - lo, hi - share)
            margins[f"{tag}__share"] = round(margin, 4)
            if margin < 0:
                violated.append(
                    f"{tag}: доля в пуле {share:.3f} вне окна [{lo}, {hi}]"
                )


def filter_passed(candidates, verdicts) -> List[Candidate]:
    ok = {v.candidate_id for v in verdicts if v.passed}
    return [c for c in candidates if c.candidate_id in ok]
