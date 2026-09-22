"""
L4. Оркестратор.

Порядок одного цикла принятия решения ровно по разделу 1 ТЗ:
  1. получить состояние
  2. проверить полноту и актуальность данных
  3. оценить качество и риск выхода за спецификацию
  4. оценить тяжесть режима
  5. сформировать варианты
  6. отбросить недопустимые
  7. сравнить оставшиеся
  8. выбрать и объяснить, либо отказаться

ПРАВИЛО РАЗРЕШЕНИЯ КОНФЛИКТА ЦЕЛЕЙ:
  1) жёсткие ограничения (бинарно, решает Gate)
  2) достаточен ли запас по жёстким спекам (бинарно, порог target_margin)
  3) величина воздействия (меньше = лучше)
  4) тяжесть режима (меньше = лучше)
  5) экономика (меньше cost_proxy = лучше)

Величина воздействия стоит выше тяжести и экономики сознательно: если
запас уже достаточен, установку трогать не надо, и ТЗ прямо требует
показать период, где лишних управляющих действий не создаётся.
Если запаса не хватает, критерий (3) меняется на "больше запас" - см. _rank().

Экономика стоит последней сознательно. Ключевой принцип ТЗ:
недопустимый режим нельзя компенсировать более высокой
производительностью или меньшими затратами.

ЖЁСТКИЕ СПЕКИ БЕРУТСЯ ИЗ КОНФИГА, не зашиты в код: это те показатели
config/constraints.yaml, у которых source: spec. Сегодня такой один -
сера. Появится второй подтверждённый предел - оркестратор менять
не придётся.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .agents.model_monitor import ModelMonitorAgent
from .agents.optimizer import OptimizerAgent
from .agents.quality import QualityAgent
from .agents.reliability import ReliabilityAgent
from .blending import blend_fractions
from .contracts import (
    REFUSE,
    Candidate,
    DecisionTrace,
    GateVerdict,
    ProcessState,
    QualityAssess,
    Recommendation,
    ReliabilityAssess,
)
from .data.tags import load_config, manipulated_vars, quality_specs, refusal_rules
from .explain import render_explanation, summarize_freshness
from .gate import ConstraintGate, filter_passed

if TYPE_CHECKING:
    import pandas as pd


class Orchestrator:
    def __init__(
        self,
        quality: Optional[QualityAgent] = None,
        reliability: Optional[ReliabilityAgent] = None,
        optimizer: Optional[OptimizerAgent] = None,
        gate: Optional[ConstraintGate] = None,
        telemetry: Optional["pd.DataFrame"] = None,
        monitor: Optional[ModelMonitorAgent] = None,
    ):
        self.quality = quality or QualityAgent()
        self.reliability = reliability or ReliabilityAgent(telemetry=telemetry)
        self.optimizer = optimizer or OptimizerAgent(self.quality)
        self.gate = gate or ConstraintGate()
        self.monitor = monitor or ModelMonitorAgent()

        self.rules = refusal_rules()
        self.decision = load_config("constraints")["decision"]
        specs = quality_specs()
        self.hard_specs = {n: s for n, s in specs.items() if s.get("source") == "spec"}
        self.soft_specs = {n: s for n, s in specs.items() if n not in self.hard_specs}

    def run_cycle(self, state: ProcessState) -> DecisionTrace:
        # Порядок важен: сначала сверяем прошлые прогнозы с новым анализом
        # ЛИМС и получаем масштаб интервала, потом считаем прогноз этим
        # масштабом, и только потом запоминаем его для будущей сверки.
        self.monitor.observe(state)
        self.quality.sulfur_scale = self.monitor.scale
        q = self.quality.assess(state)
        self.monitor.record(state, q)
        health = self.monitor.assess()
        r = self.reliability.assess(state)

        candidates: List[Candidate] = []
        verdicts: List[GateVerdict] = []

        rec = self._data_refusal(state, q)
        # Модель систематически промахивается мимо лаборатории - значит
        # опираться на её прогноз нельзя, сколько бы ни был широк интервал.
        if rec is None and health.status == "failed":
            rec = self._refuse(state, q, health.message, [])

        if rec is None:
            candidates = self._drop_micro_moves(self.optimizer.propose(state, q, r))
            verdicts = [
                self.gate.check(c, state, r.allowed_ranges,
                                blend_fractions=blend_fractions(state, c.deltas))
                for c in candidates
            ]

            survivors = filter_passed(candidates, verdicts)
            if not survivors:
                survivors = self._recovery(candidates, verdicts)

            if survivors:
                best = self._rank(survivors, verdicts)[0]
                rec = self._build_recommendation(state, q, r, best, survivors, verdicts)
            else:
                rec = self._refuse(
                    state, q, self._infeasible_reason(candidates, verdicts),
                    verdicts, baseline=self._baseline(candidates, verdicts),
                )

        return DecisionTrace(state.ts, state, q, r, candidates, verdicts, rec,
                             model_health=health)

    def _data_refusal(
        self, state: ProcessState, q: QualityAssess
    ) -> Optional[Recommendation]:
        """
        Отказ по данным наступает, когда по жёсткой спеке НЕ ОСТАЛОСЬ НИ
        ОДНОГО пригодного измерения. ТЗ задаёт приоритет источников
        ЛИМС -> ПАК -> ВАК, то есть поточный анализатор - законный
        источник, а не отсутствие данных: при живом свежем ПАК и
        устаревшем ЛИМС система обязана работать, просто с меньшим
        доверием (его считает QualityAgent).
        """
        reasons: List[str] = []
        max_age = self.rules["max_lims_age_min"]

        for name in self.hard_specs:
            lims = state.lims.get(name)
            pak = state.pak.get(name)

            if self.rules["require_healthy_pak"] and pak is not None and not pak.healthy:
                reasons.append(
                    f"поточный анализатор {name} признан неисправным (залипание)"
                )

            if any(m is not None and m.is_usable(max_age) for m in (lims, pak)):
                continue

            reasons.append(
                f"нет пригодного измерения {name}: "
                + "; ".join([
                    self._source_status("лабораторный анализ", lims, max_age),
                    self._source_status("поточный анализатор", pak, max_age),
                ])
            )

        if q.confidence < self.rules["min_confidence"]:
            reasons.append(
                f"доверие к прогнозу {q.confidence:.2f} ниже порога "
                f"{self.rules['min_confidence']}"
            )

        if not reasons:
            return None

        return self._refuse(state, q, "; ".join(reasons), [])

    @staticmethod
    def _source_status(name: str, m, max_age_min: float) -> str:
        if m is None or m.value is None:
            return f"{name} отсутствует"
        if not m.healthy:
            return f"{name} признан неисправным"
        if m.age_min is None:
            return f"{name} без отметки времени"
        if m.age_min > max_age_min:
            return (f"{name} устарел: {m.age_min / 60:.0f} ч "
                    f"при пороге {max_age_min / 60:.0f} ч")
        return f"{name} пригоден"

    def _drop_micro_moves(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        action_deadband: движение меньше порога неотличимо от шума регулятора,
        и предлагать его оператору - значит тратить его доверие впустую.
        """
        dead = float(self.decision["action_deadband"])
        mvars = manipulated_vars()
        return [
            c for c in candidates
            if all(
                abs(d) < 1e-9 or abs(d) >= dead * float(mvars[t]["max_step"])
                for t, d in c.deltas.items()
            )
        ]

    def _recovery(
        self, candidates: List[Candidate], verdicts: List[GateVerdict]
    ) -> List[Candidate]:
        """
        Режим восстановления: текущий режим УЖЕ вне спецификации.

        Gate прав, отбраковывая все варианты - спецификация не выполняется
        ни в одном из них. Но отказ здесь означал бы, что система молчит
        ровно тогда, когда нужна: продукт уже вне спеки, а лучшее доступное
        действие существует и хуже не сделает. Поэтому допускаются
        варианты, которые УЛУЧШАЮТ запас по жёсткой спеке относительно
        бездействия и не нарушают больше ничего: ни диапазонов, ни шага,
        ни долей блендинга, ни остальных показателей качества.

        Спецификация при этом НЕ объявляется выполненной: отчёт говорит,
        что за один шаг она не восстанавливается, решение пересматривается
        на следующем цикле по свежим данным.
        """
        vmap = {v.candidate_id: v for v in verdicts}
        base = next((c for c in candidates if c.is_no_action), None)
        if base is None:
            return []

        base_margin = self._hard_margin(vmap[base.candidate_id])
        if base_margin is None or base_margin >= 0:
            return []

        out: List[Candidate] = []
        for c in candidates:
            v = vmap[c.candidate_id]
            margin = self._hard_margin(v)
            if margin is None or margin <= base_margin:
                continue
            if any(
                m < 0 for k, m in v.margins.items()
                if k not in self.hard_specs and k not in self.soft_specs
            ):
                continue
            out.append(c)
        return out

    def _hard_margin(self, v: GateVerdict) -> Optional[float]:
        """
        Худший запас по жёстким спекам, одно число для сравнения вариантов.
        """
        margins = [v.margins.get(name) for name in self.hard_specs]
        if not margins or any(m is None for m in margins):
            return None
        return min(margins)

    def _is_sufficient(self, v: GateVerdict) -> bool:
        """Запас по КАЖДОЙ жёсткой спеке не ниже целевого из конфига."""
        targets = self.decision["target_margin"]
        return all(
            v.margins.get(name) is not None
            and v.margins[name] >= float(targets.get(name, 0.0))
            for name in self.hard_specs
        )

    def _rank(
        self, survivors: List[Candidate], verdicts: List[GateVerdict]
    ) -> List[Candidate]:
        """
        Лексикографический выбор. Порядок критериев - это и есть политика системы.

        1) достаточен ли запас по жёстким спекам (бинарно)
        2) среди достаточных: МЕНЬШЕ ВОЗДЕЙСТВИЕ.
           Если режим уже безопасен, трогать установку не надо.
           Без этого правила система всегда уходит в максимальную severity
           ради лишнего запаса и нарушает требование ТЗ о том,
           что в устойчивом режиме лишних действий быть не должно.
           Если запаса не хватает, вместо воздействия сравнивается сам
           запас: больше = лучше.
        3) мягче режим (меньше severity_delta)
        4) дешевле (cost_proxy)

        Экономика последняя сознательно: недопустимый или более жёсткий режим
        нельзя оправдать выгодой.
        """
        vmap = {v.candidate_id: v for v in verdicts}
        mvars = manipulated_vars()

        def effort(c: Candidate) -> float:
            """Нормированная величина воздействия, 0 = ничего не трогаем."""
            return round(sum(
                abs(d) / float(mvars[t]["max_step"]) for t, d in c.deltas.items()
            ), 4)

        def key(c: Candidate) -> Tuple[int, float, float, float, float]:
            v = vmap[c.candidate_id]
            margin = self._hard_margin(v)
            if margin is None:
                return (2, 0.0, 0.0, 0.0, 0.0)
            if self._is_sufficient(v):
                return (0, 0.0, effort(c), c.severity_delta, c.cost_proxy)
            return (1, -round(margin, 3), effort(c), c.severity_delta, c.cost_proxy)

        return sorted(survivors, key=key)

    def _baseline(self, candidates, verdicts):
        """Вариант «ничего не менять» вместе с его вердиктом, если он есть."""
        vmap = {v.candidate_id: v for v in verdicts}
        base = next((c for c in candidates if c.is_no_action), None)
        if base is None or base.candidate_id not in vmap:
            return None
        return base, vmap[base.candidate_id]

    def _infeasible_reason(self, candidates, verdicts) -> str:
        """
        Отказ по невыполнимости. Оператору мало знать, что вариантов нет:
        ему нужно, ЧТО происходит с продуктом прямо сейчас, если не
        вмешиваться, и насколько далеко ближайшее решение.
        """
        head = (f"ни один из {len(candidates)} рассмотренных вариантов "
                f"не проходит жёсткие ограничения")
        baseline = self._baseline(candidates, verdicts)
        if baseline is None:
            return head

        _, v = baseline
        if not v.violated:
            return head
        return f"{head}; при бездействии нарушается: {'; '.join(v.violated)}"

    def _refuse(
        self,
        state: ProcessState,
        q: QualityAssess,
        reason: str,
        verdicts: List[GateVerdict],
        baseline=None,
    ) -> Recommendation:
        effect: Dict[str, Any] = {}
        checked: List[str] = []
        violated: List[str] = []
        warned: List[str] = []
        if baseline is not None:
            candidate, verdict = baseline
            effect = self._quality_effect(candidate, verdict)
            checked, violated, warned = verdict.checked, verdict.violated, verdict.warnings

        return Recommendation(
            ts=state.ts,
            action=REFUSE,
            reason=reason,
            expected_effect=effect,
            checks_passed=checked,
            checks_failed=violated,
            checks_warned=warned,
            confidence=q.confidence,
            explanation=(
                "Надёжной рекомендации нет. " + reason[:1].upper() + reason[1:] + "."
                + self._rejection_summary(verdicts)
                + " Рискованное управляющее воздействие в этих условиях не выдаётся."
            ),
            alternatives=[],
            data_freshness=summarize_freshness(state),
        )

    def _rejection_summary(self, verdicts: List[GateVerdict]) -> str:
        """
        Почему отбраковали ВСЕХ, а не три случайных примера.

        Оператору важны две вещи: какое ограничение оказалось узким местом
        и насколько близко было ближайшее решение. Перечислять нарушения
        поштучно бессмысленно - их тысяча и они однотипные.
        """
        if not verdicts:
            return ""

        counts: Dict[str, int] = {}
        for v in verdicts:
            for name in self.hard_specs:
                if any(text.startswith(f"{name}:") for text in v.violated):
                    counts[name] = counts.get(name, 0) + 1

        parts = [f" Рассмотрено вариантов: {len(verdicts)}."]
        for name, n in sorted(counts.items(), key=lambda x: -x[1]):
            best_margin = max(
                (m for m in (v.margins.get(name) for v in verdicts) if m is not None),
                default=None,
            )
            near = (f", ближайший промах {abs(best_margin):.2f}"
                    if best_margin is not None and best_margin < 0 else "")
            parts.append(f" Нарушают {name}: {n}{near}.")
        return "".join(parts)

    def _build_recommendation(
        self,
        state: ProcessState,
        q: QualityAssess,
        r: ReliabilityAssess,
        best: Candidate,
        survivors: List[Candidate],
        verdicts: List[GateVerdict],
    ) -> Recommendation:
        vmap = {v.candidate_id: v for v in verdicts}
        v = vmap[best.candidate_id]
        margin = self._hard_margin(v)

        rec = Recommendation(
            ts=state.ts,
            action=best.deltas,
            reason=self._reason(q, r, best, v, margin),
            expected_effect={
                **self._quality_effect(best, v),
                "setpoints": self._setpoints(state, best),
                "margin_to_spec": None if margin is None else round(margin, 3),
                "yield_delta_tph": round(best.yield_delta, 2),
                "cost_proxy": round(best.cost_proxy, 2),
                "severity_delta": best.severity_delta,
            },
            checks_passed=v.checked,
            checks_failed=v.violated,
            checks_warned=v.warnings,
            confidence=q.confidence,
            alternatives=self._alternatives(best, survivors, verdicts),
            data_freshness=summarize_freshness(state),
        )
        rec.explanation = render_explanation(rec, q, r, best, v, len(survivors))
        return rec

    def _reason(
        self,
        q: QualityAssess,
        r: ReliabilityAssess,
        best: Candidate,
        v: GateVerdict,
        margin: Optional[float],
    ) -> str:
        if margin is not None and margin < 0:
            worst = min(
                self.hard_specs,
                key=lambda n: v.margins.get(n, math.inf),
            )
            spec = self.hard_specs[worst]
            iv = best.predicted.get(worst)
            bound = iv.conservative(spec["check_on"])
            return (
                f"продукт уже вне спецификации по {worst}: "
                f"{bound:.2f} при лимите {float(spec['limit']):.2f}. Предложено "
                f"наибольшее улучшение, достижимое за один шаг; спецификация "
                f"за один цикл не восстанавливается"
            )
        if best.is_no_action:
            return "режим устойчив, запас по всем жёстким ограничениям сохраняется"
        targets = self.decision["target_margin"]
        tight = [
            (name, v.margins[name], float(targets.get(name, 0.0)))
            for name in self.hard_specs
            if v.margins.get(name) is not None
            and v.margins[name] < float(targets.get(name, 0.0))
        ]
        if tight:
            name, got, want = min(tight, key=lambda x: x[1] - x[2])
            display = self.hard_specs[name].get("display", name)
            return (
                f"запас по спецификации ({display}) {got:.2f} ниже целевого "
                f"{want:.2f}, тяжесть режима {r.severity_class}"
            )
        return (
            f"текущий риск нарушения спецификации {q.spec_risk_prob:.0%}, "
            f"тяжесть режима {r.severity_class}"
        )

    def _quality_effect(self, best: Candidate, v: GateVerdict) -> Dict[str, Any]:
        """Прогноз по каждой жёсткой спеке: среднее и та граница, по которой проверяет Gate."""
        out: Dict[str, Any] = {}
        for name, spec in self.hard_specs.items():
            iv = best.predicted.get(name)
            if iv is None:
                continue
            on = spec["check_on"]
            out[name] = {
                "display": spec.get("display", name),
                "units": spec.get("units", ""),
                "mean": round(iv.mean, 2),
                on: round(iv.conservative(on), 2),
                "limit": float(spec["limit"]),
                "margin": v.margins.get(name),
            }
        return out

    @staticmethod
    def _setpoints(state: ProcessState, best: Candidate) -> Dict[str, Any]:
        """
        Блок "текущее значение -> рекомендуемое" из раздела 5 ТЗ.
        """
        mvars = manipulated_vars()
        out: Dict[str, Any] = {}
        for tag, delta in best.deltas.items():
            current = state.tag(tag)
            if current is None:
                continue
            spec = mvars.get(tag, {})
            out[tag] = {
                "name": spec.get("name", tag),
                "current": round(current, 2),
                "recommended": round(current + delta, 2),
                "delta": round(delta, 2),
                "units": spec.get("units", ""),
            }
        return out

    def _alternatives(
        self,
        best: Candidate,
        survivors: List[Candidate],
        verdicts: List[GateVerdict],
    ) -> List[Dict[str, Any]]:
        vmap = {v.candidate_id: v for v in verdicts}

        def margin_of(c: Candidate) -> float:
            m = self._hard_margin(vmap[c.candidate_id])
            return -math.inf if m is None else m

        criteria = (
            ("максимум выпуска", lambda c: -c.yield_delta),
            ("минимум затрат", lambda c: c.cost_proxy),
            ("мягчайший режим", lambda c: c.severity_delta),
            ("наибольший запас", lambda c: -margin_of(c)),
        )

        out: List[Dict[str, Any]] = []
        seen = {best.candidate_id}
        for label, key in criteria:
            pick = min(
                survivors,
                key=lambda c: (key(c), sum(abs(d) for d in c.deltas.values())),
            )
            if pick.candidate_id in seen:
                continue
            seen.add(pick.candidate_id)
            out.append({
                "label": label,
                "candidate_id": pick.candidate_id,
                "deltas": pick.deltas,
                "margin_to_spec": round(margin_of(pick), 3),
                "yield_delta_tph": round(pick.yield_delta, 2),
                "cost_proxy": round(pick.cost_proxy, 2),
                "severity_delta": pick.severity_delta,
            })
        return out
