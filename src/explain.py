"""
Сборка объяснения оператору.

ЧТО ПОКАЗЫВАЕТ ОТЧЁТ. Блоки 1-7 - ровно таблица раздела 5 ТЗ. Блоки 8-10
добавлены под критерии оценки: альтернативы с ценой выбора («чем выбранный
вариант лучше»), явные допущения и обмен между агентами. Блоки 8-10 требуют DecisionTrace:
в Recommendation этих данных нет и быть не должно, это ответ системы
оператору, а не её внутренняя кухня.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from .contracts import DecisionTrace, ProcessState, Recommendation
from .data.tags import load_config, manipulated_vars

WIDTH = 78

_EFFECT_LABELS = {
    "margin_to_spec": ("запас до спецификации", ""),
    "yield_delta_tph": ("изменение выпуска", "т/ч"),
    "cost_proxy": ("затраты (прокси)", "усл. ед."),
    "severity_delta": ("изменение тяжести режима", ""),
}


def summarize_freshness(state: ProcessState) -> Dict[str, Any]:
    """
    Блок 'Время и состояние' из раздела 5 ТЗ.

    """
    out: Dict[str, Any] = {"ts": state.ts.isoformat(), "dq_flags": list(state.dq_flags)}
    for name, store in (("lims", state.lims), ("pak", state.pak)):
        for param, m in store.items():
            out[f"{name}:{param}"] = {
                "value": m.value,
                "age_min": m.age_min,
                "healthy": m.healthy,
                "units": m.units,
            }

    out["tags"] = {
        tag: {"value": state.tag(tag), "units": spec.get("units", ""),
              "name": spec.get("name", tag)}
        for tag, spec in manipulated_vars().items()
        if state.tag(tag) is not None
    }
    return out


def _quality_items(expected_effect: Dict[str, Any]) -> Iterable[Tuple[str, Dict]]:
    """
    Показатели качества внутри expected_effect.

    Оркестратор кладёт туда по одной записи на каждую жёсткую спеку из
    конфига, поэтому здесь нет ни одного имени показателя: добавится
    второй подтверждённый предел - отчёт напечатает его сам.
    """
    for name, effect in (expected_effect or {}).items():
        if isinstance(effect, dict) and "mean" in effect:
            yield name, effect


def _target_margin(name: str) -> Optional[float]:
    targets = load_config("constraints")["decision"]["target_margin"]
    value = targets.get(name)
    return None if value is None else float(value)


def render_explanation(rec, quality, reliability, candidate, verdict, n_survivors: int) -> str:
    parts: List[str] = []

    setpoints = rec.expected_effect.get("setpoints") or {}
    if candidate.is_no_action:
        parts.append("Рекомендация: режим не менять.")
    else:
        moves = ", ".join(
            f"{tag} {sp['current']} -> {sp['recommended']} {sp['units']}".strip()
            for tag, sp in setpoints.items() if abs(sp["delta"]) > 1e-9
        ) or ", ".join(
            f"{tag} на {d:+.1f}" for tag, d in candidate.deltas.items() if abs(d) > 1e-9
        )
        parts.append(f"Рекомендация: изменить {moves}.")

    parts.append(f"Причина: {rec.reason}.")

    if quality.drivers:
        parts.append("Факторы: " + "; ".join(quality.drivers[:3]) + ".")

    for name, effect in _quality_items(rec.expected_effect):
        bound_key = "hi" if "hi" in effect else "lo"
        side = "верхняя" if bound_key == "hi" else "нижняя"
        units = effect.get("units", "")
        parts.append(
            f"Ожидаемая величина «{effect.get('display', name)}»: "
            f"{effect['mean']} {units}, {side} граница прогноза "
            f"{effect[bound_key]} при лимите {effect['limit']}, "
            f"запас {effect['margin']} {units}.".replace("  ", " ")
        )

    parts.append(
        f"Тяжесть режима {reliability.severity_class} "
        f"(индекс {reliability.severity_index}), изменение {candidate.severity_delta:+.3f}."
    )
    if verdict.violated:
        parts.append(
            f"Проверено ограничений: {len(verdict.checked)}, остаются нарушенными: "
            + "; ".join(verdict.violated) + "."
        )
        parts.append(
            f"Рассмотрено вариантов, улучшающих запас: {n_survivors}. "
            "Выбран дающий наибольший запас."
        )
    else:
        parts.append(
            f"Проверено ограничений: {len(verdict.checked)}, нарушений нет."
            + (f" Задеты модельные допущения: {'; '.join(verdict.warnings)}."
               if verdict.warnings else "")
        )
        parts.append(
            f"Допустимых альтернатив рассмотрено: {n_survivors}. "
            "Правило выбора: при достаточном запасе - наименьшее "
            "воздействие, иначе наибольший запас."
        )
    parts.append(f"Доверие к прогнозу: {rec.confidence:.2f}.")

    return " ".join(parts)

# Отчёт оператору

def print_operator_report(
    rec: Recommendation, trace: Optional[DecisionTrace] = None
) -> str:
    """
    Отчёт в терминал. Блоки 1-7 - таблица раздела 5 ТЗ.

    trace необязателен: без него печатаются только те блоки, данные для
    которых есть в самой рекомендации. С ним добавляются альтернативы
    с ценой выбора, допущения и обмен между агентами.
    """
    lines: List[str] = [
        "=" * WIDTH,
        f"РЕКОМЕНДАЦИЯ ОПЕРАТОРУ   {rec.ts:%Y-%m-%d %H:%M}",
        "=" * WIDTH,
    ]

    lines += _block_state(rec)
    lines += _block_problem(rec, trace)
    lines += _block_action(rec)
    lines += _block_effect(rec)
    lines += _block_checks(rec)
    lines += _block_confidence(rec, trace)
    lines += _block_explanation(rec)
    lines += _block_alternatives(rec)
    lines += _block_assumptions(trace)
    lines += _block_agents(trace)

    lines.append("=" * WIDTH)
    return "\n".join(lines)


def _block_state(rec: Recommendation) -> List[str]:
    out = ["\n[1] ВРЕМЯ И СОСТОЯНИЕ"]

    tags = (rec.data_freshness or {}).get("tags") or {}
    if tags:
        out.append("    режим установки")
        for tag, info in tags.items():
            out.append(
                f"      {tag:<12} {info['value']:>9.2f} {info['units']:<6} {info['name']}"
            )

    out.append("    измерения качества")
    printed = False
    for key, value in (rec.data_freshness or {}).items():
        if key in ("ts", "dq_flags", "tags") or not isinstance(value, dict):
            continue
        printed = True
        source, _, param = key.partition(":")
        age = value.get("age_min")
        age_s = f"{age / 60:.1f} ч" if age is not None else "нет"
        state = "" if value.get("healthy", True) else "  [НЕИСПРАВЕН]"
        shown = value.get("value")
        shown_s = f"{shown:>9.2f}" if isinstance(shown, (int, float)) else f"{'нет':>9}"
        out.append(
            f"      {source.upper():<5}{param:<16}{shown_s} {value.get('units',''):<6}"
            f"возраст {age_s}{state}"
        )
    if not printed:
        out.append("      нет данных")

    for flag in (rec.data_freshness or {}).get("dq_flags") or []:
        out.append(f"    ! {flag}")
    return out


def _block_problem(rec: Recommendation, trace: Optional[DecisionTrace]) -> List[str]:
    out = ["\n[2] ПРОБЛЕМА / РИСК"]
    out += [f"    {chunk}" for chunk in _wrap(rec.reason, WIDTH - 4)]

    if trace is None:
        return out

    q, r = trace.quality, trace.reliability
    out += _labeled(
        "    ",
        f"риск выхода за спецификацию {q.spec_risk_prob:.0%}, "
        f"тяжесть режима {r.severity_class} (индекс {r.severity_index})",
    )
    for driver in q.drivers[:3]:
        if driver.startswith("проверяется изменение"):
            continue            # это про кандидата, а не про текущий риск
        out += _labeled("      - ", driver)
    for factor in r.factors[:3]:
        out += _labeled("      - ", factor)
    return out


def _block_action(rec: Recommendation) -> List[str]:
    out = ["\n[3] ПРЕДЛАГАЕМОЕ ДЕЙСТВИЕ"]
    setpoints = (rec.expected_effect or {}).get("setpoints") or {}

    if rec.is_refusal:
        out.append("    ОТКАЗ ОТ РЕКОМЕНДАЦИИ")
    elif not rec.action:
        out.append("    изменений не требуется")
    elif setpoints:
        out.append(f"    {'тег':<12} {'сейчас':>9} {'станет':>9} {'шаг':>8}  параметр")
        for tag, sp in setpoints.items():
            name = sp["name"] if abs(sp["delta"]) > 1e-9 else f"{sp['name']} (без изменений)"
            label = _fit(f"{name}, {sp['units']}", 31)
            out.append(
                f"    {tag:<12} {sp['current']:>9.2f} {sp['recommended']:>9.2f} "
                f"{sp['delta']:>+8.2f}  {label}"
            )
    else:
        for tag, delta in rec.action.items():
            mark = "" if abs(delta) > 1e-9 else "   (без изменений)"
            out.append(f"    {tag:<20} {delta:+.2f}{mark}")
    return out


def _block_effect(rec: Recommendation) -> List[str]:
    out = ["\n[4] ОЖИДАЕМЫЙ ЭФФЕКТ"]
    if not rec.expected_effect:
        out.append("    не оценивается")
        return out
    if rec.is_refusal:
        out.append("    прогноз при бездействии")

    for name, effect in _quality_items(rec.expected_effect):
        bound_key = "hi" if "hi" in effect else "lo"
        units = effect.get("units", "")
        target = _target_margin(name)
        margin = effect.get("margin")
        verdict = ""
        if target is not None and margin is not None:
            verdict = (" - запаса достаточно" if margin >= target
                       else f" - ниже целевого {target}")
        out.append(f"    {effect.get('display', name)} ({name})")
        out += _labeled(
            "      ",
            f"прогноз {effect['mean']} {units}, граница {effect[bound_key]} "
            f"при лимите {effect['limit']}, запас {margin}{verdict}",
        )

    for key, value in rec.expected_effect.items():
        if isinstance(value, dict):
            continue
        label, units = _EFFECT_LABELS.get(key, (key, ""))
        out.append(f"    {label:<28} {value} {units}".rstrip())
    return out


def _block_checks(rec: Recommendation) -> List[str]:
    total = len(rec.checks_passed)
    failed, warned = len(rec.checks_failed), len(rec.checks_warned)
    head = (f"\n[5] ПРОВЕРКА ОГРАНИЧЕНИЙ   "
            f"проверок {total}, нарушено {failed}, предупреждений {warned}")
    out = [head]
    if not rec.checks_passed:
        out.append("    проверки не выполнялись")

    failed_params = {c.split(":")[0] for c in rec.checks_failed}
    warned_params = {c.split(":")[0] for c in rec.checks_warned}
    for check in rec.checks_passed:
        param = check.split()[0]
        mark = ("[!]" if param in failed_params
                else "[~]" if param in warned_params else "[OK]")
        out.append(f"    {mark} {check}")

    for check in rec.checks_failed:
        out.append(f"    [!] НАРУШЕНО: {check}")
    for check in rec.checks_warned:
        out.append(f"    [~] ВНИМАНИЕ (модельное допущение): {check}")
    return out


def _block_confidence(
    rec: Recommendation, trace: Optional[DecisionTrace]
) -> List[str]:
    """
    Причины снижения доверия берутся у агента качества, а не выводятся
    из свежести данных заново
    """
    level = ("высокая" if rec.confidence >= 0.75
             else "средняя" if rec.confidence >= 0.5 else "низкая")
    out = ["\n[6] УВЕРЕННОСТЬ", f"    {rec.confidence:.2f} - {level}"]

    reasons = list(trace.quality.confidence_drivers) if trace else []
    if reasons:
        out.append("    снижают доверие:")
        out += [f"      - {r}" for r in reasons]
    elif rec.confidence >= 0.75:
        out.append("    данные свежие, источники исправны")

    # Доверие к ДАННЫМ и доверие к МОДЕЛИ - разные вещи. Данные могут быть
    # свежими, а прогноз при этом регулярно промахиваться мимо лаборатории.
    health = getattr(trace, "model_health", None) if trace else None
    if health is not None:
        out += _labeled("    самоконтроль: ", health.message)
    return out


def _block_explanation(rec: Recommendation) -> List[str]:
    out = ["\n[7] ОБЪЯСНЕНИЕ"]
    out += [f"    {chunk}" for chunk in _wrap(rec.explanation, WIDTH - 4)]
    return out


def _block_alternatives(rec: Recommendation) -> List[str]:
    if not rec.alternatives:
        return []

    out = ["\n[8] ДОПУСТИМЫЕ АЛЬТЕРНАТИВЫ И ЦЕНА ВЫБОРА"]
    chosen = rec.expected_effect or {}
    base = {
        "margin_to_spec": chosen.get("margin_to_spec"),
        "yield_delta_tph": chosen.get("yield_delta_tph"),
        "cost_proxy": chosen.get("cost_proxy"),
    }

    for alt in rec.alternatives:
        moves = ", ".join(
            f"{tag} {d:+.1f}" for tag, d in alt["deltas"].items() if abs(d) > 1e-9
        )
        out.append(f"    {alt.get('label', alt['candidate_id'])}: {moves or 'без изменений'}")
        out.append(
            f"      запас {alt['margin_to_spec']}  выпуск {alt['yield_delta_tph']:+.1f} т/ч"
            f"  затраты {alt['cost_proxy']}  режим {alt['severity_delta']:+.3f}"
        )
        diffs = [
            _diff("запас", alt.get("margin_to_spec"), base["margin_to_spec"]),
            _diff("выпуск", alt.get("yield_delta_tph"), base["yield_delta_tph"]),
            _diff("затраты", alt.get("cost_proxy"), base["cost_proxy"]),
        ]
        diffs = [d for d in diffs if d]
        if diffs:
            out.append("      против выбранного: " + ", ".join(diffs))
    return out


def _diff(label: str, alt_value, chosen_value) -> str:
    if alt_value is None or chosen_value is None:
        return ""
    delta = alt_value - chosen_value
    if abs(delta) < 1e-9:
        return ""
    return f"{label} {delta:+.2f}"


def _block_assumptions(trace: Optional[DecisionTrace]) -> List[str]:
    if trace is None or not trace.reliability.assumptions:
        return []
    out = ["\n[9] ДОПУЩЕНИЯ"]
    for assumption in trace.reliability.assumptions:
        out += _labeled("    - ", assumption)
    return out


def _fit(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _labeled(prefix: str, text: str) -> List[str]:
    chunks = _wrap(text, WIDTH - len(prefix))
    if not chunks:
        return []
    pad = " " * len(prefix)
    return [prefix + chunks[0]] + [pad + chunk for chunk in chunks[1:]]


def _block_agents(trace: Optional[DecisionTrace]) -> List[str]:
    if trace is None:
        return []

    passed = sum(1 for v in trace.verdicts if v.passed)
    narrowed = _narrowed_ranges(trace.reliability.allowed_ranges)
    ranges_note = (
        f"сузила диапазоны: {narrowed} из "
        f"{len(trace.reliability.allowed_ranges)} переменных"
        if narrowed else
        f"диапазоны не сужены (режим не тяжёлый), "
        f"передано {len(trace.reliability.allowed_ranges)} границ"
    )

    out = ["\n[10] ОБМЕН МЕЖДУ АГЕНТАМИ"]
    out += _labeled("    качество       ", f"модель {trace.quality.model_id}, "
                    f"риск {trace.quality.spec_risk_prob}, "
                    f"доверие {trace.quality.confidence}")
    out += _labeled("    надёжность     ", f"severity {trace.reliability.severity_index} "
                    f"({trace.reliability.severity_class}), {ranges_note}")
    out += _labeled("    оптимизация    ", f"сгенерировано кандидатов "
                    f"{len(trace.candidates)} внутри допустимых границ")
    out += _labeled("    жёсткий фильтр ", f"прошло {passed} из {len(trace.verdicts)}")

    rejected = [v for v in trace.verdicts if not v.passed]
    if rejected:
        out += _labeled("    отбраковка     ", rejected[0].violated[0])
    return out


def _narrowed_ranges(allowed: Dict[str, List[float]]) -> int:
    mvars = manipulated_vars()
    count = 0
    for tag, (lo, hi) in allowed.items():
        spec = mvars.get(tag)
        if spec is None:
            continue
        base_lo, base_hi = float(spec["range"][0]), float(spec["range"][1])
        if lo > base_lo + 1e-9 or hi < base_hi - 1e-9:
            count += 1
    return count


def _wrap(text: str, width: int) -> List[str]:
    words, line, out = (text or "").split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out
