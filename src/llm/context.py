"""
Контекст решения для языковой модели.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from ..contracts import DecisionTrace
from ..data.tags import manipulated_vars, quality_specs


def build_context(trace: DecisionTrace) -> Dict[str, Any]:
    """Компактный словарь решения: всё, на что модель имеет право ссылаться."""
    rec = trace.recommendation
    return {
        "момент_решения": trace.ts.isoformat(),
        "состояние": _state(trace),
        "оценка_качества": _quality(trace),
        "оценка_надёжности": _reliability(trace),
        "перебор_вариантов": _search(trace),
        "рекомендация": {
            "действие": "ОТКАЗ" if rec.is_refusal else rec.action,
            "причина": rec.reason,
            "ожидаемый_эффект": rec.expected_effect,
            "проверки_пройдены": rec.checks_passed,
            "проверки_нарушены": rec.checks_failed,
            "предупреждения": rec.checks_warned,
            "альтернативы": rec.alternatives,
            "доверие": rec.confidence,
            "объяснение": rec.explanation,
        },
        "ограничения": _limits(),
    }


def _state(trace: DecisionTrace) -> Dict[str, Any]:
    state = trace.state
    return {
        "режим": {tag: state.tag(tag) for tag in manipulated_vars()
                  if state.tag(tag) is not None},
        "лаборатория": {
            param: {"значение": m.value, "возраст_мин": m.age_min, "исправен": m.healthy}
            for param, m in state.lims.items()
        },
        "поточные_анализаторы": {
            param: {"значение": m.value, "возраст_мин": m.age_min, "исправен": m.healthy}
            for param, m in state.pak.items()
        },
        "проблемы_с_данными": list(state.dq_flags),
    }


def _quality(trace: DecisionTrace) -> Dict[str, Any]:
    q = trace.quality
    return {
        "модель": q.model_id,
        "риск_нарушения_спецификации": q.spec_risk_prob,
        "доверие": q.confidence,
        "что_снизило_доверие": q.confidence_drivers,
        "факторы": [d for d in q.drivers if not d.startswith("проверяется изменение")],
        "прогноз": {
            name: {"среднее": iv.mean, "нижняя": iv.lo, "верхняя": iv.hi}
            for name, iv in q.predictions.items()
        },
    }


def _reliability(trace: DecisionTrace) -> Dict[str, Any]:
    r = trace.reliability
    return {
        "индекс_тяжести": r.severity_index,
        "класс": r.severity_class,
        "факторы": r.factors,
        "допущения": r.assumptions,
        "разрешённые_диапазоны": r.allowed_ranges,
    }


def _search(trace: DecisionTrace) -> Dict[str, Any]:
    """
    Статистика перебора вместо списка кандидатов.

    Оператор спрашивает "почему не сделали иначе" — для ответа нужно знать,
    сколько вариантов рассмотрено, сколько отброшено и ПО КАКОЙ причине,
    а не сами варианты.
    """
    verdicts = trace.verdicts
    reasons: Dict[str, int] = {}
    for v in verdicts:
        for text in v.violated:
            key = text.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1

    return {
        "кандидатов_всего": len(trace.candidates),
        "прошло_жёсткий_фильтр": sum(1 for v in verdicts if v.passed),
        "отбраковано_по_причинам": reasons,
        "пример_отбраковки": next(
            (v.violated[0] for v in verdicts if v.violated), None
        ),
    }


def _limits() -> Dict[str, Any]:
    specs = quality_specs()
    return {
        "спецификация": {
            name: {"лимит": spec["limit"], "тип": spec["direction"],
                   "проверка_по": spec["check_on"], "источник": spec["source"]}
            for name, spec in specs.items()
        },
        "управляемые_переменные": {
            tag: {"диапазон": spec["range"], "максимальный_шаг": spec["max_step"],
                  "единицы": spec.get("units", ""), "название": spec.get("name", tag)}
            for tag, spec in manipulated_vars().items()
        },
    }


def numbers_in(obj: Any) -> List[float]:
    """Все числа контекста — белый список для проверки ответа модели."""
    out: List[float] = []
    _collect(obj, out)
    return out


def _collect(obj: Any, out: List[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _collect(key, out)
            _collect(value, out)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect(item, out)
    elif isinstance(obj, str):
        out.extend(_numbers_in_text(obj))


def _numbers_in_text(text: str) -> Iterable[float]:
    """Числа внутри готовых формулировок ('запас 0.88 мг/кг') тоже разрешены."""
    from .validator import extract_numbers

    return extract_numbers(text)
