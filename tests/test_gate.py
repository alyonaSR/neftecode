"""
Тесты жёсткого фильтра.

Gate - единственный модуль, где баг напрямую означает опасную рекомендацию.
Тесты на него пишутся ПЕРВЫМИ.

Запуск: python -m pytest tests -q   (или python tests/test_gate.py без pytest)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

from src.contracts import Candidate, Interval, ProcessState
from src.gate import ConstraintGate


def _state():
    return ProcessState(
        ts=datetime(2026, 3, 14, 8, 20),
        tags={"242000:T5": 365.0, "AVT:F30": 61.0, "AVT:T33": 348.0},
    )


def _cand(sulfur_mean, sulfur_half=0.5, deltas=None):
    return Candidate(
        candidate_id="c_test",
        deltas=deltas if deltas is not None else {"242000:T5": 0.0},
        predicted={
            "sulfur_mgkg": Interval(sulfur_mean, sulfur_mean - sulfur_half, sulfur_mean + sulfur_half),
            "flash_c": Interval(68, 65, 71),
            "cfpp_c": Interval(-6, -8, -4),
            "d15_kgm3": Interval(836, 833, 839),
        },
    )


def test_sulfur_checked_on_upper_bound_not_mean():
    """Ключевой тест: mean=9.6 проходит по среднему, но hi=10.4 обязан упасть."""
    g = ConstraintGate()
    v = g.check(_cand(9.6, 0.8), _state())
    assert not v.passed
    assert any("sulfur" in x for x in v.violated)


def test_sulfur_with_margin_passes():
    g = ConstraintGate()
    v = g.check(_cand(8.0, 0.5), _state())
    assert v.passed
    assert v.margins["sulfur_mgkg"] > 0


def test_step_limit_enforced():
    g = ConstraintGate()
    v = g.check(_cand(8.0, 0.5, deltas={"242000:T5": 5.0}), _state())
    assert not v.passed
    assert any("шаг" in x for x in v.violated)


def test_allowed_ranges_from_reliability_agent_narrow_the_gate():
    """Агент надёжности реально ограничивает оптимизатор."""
    g = ConstraintGate()
    c = _cand(8.0, 0.5, deltas={"242000:T5": 1.5})
    assert g.check(c, _state()).passed
    v = g.check(c, _state(), allowed_ranges={"242000:T5": [340.0, 366.0]})
    assert not v.passed


def test_blending_must_sum_to_one():
    g = ConstraintGate()
    ok = g.check(_cand(8.0), _state(), blend_fractions=[0.6, 0.4])
    bad = g.check(_cand(8.0), _state(), blend_fractions=[0.6, 0.5])
    assert ok.passed and not bad.passed


def test_blending_share_outside_history_is_rejected():
    """Сумма долей равна единице всегда: содержательна проверка пропорции."""
    g = ConstraintGate()
    ok = g.check(_cand(8.0), _state(), blend_fractions={"AVT:F30": 0.62, "AVT:F32": 0.38})
    bad = g.check(_cand(8.0), _state(), blend_fractions={"AVT:F30": 0.85, "AVT:F32": 0.15})
    assert ok.passed and not bad.passed


def test_blend_sum_margin_is_a_margin_not_a_difference():
    """Отклонение внутри допуска обязано давать ПОЛОЖИТЕЛЬНЫЙ запас."""
    g = ConstraintGate()
    v = g.check(_cand(8.0), _state(), blend_fractions={"AVT:F30": 0.6000005, "AVT:F32": 0.4})
    assert v.passed and v.margins["blend_sum"] > 0


def _cand_cfpp(cfpp: float):
    c = _cand(8.0)
    c.predicted["cfpp_c"] = Interval(cfpp, cfpp - 2, cfpp + 2)
    return c


def test_assumption_limit_warns_but_does_not_reject():
    """
    ПТФ, вспышка и плотность - НАШИ допущения, а не промышленные пределы.
    ТЗ запрещает выдавать одно за другое, поэтому отбраковывать по ним
    нельзя: лимит ПТФ 0 C для летнего сорта зарезал бы любой зимний режим.
    """
    v = ConstraintGate().check(_cand_cfpp(3.0), _state())
    assert v.passed
    assert v.warnings and "cfpp_c" in v.warnings[0]
    assert not v.violated


def test_spec_limit_still_rejects():
    """Сера - единственное требование спецификации, по ней отбраковка жёсткая."""
    v = ConstraintGate().check(_cand(10.5), _state())
    assert not v.passed and not v.warnings


def test_nan_prediction_never_passes():
    """
    Любое сравнение с NaN даёт False, поэтому испорченный прогноз проходил
    все проверки насквозь. Фильтр, пропускающий NaN, не фильтр.
    """
    g = ConstraintGate()
    assert not g.check(_cand(float("nan")), _state()).passed
    assert not g.check(_cand(8.0, deltas={"242000:T5": float("nan")}), _state()).passed


def test_unknown_tag_is_rejected_not_crashed():
    """
    Правило границ ТЗ: без подтверждённого диапазона параметр не трогаем.
    Раньше здесь падал KeyError и рушил весь цикл.
    """
    v = ConstraintGate().check(_cand(8.0, deltas={"AVT:T33": 1.0}), _state())
    assert not v.passed
    assert "не входит в список управляемых" in v.violated[0]


def test_gate_does_not_depend_on_agents_or_llm():
    """
    Архитектурное свойство, на котором держится безопасность: вердикт не
    зависит от того, что сгенерировала языковая модель. Проверяется тестом,
    а не обещанием в докстринге.
    """
    import ast

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "src", "gate.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    imports = [
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ] + [
        alias.name
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any("agent" in m or "llm" in m.lower() for m in imports), imports


def test_missing_prediction_is_a_violation():
    g = ConstraintGate()
    c = _cand(8.0)
    del c.predicted["sulfur_mgkg"]
    assert not g.check(c, _state()).passed


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"  OK  {name}")
    print("\nвсе тесты прошли")
