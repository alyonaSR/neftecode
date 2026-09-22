"""
Тесты агента оптимизации и отчёта по Парето-фронту.

Тесты фиксируют конкретные баги, которые уже были пойманы и исправлены,
включая регрессию на время полного цикла.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

from src.agents.optimizer import OptimizerAgent
from src.agents.pareto_report import pareto_table
from src.agents.quality import QualityAgent
from src.agents.reliability import ReliabilityAgent
from src.contracts import Candidate
from src.data.state_builder import build_demo_state
from src.data.tags import manipulated_vars
from src.orchestrator import Orchestrator


def _propose(scenario="normal"):
    state = build_demo_state(scenario)
    quality = QualityAgent()
    reliability = ReliabilityAgent()
    q = quality.assess(state)
    r = reliability.assess(state)
    optimizer = OptimizerAgent(quality)
    return optimizer.propose(state, q, r)


def test_default_active_vars_come_from_config_not_code():
    """
    РЕГРЕССИОННЫЙ ТЕСТ на баг Ксении: active_vars раньше был списком,
    зашитым прямо в propose(). Когда AVT:T33 убрали из constraints.yaml,
    а из кода - нет, всё падало с KeyError у всей команды. Теперь
    единственный источник истины - поле active: true в constraints.yaml.
    """
    expected = {tag for tag, spec in manipulated_vars().items() if spec.get("active")}
    assert expected, "ни одна переменная не помечена active: true в constraints.yaml"

    candidates = _propose()
    tags_seen = {t for c in candidates for t in c.deltas}
    assert tags_seen == expected


def test_grid_respects_current_real_operating_point():
    """
    РЕГРЕССИОННЫЙ ТЕСТ: allowed_ranges
    строится из статического config/constraints.yaml, а не от текущего
    значения тега. Демо-состояние AVT:F30=128.4 (normal) и 141.0
    (quality_risk) - если диапазон в constraints.yaml не накрывает эти
    значения, _within_allowed() бракует вообще все кандидаты.
    """
    for scenario in ("normal", "quality_risk"):
        candidates = _propose(scenario)
        assert len(candidates) > 0, (
            f"0 кандидатов на сценарии {scenario} - allowed_ranges снова "
            "не накрывает текущее состояние установки"
        )


def test_full_cycle_is_fast_enough_for_a_10_minute_step():
    """
    РЕГРЕССИОННЫЙ ТЕСТ: после подключения
    обученной модели (FormulaPlusResidual, pandas-индексация
    внутри predict_one), полный грид по T5/F30/F32 со старыми
    grid_points 5/4/5 (1089 кандидатов) давал ~17с на один цикл - вся
    тестовая связка переставала укладываться в разумное время. ТЗ
    предполагает цикл принятия решения раз в 10 минут, не раз в 17
    секунд ради одной рекомендации. Порог 5с - с большим запасом
    от текущих ~1.5-2с, но ловит повторный разрастание сетки.
    """
    state = build_demo_state("quality_risk")
    quality = QualityAgent()
    reliability = ReliabilityAgent()
    q = quality.assess(state)
    r = reliability.assess(state)
    optimizer = OptimizerAgent(quality)

    t0 = time.time()
    candidates = optimizer.propose(state, q, r)
    elapsed = time.time() - t0

    assert len(candidates) > 0
    assert elapsed < 5.0, f"один цикл propose() занял {elapsed:.1f}с - сетка снова разрослась"


def test_cost_proxy_does_not_double_count_yield():
    assert OptimizerAgent._cost_proxy({"AVT:F30": 3.0, "AVT:F32": 2.0}) == 0.0
    assert OptimizerAgent._cost_proxy({"242000:T5": 1.0}) == 1.0


def test_cost_proxy_charges_for_steam_if_f28_gets_activated():
    """
    F28 не в active_vars по умолчанию, но если кто-то явно включит его
    (active_vars=[..., "AVT:F28"]), расход пара не должен быть бесплатным.
    """
    assert OptimizerAgent._cost_proxy({"AVT:F28": 10.0}) == 1.0


def test_yield_delta_sums_both_diesel_pool_tags():
    """F30 и F32 вместе формируют объём дизельного пула."""
    candidates = _propose()
    sample = next(
        c for c in candidates
        if abs(c.deltas["AVT:F30"]) > 1e-9 and abs(c.deltas["AVT:F32"]) > 1e-9
    )
    expected = sample.deltas["AVT:F30"] + sample.deltas["AVT:F32"]
    assert abs(sample.yield_delta - expected) < 1e-9


def test_severity_delta_only_from_reactor_temperature():
    assert OptimizerAgent._severity_delta({"AVT:F32": 2.0}) == 0.0
    assert OptimizerAgent._severity_delta({"242000:T5": 1.0}) == 0.04


def test_within_allowed_respects_reliability_ranges():
    for scenario in ("normal", "quality_risk"):
        state = build_demo_state(scenario)
        reliability = ReliabilityAgent()
        r = reliability.assess(state)
        for tag, (lo, hi) in r.allowed_ranges.items():
            cur = state.tag(tag)
            if cur is None:
                continue
            assert lo <= cur <= hi, f"{scenario}: {tag}={cur} вне {[lo, hi]}"


def test_pareto_front_key_includes_yield_and_breaks_degenerate_ties():
    """
    РЕГРЕССИОННЫЙ ТЕСТ на находку коллеги: раньше ключ фронта был
    (cost_proxy, severity_delta, sulfur_hi) - все три зависят только от
    242000:T5, и варианты с одинаковой температурой, но разным отбором,
    были неразличимы. Теперь -yield_delta - четвёртая ось: два кандидата
    с одинаковыми первыми тремя, но разным выпуском, не должны оба
    остаться во фронте как "одна и та же точка".
    """
    same_temp = {"242000:T5": 1.0, "AVT:F30": 0.0, "AVT:F32": 0.0}
    more_yield = {"242000:T5": 1.0, "AVT:F30": 2.0, "AVT:F32": 0.0}

    def fake(deltas, sulfur_hi=9.0):
        return Candidate(
            candidate_id="x",
            deltas=deltas,
            predicted={"sulfur_mgkg": type("I", (), {"mean": 8.0, "lo": 7.5, "hi": sulfur_hi})()},
            cost_proxy=OptimizerAgent._cost_proxy(deltas),
            severity_delta=OptimizerAgent._severity_delta(deltas),
            yield_delta=deltas.get("AVT:F30", 0.0) + deltas.get("AVT:F32", 0.0),
        )

    low_yield = fake(same_temp)
    high_yield = fake(more_yield)
    front = OptimizerAgent.pareto_front([low_yield, high_yield])

    # одинаковый cost/severity/sulfur, но high_yield строго лучше по выпуску
    # -> low_yield должен быть отброшен как доминируемый
    assert high_yield in front
    assert low_yield not in front


def test_pareto_front_is_nondominated():
    candidates = _propose("quality_risk")
    front = OptimizerAgent.pareto_front(candidates)
    assert 0 < len(front) <= len(candidates)

    def key(c: Candidate):
        s = c.predicted.get("sulfur_mgkg")
        return (c.cost_proxy, c.severity_delta, s.hi if s else 0.0, -c.yield_delta)

    for a in front:
        ka = key(a)
        for b in front:
            if a is b:
                continue
            kb = key(b)
            dominated = all(x <= y for x, y in zip(kb, ka)) and kb != ka
            assert not dominated, f"{a.candidate_id} доминируется {b.candidate_id}"


def test_pareto_front_handles_empty_input():
    assert OptimizerAgent.pareto_front([]) == []


def test_pareto_report_refuses_when_orchestrator_refuses():
    trace = Orchestrator().run_cycle(build_demo_state("degraded_data"))
    assert trace.recommendation.is_refusal
    report = pareto_table(trace)
    assert "Отказ от рекомендации" in report
    assert "ПАРЕТО-ФРОНТ" not in report


def test_pareto_report_shows_front_for_normal_and_quality_risk():
    orch = Orchestrator()
    for scenario in ("normal", "quality_risk"):
        trace = orch.run_cycle(build_demo_state(scenario))
        assert not trace.recommendation.is_refusal
        report = pareto_table(trace)
        assert "ПАРЕТО-ФРОНТ" in report


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"  OK  {name}")
    print("\nвсе тесты оптимизатора прошли")
