"""
Интеграционные тесты полного цикла.

Проверяют ровно те три сценария, которые ТЗ требует показать на демо.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.contracts import REFUSE
from src.data.state_builder import build_demo_state
from src.orchestrator import Orchestrator


def test_normal_mode_produces_no_action():
    """Устойчивый режим: система не создаёт лишних управляющих действий."""
    t = Orchestrator().run_cycle(build_demo_state("normal"))
    rec = t.recommendation
    assert not rec.is_refusal
    assert all(abs(v) < 1e-9 for v in rec.action.values())


def test_quality_risk_produces_corrective_action():
    """Риск качества: система предлагает действие и оно проходит Gate."""
    t = Orchestrator().run_cycle(build_demo_state("quality_risk"))
    rec = t.recommendation
    assert not rec.is_refusal
    assert any(abs(v) > 1e-9 for v in rec.action.values())
    assert rec.expected_effect["sulfur_mgkg"]["hi"] <= 10.0


def test_already_out_of_spec_gets_best_improvement_not_refusal():
    """
    Продукт уже вне спецификации: отказ здесь означал бы молчание ровно
    тогда, когда система нужнее всего. Ожидается лучшее улучшение за шаг,
    честно помеченное как не восстанавливающее спеку.
    """
    st = build_demo_state("quality_risk")
    st.pak["sulfur_mgkg"].value = 10.4
    st.lims["sulfur_mgkg"].value = 10.3

    rec = Orchestrator().run_cycle(st).recommendation
    assert not rec.is_refusal
    assert rec.action["242000:T5"] > 0          # температура вверх = сера вниз
    assert rec.checks_failed                     # нарушение не спрятано
    assert rec.expected_effect["margin_to_spec"] < 0


def test_scenario_results_do_not_depend_on_order():
    """
    Агент самоконтроля копит память между циклами, а сценарии ТЗ - это не
    последовательность моментов, а три разных "что если" в один и тот же
    момент. Если память течёт между ними, результат начинает зависеть от
    порядка запуска, причём молча. Демо раздаёт каждому сценарию копию
    одной и той же прогретой памяти - этот тест стережёт правило.
    """
    import copy
    from datetime import timedelta

    from src.data.state_builder import SCENARIOS

    orch = Orchestrator()

    # Память: прогноз за 50 минут до лабораторного анализа сценария normal.
    # Без него сверок не происходит и тест проходил бы впустую.
    ts = build_demo_state("normal").ts
    early = build_demo_state("normal", ts=ts - timedelta(hours=6, minutes=50))
    orch.monitor.record(early, orch.quality.assess(early))
    warmed = copy.deepcopy(orch.monitor)

    def run(order):
        out = {}
        for name in order:
            orch.monitor = copy.deepcopy(warmed)
            trace = orch.run_cycle(build_demo_state(name))
            out[name] = (str(trace.recommendation.action),
                         trace.recommendation.expected_effect.get("margin_to_spec"),
                         trace.model_health.scale)
        return out

    assert run(SCENARIOS) == run(tuple(reversed(SCENARIOS)))


def test_broken_forecast_blocks_recommendation():
    """
    Агент самоконтроля сверяет прошлые прогнозы с лабораторией. Если модель
    систематически промахивается, доверять ей нельзя, сколько бы ни был
    широк интервал: данные при этом могут быть свежими, поэтому обычный
    отказ по данным такую ситуацию не ловит.
    """
    from src.agents.model_monitor import ModelHealthAssess, ModelMonitorAgent

    class Broken(ModelMonitorAgent):
        def assess(self):
            return ModelHealthAssess("failed", 0.55, 20, 3.0,
                                     "интервал серы накрыл лабораторию в 55% из 20 анализов")

    trace = Orchestrator(monitor=Broken()).run_cycle(build_demo_state("quality_risk"))

    assert trace.recommendation.is_refusal
    assert "55%" in trace.recommendation.reason
    assert trace.model_health.status == "failed"


def test_model_health_reaches_report_and_trace():
    """Оценка самоконтроля обязана дойти и до оператора, и до трейса."""
    import json

    from src.explain import print_operator_report

    trace = Orchestrator().run_cycle(build_demo_state("normal"))

    assert trace.model_health is not None
    assert "самоконтроль" in print_operator_report(trace.recommendation, trace)
    assert json.loads(json.dumps(trace.to_dict(), ensure_ascii=False))["model_health"]


def test_widened_interval_shrinks_the_margin():
    """
    Масштаб от самоконтроля обязан доходить до прогноза: иначе расширение
    интервала останется числом в отчёте и ни на что не повлияет.
    """
    state = build_demo_state("quality_risk")

    tight = Orchestrator().run_cycle(state).quality.predictions["sulfur_mgkg"]

    widened = Orchestrator()
    widened.monitor.scale = 2.0
    loose = widened.run_cycle(state).quality.predictions["sulfur_mgkg"]

    assert loose.width > tight.width
    assert loose.hi > tight.hi          # Gate проверяет серу именно по hi


def test_no_viable_option_explains_itself():
    """
    Отказ по невыполнимости: спека нарушена и ни одно действие не помогает.
    Отказ обязан остаться содержательным - что происходит при бездействии,
    какая проверка падает и насколько промахнулись, - иначе оператор
    получает пустой отчёт ровно в самой тяжёлой ситуации.
    """
    from src.agents.quality import QualityAgent
    from src.contracts import Interval

    class Hopeless(QualityAgent):
        """Сера вне спеки, и ни одно действие её не меняет."""
        def _predict(self, state, deltas):
            return {
                "sulfur_mgkg": Interval(11.5, 10.8, 12.2),
                "flash_c": Interval(68, 65, 71),
                "cfpp_c": Interval(-6, -8, -4),
                "d15_kgm3": Interval(836, 833, 839),
            }

    trace = Orchestrator(quality=Hopeless()).run_cycle(build_demo_state("quality_risk"))
    rec = trace.recommendation

    assert rec.is_refusal
    assert "при бездействии нарушается" in rec.reason
    assert rec.checks_passed and rec.checks_failed       # блоки отчёта не пустые
    assert rec.expected_effect["sulfur_mgkg"]["margin"] < 0
    assert "ближайший промах" in rec.explanation


def test_stale_lims_with_live_pak_still_works():
    """
    ТЗ задаёт приоритет ЛИМС -> ПАК, то есть ПАК - законный источник.
    Устаревший лабораторный анализ при живом поточном не повод молчать.
    """
    st = build_demo_state("normal")
    st.lims["sulfur_mgkg"].age_min = 60 * 60      # 60 ч, сильно за порогом

    rec = Orchestrator().run_cycle(st).recommendation
    assert not rec.is_refusal
    assert rec.confidence < 0.9                    # доверие снижено, но работаем


def test_no_usable_sulfur_source_produces_refusal():
    """Ни одного пригодного источника серы - отказ с разбором по источникам."""
    st = build_demo_state("normal")
    st.lims["sulfur_mgkg"].age_min = 60 * 60
    st.pak["sulfur_mgkg"].healthy = False

    rec = Orchestrator().run_cycle(st).recommendation
    assert rec.is_refusal
    assert "устарел" in rec.reason and "неисправ" in rec.reason


def test_require_healthy_pak_switches_to_strict_mode():
    """
    Строгий режим из конфига: мёртвый ПАК - повод отказаться даже при
    свежем лабораторном анализе. Ключ read-only проверяем через агента,
    чтобы правило не осталось декларацией в конфиге.
    """
    st = build_demo_state("normal")
    st.pak["sulfur_mgkg"].healthy = False

    orch = Orchestrator()
    assert not orch.run_cycle(st).recommendation.is_refusal

    orch.rules = dict(orch.rules, require_healthy_pak=True)
    rec = orch.run_cycle(st).recommendation
    assert rec.is_refusal and "неисправ" in rec.reason


def test_blending_shares_are_checked():
    """Доли компонентов пула - обязательная жёсткая проверка раздела 4 ТЗ."""
    t = Orchestrator().run_cycle(build_demo_state("normal"))
    assert any("доля AVT:F30" in c for c in t.recommendation.checks_passed)


def test_degraded_data_produces_refusal():
    """Устаревшие и мёртвые данные: корректный отказ, а не риск."""
    t = Orchestrator().run_cycle(build_demo_state("degraded_data"))
    assert t.recommendation.action == REFUSE
    assert "устарел" in t.recommendation.reason or "неисправ" in t.recommendation.reason


def test_agents_actually_exchange_information():
    """Мультиагентность: allowed_ranges надёжности реально ограничивают оптимизатор."""
    t = Orchestrator().run_cycle(build_demo_state("normal"))
    assert t.reliability.allowed_ranges
    for c in t.candidates:
        for tag, d in c.deltas.items():
            lo, hi = t.reliability.allowed_ranges[tag]
            assert lo <= t.state.tag(tag) + d <= hi


def test_trace_is_serializable():
    """Воспроизводимость: весь цикл сериализуется в JSON."""
    import json
    t = Orchestrator().run_cycle(build_demo_state("quality_risk"))
    assert json.dumps(t.to_dict(), ensure_ascii=False)


def test_run_is_deterministic():
    a = Orchestrator().run_cycle(build_demo_state("quality_risk")).recommendation.action
    b = Orchestrator().run_cycle(build_demo_state("quality_risk")).recommendation.action
    assert a == b


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(); print(f"  OK  {name}")
    print("\nвсе интеграционные тесты прошли")
