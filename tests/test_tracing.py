"""
Тесты трейсов и контрактов.

Трейс - это доказательство критерия «Воспроизводимость» из ТЗ: по нему
через неделю должно быть понятно не только КАКОЕ решение принято, но и
ЧЕМ оно получено - какой код, какие конфиги, какие модели.
"""
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.contracts import Interval, Measurement, ProcessState
from src.data.state_builder import build_demo_state
from src.explain import print_operator_report
from src.orchestrator import Orchestrator
from src.tracing import load_trace, recommendation_from_dict, save_trace


def _trace():
    return Orchestrator().run_cycle(build_demo_state("quality_risk"))


# ----------------------------------------------------------------------
def test_trace_records_what_produced_the_result(tmp_path, monkeypatch):
    import src.tracing as tracing

    monkeypatch.setattr(tracing, "TRACE_DIR", str(tmp_path))
    data = load_trace(save_trace(_trace(), tag="test"))

    meta = data["meta"]
    assert meta["конфиги"]["constraints.yaml"]        # пороги - часть решения
    assert meta["модели"]
    assert meta["schema_version"] == data["schema_version"]


def test_two_runs_do_not_overwrite_each_other(tmp_path, monkeypatch):
    """У демо-сценариев один и тот же момент решения, файлы обязаны различаться."""
    import src.tracing as tracing

    monkeypatch.setattr(tracing, "TRACE_DIR", str(tmp_path))
    trace = _trace()
    first, second = save_trace(trace, tag="x"), save_trace(trace, tag="x")

    assert first != second
    assert len(os.listdir(tmp_path)) == 2


def test_compact_trace_keeps_the_audit_trail(tmp_path, monkeypatch):
    """
    Сокращения не должны стирать доказательство: каждый кандидат и его
    запасы остаются, иначе нельзя проверить, почему вариант отброшен.
    """
    import src.tracing as tracing

    monkeypatch.setattr(tracing, "TRACE_DIR", str(tmp_path))
    trace = _trace()
    data = load_trace(save_trace(trace, tag="compact"))

    assert len(data["candidates"]) == len(trace.candidates)
    assert data["verdicts"][0]["margins"]
    assert data["проверки_цикла"]                      # проверки хранятся один раз
    assert "checked" not in data["verdicts"][0]


def test_report_can_be_replayed_from_file(tmp_path, monkeypatch):
    import src.tracing as tracing

    monkeypatch.setattr(tracing, "TRACE_DIR", str(tmp_path))
    trace = _trace()
    data = load_trace(save_trace(trace, tag="replay"))

    report = print_operator_report(recommendation_from_dict(data["recommendation"]))
    assert "РЕКОМЕНДАЦИЯ ОПЕРАТОРУ" in report
    assert "ПРЕДЛАГАЕМОЕ ДЕЙСТВИЕ" in report


def test_trace_from_newer_schema_still_opens(tmp_path, monkeypatch):
    """Разбор инцидента не должен упираться в несовпадение версий."""
    import src.tracing as tracing

    monkeypatch.setattr(tracing, "TRACE_DIR", str(tmp_path))
    path = save_trace(_trace(), tag="future")

    data = json.load(open(path, encoding="utf-8"))
    data["schema_version"] = "99.0.0"
    data["recommendation"]["поле_из_будущего"] = 1
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False)

    rec = recommendation_from_dict(load_trace(path)["recommendation"])
    assert rec.reason


# ----------------------------------------------------------------------
def test_two_source_policies_are_different_on_purpose():
    """
    best_quality - приоритет достоверности (ЛИМС), freshest_usable -
    приоритет свежести (ПАК). Для якоря по сере нужен свежий факт.
    """
    ts = datetime(2026, 3, 14, 8, 20)
    state = ProcessState(
        ts=ts,
        lims={"sulfur_mgkg": Measurement(9.4, ts - timedelta(hours=9), 540.0, "LIMS", "mg/kg")},
        pak={"sulfur_mgkg": Measurement(9.6, ts, 0.0, "PAK", "mg/kg")},
    )

    assert state.best_quality("sulfur_mgkg").source == "LIMS"
    assert state.freshest_usable("sulfur_mgkg").source == "PAK"
    assert len(state.usable_sources("sulfur_mgkg")) == 2


def test_conservative_bound_takes_config_value_as_is():
    iv = Interval(9.0, 8.0, 10.0)
    assert iv.conservative("hi") == 10.0 and iv.conservative("lo") == 8.0
    assert iv.conservative("upper") == 10.0 and iv.conservative("lower") == 8.0


if __name__ == "__main__":
    print("запускать через pytest: нужны фикстуры tmp_path и monkeypatch")
