"""Тесты агента самоконтроля. Тест на истории пропускается, если data/ пуста."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.agents.model_monitor import ModelMonitorAgent
from src.contracts import Interval, Measurement, ProcessState, QualityAssess

T0 = datetime(2025, 1, 1, 12, 0)
CFG = {"target_coverage": 0.90, "step": 0.2, "scale_min": 0.8, "scale_max": 3.0,
       "window": 30, "min_obs": 10, "degraded_below": 0.80, "degraded_scale": 2.0,
       "failed_below": 0.65,
       "match_max_lag_min": 60}


def _state(ts, lims_value=None, lims_ts=None):
    """Состояние с последним анализом серы или без него."""
    lims = {}
    if lims_value is not None:
        lims["sulfur_mgkg"] = Measurement(lims_value, lims_ts, (ts - lims_ts).total_seconds() / 60,
                                          "LIMS", "mg/kg")
    return ProcessState(ts=ts, tags={}, lims=lims, pak={}, dq_flags=[])


def _q(mean, hi):
    """Оценка качества с одним интервалом серы."""
    return QualityAssess(predictions={"sulfur_mgkg": Interval(mean, 2 * mean - hi, hi)})


def _cycle(mon, k, lims_value, hi=9.5):
    """Прогноз за 10 минут до пробы, затем состояние с этой пробой."""
    t_pred = T0 + timedelta(days=k)
    t_lims = t_pred + timedelta(minutes=10)
    mon.observe(_state(t_pred))
    mon.record(_state(t_pred), _q(8.5, hi))
    mon.observe(_state(t_lims + timedelta(minutes=5), lims_value, t_lims))


def test_no_labs_keeps_scale():
    """Пока анализов нет, масштаб не меняется, статус ok."""
    mon = ModelMonitorAgent(CFG)
    mon.record(_state(T0), _q(8.5, 9.5))
    assert mon.scale == 1.0
    assert mon.assess().status == "ok"


def test_misses_widen_hits_narrow():
    """Промахи расширяют интервал, серия попаданий сужает."""
    mon = ModelMonitorAgent(CFG)
    for k in range(5):
        _cycle(mon, k, lims_value=12.0)
    widened = mon.scale
    assert widened > 1.0
    for k in range(5, 40):
        _cycle(mon, k, lims_value=8.0)
    assert mon.scale < widened


def test_forecast_after_sample_is_not_used():
    """Прогноз, сделанный после отбора пробы, для сверки не годится."""
    mon = ModelMonitorAgent(CFG)
    t_lims = T0
    mon.record(_state(t_lims + timedelta(minutes=5)), _q(8.5, 9.5))
    mon.observe(_state(t_lims + timedelta(minutes=10), 12.0, t_lims))
    assert mon.assess().n_obs == 0


def test_stale_forecast_is_not_used():
    """Прогноз старше match_max_lag_min до пробы не сверяется."""
    mon = ModelMonitorAgent(CFG)
    mon.record(_state(T0), _q(8.5, 9.5))
    t_lims = T0 + timedelta(hours=3)
    mon.observe(_state(t_lims + timedelta(minutes=5), 12.0, t_lims))
    assert mon.assess().n_obs == 0


def test_same_lab_counted_once():
    """Один анализ сверяется один раз, сколько бы циклов он ни висел в состоянии."""
    mon = ModelMonitorAgent(CFG)
    _cycle(mon, 0, lims_value=12.0)
    s = _state(T0 + timedelta(minutes=30), 12.0, T0 + timedelta(minutes=10))
    for _ in range(5):
        mon.observe(s)
    assert mon.assess().n_obs == 1


def test_status_failed_when_coverage_collapses():
    """При систематических промахах статус failed."""
    cfg = dict(CFG, step=0.0)
    mon = ModelMonitorAgent(cfg)
    for k in range(15):
        _cycle(mon, k, lims_value=12.0)
    h = mon.assess()
    assert h.status == "failed"
    assert h.coverage == 0.0


def test_wide_interval_is_degraded():
    """Сильно расширенный интервал означает деградацию, даже если покрытие в норме."""
    mon = ModelMonitorAgent(CFG)
    for k in range(15):
        _cycle(mon, k, lims_value=8.0)
    mon.scale = 2.5
    assert mon.assess().status == "degraded"


def test_apply_keeps_center():
    """Масштаб меняет ширину, но не центр интервала."""
    mon = ModelMonitorAgent(CFG)
    mon.scale = 2.0
    iv = mon.apply(Interval(8.0, 7.0, 9.0))
    assert iv.mean == 8.0 and iv.lo == 6.0 and iv.hi == 10.0


def _have_data() -> bool:
    try:
        from src.data.loaders import load_lims
        load_lims()
        return True
    except (FileNotFoundError, ValueError):
        return False


@pytest.mark.skipif(not _have_data(), reason="нет файлов организаторов в data/")
def test_warm_up_before_demo():
    """Прогрев до момента демо даёт накопленную историю сверок."""
    import pandas as pd
    from src.agents.quality import QualityAgent
    from src.data.loaders import load_lims, load_pak, load_telemetry
    from src.data.state_builder import build_demo_state
    from src.models.avt import AVTModel
    from src.models.go import GOModel

    tel = pd.concat([load_telemetry("AVT"), load_telemetry("242000")], axis=1)
    qa = QualityAgent(avt_model=AVTModel(), go_model=GOModel())
    mon = ModelMonitorAgent()
    n = mon.warm_up(qa, tel, load_lims(), load_pak(), build_demo_state("normal").ts)
    h = mon.assess()
    assert n >= mon.cfg["min_obs"]
    assert h.coverage is not None
    assert qa.sulfur_scale == 1.0


@pytest.mark.skipif(not _have_data(), reason="нет файлов организаторов в data/")
def test_replay_restores_coverage():
    """На истории агент поднимает покрытие до заявленного."""
    import pandas as pd
    from scripts.replay_model_monitor import replay
    from src.data.loaders import load_lims, load_pak, load_telemetry

    tel = pd.concat([load_telemetry("AVT"), load_telemetry("242000")], axis=1)
    d = replay(tel, load_lims(), load_pak())

    assert d.hit_mon.mean() >= 0.90
    assert d.hit_mon.mean() > d.hit_raw.mean()
    assert d.hit_mon.rolling(30).mean().min() > d.hit_raw.rolling(30).mean().min()