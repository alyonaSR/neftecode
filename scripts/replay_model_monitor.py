"""Прогон агента самоконтроля по истории ЛИМС. Запуск: python -m scripts.replay_model_monitor"""
from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
import pandas as pd

from src.agents.model_monitor import ModelMonitorAgent
from src.agents.quality import QualityAgent
from src.contracts import Interval, QualityAssess
from src.data.loaders import TARGET_POINT, load_lims, load_pak, load_telemetry
from src.data.state_builder import build_state

OUT = os.path.join("artifacts", "model_monitor_replay.png")


def anchored_interval(state) -> Interval:
    """Интервал серы для текущего режима, модель не нужна."""
    dummy = Interval(8.5, 8.0, 9.0)
    return QualityAgent._sulfur_anchored(None, state, dummy, dummy)


def replay(tel, lims, pak) -> pd.DataFrame:
    """Попадание интервала в каждую пробу без агента и с агентом."""
    y = lims[(lims.sample_point == TARGET_POINT) & (lims.param == "sulfur_mgkg")
             & (~lims.outlier) & (lims.value <= 20)].set_index("ts")["value"].sort_index()
    mon = ModelMonitorAgent()
    rows = []
    for ts, value in y.items():
        state = build_state(ts - timedelta(minutes=10), tel, lims, pak)
        mon.observe(state)
        if not state.usable_sources("sulfur_mgkg", 1440):
            continue
        raw = anchored_interval(state)
        adj = mon.apply(raw)
        mon.record(state, QualityAssess(predictions={"sulfur_mgkg": adj}, spec_risk_prob=0.0,
                                        confidence=1.0, drivers=[], model_id="replay"))
        rows.append({"ts": ts, "lims": value, "hi_raw": raw.hi, "hi_mon": adj.hi,
                     "scale": mon.scale, "status": mon.assess().status})
    d = pd.DataFrame(rows).set_index("ts")
    d["hit_raw"] = d.lims <= d.hi_raw
    d["hit_mon"] = d.lims <= d.hi_mon
    return d


def summary(d: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Покрытие, худшее окно и полуширина без агента и с агентом."""
    out = {}
    for name, hit, hi in (("без агента", "hit_raw", "hi_raw"), ("с агентом", "hit_mon", "hi_mon")):
        roll = d[hit].rolling(window).mean()
        out[name] = {"покрытие": d[hit].mean(), "худшее окно": roll.min(),
                     "окон ниже 80%": (roll < 0.8).mean()}
    return pd.DataFrame(out).T


def plot(d: pd.DataFrame, window: int = 30) -> None:
    """График скользящего покрытия."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(d.index, d.hit_raw.rolling(window).mean(), label="без агента", color="#c0392b")
    ax.plot(d.index, d.hit_mon.rolling(window).mean(), label="с агентом", color="#1f6f8b")
    ax.axhline(0.9, ls="--", color="gray", lw=1, label="заявлено 90%")
    ax.set_ylabel(f"покрытие, окно {window} проб")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower left")
    ax.set_title("Накрывает ли интервал серы лабораторный анализ")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)


if __name__ == "__main__":
    tel = pd.concat([load_telemetry("AVT"), load_telemetry("242000")], axis=1)
    d = replay(tel, load_lims(), load_pak())
    print(summary(d).round(3).to_string())
    print(f"\nмасштаб интервала с агентом: в среднем {d.scale.mean():.2f}, "
          f"от {d.scale.min():.2f} до {d.scale.max():.2f}")
    print("статусы:", d.status.value_counts().to_dict())
    plot(d)
    print(f"график: {OUT}")