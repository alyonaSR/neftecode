#!/usr/bin/env python3
"""
Точка входа.

    python run.py demo              все три сценария ТЗ
    python run.py demo --scenario quality_risk
    python run.py demo --no-telemetry    быстро, без исторических рядов

    python run.py ask "почему не подняли температуру сильнее?"
    python run.py ask --scenario normal "почему ничего не меняем?"

Воспроизводимость: seed фиксируется, трейсы пишутся в artifacts/traces/.
Сценарии независимы друг от друга: перед каждым память агента самоконтроля
возвращается в одно и то же состояние, см. main().
Команда ask объясняет УЖЕ принятое решение и на него не влияет: без ключа
языковой модели она отвечает шаблоном по тем же числам.
"""
from __future__ import annotations

import argparse
import copy
import glob
import os
import random

import numpy as np

from src.data.state_builder import SCENARIOS, build_demo_state
from src.explain import print_operator_report
from src.llm import ask
from src.orchestrator import Orchestrator
from src.tracing import TRACE_DIR, load_trace, recommendation_from_dict, save_trace

TITLES = {
    "normal": "СЦЕНАРИЙ 1. Устойчивый режим (лишних воздействий быть не должно)",
    "quality_risk": "СЦЕНАРИЙ 2. Риск ухудшения качества",
    "degraded_data": "СЦЕНАРИЙ 3. Неполные / устаревшие / аномальные данные",
}


def run_scenario(name: str, orch: Orchestrator) -> None:
    print("\n" + "#" * 78)
    print("#  " + TITLES[name])
    print("#" * 78)

    state = build_demo_state(name)
    trace = orch.run_cycle(state)

    # Обмен между агентами печатает сам отчёт (блок 10): он часть ответа
    # системы, а не отладочный вывод демо-скрипта.
    print()
    print(print_operator_report(trace.recommendation, trace))
    path = save_trace(trace, tag=name)
    print(f"\ntrace -> {path}")


def load_demo_telemetry():
    """
    Обе установки в одном кадре: агент надёжности смотрит и АВТ, и 24-2000.

    Возвращает None, если файлов нет. Данные организаторов в репозиторий не
    коммитятся, поэтому на свежем клоне их не будет - демо обязано
    запускаться и без них, просто с двумя факторами тяжести вместо пяти.
    """
    import pandas as pd

    from src.data.loaders import load_telemetry

    try:
        print("загрузка телеметрии из data/ ...")
        frame = pd.concat([load_telemetry("AVT"), load_telemetry("242000")], axis=1)
    except FileNotFoundError as e:
        print(f"  телеметрии нет ({e.filename}), работаем без исторических рядов")
        return None

    print(f"  {len(frame)} точек, {frame.index.min():%Y-%m-%d} .. {frame.index.max():%Y-%m-%d}")
    return frame


def warm_up_monitor(orch: Orchestrator, telemetry) -> None:
    """Прогрев агента самоконтроля на реальных анализах до момента демо."""
    from src.data.loaders import load_lims, load_pak

    until = build_demo_state("normal").ts
    try:
        n = orch.monitor.warm_up(orch.quality, telemetry, load_lims(), load_pak(), until)
    except FileNotFoundError as e:
        print(f"самоконтроль: нет данных для прогрева ({e.filename})")
        return
    print(f"самоконтроль: {n} сверок с лабораторией до {until:%Y-%m-%d %H:%M}, "
          f"{orch.monitor.assess().message}")


def run_question(question: str, scenario: str, orch: Orchestrator) -> None:
    trace = orch.run_cycle(build_demo_state(scenario))
    answer = ask(question, trace)

    # Ответ относится к конкретному решению, поэтому решение печатается
    # рядом с вопросом: иначе непонятно, о каком режиме речь.
    rec = trace.recommendation
    action = ("отказ от рекомендации" if rec.is_refusal else
              ", ".join(f"{tag} {d:+.2f}" for tag, d in rec.action.items()
                        if abs(d) > 1e-9) or "режим не менять")

    print("\n" + "=" * 78)
    print(f"ВОПРОС ОПЕРАТОРА: {question}")
    print(f"режим: {scenario}, состояние на {trace.ts:%Y-%m-%d %H:%M}")
    print(f"решение системы: {action}")
    print("=" * 78)
    print(answer.text)
    print(f"\n[источник ответа: {answer.source}]", end="")
    if answer.unverified:
        # Числа, которых нет в решении. Прятать нельзя: оператор должен
        # видеть, какой части текста верить не следует.
        print(f"  [!] не подтверждены данными: {answer.unverified}", end="")
    print()


def replay(path: str = None) -> None:
    """
    Отчёт из сохранённого трейса. Воспроизводимость по ТЗ - это в том
    числе возможность показать решение недельной давности как есть,
    ничего не пересчитывая.
    """
    if path is None:
        traces = sorted(glob.glob(os.path.join(TRACE_DIR, "*.json")))
        if not traces:
            raise SystemExit("трейсов нет: сначала `python run.py demo`")
        path = traces[-1]

    data = load_trace(path)
    meta = data.get("meta") or {}

    print(f"\nтрейс: {path}")
    if meta:
        print(f"записан {meta.get('записан')}, коммит {meta.get('код', {}).get('git_commit')}, "
              f"модели {meta.get('модели')}")
        print(f"конфиги: {meta.get('конфиги')}")
    print(print_operator_report(recommendation_from_dict(data["recommendation"])))


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument(
        "--telemetry", action=argparse.BooleanOptionalAction, default=True,
        help="исторические ряды из data/ (по умолчанию включено): агент "
             "надёжности считает все пять факторов вместо двух, агент "
             "самоконтроля прогревается на последних анализах ЛИМС. "
             "--no-telemetry для быстрого прогона без данных",
    )

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    demo_parser = sub.add_parser("demo", parents=[common], help="сценарии раздела 6 ТЗ")
    demo_parser.add_argument("--scenario", choices=list(SCENARIOS), default=None,
                             help="один сценарий вместо всех трёх")

    ask_parser = sub.add_parser(
        "ask", parents=[common], help="вопрос по принятому решению"
    )
    # Сценарий обязателен: ответ относится к конкретному решению, и раньше
    # молчаливый выбор по умолчанию приводил к тому, что оператор получал
    # ответ про режим, о котором не спрашивал.
    ask_parser.add_argument("--scenario", choices=list(SCENARIOS), required=True,
                            help="режим, о решении по которому идёт вопрос")
    ask_parser.add_argument("question", nargs="+", help="вопрос оператора")

    replay_parser = sub.add_parser("replay", help="отчёт из сохранённого трейса")
    replay_parser.add_argument("path", nargs="?", default=None,
                               help="путь к трейсу; без аргумента - последний")

    args = ap.parse_args()

    if args.command == "replay":
        replay(args.path)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)

    telemetry = load_demo_telemetry() if args.telemetry else None
    orch = Orchestrator(telemetry=telemetry)
    if telemetry is not None:
        warm_up_monitor(orch, telemetry)

    if args.command == "ask":
        run_question(" ".join(args.question), args.scenario, orch)
        return

    # Агент самоконтроля - единственный, кто копит память между циклами.
    # Сценарии ТЗ это не последовательность моментов, а три разных "что
    # если" в один и тот же момент, поэтому каждый стартует с одной и той
    # же прогретой памяти. Иначе результат зависел бы от порядка запуска.
    warmed = copy.deepcopy(orch.monitor)
    for name in ([args.scenario] if args.scenario else SCENARIOS):
        orch.monitor = copy.deepcopy(warmed)
        run_scenario(name, orch)


if __name__ == "__main__":
    main()
