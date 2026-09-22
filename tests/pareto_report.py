"""
Демо-визуализация Парето-фронта.

Подключает OptimizerAgent.pareto_front() к демонстрации: показывает
недоминируемые варианты режима и цену компромисса между ними.

НАМЕРЕННО отдельный файл, а не правка run.py / orchestrator.py / explain.py:
визуализация нужна только для демо и не должна тянуть за собой
правки в рабочем цикле.

ВАЖНО, почему строится через Orchestrator.run_cycle(), а не напрямую через
QualityAgent/ReliabilityAgent/OptimizerAgent: первая версия этого файла
звала агентов напрямую и обходила Orchestrator._data_refusal(). Итог -
для сценария degraded_data (устаревший ЛИМС, залипший ПАК) отчёт всё равно
рисовал красивый Парето-фронт, хотя система обязана отказаться от
рекомендации ещё ДО генерации кандидатов. Дублировать правило отказа
здесь второй раз - верный способ незаметно рассинхронизировать его с
orchestrator.py. Поэтому используем готовый DecisionTrace: если
Orchestrator отказался, candidates там уже пустой список, и фронт
корректно окажется пустым без отдельной проверки.

Запуск:
    python -m src.agents.pareto_report                     # все 3 сценария
    python -m src.agents.pareto_report --scenario quality_risk
"""
from __future__ import annotations

import argparse

from ..contracts import DecisionTrace
from ..data.state_builder import SCENARIOS, build_demo_state
from ..gate import filter_passed
from ..orchestrator import Orchestrator
from .optimizer import OptimizerAgent


def pareto_table(trace: DecisionTrace) -> str:
    """
    Строит Парето-фронт среди кандидатов, прошедших Gate, и печатает
    его как таблицу 'было -> стало' относительно варианта 'ничего не менять'.

    Недоминируемая точка = нет другого допустимого варианта, который
    одновременно дешевле (cost_proxy), мягче для оборудования
    (severity_delta) И безопаснее по сере (sulfur hi), не проиграв ни
    по одному из трёх критериев. Те же три величины, которые
    Orchestrator сравнивает лексикографически, - фронт показывает
    то же компромиссное множество целиком, а не одну выбранную точку.

    Если trace.recommendation.is_refusal - Orchestrator уже отказался
    до генерации кандидатов (устаревшие/мёртвые данные), candidates
    пуст, и это явно печатается, а не маскируется пустым фронтом.
    """
    if trace.recommendation.is_refusal:
        return (
            "Отказ от рекомендации ДО генерации кандидатов: "
            f"{trace.recommendation.reason}\n"
            "Парето-фронт не строится - рискованное решение в этих условиях "
            "не предлагается в принципе, показывать тут нечего."
        )

    survivors = filter_passed(trace.candidates, trace.verdicts)
    if not survivors:
        return "Парето-фронт пуст: ни один кандидат не прошёл Gate."

    baseline = next((c for c in survivors if c.is_no_action), None)
    base_sulfur = (
        baseline.predicted["sulfur_mgkg"].mean
        if baseline is not None
        else trace.quality.predictions["sulfur_mgkg"].mean
    )

    front = sorted(
        OptimizerAgent.pareto_front(survivors),
        key=lambda c: c.predicted["sulfur_mgkg"].hi,
    )

    w = 98
    lines = [
        "=" * w,
        f"ПАРЕТО-ФРОНТ  ({len(front)} недоминируемых из {len(survivors)} допустимых кандидатов)",
        "=" * w,
        f"{'candidate':<10}{'изменения':<32}{'сера mean':>11}{'сера hi':>10}"
        f"{'Δ к базе':>11}{'cost':>8}{'severityΔ':>11}{'yieldΔ':>8}",
        "-" * w,
    ]

    if baseline is not None:
        s = baseline.predicted["sulfur_mgkg"]
        lines.append(
            f"{'БАЗА':<10}{'без изменений':<32}{s.mean:>11.2f}{s.hi:>10.2f}"
            f"{'-':>11}{0.0:>8.2f}{0.0:>11.3f}{0.0:>8.1f}"
        )

    for c in front:
        s = c.predicted["sulfur_mgkg"]
        deltas_str = ", ".join(
            f"{tag.split(':')[-1]}{d:+.1f}" for tag, d in c.deltas.items() if abs(d) > 1e-9
        ) or "без изменений"
        lines.append(
            f"{c.candidate_id:<10}{deltas_str:<32}{s.mean:>11.2f}{s.hi:>10.2f}"
            f"{s.mean - base_sulfur:>+11.2f}{c.cost_proxy:>8.2f}"
            f"{c.severity_delta:>11.3f}{c.yield_delta:>8.1f}"
        )

    lines.append("=" * w)
    lines.append(
        "Читать так: любая точка ниже - снижает риск по сере ценой роста cost_proxy\n"
        "и/или severity_delta. Оператор выбирает из фронта сам, Orchestrator по\n"
        "умолчанию берёт компромисс по лексикографическому правилу (см. orchestrator.py)."
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", choices=list(SCENARIOS), default=None)
    args = ap.parse_args()

    orch = Orchestrator()
    for name in ([args.scenario] if args.scenario else SCENARIOS):
        trace = orch.run_cycle(build_demo_state(name))
        print(f"\n### сценарий: {name} ###")
        print(pareto_table(trace))


if __name__ == "__main__":
    main()
