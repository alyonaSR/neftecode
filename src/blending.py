"""
Доли компонентов дизельного пула.

Явных компонентов смешения в выданных данных нет, поэтому принято
модельное допущение: дизельный пул = два потока с АВТ, тяжёлый AVT:F30
(фр. 290-350 C) и лёгкий AVT:F32 (фр. 240-290 C). Доли - их массовая
пропорция. Состав и границы долей лежат в config/constraints.yaml,
раздел blending.

"""
from __future__ import annotations

from typing import Dict, Optional

from .contracts import ProcessState
from .data.tags import load_config


def blend_fractions(
    state: ProcessState, deltas: Optional[Dict[str, float]] = None
) -> Optional[Dict[str, float]]:
    """
    Доли компонентов после применения дельт кандидата.

    None означает "не знаю": нет расхода хотя бы одного компонента или
    суммарный расход ниже min_pool_tph. Gate получает честное незнание
    вместо деления на почти ноль - проверка при этом просто не
    выполняется, а не проходит молча.
    """
    cfg = load_config("constraints")["blending"]
    deltas = deltas or {}

    flows: Dict[str, float] = {}
    for tag in cfg["components"]:
        current = state.tag(tag)
        if current is None:
            return None
        flows[tag] = current + deltas.get(tag, 0.0)

    total = sum(flows.values())
    if total < float(cfg["min_pool_tph"]):
        return None

    return {tag: flow / total for tag, flow in flows.items()}
