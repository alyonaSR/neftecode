"""
Резолвер тегов и загрузка конфигов из config/*.yaml.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Tuple

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "config")


@lru_cache(maxsize=None)
def load_config(name: str) -> Dict[str, Any]:
    """Читает config/<name>.yaml с кэшированием. Возвращает содержимое как словарь."""
    with open(os.path.join(CONFIG_DIR, f"{name}.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_tag(key: str) -> Tuple[str, str]:
    """'242000:T5' -> ('242000', 'T5'). Возвращает (установка, код); ключ без установки вызывает KeyError."""
    if ":" not in key:
        raise KeyError(
            f"Тег '{key}' без установки. Код T6 на АВТ и на 24-2000 означает разные величины."
        )
    unit, code = key.split(":", 1)
    return unit, code


def make_tag(unit: str, code: str) -> str:
    """Склеивает установку и код в ключ. Возвращает строку вида '242000:T5'."""
    return f"{unit}:{code}"


def tag_info(key: str) -> Dict[str, Any]:
    """Берёт запись тега из catalog в config/tags.yaml. Для неизвестного тега возвращает заглушку с verified=False."""
    cat = load_config("tags").get("catalog", {})
    return cat.get(key, {"desc": "нет в справочнике", "units": "?", "verified": False})


def is_verified(key: str) -> bool:
    """Проверяет, что описание тега согласовано с данными. Возвращает bool."""
    return bool(tag_info(key).get("verified", False))


def manipulated_vars() -> Dict[str, Any]:
    """Возвращает раздел manipulated_vars из config/constraints.yaml: управляемые переменные и их диапазоны."""
    return load_config("constraints")["manipulated_vars"]


def quality_specs() -> Dict[str, Any]:
    """Возвращает раздел quality_specs из config/constraints.yaml: спецификации качества."""
    return load_config("constraints")["quality_specs"]


def refusal_rules() -> Dict[str, Any]:
    """Возвращает раздел refusal_rules из config/constraints.yaml: правила отказа (например, max_lims_age_min)."""
    return load_config("constraints")["refusal_rules"]