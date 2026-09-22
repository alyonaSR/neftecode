"""
Трейс-логирование цикла.

Каждый прогон пишет JSON: вход -> оценки агентов -> кандидаты -> вердикты
Gate -> финал. Закрывает критерии ТЗ «Воспроизводимость» и «Объяснимость»:
логику любого решения можно проверить постфактум, а отчёт - воспроизвести
из файла командой `python run.py replay <trace.json>`.

"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import fields
from datetime import datetime
from typing import Any, Dict, Optional

from .contracts import SCHEMA_VERSION, DecisionTrace, Recommendation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR = os.path.join(ROOT, "artifacts", "traces")
CONFIG_DIR = os.path.join(ROOT, "config")


def save_trace(trace: DecisionTrace, tag: str = "", full: bool = False) -> str:
    """
    Записать трейс. Возвращает путь к файлу.

    Имя включает и момент решения, и момент записи: у демо-сценариев
    момент решения один и тот же, и без второй метки повторный прогон
    молча затирал предыдущий.
    """
    os.makedirs(TRACE_DIR, exist_ok=True)

    payload: Dict[str, Any] = trace.to_dict()
    payload["meta"] = build_meta(trace, full=full)
    if not full:
        payload = _compact(payload)

    path = _free_path(trace, tag)
    with open(path, "w", encoding="utf-8") as f:
        if full:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    return path


def _free_path(trace: DecisionTrace, tag: str) -> str:
    """
    Свободное имя файла. Секунды недостаточно: три сценария демо пишутся
    в один и тот же момент времени, и второй затирал первый.
    """
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    base = f"{trace.ts:%Y%m%dT%H%M}{'_' + tag if tag else ''}_{stamp}"
    path = os.path.join(TRACE_DIR, f"{base}.json")

    suffix = 1
    while os.path.exists(path):
        path = os.path.join(TRACE_DIR, f"{base}-{suffix}.json")
        suffix += 1
    return path


def load_trace(path: str) -> Dict[str, Any]:
    """
    Прочитать трейс. Расхождение версии схемы - предупреждение, не отказ:
    старый трейс должен открываться, просто в нём может не быть новых полей.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    version = data.get("schema_version")
    if version and version != SCHEMA_VERSION:
        print(
            f"[!] трейс записан схемой {version}, текущая {SCHEMA_VERSION}: "
            "часть полей может отсутствовать",
            file=sys.stderr,
        )
    return data


def recommendation_from_dict(data: Dict[str, Any]) -> Recommendation:
    """
    Собрать рекомендацию обратно для печати отчёта из файла.

    Неизвестные поля игнорируются: трейс, записанный более новой версией,
    обязан открываться старым кодом, иначе разбор инцидента упрётся в
    несовпадение версий ровно тогда, когда он нужен.
    """
    known = {f.name for f in fields(Recommendation)}
    kwargs = {k: v for k, v in data.items() if k in known}
    kwargs["ts"] = datetime.fromisoformat(kwargs["ts"])
    return Recommendation(**kwargs)


def build_meta(trace: DecisionTrace, full: bool = False) -> Dict[str, Any]:
    """Чем именно получен результат: код, конфиги, модели, окружение."""
    return {
        "записан": datetime.now().isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
        "полный_дамп": full,
        "код": {"git_commit": _git_commit(), "python": sys.version.split()[0]},
        "модели": trace.quality.model_id,
        "конфиги": _config_checksums(),
        "библиотеки": _library_versions(),
    }


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _config_checksums() -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not os.path.isdir(CONFIG_DIR):
        return out
    for name in sorted(os.listdir(CONFIG_DIR)):
        if not name.endswith(".yaml"):
            continue
        with open(os.path.join(CONFIG_DIR, name), "rb") as f:
            out[name] = hashlib.sha256(f.read()).hexdigest()[:12]
    return out


def _library_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for module in ("numpy", "pandas", "lightgbm", "sklearn"):
        try:
            out[module] = __import__(module).__version__
        except Exception:
            # Отсутствие библиотеки - факт об окружении, а не ошибка:
            # система работает и без обученных моделей.
            out[module] = "нет"
    return out


TRACE_PRECISION = 4


def _compact(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Три сокращения, ни одно не теряет смысла решения:

    1. список выполненных проверок одинаков у всех кандидатов - храним
       один раз на цикл;
    2. числа округляются до четырёх знаков - запасы считаются с точностью
       до тысячных, а прогноз серы с семнадцатью знаками занимал половину
       файла;
    3. версия схемы пишется один раз в корне, а не у каждого из 1089
       кандидатов.

    Сами кандидаты, их прогнозы и запасы остаются полностью: без них
    трейс перестал бы доказывать, почему вариант отброшен.
    """
    verdicts = payload.get("verdicts") or []
    if verdicts:
        payload["проверки_цикла"] = verdicts[0].get("checked", [])
        for verdict in verdicts:
            verdict.pop("checked", None)

    for key in ("candidates", "verdicts"):
        for item in payload.get(key) or []:
            item.pop("schema_version", None)

    return _round_floats(payload)


def _round_floats(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, TRACE_PRECISION)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v) for v in obj]
    return obj
