"""
Подключение обученных артефактов вместо заглушек по умолчанию.

Раньше QualityAgent по умолчанию создавал
AVTModel()/GOModel() без аргументов -- необученные, formula-only, с
намеренно широким интервалом (+-8 у серы). Из-за этого Orchestrator()
без явного wiring ВСЕГДА получал верхнюю границу серы выше лимита 10 мг/кг
и отбраковывал все кандидаты, включая "ничего не менять" -- демо и
test_integration.py::test_normal_mode_produces_no_action отказывали
даже в устойчивом режиме.

Эти две функции -- единственное, что меняется: они подставляются как
default в QualityAgent.__init__ вместо голых конструкторов. Если файла
артефакта нет (например, scripts/train_go.py ещё не запускали), тихо
откатываются на formula-only модель -- цикл не падает, просто прогноз
менее точен, ровно как раньше.
"""
from __future__ import annotations

import os

from .avt import AVTModel
from .go import GOModel

ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "artifacts", "models",
)


def load_default_avt() -> AVTModel:
    path = os.path.join(ARTIFACTS_DIR, "avt_v1.joblib")
    return AVTModel.load(path) if os.path.exists(path) else AVTModel()


def load_default_go() -> GOModel:
    path = os.path.join(ARTIFACTS_DIR, "go_v1.joblib")
    return GOModel.load(path) if os.path.exists(path) else GOModel()
