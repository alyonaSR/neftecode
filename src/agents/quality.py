"""
L1a. Агент качества.

    src/models/   признаки на входе -> Interval на выходе.
                  Ничего не знает про ProcessState и спецификации.

    QualityAgent  вызывает обе модели, сцепляет их в цепочку АВТ -> ГО,
                  считает confidence по возрасту данных, считает
                  spec_risk_prob по конфигу, собирает QualityAssess.

ЦЕПОЧКА: AVTModel предсказывает качество дизельной фракции, уходящей
в гидроочистку. Её конец кипения (EBP) становится ВХОДОМ GOModel. Чем тяжелее хвост,
тем труднее удаляемая сера и тем более жёсткий режим нужен на ГО.
Это и есть связанность цепочки из ТЗ, выраженная в коде.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..contracts import Interval, ProcessState, QualityAssess
from ..data.tags import load_config, quality_specs, refusal_rules
from ..models import load_default_avt, load_default_go
from ..models.features import catalyst_age_days_scalar


class QualityAgent:
    """Заглушка с физически осмысленным поведением, чтобы цикл работал уже сегодня."""

    def __init__(self, avt_model=None, go_model=None):
        # Person 3: подмена на обученные модели, если артефакты в
        # artifacts/models/ есть (load_default_* сами проверяют файл и
        # откатываются на formula-only, если его нет -- см. models/loading.py).
        # Сигнатура predict(features) -> dict[str, Interval] не меняется,
        # поэтому подмена не затрагивает ни один другой слой.
        self.avt = avt_model or load_default_avt()
        self.go = go_model or load_default_go()
        self.model_id = f"{self.avt.model_id}+{self.go.model_id}"

    # ------------------------------------------------------------------
    def assess(
        self,
        state: ProcessState,
        deltas: Optional[Dict[str, float]] = None,
    ) -> QualityAssess:
        """
        deltas=None  -> оценка текущего состояния
        deltas={...} -> прогноз для гипотетического режима (вызывает оптимизатор)
        """
        deltas = deltas or {}
        preds = self._predict(state, deltas)
        conf, conf_drivers = self._confidence(state)
        risk = self._spec_risk(preds)
        return QualityAssess(
            predictions=preds,
            spec_risk_prob=risk,
            confidence=conf,
            drivers=self._drivers(state, deltas),
            confidence_drivers=conf_drivers,
            model_id=self.model_id,
        )

    def _predict(self, state: ProcessState, deltas: Dict[str, float]) -> Dict[str, Interval]:
        """
        Цепочка АВТ -> ГО. Обе модели вызываются ДВАЖДЫ — на текущем
        режиме и на предлагаемом — разницу берёт вызывающий код, не
        сама модель.

        """
        # 1. текущий режим АВТ
        f_now = {t: state.tag(t) for t in self.avt.required_features}
        avt_now = self.avt.predict({k: v for k, v in f_now.items() if v is not None})

        # 2. режим АВТ после предлагаемого изменения
        f_new = {t: (v + deltas.get(t, 0.0)) for t, v in f_now.items() if v is not None}
        avt_new = self.avt.predict(f_new)

        # 3. гидроочистка: сырые теги 24-2000
        go_raw_now = {t: state.tag(t) for t in self.go.required_features
                      if t.startswith("242000:")}
        go_raw_now["catalyst_age_days"] = catalyst_age_days_scalar(state.ts)
        go_raw_now = {k: v for k, v in go_raw_now.items() if v is not None}

        go_raw_new = {t: (v + deltas.get(t, 0.0)) for t, v in go_raw_now.items()}

        def go_features(avt_out, go_raw):
            return {
                **go_raw,
                "feed_ebp_c": avt_out["feed_ebp_c"].mean,
                "feed_d15_kgm3": avt_out["feed_d15_kgm3"].mean,
                "feed_cfpp_c": avt_out["feed_cfpp_c"].mean,
                "feed_flash_c": avt_out["feed_flash_c"].mean,
            }

        go_now = self.go.predict(go_features(avt_now, go_raw_now))
        go_new = self.go.predict(go_features(avt_new, go_raw_new))
        self._last_go_delta = {
            k: go_new[k].mean - go_now[k].mean for k in go_new
        }

        go_new["sulfur_mgkg"] = self._sulfur_anchored(
            state, go_now["sulfur_mgkg"], go_new["sulfur_mgkg"]
        )
        return go_new

    def _sulfur_anchored(
        self, state: ProcessState, now_iv: Interval, new_iv: Interval
    ) -> Interval:
        cfg = load_config("constraints")["sulfur_anchor"]
        max_age = refusal_rules()["max_lims_age_min"]
        usable = state.usable_sources("sulfur_mgkg", max_age)
        if not usable:
            return new_iv

        anchor = state.freshest_usable("sulfur_mgkg", max_age)
        effect = new_iv.mean - now_iv.mean
        horizon_min = cfg["cycle_min"] + anchor.age_min
        grid = cfg["growth_q90_mgkg"]

        half = float(np.interp(horizon_min, [p[0] for p in grid], [p[1] for p in grid]))
        if len(usable) == 2:
            half += abs(usable[0].value - usable[1].value)
        half += cfg["effect_uncertainty_share"] * abs(effect)

        mean = anchor.value + effect
        return Interval(mean, mean - half, mean + half)

    def _confidence(self, state: ProcessState) -> tuple:
        rules = refusal_rules()
        conf = 0.9
        reasons = []

        lims = state.lims.get("sulfur_mgkg")
        if lims and lims.age_min is not None:
            over = lims.age_min / rules["max_lims_age_min"]
            if over > 1.0:
                penalty = min(0.45, 0.25 * over)
                conf -= penalty
                reasons.append(
                    f"лабораторный анализ серы старше порога: "
                    f"{lims.age_min / 60:.0f} ч (-{penalty:.2f})"
                )

        pak = state.pak.get("sulfur_mgkg")
        if pak is not None and not pak.healthy:
            conf -= 0.30
            reasons.append("поточный анализатор серы неисправен (-0.30)")

        if state.dq_flags:
            conf -= 0.05 * len(state.dq_flags)
            reasons.append(
                f"флагов качества данных: {len(state.dq_flags)} "
                f"(-{0.05 * len(state.dq_flags):.2f})"
            )

        return max(0.0, round(conf, 3)), reasons

    def _spec_risk(self, preds: Dict[str, Interval]) -> float:
        """
        Грубая вероятность нарушить хотя бы одно требование.
        """
        worst = 0.0
        for param, spec in quality_specs().items():
            iv = preds.get(param)
            if iv is None or iv.width <= 0:
                continue
            if spec["direction"] == "max":
                p = (iv.hi - spec["limit"]) / iv.width
            else:
                p = (spec["limit"] - iv.lo) / iv.width
            worst = max(worst, min(1.0, max(0.0, p)))
        return round(worst, 3)

    def _drivers(self, state: ProcessState, deltas: Dict[str, float]) -> list:
        out = []
        # пороги = p75 и p25 по очищенной истории, см. config/constraints.yaml
        f30 = state.tag("AVT:F30")
        if f30 and f30 > 140.0:
            out.append(f"высокий отбор дизельной фракции AVT:F30={f30:.1f} т/ч, хвост тяжелее")
        f32 = state.tag("AVT:F32")
        if f32 and f32 > 90.0:
            out.append(f"высокий отбор AVT:F32={f32:.1f} т/ч")
        t5 = state.tag("242000:T5")
        if t5 and t5 < 367.0:
            out.append(f"температура реактора 242000:T5={t5:.1f} C ниже обычной, severity недостаточна")
        for k, v in deltas.items():
            out.append(f"проверяется изменение {k} на {v:+.1f}")
        return out
