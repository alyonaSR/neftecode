"""
L2. Агент оптимизации.
 
ЗАДАЧА: сгенерировать допустимых кандидатов внутри trust region
и оценить каждого. Отбраковку делает НЕ он, а Gate.
 
TRUST REGION: кандидаты строятся как дельты не больше max_step
из config/constraints.yaml и не выходят за allowed_ranges агента
надёжности. Это защита от экстраполяции модели в режимы,
которых в истории не было.
 
ПЕРЕБОР: сетка по активным управляемым переменным. Полный перебор по
всем шести дал бы 1.3 млн комбинаций, поэтому активных по умолчанию три,
остальные подключаются явно через active_vars.
"""
 
from __future__ import annotations
 
import itertools
from typing import Dict, List
 
from ..contracts import Candidate, ProcessState, QualityAssess, ReliabilityAssess
from ..data.tags import manipulated_vars
 
class OptimizerAgent:
    def __init__(self, quality_agent, n_steps: int = 3):
        self.quality = quality_agent
        self.n_steps = n_steps      # число шагов в каждую сторону по каждой переменной
 
    # ------------------------------------------------------------------
    def propose(
        self,
        state: ProcessState,
        quality: QualityAssess,
        reliability: ReliabilityAssess,
        active_vars: List[str] = None,
    ) -> List[Candidate]:
        # active_vars больше не зашиты в коде -
        # источник истины один, config/constraints.yaml, поле active: true.
        # Ловушка, на которую уже наступали: код и конфиг расходились (T33 убрали
        # из конфига, из кода - нет), у всей команды падало с KeyError.
        # Явный active_vars=[...] по-прежнему работает и имеет приоритет
        # (например, шаг 2 с optuna может гонять весь набор из 6).
        specs = manipulated_vars()
        active_vars = active_vars or [tag for tag, spec in specs.items() if spec.get("active")]
        if not active_vars:
            raise ValueError(
                "ни одна управляемая переменная не помечена active: true "
                "в config/constraints.yaml, и active_vars не передан явно"
            )
 
        grids: Dict[str, List[float]] = {}
        for tag in active_vars:
            spec = specs[tag]
            # grid_points задаётся в constraints.yaml персонально по переменной,
            # чтобы шаг получался круглым числом (0.5 degC, 1.0 t/h), а не 1.333.
            # Раньше единый self.n_steps делил max_step без оглядки на переменную.
            n = int(spec.get("grid_points", self.n_steps))
            step = spec["max_step"] / n
            grids[tag] = [round(step * k, 3) for k in range(-n, n + 1)]
 
        candidates: List[Candidate] = []
        for i, combo in enumerate(itertools.product(*[grids[t] for t in active_vars])):
            deltas = {t: d for t, d in zip(active_vars, combo)}
            if not self._within_allowed(state, deltas, reliability):
                continue
            pred = self.quality.assess(state, deltas=deltas).predictions
            candidates.append(
                Candidate(
                    candidate_id=f"c_{i:03d}",
                    deltas=deltas,
                    predicted=pred,
                    cost_proxy=self._cost_proxy(deltas),
                    severity_delta=self._severity_delta(deltas),
                    # AVT:F32 тоже часть дизельного
                    # пула, раньше в yield_delta не учитывался.
                    yield_delta=deltas.get("AVT:F30", 0.0) + deltas.get("AVT:F32", 0.0),
                )
            )
        return candidates
 
    # ------------------------------------------------------------------
    @staticmethod
    def _within_allowed(state, deltas, reliability) -> bool:
        """Кандидат не должен выводить параметр за allowed_ranges."""
        for tag, d in deltas.items():
            cur = state.tag(tag)
            if cur is None:
                return False
            lo, hi = reliability.allowed_ranges.get(tag, [-1e9, 1e9])
            if not (lo <= cur + d <= hi):
                return False
        return True
 
    # ------------------------------------------------------------------
    @staticmethod
    def _cost_proxy(deltas: Dict[str, float]) -> float:
        """
        Прозрачный стоимостной прокси ЧИСТЫХ ЗАТРАТ. Фактических
        экономических данных в пакете нет, ТЗ такое разрешает при явном
        описании допущений.
 
        Раньше сюда подмешивался -0.6*ΔF30 - рост
        отбора дизельной фракции одновременно снижал cost_proxy И
        увеличивал yield_delta. Это двойной учёт одной и той же выгоды:
        один раз как "рост выпуска", второй раз как "мнимая экономия".
        Orchestrator сравнивает их как разные критерии лексикографической
        цепочки (сначала запас, потом воздействие, потом severity, и
        только в конце cost_proxy) - cost_proxy обязан быть чистыми
        затратами, без части выгоды внутри.
 
        Условные единицы за цикл:
          +1.0 за каждый градус температуры реактора 242000:T5 (топливо + водород)
          +0.1 за каждую т/ч пара в стриппинг AVT:F28 (пар стоит денег)
        Рост отбора (AVT:F30, AVT:F32) в cost_proxy не входит - он есть
        в yield_delta. AVT:T33 убрана вместе с исключением из active_vars
        (см. propose()) - она не рычаг, а только признак модели.

        AVT:F28 не входит в
        active_vars по умолчанию, но если кто-то передаст его явным
        active_vars=[...] (например, при переходе на optuna), пар не
        должен доставаться бесплатно - раньше при активации F28 оптимизатор
        считал бы расход пара нулевой ценой. Оба веса условные: заменить их на реальные
        энергозатраты можно, когда появится экономический источник данных.
        """
        return (
            1.0 * deltas.get("242000:T5", 0.0)
            + 0.1 * deltas.get("AVT:F28", 0.0)
        )
 
    @staticmethod
    def _severity_delta(deltas: Dict[str, float]) -> float:
        """Рост температуры реактора = более жёсткий режим = быстрее деактивация."""
        return round(0.04 * deltas.get("242000:T5", 0.0), 4)
 
    # ------------------------------------------------------------------
    @staticmethod
    def pareto_front(candidates: List[Candidate]) -> List[Candidate]:
        """
        Недоминируемые точки по (cost_proxy, severity_delta, sulfur hi, -yield).
        Первые три минимизируются, выпуск максимизируется (поэтому со знаком
        минус - единый порядок сравнения "меньше = лучше" по всем осям).

        Раньше выпуск в ключ не входил, а
        cost_proxy и severity_delta оба зависят только от 242000:T5. Все
        точки с одинаковой температурой были неотличимы по ключу - во
        фронт попадали варианты с буквально одинаковыми (cost, severity,
        sulfur), различавшиеся только отбором, который на исход не влиял.
        Теперь отбор - четвёртая ось, вырождение снято.
        """
        def key(c: Candidate):
            s = c.predicted.get("sulfur_mgkg")
            return (c.cost_proxy, c.severity_delta, s.hi if s else 0.0, -c.yield_delta)

        front: List[Candidate] = []
        for c in candidates:
            kc = key(c)
            if not any(
                all(ko <= kk for ko, kk in zip(key(o), kc)) and key(o) != kc
                for o in candidates
            ):
                front.append(c)
        return front
