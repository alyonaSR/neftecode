"""
Тесты слоя языковой модели.

НИ ОДИН тест не ходит в сеть: модель подменяется заглушкой, а режим без
модели проверяется явно. Иначе тесты падали бы на площадке без интернета,
а вместе с ними и критерий воспроизводимости.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.state_builder import build_demo_state
from src.llm import build_context, numbers_in
from src.llm.qa import ask
from src.llm.validator import extract_numbers, unknown_numbers
from src.orchestrator import Orchestrator

OFFLINE = {"enabled": False}
ONLINE = {"enabled": True, "verify_numbers": True, "strict": False}


class FakeClient:
    """Заглушка модели: возвращает заранее заданный текст."""
    def __init__(self, text):
        self.text = text
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.text


def _trace(scenario="quality_risk"):
    return Orchestrator().run_cycle(build_demo_state(scenario))


# ----------------------------------------------------------------------
def test_numbers_are_extracted_with_both_separators():
    assert extract_numbers("запас 0.88 мг/кг, риск 29%, шаг -2,5") == [0.88, 29.0, -2.5]


def test_rounding_and_percent_are_not_hallucinations():
    """Модель округляет 0.262 до 0.26 и пишет 29% вместо 0.291 - это те же числа."""
    allowed = [0.262, 0.291, 10.0]
    assert unknown_numbers("запас 0.26, риск 29%, лимит 10", allowed) == []


def test_invented_number_is_detected():
    assert unknown_numbers("сера станет 7.10 мг/кг", [8.94, 9.74, 10.0]) == [7.10]


# ----------------------------------------------------------------------
def test_context_carries_decision_numbers():
    """Модель имеет право ссылаться только на числа решения - они все здесь."""
    context = build_context(_trace())
    numbers = numbers_in(context)

    assert 10.0 in numbers                       # лимит серы
    assert context["рекомендация"]["причина"]
    assert context["перебор_вариантов"]["кандидатов_всего"] > 0
    assert context["оценка_надёжности"]["допущения"]


def test_works_without_model():
    """Без ключа и без сети оператор всё равно получает ответ по числам."""
    answer = ask("почему так решили?", _trace(), cfg=OFFLINE)

    assert answer.source == "шаблон"
    assert "Рассмотрено вариантов" in answer.text
    assert answer.verified


def test_model_answer_passes_through_when_numbers_check_out():
    trace = _trace()
    margin = trace.recommendation.expected_effect["margin_to_spec"]
    client = FakeClient(f"Запас по сере {margin} мг/кг при лимите 10.")

    answer = ask("какой запас?", trace, client=client, cfg=ONLINE)

    assert answer.source == "llm" and answer.verified
    assert "ДАННЫЕ" in client.calls[0][1]        # модель получила контекст решения


def test_prompt_names_controllable_variables_explicitly():
    """
    Список управляемых переменных идёт отдельным блоком, а не только внутри
    JSON. Пока он был спрятан в контексте, модель на вопрос про температуру
    печи, которой система не управляет, отвечала про температуру реактора.
    """
    client = FakeClient("ответ")
    ask("что будет, если прибавить жара в печь?", _trace(), client=client, cfg=ONLINE)

    system, user = client.calls[0]
    assert "УПРАВЛЯЕМЫЕ ПЕРЕМЕННЫЕ" in user
    assert "242000:T5" in user.split("ДАННЫЕ")[0]      # до JSON, а не в нём
    assert "НЕ ПОДМЕНЯЙ ПАРАМЕТРЫ" in system


def test_ask_requires_explicit_scenario(monkeypatch):
    """
    Ответ относится к конкретному решению. Молчаливый режим по умолчанию
    приводил к ответу про сценарий, о котором оператор не спрашивал.
    Проверка упирается в разбор аргументов и до расчётов не доходит.
    """
    import run

    monkeypatch.setattr(sys, "argv", ["run.py", "ask", "вопрос"])
    try:
        run.main()
    except SystemExit as exit_code:
        assert exit_code.code != 0
    else:
        raise AssertionError("ask без --scenario обязан завершиться ошибкой")


def test_invented_number_is_flagged_not_hidden():
    """
    Тот же принцип, что у Gate: убедительность текста не даёт прав.
    Число, которого нет в решении, помечается для оператора.
    """
    client = FakeClient("Сера упадёт до 3.14 мг/кг, всё отлично.")

    answer = ask("что будет с серой?", _trace(), client=client, cfg=ONLINE)

    assert not answer.verified and 3.14 in answer.unverified


def test_strict_mode_replaces_unverified_answer():
    client = FakeClient("Сера упадёт до 3.14 мг/кг.")
    cfg = dict(ONLINE, strict=True)

    answer = ask("что будет с серой?", _trace(), client=client, cfg=cfg)

    # Текст модели не показывается совсем; вместо него факты решения и
    # причина отклонения, включая само выдуманное число - оператор должен
    # знать, что именно забраковано.
    assert answer.source == "шаблон"
    assert "Сера упадёт" not in answer.text
    assert "Решение:" in answer.text


def test_llm_does_not_affect_the_decision():
    """
    Ключевое архитектурное свойство: рекомендация считается кодом.
    Вопрос к модели её не меняет - иначе рушится воспроизводимость из ТЗ.
    """
    trace = _trace()
    before = dict(trace.recommendation.action)

    ask("а можно поднять выпуск?", trace, client=FakeClient("можно"), cfg=ONLINE)

    assert trace.recommendation.action == before


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"  OK  {name}")
    print("\nвсе тесты слоя LLM прошли")
