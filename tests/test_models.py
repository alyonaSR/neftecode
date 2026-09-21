"""
Тесты предсказателей качества.

Зона ответственности: Person 3.
Эти тесты проверяют не точность, а ЗНАКИ и интерфейс. Они должны
продолжать проходить после замены заглушек на обученные модели —
если перестали, модель выучила контур регулирования вместо физики.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.contracts import Interval
from src.models import AVTModel, GOModel
from src.models.base import monotone_vector

# Медианы по очищенной истории 2023-2026 (см. src/data/state_builder.py,
# те же значения). Формулы ВАК работают на абсолютных тегах, не на
# приращениях -- untrained AVTModel без полного набора входов формулу
# не считает (formula_residual.baseline() ловит KeyError и отдаёт 0.0),
# поэтому тесты знаков дают полный вектор признаков.
AVT_BASE = {
    "AVT:F30": 127.91, "AVT:T33": 338.24, "AVT:F36": 131.32,
    "AVT:T37": 60.9, "AVT:T40": 177.61, "AVT:T58": 58.32,
    "AVT:F32": 81.19, "AVT:T66": 254.14, "AVT:F65": 920.0,
    "AVT:P67": 1.12, "AVT:P4": 3.85,
}


def test_avt_interface():
    m = AVTModel()
    out = m.predict(AVT_BASE)
    assert set(out) == set(m.outputs)
    assert all(isinstance(v, Interval) for v in out.values())
    assert all(v.lo <= v.mean <= v.hi for v in out.values())


def test_avt_more_diesel_draw_means_heavier_tail():
    """Больше отбор дизельной фракции -> EBP растёт (AVT6:240-350:EBP)."""
    m = AVTModel()
    low = m.predict({**AVT_BASE, "AVT:F30": 110.0})["feed_ebp_c"].mean
    high = m.predict({**AVT_BASE, "AVT:F30": 150.0})["feed_ebp_c"].mean
    assert high > low


# Stage 2: GOModel больше не работает через anchor+дельты (см. докстринг
# go.py -- обучающих примеров "что было бы при таком-то Δ" в истории нет).
# Вход теперь абсолютный: сырые теги 24-2000 (в т.ч. предпосчитанные
# лаг/волатильность-ключи) + выход AVTModel.
#
# catalyst_age_days убран из sulfur_mgkg (эксперимент 2,
# scripts/experiments_sulfur.py, см. память hackathon_neftecode_ml_stage3):
# буквально функция календарного времени, главный подозреваемый в переносе
# temporal drift между train- и calib-частью сплита.
GO_BASE = {
    "242000:T5": 370.4, "242000:T5__lag3h": 366.4, "242000:T5__lag6h": 367.8,
    "242000:T5__std3h": 1.95, "242000:T5__std6h": 1.68,
    "feed_ebp_c": 365.0, "feed_d15_kgm3": 838.0,
    "242000:T23": 238.2, "242000:P8": 0.186, "242000:F9": 189.2,
    "242000:W7": 0.188, "242000:P24": 0.62,
}


def _artifact_path(name):
    import os
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "artifacts", "models", name)


def _load_go_or_skip(test_name):
    import os
    path = _artifact_path("go_v1.joblib")
    if not os.path.exists(path):
        print(f"  SKIP {test_name} (нет artifacts/models/go_v1.joblib, запусти scripts/train_go.py)")
        return None
    return GOModel.load(path)


def test_go_interface():
    m = GOModel()
    out = m.predict(GO_BASE)
    assert set(out) == set(m.outputs)
    assert all(isinstance(v, Interval) for v in out.values())
    assert all(v.lo <= v.mean <= v.hi for v in out.values())


def test_go_hotter_reactor_means_less_sulfur():
    """
    Аррениус: горячее -> глубже обессеривание. Знак критичен.

    Stage 2 находка: без формулы-baseline LightGBM выучивал contour
    регулирования и вообще не использовал T5 (feature_importance=0,
    вся сила уходила в волатильность std3h). Поэтому сера теперь
    formula+residual (go.sulfur_arrhenius_baseline), не чистый ML --
    это как раз то, что тест обязан ловить, если кто-то уберёт baseline.
    """
    m = GOModel()
    cold = m.predict({**GO_BASE, "242000:T5": 365.0})["sulfur_mgkg"].mean
    hot = m.predict({**GO_BASE, "242000:T5": 375.0})["sulfur_mgkg"].mean
    assert hot < cold


def test_go_heavier_feed_means_more_sulfur():
    """
    Тяжелее хвост сырья (EBP выше) -> труднее удаляемая сера.

    Как и в test_chain_avt_output_feeds_go_input: feed_ebp_c влияет на
    серу только через обученный остаток (baseline знает только T5),
    поэтому нужен обученный артефакт, иначе связь физически отсутствует.
    """
    m = _load_go_or_skip("test_go_heavier_feed_means_more_sulfur")
    if m is None:
        return
    light = m.predict({**GO_BASE, "feed_ebp_c": 355.0})["sulfur_mgkg"].mean
    heavy = m.predict({**GO_BASE, "feed_ebp_c": 385.0})["sulfur_mgkg"].mean
    # >=, не >: см. комментарий в test_chain_avt_output_feeds_go_input --
    # monotone_constraints гарантирует неубывание, не строгий рост.
    assert heavy >= light


def test_effect_of_heating_lowers_sulfur_on_trained_model():
    """
    Регрессия на H-L1 (2026-09-22). Соседний тест
    test_go_hotter_reactor_means_less_sulfur строит GOModel() ПУСТОЙ, то
    есть проверяет одну формулу -- и потому годами проходил, пока реальный
    обученный артефакт давал ОБРАТНЫЙ знак: отклик на +2 C был +0.197 мг/кг
    вместо -0.254, неверный знак у 59.4% точек.

    Причина -- monotone_constraints в LightGBM действуют ПОКОЛОНОЧНО:
    знак задан для 242000:T5, но производные от неё (arrhenius_t5,
    q20_x_arrhenius, T5_T6_quench_delta) знака не имеют, и ограничение
    обходится через них.

    Здесь проверяется ОБУЧЕННЫЙ артефакт и именно тот путь, которым
    пользуется Optimizer -- predict_effect, где эффект берётся из формулы.
    """
    m = _load_go_or_skip("test_effect_of_heating_lowers_sulfur_on_trained_model")
    if m is None:
        return
    for t5 in (360.0, 370.0, 380.0):
        now = {**GO_BASE, "242000:T5": t5}
        hotter = {**GO_BASE, "242000:T5": t5 + 2.0}
        effect = m.predict_effect(now, hotter)["sulfur_mgkg"]
        assert effect < 0, (
            f"нагрев на +2 C при T5={t5} должен СНИЖАТЬ серу, получено {effect:+.3f}"
        )


def test_derived_features_are_computed_from_raw_tags():
    """
    Регрессия на training/serving skew (2026-09-22). Фичеинжиниринг серы
    жил только в обучающих скриптах, а путь инференса клал в модель сырые
    теги -- 10 из 12 телеметрийных признаков приходили как NaN, и прогноз
    в демо выходил 3.65 мг/кг при реальной медиане 8.6.
    """
    raw = {"242000:T5": 370.0, "242000:T6": 364.0, "242000:Q20": 9500.0,
           "242000:F25": 22000.0, "242000:F9": 190.0}
    d = GOModel.derive_features(raw)
    assert d["242000:T5_T6_quench_delta"] == 6.0
    assert d["242000:h2_oil_ratio"] == 22000.0 / 190.0
    assert 0.0 < d["242000:arrhenius_t5"] < 1.0
    assert d["242000:q20_x_arrhenius"] == 9500.0 * d["242000:arrhenius_t5"]
    # уже посчитанное вызывающим кодом не перетирается
    assert GOModel.derive_features({**raw, "242000:h2_oil_ratio": 1.0})[
        "242000:h2_oil_ratio"] == 1.0
    # отсутствующее сырьё не роняет
    GOModel.derive_features({"242000:T5": 370.0})


def test_missing_features_are_reported_not_raised():
    m = GOModel()
    assert "242000:T5" in m.check_features({"catalyst_age_days": 500.0})


def test_missing_feature_in_predict_does_not_crash_lightgbm():
    """
    Найдено Person 1 (полный прогон цикла): отсутствующий тег ->
    features.get(c) -> None -> колонка DataFrame dtype=object ->
    LightGBM.predict() падает ValueError вместо штатной деградации.
    В эксплуатации дырка в теге -- рутина (поверка датчика, обрыв связи),
    не повод ронять весь цикл принятия решения.
    """
    m = _load_go_or_skip("test_missing_feature_in_predict_does_not_crash_lightgbm")
    if m is None:
        return
    incomplete = {k: v for k, v in GO_BASE.items() if k != "242000:T5__lag3h"}
    out = m.predict(incomplete)
    assert isinstance(out["sulfur_mgkg"], Interval)


def test_monotone_vector_for_lightgbm():
    cols = ["242000:T5", "feed_ebp_c", "some_unknown_feature"]
    assert monotone_vector(cols, "sulfur_mgkg") == [-1, 1, 0]


def test_chain_avt_output_feeds_go_input():
    """
    Цепочка реальна: более тяжёлый режим АВТ поднимает серу на выходе ГО.

    ВАЖНО: feed_ebp_c влияет на серу только через обученный остаток --
    у него нет формулы-baseline (только у T5 есть, см. go.py). На
    необученной GOModel() эта связь физически отсутствует, поэтому тест
    грузит artifacts/models/go_v1.joblib и мягко пропускается, если
    scripts/train_go.py ещё не запускали.
    """
    import os
    go = _load_go_or_skip("test_chain_avt_output_feeds_go_input")
    if go is None:
        return
    avt_path = _artifact_path("avt_v1.joblib")
    avt = AVTModel.load(avt_path) if os.path.exists(avt_path) else AVTModel()
    a = avt.predict(AVT_BASE)
    b = avt.predict({**AVT_BASE, "AVT:F30": 150.0})
    assert b["feed_ebp_c"].mean > a["feed_ebp_c"].mean

    s_a = go.predict({**GO_BASE, "feed_ebp_c": a["feed_ebp_c"].mean})["sulfur_mgkg"].mean
    s_b = go.predict({**GO_BASE, "feed_ebp_c": b["feed_ebp_c"].mean})["sulfur_mgkg"].mean
    # >=, не >: monotone_constraints гарантирует НЕубывание, не строгий
    # рост -- если обе точки попали в один лист дерева, s_a == s_b точно
    # (наблюдалось после ретрейна), и это не нарушение монотонности.
    assert s_b >= s_a


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(); print(f"  OK  {name}")
    print("\nвсе тесты моделей прошли")
