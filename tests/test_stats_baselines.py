"""Tests de eval/stats.py y de los baselines deterministas (sin arena)."""
import math

import numpy as np
import pytest

from bringvgc.baselines.methods import RandomMethod, TiposMethod, UsageMethod
from bringvgc.brings import BRINGS
from bringvgc.data.usage import UsageStats
from bringvgc.eval.stats import decision, paired_difference, t975, t_interval, wilson
from bringvgc.parsing.showdown import parse_packed_team
from tests.test_data import TEAM1, TEAM2


def test_wilson_known_value():
    p, lo, hi = wilson(550, 1000)
    assert p == 0.55
    assert lo == pytest.approx(0.5191, abs=1e-3) and hi == pytest.approx(0.5805, abs=1e-3)
    assert hi - lo == pytest.approx(0.0614, abs=2e-3)   # +-0.03 con n=1000


def test_t_interval_and_quantiles():
    assert t975(19) == pytest.approx(2.093)
    assert t975(200) == pytest.approx(1.972, abs=2e-3)
    m, lo, hi, sd = t_interval([0.5, 0.6, 0.7])
    assert m == pytest.approx(0.6) and sd == pytest.approx(0.1)
    assert hi - m == pytest.approx(4.303 * 0.1 / math.sqrt(3))


def test_paired_difference():
    d, se = paired_difference([0.6, 0.7, 0.5], [0.5, 0.5, 0.5])
    assert d == pytest.approx(0.1)
    assert se == pytest.approx(np.std([0.1, 0.2, 0.0], ddof=1) / math.sqrt(3))


def test_decision_rule():
    assert decision([0.60] * 10 + [0.62] * 10)["verdict"] == "supera el 55 %"
    assert decision([0.40, 0.75] * 10)["verdict"] == "compatible, no concluyente"
    assert decision([0.50] * 20)["verdict"] == "se rechaza"


def test_random_uniform_and_usage_distribution():
    t1, t2 = parse_packed_team(TEAM1), parse_packed_team(TEAM2)
    assert np.allclose(RandomMethod().probs(t1, t2), 1 / 90)
    u = UsageMethod(UsageStats({}, prior=5.0))
    p = u.probs(t1, t2)
    assert p.shape == (90,) and p.sum() == pytest.approx(1)
    am = UsageMethod(UsageStats({}, prior=5.0), argmax=True).probs(t1, t2)
    assert am.sum() == 1 and am.max() == 1


def test_tipos_is_deterministic_and_legal():
    t1, t2 = parse_packed_team(TEAM1), parse_packed_team(TEAM2)
    m = TiposMethod()
    p1, p2 = m.probs(t1, t2), m.probs(t1, t2)
    assert (p1 == p2).all() and p1.sum() == 1
    L, R = BRINGS[int(p1.argmax())]
    s = m.mon_scores(t1, t2)
    assert min(s[list(L)]) >= max(s[list(R)]) - 1e-12
