import numpy as np

import pytest

import jutil.surface


def test_surface():
    x = np.full((5, 5), np.nan)
    x = np.ma.masked_invalid(x)
    x[1, 1] = 2
    x[3, 3] = -2
    x_p = jutil.surface.minimum_curvature_surface(x, lam=None)
    assert x_p[1, 1] == pytest.approx(x[1, 1])
    assert x_p[3, 3] == pytest.approx(x[3, 3])
    assert x_p[2, 2] == pytest.approx(0)
    x_p = jutil.surface.minimum_curvature_surface(x, T=1)
    assert x_p[1, 1] == pytest.approx(x[1, 1])
    assert x_p[3, 3] == pytest.approx(x[3, 3])
    assert x_p[2, 2] == pytest.approx(0)
    x_p = jutil.surface.minimum_curvature_surface(x, T=0)
    assert x_p[1, 1] == pytest.approx(x[1, 1])
    assert x_p[3, 3] == pytest.approx(x[3, 3])
    assert x_p[2, 2] == pytest.approx(0)
    x_p = jutil.surface.minimum_curvature_surface(x, lam=1e6)
    assert x_p[1, 1] == pytest.approx(x[1, 1])
    assert x_p[3, 3] == pytest.approx(x[3, 3])
    assert x_p[2, 2] == pytest.approx(0)
