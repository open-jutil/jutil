import pytest
import numpy as np
import jutil.regularization as reg


@pytest.mark.parametrize(
    "xs,rel", [
        (np.linspace(0, 2 * np.pi, 50), 1e-2),
        (np.linspace(0, 2 * np.pi, 3000), 1e-3),
        (np.concatenate([
            np.linspace(0, np.pi, 30, endpoint=False),
            np.linspace(np.pi, 2 * np.pi, 30)
        ]), 1e-2),
        (np.concatenate([
            [0],
            np.exp(np.linspace(-2, np.log(2 * np.pi), 199))
        ]), 1e-2),
        (np.concatenate([
            [0],
            np.exp(np.linspace(-5, np.log(2 * np.pi), 2999))
        ]), 1e-3),
    ])
@pytest.mark.parametrize(
    "func,val0,val1,val2,val3",
    [(reg.create_l0, np.pi, 1958.5, 1.52655, 566.764),
     (reg.create_l1, np.pi, 330.73, 2.44537, 226.486),
     (reg.create_l2, np.pi, 8 * np.pi, 4.12855, 33.0384)
     ])
def test_l(xs, rel, func, val0, val1, val2, val3):
    L = func(xs, np.ones_like(xs), 1)

    ys = np.sin(xs)
    temp = L.dot(ys)
    # int_0^2pi sin(x)^2 dx = pi
    # int_0^2pi cos(x)^2 dx = pi
    # int_0^2pi (-sin(x))^2 dx = pi
    assert temp.dot(temp).sum() == pytest.approx(val0, rel=rel)

    ys = xs ** 2
    temp = L.dot(ys)
    # int_0^2pi x^4 dx = 1958.5
    # int_0^2pi (2x)^2 dx = 330.73
    # int_0^2pi 4 dx = 8 * pi
    assert temp.dot(temp).sum() == pytest.approx(val1, rel=rel)

    # test with variable corr and std
    corr = (0.1 * (xs[1:] + xs[:-1]) / 2) + 1
    std = (0.1 * xs) + 1

    L = func(xs, std, corr)

    if func == reg.create_l2:
        return True

    ys = np.sin(xs)
    temp = L.dot(ys)
    # int_0^2pi (0.1x+1)^(-3) sin(x)^2 dx = 1.52655
    # int_0^2pi (0.1x+1)^(-1) cos(x)^2 dx = 2.44537
    # int_0^2pi (0.1x+1)^(1) (-sin(x))^2 dx = 4.12855
    assert temp.dot(temp).sum() == pytest.approx(val2, rel=rel)

    ys = xs ** 2
    temp = L.dot(ys)
    # int_0^2pi (0.1x+1)^(-3) x^4 dx = 566.764
    # int_0^2pi (0.1x+1)^(-1) (2x)^2 dx = 226.486
    # int_0^2pi (0.1x+1)^(1) 4 dx = 33.0384
    assert temp.dot(temp).sum() == pytest.approx(val3, rel=rel)


def test_generate_regblock():
    xs = np.linspace(0, 2 * np.pi, 100)
    ys = np.sin(xs)

    Ls = reg.generate_regblock(xs, scale0=1., scale1=1., scale2=1.)
    Sa_inv = reg.generate_inverse_covmatrix(Ls)
    assert ys.T.dot(Sa_inv.dot(ys)) == pytest.approx(1.5 * np.pi, rel=1e-3)


def test_generate_regularization():
    axis = np.arange(2)
    axes = [axis, axis]
    scales0 = [10., 10.]
    scales1 = [0., 0.]
    Sa_inv_a = np.array([[25, 0, 0, 0],
                         [0, 25, 0, 0],
                         [0, 0, 25, 0],
                         [0, 0, 0, 25]])
    Ls = reg.generate_regularization(axes, scales0=scales0, scales1=scales1)
    Sa_inv_t = reg.generate_inverse_covmatrix(Ls)
    assert np.allclose(Sa_inv_a, Sa_inv_t.toarray())

    tmp = 10 * np.sqrt(0.5)
    L0_a = np.array([[tmp, 0, 0, 0, 0],
                     [0, 10, 0, 0, 0],
                     [0, 0, tmp, 0, 0],
                     [0, 0, 0, tmp, 0],
                     [0, 0, 0, 0, tmp]])
    L1_a = np.array([[-5, 5, 0, 0, 0],
                     [0, -5, 5, 0, 0],
                     [0, 0, 0, -10, 10]])
    axes = [np.arange(3), np.arange(2)]
    scales0 = [10., 10.]
    scales1 = [5., 10.]
    Ls = reg.generate_regularization(axes, scales0=scales0, scales1=scales1)
    assert np.allclose(L0_a, Ls[0].toarray())
    assert np.allclose(L1_a, Ls[1].toarray())

    Sa_inv_a = np.array([[1.5, -1, 0, 0, 0, 0],
                         [-1, 1.5, 0, 0, 0, 0],
                         [0, 0, 1.5, -1, 0, 0],
                         [0, 0, -1, 1.5, 0, 0],
                         [0, 0, 0, 0, 1.5, -1],
                         [0, 0, 0, 0, -1, 1.5]])
    axis = np.arange(2)
    axes = [axis, axis, axis]
    std = np.sqrt(0.5)
    list_stds = [std, std, std]
    list_corr = [1, 1, 1]
    Ls = reg.generate_regularization(axes, list_stds=list_stds, list_corr=list_corr)
    Sa_inv_t = reg.generate_inverse_covmatrix(Ls)
    assert np.allclose(Sa_inv_a, Sa_inv_t.toarray())
