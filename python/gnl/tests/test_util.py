import numpy as np
from gnl.util import combine_axes, dftderiv, phaseshift, linearf2matrix


def test_combine_axes():

    a = np.arange(12).reshape(3, 2, 2)

    b = combine_axes(a, (1, 2, 0))
    assert b.shape == (2, 2, 3)

    b = combine_axes(a, ((1, 2), 0))
    assert b.shape == (4, 3)

    b = combine_axes(a, (0, (1, 2)))
    assert b.shape == (3, 4)
    assert b[0].tolist() == [0, 1, 2, 3]
    assert b[1].tolist() == [4, 5, 6, 7]


def test_dftderiv():
    n = 200
    d = 2 * np.pi / n

    x = np.r_[:n] * d
    y = np.sin(x)

    yp = np.fft.ifft(dftderiv(n, d) * np.fft.fft(y))
    yp = yp.real

    np.testing.assert_allclose(yp, np.cos(x), atol=1e-6)


def test_phaseshift():

    x = np.arange(20)
    t = np.array([0, 1, 2])

    xx, tt = np.meshgrid(x, t)

    c = 10
    u = np.sin(2 * np.pi * 2 * (xx - c * tt)/20)
    u_flat = np.sin(2 * np.pi * 2 * xx/20)

    u_phase_shifted = phaseshift(x, t, u, c=c)
    np.testing.assert_allclose(u_flat, u_phase_shifted, atol=1e-4)

    # import matplotlib.pyplot as plt
    # plt.subplot(211)
    # plt.pcolormesh(u_phase_shifted)
    #
    # plt.subplot(212)
    # plt.pcolormesh(u_flat)
    # plt.show()



def test_linearf2matrix():
    n, m = 100, 200

    R = np.random.rand(m, n)

    def linearfunc(x):
        y = np.zeros(m)

        for i in range(0, y.shape[0]):
            y[i] = R[i, :].dot(x)

        return y

    A = linearf2matrix(linearfunc, n)
    np.testing.assert_allclose(A, R)
