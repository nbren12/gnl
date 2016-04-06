from gnl.util import phaseshift


def test_phaseshift():
    import numpy as np

    x = np.linspace(0, 1, 101)[:-1]
    t = np.linspace(0, 10, 101)

    xx, tt = np.meshgrid(x, t)

    c = 2
    u = np.sin(2 * np.pi * 8 * (xx - c * tt))
    u_flat = np.sin(2 * np.pi * 8 * xx)


    u_phase_shifted = phaseshift(x, t, u, c=c, x_index=-1, time_index=0)
    np.testing.assert_allclose(u_flat, u_phase_shifted, atol=1e-4)

    # import matplotlib.pyplot as plt
    # plt.subplot(211)
    # plt.contourf(u_phase_shifted)
    #
    # plt.subplot(212)
    # plt.contourf(u_flat)
    # plt.show()


    # test for transpose
    u_phase_shifted = phaseshift(x, t, u.T, c=c, x_index=0, time_index=1)
    np.testing.assert_allclose(u_flat.T, u_phase_shifted, atol=1e-4)
