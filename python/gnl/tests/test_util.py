import numpy as np
from gnl.util import combine_axes


def test_combine_axes():

    a = np.arange(12).reshape(3,2,2)

    b = combine_axes(a, (1,2,0))
    assert b.shape == (2, 2, 3)

    b = combine_axes(a, ((1,2),0))
    assert b.shape == (4, 3)

    b = combine_axes(a, (0, (1, 2)))
    assert b.shape == (3, 4)
    assert b[0].tolist() == [0, 1, 2, 3]
    assert b[1].tolist() == [4, 5, 6, 7]


