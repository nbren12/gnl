import numpy as np

from .calculus import diff_to_right
import pytest


@pytest.mark.parametrize('mode,expected', [
    ('wrap', [1, 1, -2]),
    ('neumann', [1, 1, -3]),
])
def test_diff_to_right(mode, expected):
    a = np.array([1.0, 2, 3])
    v = diff_to_right(a, mode=mode)
    np.testing.assert_allclose(v, expected)
