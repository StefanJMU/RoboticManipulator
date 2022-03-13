import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from ManipulatorCore import Joint, ManipulatorCore


def test_arm_1():

    ph = np.pi / 2
    bot = ManipulatorCore([

        Joint('prismatic', -ph, 10, 0, ph),
        Joint('prismatic', -ph, 20, 0, ph),
        Joint('prismatic', np.pi, 30, 0, 0)
    ])
    assert_array_almost_equal(bot.arm_matrix,
                              np.array([[0, 1, 0, -20],
                                        [0, 0, 1, 30],
                                        [1, 0, 0, 10],
                                        [0, 0, 0, 1]]))


def test_arm_2():

    ph = np.pi / 2
    bot = ManipulatorCore([

        Joint('revolute', ph, 450, 0, ph),
        Joint('prismatic', 0, 20, 0, ph),
        Joint('revolute', 0, 250, 0, 0)
    ])

    assert_array_almost_equal(bot.arm_matrix,
                              np.array([[0, 1, 0, 20],
                                        [1, 0, 0, 0],
                                        [0, 0, -1, 200],
                                        [0, 0, 0, 1]]))


def test_arm_3():
    ph = np.pi / 2
    bot = ManipulatorCore([

        Joint('revolute', 0, 0, 0, 0),
        Joint('prismatic', 0, 20, 0, ph),
        Joint('prismatic', 0, 30, 0, ph),
        Joint('revolute',  ph, 40, 0, 0)
    ])

    assert_array_almost_equal(bot.arm_matrix,
                              np.array([[0, -1, 0, 0],
                                        [-1, 0, 0, -30],
                                        [0, 0, -1, -20],
                                        [0, 0, 0, 1]]))