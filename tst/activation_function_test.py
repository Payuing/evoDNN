import numpy as np
import unittest as ut

from src import activation_funciton as af


class TestActivationFunctionMethods(ut.TestCase):

    def test_sigmoid(self):
        x = np.array([1, 2, 3])
        expected = 1 / (1 + np.exp(-x))

        actual = af.sigmoid(x)

        self.assertEqual(expected, actual)

    def test_tanh(self):
        x = np.array([1, 2, 3])
        expected = np.tanh(x)

        actual = af.tanh(x)

        self.assertEqual(expected, actual)

    def test_relu(self):
        x = np.array([-1, -2, 3])
        expected = np.array([0, 0, 3])

        actual = af.relu(x)

        self.assertEqual(expected, actual)

    def lrelu_test(self):
        alpha = 0.01
        x = np.array([-1, -2, 3])
        expected = np.array([-0.01, -0.02, 3])

        actual = af.lrelu(x, alpha)

        self.assertEqual(expected, actual)


if __name__ == "__main__":
    ut.main()
