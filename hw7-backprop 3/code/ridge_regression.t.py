import unittest
import logging
import ridge_regression as ridge
import numpy as np
import test_utils
import graph
import nodes

logging.basicConfig(format='%(levelname)s: %(message)s',level=logging.DEBUG)

class TestAll(unittest.TestCase):

    def test_SumNode(self):
        max_allowed_rel_err = 1e-5
        a = nodes.ValueNode("a")
        b = nodes.ValueNode("b")
        dims = ()
        a_val = np.array(np.random.standard_normal(dims))
        b_val = np.array(np.random.standard_normal(dims))
        sum_node = nodes.SumNode(a, b, "sum node")

        init_vals = {"a":a_val, "b":b_val}
        max_rel_err = test_utils.test_node_backward(sum_node, init_vals, delta=1e-7)
        self.assertTrue(max_rel_err < max_allowed_rel_err)

    def test_L2NormPenaltyNode(self):
        max_allowed_rel_err = 1e-5
        l2_reg = np.array(4.0)
        w = nodes.ValueNode("w")
        l2_norm_node = nodes.L2NormPenaltyNode(l2_reg, w, "l2 norm node")
        d = (5)
        init_vals = {"w":np.array(np.random.standard_normal(d))}
        max_rel_err = test_utils.test_node_backward(l2_norm_node, init_vals, delta=1e-7)
        self.assertTrue(max_rel_err < max_allowed_rel_err)

    def test_ridge_regression_gradient(self):
        estimator = ridge.RidgeRegression(l2_reg=.01)
        d = 5
        input_vals = {"x": np.random.randn(d)}
        outcome_vals = {"y": np.array(np.random.randn())}
        parameter_vals = {"w": np.random.randn(d), "b":np.array(np.random.randn())}

        test_utils.test_ComputationGraphFunction(estimator.graph, input_vals, outcome_vals, parameter_vals)
        self.assertTrue(1 == 1)

if __name__ == "__main__":
    unittest.main()
