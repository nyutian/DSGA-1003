""" Test cases for nodes.py """
import pdb
#        pdb.set_trace()
#
import unittest
import logging
import mlp_regression
import nodes
import numpy as np
import test_utils

class TestNodes(unittest.TestCase):

    def test_AffineNode(self):
        W = nodes.ValueNode(node_name="W")
        x = nodes.ValueNode(node_name="x")
        b = nodes.ValueNode(node_name="b")
        affine_node = nodes.AffineNode(W, x, b, "affine")
        m = 8
        d = 5
        init_vals = {"W":np.random.randn(m,d),
                     "b":np.random.randn(m),
                     "x":np.random.randn(d)}

        max_rel_err = test_utils.test_node_backward(affine_node, init_vals, delta=1e-7)
        max_allowed_rel_err = 1e-5
        self.assertTrue(max_rel_err < max_allowed_rel_err)

    def test_TanhNode(self):
        a = nodes.ValueNode(node_name="a")
        tanh_node = nodes.TanhNode(a, "tanh")
        m = 8
        d = 5
        init_vals = {"a":np.random.randn(m,d)}

        max_rel_err = test_utils.test_node_backward(tanh_node, init_vals, delta=1e-7)
        max_allowed_rel_err = 1e-5
        self.assertTrue(max_rel_err < max_allowed_rel_err)

    def test_mlp_regression_gradient(self):
        estimator = mlp_regression.MLPRegression()
        num_hidden_units = 4
        num_ftrs = 5
        input_vals = {"x": np.random.randn(num_ftrs)}
        outcome_vals = {"y": np.array(np.random.randn())}
        parameter_vals = {"W1": np.random.standard_normal((num_hidden_units, num_ftrs)),
                          "b1": np.random.standard_normal((num_hidden_units)),
                          "w2": np.random.standard_normal((num_hidden_units)),
                          "b2": np.array(np.random.randn()) }

        max_rel_err = test_utils.test_ComputationGraphFunction(estimator.graph, input_vals, outcome_vals, parameter_vals)

        max_allowed_rel_err = 1e-5
        self.assertTrue(max_rel_err < max_allowed_rel_err)



if __name__ == "__main__":
    unittest.main()
