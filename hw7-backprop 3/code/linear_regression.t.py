import unittest
import logging
import linear_regression as linear
import numpy as np
import test_utils
import graph
import nodes

logging.basicConfig(format='%(levelname)s: %(message)s',level=logging.DEBUG)

class TestNodes(unittest.TestCase):

    def test_backward_SquaredL2DistanceNode(self):
        max_allowed_rel_err = 1e-5
        a = nodes.ValueNode("a")
        b = nodes.ValueNode("b")
        node = nodes.SquaredL2DistanceNode(a,b,"L2 dist node")
        dims = ()
        init_vals = {"a":np.array(np.random.standard_normal(dims)),
                     "b":np.array(np.random.standard_normal(dims))}
        max_rel_err = test_utils.test_node_backward(node,
                                                    init_vals,
                                                    delta=1e-7)
        self.assertTrue(max_rel_err < max_allowed_rel_err)

        # Not used for linear regression, but can also apply the
        # node to higher dimensional arrays
        dims = (10)
        init_vals = {"a":np.array(np.random.standard_normal(dims)),
                     "b":np.array(np.random.standard_normal(dims))}
        max_rel_err = test_utils.test_node_backward(node,
                                                    init_vals,
                                                    delta=1e-7)
        self.assertTrue(max_rel_err < max_allowed_rel_err)

        dims = (10,10)
        init_vals = {"a":np.array(np.random.standard_normal(dims)),
                     "b":np.array(np.random.standard_normal(dims))}
        max_rel_err = test_utils.test_node_backward(node,
                                                    init_vals,
                                                    delta=1e-7)
        self.assertTrue(max_rel_err < max_allowed_rel_err)


    def test_backward_VectorScalarAffineNode(self):
        max_allowed_rel_err = 1e-5
        w = nodes.ValueNode("w")
        x = nodes.ValueNode("x")
        b = nodes.ValueNode("b")
        affine_node = nodes.VectorScalarAffineNode(x,w,b,"affine node")
        num_ftrs = 5
        init_vals = {"w":np.random.randn(num_ftrs),
                     "x":np.random.randn(num_ftrs),
                     "b":np.array(np.random.randn())}
        max_rel_err = test_utils.test_node_backward(affine_node,
                                                    init_vals,
                                                    delta=1e-7)
        self.assertTrue(max_rel_err < max_allowed_rel_err)

    def test_linear_regression_gradient(self):
        estimator = linear.LinearRegression()
        d = 5
        input_vals = {"x": np.random.randn(d)}
        outcome_vals = {"y": np.array(np.random.randn())}
        parameter_vals = {"w": np.random.randn(d), "b":np.array(np.random.randn())}

        test_utils.test_ComputationGraphFunction(estimator.graph, input_vals, outcome_vals, parameter_vals)
        self.assertTrue(1 == 1)



if __name__ == "__main__":
    unittest.main()
