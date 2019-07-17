"""Computation graph test utilities

Below are functions that can assist in testing the implementation of backward
for individual nodes, as well as the gradient computation in a
ComputationGraphFunction. The approach is to use the secant approximation to
compute the gradient numerically, and this is then compared to the gradient
computed by the node or ComputationGraphFunction. Note that this is really only
verifying that the gradient corresponds to the function value that is being
computed. So this is only useful if you also check that the forward() is
correct in nodes, and that the get_objective() is correct in the
CopmputationGraphFunction.

This computation graph framework was designed and implemented by
Philipp Meerkamp, Pierre Garapon, and David Rosenberg.
License: Creative Commons Attribution 4.0 International License

"""

import logging
import graph
import numpy as np
logging.basicConfig(format='%(levelname)s: %(message)s',level=logging.DEBUG)

def relative_error(a,b):
    if a==0.0 and b==0.0:
        return 0.0
    # approach described here http://cs231n.github.io/neural-networks-3/
    rel_err = np.abs(a-b) / max(np.abs(a), np.abs(b))
    return rel_err

def test_node_backward(node, init_vals, delta=1e-7):

    for parent_node in node.get_predecessors():
        parent_node.out = init_vals[parent_node.node_name]

    out = graph.forward_graph(node) 
    d_out = np.random.standard_normal(out.shape) # simulate a random derivative w.r.t. out
    node.d_out = d_out 
    node.backward() # sets partials in parent.d_node for each parent

    # Numerically estimate partial derivative of J w.r.t. each entry of each parent_node.out,
    # Assuming partial of J w.r.t. node output is given by node.d_out.

    overall_max_rel_err = -1
    for parent in node.get_predecessors():
        parent_out =  np.copy(parent.out) # save original parent.out
        it = np.nditer(parent_out, flags=['multi_index'])
        max_rel_err = -1
        while not it.finished:
            parent.out[it.multi_index] = parent_out[it.multi_index] + delta
            out_plus_delta = node.forward()
            parent.out[it.multi_index] = parent_out[it.multi_index] - delta
            out_minus_delta = node.forward()
            parent.out[it.multi_index] = parent_out[it.multi_index] # set it back to how it 
            local_partial_est = (out_plus_delta - out_minus_delta) / (2.0 * delta)

            partial_est = np.sum(d_out * local_partial_est) # this is the chain rule
            partial_backward = parent.d_out[it.multi_index] # partial as computed by backward
            #pdb.set_trace()
            rel_err = relative_error(partial_est, partial_backward)
            max_rel_err = max(max_rel_err, rel_err)
            it.iternext()
        logging.debug("(Node %s) Max rel error for partial deriv w.r.t. %s is %s." % (node.node_name, parent.node_name, max_rel_err))
        overall_max_rel_err = max(max_rel_err, overall_max_rel_err)
    return overall_max_rel_err


def test_ComputationGraphFunction(graph, input_vals, outcome_vals, parameter_vals, delta=1e-7):
    graph.set_parameters(parameter_vals)
    _, gradients = graph.get_gradients(input_vals, outcome_vals)

    overall_max_rel_err = -1
    for param in parameter_vals:
        val = parameter_vals[param]
        it = np.nditer(val, flags=['multi_index'])
        max_rel_err = -1
        while not it.finished:
            step = np.zeros(val.shape)
            step[it.multi_index] = delta
            graph.increment_parameters({param:step})
            obj_plus_delta = graph.get_objective(input_vals, outcome_vals)
            step[it.multi_index] = -2 * delta  #so a step of -delta from original value
            graph.increment_parameters({param:step})
            obj_minus_delta = graph.get_objective(input_vals, outcome_vals)
            step[it.multi_index] = delta  #bring it back to original value
            graph.increment_parameters({param:step})

            partial_est = (obj_plus_delta - obj_minus_delta) / (2.0 * delta)
            partial_backprop = gradients[param][it.multi_index] # partial from ComputationGraphFunction
            rel_err = relative_error(partial_est, partial_backprop)
            max_rel_err = max(max_rel_err, rel_err)
            it.iternext()
        logging.debug("(Parameter %s) Max rel error for partial deriv %s." % (param, max_rel_err))
        overall_max_rel_err = max(overall_max_rel_err, max_rel_err)
    return overall_max_rel_err

