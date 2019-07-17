"""Computation graph function and utilities

By linking nodes together, one creates a computation graph representing a
function, and one can use backpropagation to easily compute the gradient of the
graph output with respect all input values. However, when doing machine
learning, different nodes of the computation graph maybe treated differently
and have special meaning. For example, if we represent a linear function in a
computation graph, we will want the gradient w.r.t. the node representing the
parameter vector, we'll frequently want to access the node that is the linear
function, since that is our predictions, but we'll also need access to the
graph output node, since that contains the objective function value. In the
class ComputationGraphFunction below, we create a wrapper around a computation
graph to handle many of the standard things we need to do in ML. Once graph is
constructed, in the sense of constructing the nodes and linking them together,
we can construct a ComputationGraphFunction below by passing the nodes in
different lists, specifying whether a node is an input, outcome (i.e. label or
response), parameter, prediction, or objective node. [Note that not all nodes
of the graph will be one of these types. The nodes that are not explicitly
passed in one of these lists are still accessible, since they are linked to
other nodes.]

This computation graph framework was designed and implemented by Philipp
Meerkamp, Pierre Garapon, and David Rosenberg.
License: Creative Commons Attribution 4.0 International License
"""

import numpy as np

class ComputationGraphFunction:
    def __init__(self, inputs, outcomes, parameters, prediction, objective):
        """ 
        Parameters:
        inputs: list of ValueNode objects containing inputs (in the ML sense)
        outcomes: list of ValueNode objects containing outcomes (in the ML sense)
        parameters: list of ValueNode objects containing values we will optimize over
        prediction: node whose 'out' variable contains our prediction
        objective:  node containing the objective for which we compute the gradient
        """

        self.inputs = inputs
        self.outcomes = outcomes
        self.parameters = parameters
        self.prediction = prediction
        self.objective = objective

        # Create name to node lookup, so users can just supply node_name to set parameters
        self.name_to_node = {}
        self.name_to_node[self.prediction.node_name] = self.prediction
        self.name_to_node[self.objective.node_name] = self.objective
        for node in self.inputs + self.outcomes + self.parameters:
            self.name_to_node[node.node_name] = node

        # Precompute the topological and reverse topological sort of the nodes
        self.objective_node_list_forward = sort_topological(self.objective)
        self.objective_node_list_backward = sort_topological(self.objective)
        self.objective_node_list_backward.reverse()
        self.prediction_node_list_forward = sort_topological(self.prediction)

    def __set_values__(self, node_values):
        for node_name in node_values:
            node = self.name_to_node[node_name]
            node.out = node_values[node_name]

    def set_parameters(self, parameter_values):
        self.__set_values__(parameter_values)

    def increment_parameters(self, parameter_steps):
        for node_name in parameter_steps:
            node = self.name_to_node[node_name]
            node.out += parameter_steps[node_name]

    def get_objective(self, input_values, outcome_values):
        self.__set_values__(input_values)
        self.__set_values__(outcome_values)
        obj = forward_graph(self.objective, node_list=self.objective_node_list_forward)
        return obj

    def get_gradients(self, input_values, outcome_values):
        obj = self.get_objective(input_values, outcome_values) #need forward pass anyway
        #print("backward node list: ",self.objective_node_list_backward)
        backward_graph(self.objective, node_list=self.objective_node_list_backward)
        parameter_gradients = {}
        for node in self.parameters:
            parameter_gradients[node.node_name] = node.d_out
        return obj, parameter_gradients

    def get_prediction(self, input_values):
        self.__set_values__(input_values)
        pred = forward_graph(self.prediction, node_list=self.prediction_node_list_forward)
        return pred

###### Computation graph utilities

def sort_topological(sink):
    """Returns a list of the sink node and all its ancestors in topologically sorted order.
    Subgraph of these nodes must form a DAG."""
    L = [] # Empty list that will contain the sorted nodes
    T = set() # Set of temporarily marked nodes
    P = set() # Set of permanently marked nodes

    def visit(node):
        if node in P:
            return
        if node in T:
            raise 'Your graph is not a DAG!'
        T.add(node) # mark node temporarily
        for predecessor in node.get_predecessors():
            visit(predecessor)
        P.add(node) # mark node permanently
        L.append(node)

    visit(sink)
    return L

def forward_graph(graph_output_node, node_list=None):
    # If node_list is not None, it should be sort_topological(graph_output_node)
    if node_list is None:
        node_list = sort_topological(graph_output_node)
    for node in node_list:
        out = node.forward()
    return out

def backward_graph(graph_output_node, node_list=None):
    """
    If node_list is not None, it should be the reverse of sort_topological(graph_output_node).
    Assumes that forward_graph has already been called on graph_output_node.
    Sets d_out of each node to the appropriate derivative.
    """ 
    if node_list is None:
        node_list = sort_topological(graph_output_node)
        node_list.reverse()

    graph_output_node.d_out = np.array(1) # Derivative of graph output w.r.t. itself is 1

    for node in node_list:
        node.backward()
