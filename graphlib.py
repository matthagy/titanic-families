'''Graph data structures and utilities for graph construction
'''

import operator

class Node(object):

    component = None

    def __init__(self, value=None):
        self.value = value
        self.edges = []

class Edge(object):

    def __init__(self, node_i, node_j):
        self.node_i = node_i
        self.node_j = node_j
        node_i.edges.append(self)
        node_j.edges.append(self)

    def other(self, n):
        if n is self.node_i:
            return self.node_j
        elif n is self.node_j:
            return self.node_i
        else:
            raise ValueError


class BaseComponent(object):

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

class Component(BaseComponent):

    graph = None

    def __init__(self, nodes, edges):
        super(Component, self).__init__(nodes, edges)
        for node in self.nodes:
            assert node.component is None
            node.component = self

    def tear_down(self):
        for node in self.nodes:
            node.component = None
        nodes, edges = self.nodes, self.edges
        self.nodes = self.edges = None
        return nodes, edges


def join_lists(lists):
    return reduce(operator.add, lists, [])

class Graph(BaseComponent):

    def __init__(self, components):
        super(Graph, self).__init__(
            join_lists(component.nodes for component in components),
            join_lists(component.edges for component in components))
        self.components = components
        for component in components:
            assert component.graph is None
            component.graph = self
        self.graph = self

    def tear_down(self):
        nodes_edges = [component.tear_down for component in self.components]
        for component in self.components:
            component.graph = None
        self.components = None
        nodes, edges = map(join, zip(*nodes_edges))
        return nodes, edges


class GraphBuilder(object):

    def __init__(self, node_factory=None, edge_factory=None, component_factory=None, graph_factory=None):
        if node_factory is None:
            node_factory = Node
        if edge_factory is None:
            edge_factory = Edge
        if component_factory is None:
            component_factory = Component
        if graph_factory is None:
            graph_factory = Graph

        self.node_factory = node_factory
        self.edge_factory = edge_factory
        self.component_factory = component_factory
        self.graph_factory = graph_factory

        self.components = []
        self.nodes_to_components = {}
        self.values_to_nodes = {}

    def get_graph(self):
        return self.graph_factory(self.components + self.get_singleton_components())

    def get_singleton_components(self):
        acc = []
        for node in self.values_to_nodes.itervalues():
            if not node.edges:
                assert not node in self.nodes_to_components
                acc.append(self.component_factory([node], []))
        return acc

    def get_node(self, value):
        if value not in self.values_to_nodes:
            self.values_to_nodes[value] = self.node_factory(value)
        return self.values_to_nodes[value]

    def add_value_edges(self, edges):
        self.add_edges([(self.get_node(i), self.get_node(j))
                        for i,j in edges])

    def add_edges(self, edges):
        for i,j in edges:
            self.add_edge(i, j)

    def add_edge(self, i, j):
        assert isinstance(i, Node)
        assert isinstance(j, Node)
        component = None
        component_i = self.nodes_to_components.get(i)
        component_j = self.nodes_to_components.get(j)
        if component_i is None and component_j is None:
            component,e = self.make_component(i, j)
        elif component_i is None:
            component,e = self.make_edge(component_j, j, i, (i,j))
        elif component_j is None:
            component,e = self.make_edge(component_i, i, j, (i,j))
        elif component_j is component_i:
            component = component_i
            e = self.edge_factory(i, j)
            component.edges.append(e)
        else:
            component,e = self.combine_components(i, j, component_i, component_j)
        self.nodes_to_components[i] = self.nodes_to_components[j] = component
        return e

    def make_component(self, i, j):
        e = self.edge_factory(i, j)
        component = self.component_factory([i, j], [e])
        self.components.append(component)
        return component,e

    def make_edge(self, component, node_in, node_out, (i,j)):
        edge = self.edge_factory(i, j)
        component.nodes.append(node_out)
        component.edges.append(edge)
        return component,edge

    def combine_components(self, node_i, node_j, component_i, component_j):
        self.components.remove(component_i)
        self.components.remove(component_j)
        nodes_i, edges_i = component_i.tear_down()
        nodes_j, edges_j = component_j.tear_down()
        e = self.edge_factory(node_i, node_j)
        component = self.component_factory(nodes_i + nodes_j,
                                           edges_i + edges_j +
                                           [e])
        self.components.append(component)
        for node in component.nodes:
            self.nodes_to_components[node] = component
        return component, e

