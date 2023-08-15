from collections import defaultdict
from statistics import mode
from constraint import Problem
from svgpathtools import parse_path, Path
from networkx import Graph, number_of_nodes, connected_components
from common_utils import BaseFillExtension, pattern_vector_to_d

TOLERANCE = 0.2


class FourColorFill(BaseFillExtension):
    def __init__(self):
        BaseFillExtension.__init__(self, self.four_color_fill, self.assign_color)
        self.edge_map = defaultdict(dict)
        self.all_edges = defaultdict(Path)
        self.paths_to_nodes = defaultdict(list)
        self.solution = []
        self.graph = Graph()

    def assign_color(self):
        if not self.svg.selected:
            return
        lengths = []
        for i, shape in self.svg.selected.items():
            path_id = shape.get("id")
            if path_id not in self.all_edges:
                self.all_edges[path_id] = parse_path(pattern_vector_to_d(shape))
            for j, edge in enumerate(self.all_edges[path_id]):
                start_node = number_of_nodes(self.graph)
                self.graph.add_node(start_node, loc=edge.start)
                end_node = number_of_nodes(self.graph)
                _length = edge.length()
                if _length > 1:
                    lengths.append(_length)
                self.paths_to_nodes[path_id].append(start_node)
                self.paths_to_nodes[path_id].append(end_node)
                self.graph.add_node(end_node, loc=edge.end)
                self.graph.add_edge(start_node, end_node, path1=path_id)
        avg_length = mode(lengths)
        # 1. find the groups of nodes - by finding the subgraphs
        cluster_graph = Graph()
        for node_i, loc_i in self.graph.nodes(data="loc"):
            cluster_graph.add_node(node_i)
        for node_i, loc_i in self.graph.nodes(data="loc"):
            for node_j, loc_j in self.graph.nodes(data="loc"):
                if node_i == node_j:
                    continue
                diff_length = abs(loc_i - loc_j)
                if diff_length < avg_length*TOLERANCE:
                    cluster_graph.add_edge(node_i, node_j)
        node_reduction_map = dict()
        print(f"modal length is {avg_length}")
        # 2. for each subgraph, make a mapping of the nodes to the first node
        for subgraph in list(connected_components(cluster_graph)):
            print(f"evaluating cluster subgraph {subgraph}")
            first_node = None
            for node_i in subgraph:
                if not first_node:
                    first_node = node_i
                node_reduction_map[node_i] = first_node

        # 3. go through the paths_to_nodes, and map all nodes to the reduced nodes
        for path_id in self.paths_to_nodes:
            mapped_nodes = set()
            for node_i in self.paths_to_nodes[path_id]:
                mapped_nodes.add(node_reduction_map[node_i])
            self.paths_to_nodes[path_id] = mapped_nodes

        # 4. for each path_id, if it shares two nodes in common with the nodes on any other path_id, then add the constraint
        problem = Problem()
        _polygons = []
        for path_id in self.paths_to_nodes:
            problem.addVariable(path_id, [0, 1, 2, 3])
        for path_id_i in self.paths_to_nodes:
            for path_id_j in self.paths_to_nodes:
                if path_id_i == path_id_j:
                    continue

                if len(self.paths_to_nodes[path_id_i].intersection(self.paths_to_nodes[path_id_j])) > 1:
                    problem.addConstraint(lambda x, y: x != y, (path_id_i, path_id_j))
        self.solution = problem.getSolution()

    def four_color_fill(self, node):
        path_id = node.get("id")
        pattern_style = node.get("style")
        color = ["red", "yellow", "blue", "green"][self.solution[path_id]]
        pattern_style = pattern_style.replace("fill:#fce5a3", f"fill:{color}")
        if "fill" not in pattern_style:
            pattern_style += f";fill:{color}"
        self.add_path_node(node.get("d"), pattern_style, path_id)
        self.remove_path_node(node)
