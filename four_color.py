from collections import defaultdict

from constraint import Problem
from svgpathtools import parse_path, Path

from common_utils import BaseFillExtension, pattern_vector_to_d

TOLERANCE = 0.1


class FourColorFill(BaseFillExtension):
    def __init__(self):
        BaseFillExtension.__init__(self, self.four_color_fill, self.assign_color)
        self.edge_map = defaultdict(dict)
        self.all_edges = defaultdict(Path)
        self.solution = []

    def assign_color(self):
        if not self.svg.selected:
            return

        for i, shape in self.svg.selected.items():
            path_id = shape.get("id")
            if path_id not in self.all_edges:
                self.all_edges[path_id] = parse_path(pattern_vector_to_d(shape))
            for j, edge in enumerate(self.all_edges[path_id]):
                if j in self.edge_map[path_id]:
                    continue
                diff_size = TOLERANCE * len(edge)
                # TODO: skip over edge pieces
                # if edge.start.real == edge.end.real:
                for other_i, other_shape in self.svg.selected.items():
                    if other_i == i:
                        continue
                    other_path_id = other_shape.get("id")
                    if other_path_id not in self.all_edges:
                        self.all_edges[other_path_id] = parse_path(
                            pattern_vector_to_d(other_shape)
                        )

                    for other_j, other_edge in enumerate(self.all_edges[other_path_id]):
                        if other_j in self.edge_map[other_path_id]:
                            continue
                        if (
                            abs(edge.start - other_edge.start) < diff_size
                            and abs(edge.end - other_edge.end) < diff_size
                        ):
                            self.edge_map[path_id][j] = other_path_id
                            self.edge_map[other_path_id][other_j] = path_id
                            break
                        if (
                            abs(edge.start - other_edge.end) < diff_size
                            and abs(edge.start - other_edge.end) < diff_size
                        ):
                            self.edge_map[path_id][j] = other_path_id
                            self.edge_map[other_path_id][other_j] = path_id
                            break
        problem = Problem()
        _polygons = []
        for path_id in self.edge_map:
            problem.addVariable(path_id, [0, 1, 2, 3])
        for path_id in self.edge_map:
            connected_paths = self.edge_map[path_id].values()
            for connected_path in connected_paths:
                problem.addConstraint(lambda x, y: x != y, (path_id, connected_path))
        self.solution = problem.getSolution()

    def four_color_fill(self, node):
        path_id = node.get("id")
        pattern_style = node.get("style")
        color = ["red", "yellow", "blue", "green"][self.solution[path_id]]
        pattern_style = pattern_style.replace("fill:none", f"fill:{color}")
        if "fill" not in pattern_style:
            pattern_style += f";fill:{color}"
        self.add_path_node(node.d(), pattern_style, path_id)
