import inkex
from constraint import Problem

from common_utils import (
    pattern_vector_to_d,
    BaseFillExtension,
    get_clockwise
)

from collections import defaultdict
import random
from svgpathtools.path import Path, Line


class JewelTexture(BaseFillExtension):
    def __init__(self):
        BaseFillExtension.__init__(self, self.jewel_texture)
        self.all_paths = []
        assert self.effect_handle == self.jewel_texture
        self.edges = []
        self.locations = []
        self.cycle_edges = []

    def add_arguments(self, pars):
        pars.add_argument("--maximum", type=float, default=30, help="The maximum facet length")
        pars.add_argument(
            "--minimum",
            type=float,
            default=10,
            help="The minimum facet length",
        )

    def plot_graph(self):
        for i, _edge in enumerate(self.edges):
            self.add_path_node(Path(_edge[2]).d(), f"stroke:red;stroke-width:2;fill:none",
                               f"edge_{i}_{_edge[0]}_{_edge[1]}")
        return

    def generate_permutated_graph(self):
        even = True
        _x = self.bbox.left
        while _x <= self.bbox.right:
            _x += self.spacing
            _y = self.bbox.top + self.spacing*even/2.0
            while _y <= self.bbox.bottom:
                _y += self.spacing
                self.locations.append(_x + _y*1j)
            even = not even
        assert self.locations

        def chain_cycle(a, b, c):
            if not b or not c:
                return
            _edge1 = [j for j, _edge in enumerate(self.edges) if
                      _edge in [(a, b), (b, a)]]
            if _edge1:
                _edge1 = _edge1.pop()
            else:
                _edge1 = len(self.edges)
                self.edges.append((a, b))
            _edge2 = [j for j, _edge in enumerate(self.edges) if
                      _edge in [(b, c), (c, b)]]
            if _edge2:
                _edge2 = _edge2.pop()
            else:
                _edge2 = len(self.edges)
                self.edges.append((b, c))
            _edge3 = [j for j, _edge in enumerate(self.edges) if
                      _edge in [(c, a), (a, c)]]
            if _edge3:
                _edge3 = _edge3.pop()
            else:
                _edge3 = len(self.edges)
                self.edges.append((c, a))
            self.cycle_edges.append((_edge1, _edge2, _edge3))

        for i, _location in enumerate(self.locations):
            _left = self.locations[i]-self.spacing+self.spacing*1j/2
            _bottom_left = self.locations[i]+self.spacing*1j
            _bottom_right = self.locations[i]+self.spacing +self.spacing*1j/2
            _left_index = [j for j in range(len(self.locations)) if abs(self.locations[j]-_left) < self.spacing/8.0]
            if _left_index:
                _left_index = _left_index.pop()
            _bottom_left_index = \
            [j for j in range(len(self.locations)) if abs(self.locations[j] - _bottom_left) < self.spacing / 8.0]
            if _bottom_left_index:
                _bottom_left_index = _bottom_left_index.pop()
            _bottom_right_index = \
                [j for j in range(len(self.locations)) if abs(self.locations[j] - _bottom_right) < self.spacing / 8.0]
            if _bottom_right_index:
                _bottom_right_index = _bottom_right_index.pop()
            # if _bottom_right_index and _bottom_left_index and _left_index:
            #     self.add_marker(self.locations[_bottom_right_index], label="bottom_right")
            #     self.add_marker(self.locations[_bottom_left_index], color="blue", label="bottom_left")
            #     self.add_marker(self.locations[_left_index], color="green",  label="left")
            #     self.add_marker(self.locations[i], color="white", label="i")
            #     return
            #chain_cycle(i, _left_index, _bottom_left_index)
            chain_cycle(i, _bottom_left_index, _left_index)
        if not self.cycle_edges:
            for i, _location in enumerate(self.locations):
                self.add_marker(_location, label=f"location{i}")

    def populate_locations(self, shape):

        container_path = Path(pattern_vector_to_d(shape))

        # first, place the nodes
        # how do you ensure that the maximum is in effect? i.e. that there is at least one shape that is the maximum size?
        # maybe that comes after building the graph - i.e. merge smaller pieces to make a bigger one if there are two many small pieces
        # TODO: modularize packing algorithm
        MAX_RETRIES = 10
        current_retry = 0

        self.locations = []
        curr_i = 0
        while current_retry < MAX_RETRIES:
            curr_i += 1
            if curr_i > 50000:
                inkex.utils.errormsg(f"low number of points, {self.bbox.width= }, {self.bbox.height= } {self.spacing= }")
                break
            _x = random.random() * self.bbox.width + self.bbox.left
            _y = random.random() * self.bbox.height + self.bbox.top
            _test_location = _x + _y * 1j
            # first confirm that the _x, _y is inside
            is_inside = inkex.boolean_operations.segment_is_inside(container_path, _test_location, tolerance=0.005)
            if not is_inside:
                inkex.boolean_operations.segment_is_inside(container_path, _test_location, debug=True, tolerance=0.005)
                # raise ValueError(f"skipping, wasn't inside {_x=} {_y=}, {container_path=}")
                continue
            too_close = False
            for _location in self.locations:

                distance = abs(_location - _test_location)
                if distance < self.spacing:  # the current circle is too close to an existing location
                    current_retry += 1
                    too_close = True
                    break

            if too_close:
                continue
            self.locations.append(_test_location)
        assert self.locations

    def populate_edges(self):
        # now build up all the edges - values are tuples of start location index, end location index, svgpathtools Line
        self.edges = []
        """ only the nearest 6 - doesn't give the correct effect
        for i, _loc_i in enumerate(self.locations):
            _potential_edges = []
            for j, _loc_j in enumerate(self.locations):
                if i == j:
                    continue
                _potential_edges.append((j, Line(start=self.locations[i], end=self.locations[j])))
            _potential_edges.sort(key=lambda x: x[1].length())
            for j in range(min(len(_potential_edges), 6)):
                self.edges.append((i, _potential_edges[j][0], _potential_edges[j][1]))
        """
        self.untrimmed_edges = []
        for i, _loc_i in enumerate(self.locations):

            for j, _loc_j in enumerate(self.locations):
                if i == j:
                    continue
                self.untrimmed_edges.append((i, j, Line(start=self.locations[i], end=self.locations[j])))
        start_i = 0
        self.untrimmed_edges.sort(key=lambda x: x[2].length())
        self.edges = []
        # compare the longest edges against the shortest ones
        print(f"{len(self.untrimmed_edges)} to evaluate")
        skipped = []
        for i, _edge in enumerate(self.untrimmed_edges):
            keep = False
            if i % 500 == 0:
                print(i)

            for j, _comparison_edge in enumerate(self.untrimmed_edges):
                if i == j:
                    continue
                if j in skipped:
                    continue
                if _comparison_edge[2] == _edge[2]:
                    continue
                if not _edge[2].intersect(_comparison_edge[2]):
                    keep = True
                    continue
                if _edge[2].length() > _comparison_edge[2].length():
                    keep = False
                    skipped.append(i)
                    break
            if keep:
                self.edges.append(_edge)
        assert self.edges

    def populate_graph(self):
        # build the graph - first the node graph, then the edge graph
        self.graph = defaultdict(list)
        self.graph_edges = defaultdict(list)

        for i, _edge in enumerate(self.edges):
            if _edge[1] not in self.graph[_edge[0]]:
                self.graph[_edge[0]].append(_edge[1])
            if _edge[0] not in self.graph[_edge[1]]:
                self.graph[_edge[1]].append(_edge[0])
            self.graph_edges[i].append(_edge[0])
            self.graph_edges[i].append(_edge[1])

    def find_cycles(self):
        self.cycles = []

        def get_cycles(counterclockwise=False):
            for i, _edge in enumerate(self.edges):
                # get clockwise
                start_point = _edge[0]
                last_point = _edge[0]
                curr_point = _edge[1]
                pieces = [last_point, curr_point]
                closed = False
                while not closed:
                    branches = [_branch for _branch in self.graph[curr_point] if _branch not in [last_point, curr_point]]
                    if not branches:
                        print(f"there were no branches! {pieces=} {self.graph[curr_point]=}")
                        break
                    branch_locations = [self.locations[_branch] for _branch in branches]
                    branch_index, clockwise = get_clockwise(last_point, curr_point, branch_locations, counterclockwise)
                    next_branch = branches[branch_index]
                    if next_branch == pieces[-1]:
                        print(f"got a weird branch {last_point=} {curr_point=} {next_branch=} {branches=}")

                    pieces.append(branches[branch_index])
                    if len(pieces) >= 3 and next_branch == start_point:
                        self.cycles.append(pieces)
                        break
                    if len(pieces) > 4:
                        print(f"more than 4 pieces and failed to get back to start! {pieces=}")
                        break
                    last_point = curr_point
                    curr_point = next_branch

        get_cycles(False)
        get_cycles(True)

    def dedupe_cycles(self):
        _cycle_index = 0
        # dedupe the list of cycles
        while _cycle_index < len(self.cycles):
            for i, cycle in enumerate(self.cycles):
                if i == _cycle_index:
                    continue
                if set(self.cycles[_cycle_index]) == set(cycle):
                    del self.cycles[_cycle_index]
                    continue
            _cycle_index += 1

    def generate_permutated_paths(self):
        self.paths = []
        for _cycle in self.cycle_edges:
            _path = []
            for _cycle_index in _cycle:
                _edge = self.edges[_cycle_index]
                _path.append(Line(start=self.locations[_edge[0]],end=self.locations[_edge[1]]))
            self.paths.append(Path(*_path))

    def build_paths(self):
        # convert the self.cycles by self.edges - and also reverse the paths if need be
        self.paths = []
        self.cycle_edges = []
        for _cycle in self.cycles:
            _path = []
            _cycle_edge = []
            for j in range(len(_cycle) - 1):
                found_edge = False
                for i, _edge in enumerate(self.edges):
                    if _cycle[j] == _edge[0] and _cycle[j+1] == _edge[1]:
                        _path.append(_edge[2])
                        _cycle_edge.append(i)
                        found_edge = True
                        break
                    elif _cycle[j] == _edge[1] and _cycle[j+1] == _edge[0]:
                        _path.append(_edge[2].reversed())
                        _cycle_edge.append(i)
                        found_edge = True
                        break
                if not found_edge:
                    raise ValueError(f"could not identify edge corresponding to {_cycle[j:j+2]} {_cycle=}")
            self.paths.append(_path)
            self.cycle_edges.append(_cycle_edge)

    def find_solution(self):
        # do I need to stack the pieces?
        problem = Problem()
        _polygons = []
        for i in range(len(self.cycle_edges)):
            problem.addVariable(i, [0, 1, 2, 3])
        for i, cycle_i in enumerate(self.cycle_edges):
            for j, cycle_j in enumerate(self.cycle_edges):
                if i == j:
                    continue
                if set(cycle_i).intersection(cycle_j):
                    problem.addConstraint(lambda x, y: x != y, (i, j))
        self.solution = problem.getSolution()

    def jewel_texture(self, shape):
        #print(bbox, container_path.bbox(), container_path.d())
        #c_bbox = container_path.bbox()
        #bbox = inkex.transforms.BoundingBox(x=(c_bbox[0], c_bbox[1]), y=(c_bbox[2], c_bbox[3]))
        self.spacing = float(self.options.minimum)
        self.options.length = self.spacing
        self.bbox = shape.bounding_box()

        random_packing = False
        if random_packing:
            self.populate_locations(shape)
            self.populate_edges()
            self.populate_graph()
            self.find_cycles()
            self.dedupe_cycles()
            if not self.cycles:
                self.plot_graph()
                return
            self.build_paths()
        else:
            self.generate_permutated_graph()
            self.generate_permutated_paths()
        self.find_solution()

        if not self.solution:
            self.plot_graph()
            return
        colors = {0: "black", 1:"red", 2:"pink", 3:"maroon"}
        color_paths = defaultdict(list)
        for i, cycle in enumerate(self.cycle_edges):
            color_paths[self.solution[i]] += self.paths[i]

        for color_i in color_paths:

            _color = inkex.Color(colors[color_i])
            output = Path(*color_paths[color_i])
            print(color_i, len(output))
            self.add_path_node(output.d(), style=f"fill:{_color}", id=f"shapes{color_i}")


if __name__ == "__main__":
    JewelTexture().run()


