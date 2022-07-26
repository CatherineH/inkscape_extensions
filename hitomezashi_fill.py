#!/usr/bin/env python3
from collections import defaultdict
from typing import List

from constraint import Problem

try:
    import inkex
except ImportError:
    import sys

    raise ValueError(f"svgpathtools is not available on {sys.executable}")
from time import time
from common_utils import (
    pattern_vector_to_d,
    BaseFillExtension,
    debug_screen,
    combine_segments,
    format_complex,
    stack_lines,
    is_inside,
    intersect_over_all,
    find_orientation,
    svgpath_to_shapely_polygon,
)
from enum import Enum
from random import random
from svgpathtools import Line, Path, parse_path
from functools import lru_cache
import sys
from copy import deepcopy
import colorsys
from pickle import dump

TOLERANCE = 0.2


class Corner(Enum):
    top_left = 0
    top_right = 1
    bottom_right = 2
    bottom_left = 3

    def project(self, container: inkex.paths.Path) -> float:
        # get the bbox corner coordinates
        bbox = container.bounding_box()
        if self.value == self.top_left:
            return bbox.left + bbox.top * 1j
        if self.value == self.top_right:
            return bbox.right + bbox.top * 1j
        if self.value == self.bottom_left:
            return bbox.left + bbox.bottom * 1j
        if self.value == self.bottom_right:
            return bbox.right + bbox.bottom * 1j


class NearestEdge(Enum):
    left = 0
    top = 1
    right = 2
    bottom = 3

    def is_opposite(self, other: Enum) -> bool:
        if self.value % 2 == 0 and other.value % 2 == 0:
            return True
        if self.value % 2 == 1 and other.value % 2 == 1:
            return True
        return False

    def corners_between(self, other: Enum) -> List[Corner]:
        if self.value == other.value:
            return []
        if self.is_opposite(other):
            return [Corner(self.value), Corner(NearestEdge((self.value + 1) % 3).value)]
        if (self == self.left and other == other.top) or (
            self == self.top and other == other.left
        ):  # 0, 1 -> 0
            return [Corner(Corner.top_left)]
        if (self == self.left and other == other.bottom) or (
            self == self.bottom and other == other.left
        ):  # 0, 3 -> 3
            return [Corner(Corner.bottom_left)]
        if (self == self.right and other == other.top) or (
            self == self.top and other == other.right
        ):  # 1, 2 -> 1
            return [Corner(Corner.top_right)]
        if (self == self.right and other == other.bottom) or (
            self == self.bottom and other == other.right
        ):  # 2, 3 -> 2
            return [Corner(Corner.bottom_right)]
        raise ValueError(f"unknown edge combination: {self.value} {other.value}")

    def project(self, point: float, container: inkex.paths.Path) -> float:
        # project the point onto the bbox based on the nearest edge
        if self.value == self.left:
            return point.imag() * 1j + container.left()
        if self.value == self.right:
            return point.imag() * 1j + container.right()
        if self.value == self.top:
            return point.real() + container.top() * 1j
        if self.value == self.bottom:
            return point.real() + container.bottom() * 1j


class HitomezashiFill(BaseFillExtension):
    def __init__(self):
        BaseFillExtension.__init__(self, self.hitomezashi_fill)
        self.xmax = None
        self.xmin = None
        self.ymin = None
        self.ymax = None
        self.container = None
        self.outline_intersections = []
        self.outline_nodes = []
        # build a graph of which edge points connect where
        self.graph = defaultdict(list)
        self.edges = []
        self.graph_locs = []
        self.edges_to_visit = []
        self.visited_edges = []
        self.chained_lines = []
        self.last_branches = []  # keep track of the last possible branch
        self._debug_branches = []
        self._evaluated = []

    def add_arguments(self, pars):
        pars.add_argument("--length", type=float, default=1, help="Length of segments")
        pars.add_argument(
            "--weight_x",
            type=float,
            default=0.5,
            help="The probability of getting a 1 along the x axis",
        )
        pars.add_argument(
            "--weight_y",
            type=float,
            default=0.5,
            help="The probability of getting a 1 along the y axis",
        )
        pars.add_argument(
            "--gradient", type=str, default="false", help="fill the stitch shapes"
        )
        pars.add_argument(
            "--fill", type=str, default="false", help="fill the stitch shapes"
        )

    def add_chained_line(self, chained_line, label="chained-line", color="red"):
        segments = []
        for i in range(1, len(chained_line)):
            try:
                segments.append(self.graph[chained_line[i - 1]][chained_line[i]])
            except KeyError as e:
                self.add_marker(
                    chained_line[i], label=f"missing-segment-{i}", color="blue"
                )
                self.add_marker(
                    chained_line[i - 1], label=f"missing-segment-{i}", color="green"
                )
                print(f"outline nodes are {self.outline_nodes}")
                print(
                    f"got key error on: {chained_line[i]} not in {self.graph.get(chained_line[i-1], {}).keys()} {self.graph[chained_line[i]].keys()}"
                )
        stroke_length = self.options.length / 10
        self.add_path_node(
            combine_segments(segments).d(),
            f"stroke:{color};stroke-width:{stroke_length};fill:none",
            label,
        )

    def add_marker(self, point_i, label="marker", color="red"):
        point = self.graph_locs[point_i]
        marker_size = self.options.length / 10

        marker = [
            Line(
                point + marker_size + marker_size * 1j,
                point - marker_size + marker_size * 1j,
            ),
            Line(
                point - marker_size + marker_size * 1j,
                point - marker_size - marker_size * 1j,
            ),
            Line(
                point - marker_size - marker_size * 1j,
                point + marker_size - marker_size * 1j,
            ),
            Line(
                point + marker_size - marker_size * 1j,
                point + marker_size + marker_size * 1j,
            ),
        ]
        self.add_path_node(Path(*marker).d(), f"fill:{color};stroke:none", label)

    def plot_graph(self, color="gray", label="graph", connected=True):
        # dump the graph
        if connected:
            all_graph_segments = [
                segment for branch in self.graph.values() for segment in branch.values()
            ]
            self.add_path_node(
                combine_segments(all_graph_segments).d(),
                f"stroke:{color};stroke-width:2;fill:none",
                label,
            )
        else:
            for start_i in self.graph.keys():
                for end_i in self.graph[start_i].keys():
                    try:
                        self.add_path_node(
                            self.graph[start_i][end_i].d(),
                            f"stroke:{color};stroke-width:2;fill:none",
                            f"{label}-{start_i}-{end_i}",
                        )
                    except AttributeError as e:
                        print(f"skipping {self.graph[start_i][end_i]}")
                        pass

    def chop_shape(self, lines):
        final_lines = []
        for i, line in enumerate(lines):
            # determine whether each point on the line is inside or outside the shape
            start_inside = self.is_inside(line.start)
            end_inside = self.is_inside(line.end)
            intersections = intersect_over_all(line, self.container)
            if (
                not start_inside and not end_inside and not len(intersections)
            ):  # skip this line, it's not inside the pattern
                continue
            if (
                start_inside and end_inside and not intersections
            ):  # add this line and then continue
                final_lines.append(line)

            # if it has intersections, it's trickier
            curr_start = line.start
            for (t1, t2, seg_i) in intersections:
                self.outline_intersections.append((t2, seg_i))
                if start_inside:
                    final_lines.append(Line(curr_start, line.point(t1)))
                start_inside = not start_inside
                curr_start = line.point(t1)
            if start_inside:
                final_lines.append(Line(curr_start, line.end))

        for line in final_lines:
            if (
                line.length() < TOLERANCE * self.options.length
            ):  # skip this one because it's too short
                print("skipping: ", line)
                continue

            line_i = self.snap_nodes(line.start)
            line_j = self.snap_nodes(line.end)

            self.graph[line_i][line_j] = Path(Line(line.start, line.end))
            self.graph[line_j][line_i] = Path(Line(line.end, line.start))

        self.outline_intersections = list(set(self.outline_intersections))
        self.outline_intersections.sort(key=lambda x: -x[0] - x[1])
        intersections_copy = self.outline_intersections
        start_intersection = intersections_copy.pop()
        intersections_copy.insert(
            0, start_intersection
        )  # add the end back onto the front so that we'll close the loop
        while intersections_copy:
            end_intersection = intersections_copy.pop()
            start_i = self.snap_nodes(
                self.container[start_intersection[1]].point(start_intersection[0])
            )
            end_i = self.snap_nodes(
                self.container[end_intersection[1]].point(end_intersection[0])
            )

            if start_intersection[1] == end_intersection[1]:
                segment = self.container[start_intersection[1]].cropped(
                    start_intersection[0], end_intersection[0]
                )
            else:
                segments = []
                if start_intersection[0] != 1:
                    segments = [
                        self.container[start_intersection[1]].cropped(
                            start_intersection[0], 1
                        )
                    ]
                index_i = start_intersection[1] + 1
                while index_i < end_intersection[1]:
                    segments.append(self.container[index_i])
                    index_i += 1
                if end_intersection[0] != 0:
                    segments.append(
                        self.container[end_intersection[1]].cropped(
                            0, end_intersection[0]
                        )
                    )
                segment = Path(*segments)
            if (
                segment.length() < TOLERANCE * self.options.length
            ):  # skip this one because it's too short
                print("skipping: ", start_intersection, end_intersection)
                continue
            self.graph[start_i][end_i] = segment
            self.graph[end_i][start_i] = segment.reversed()
            if start_i not in self.outline_nodes:
                self.outline_nodes.append(start_i)
            if end_i not in self.outline_nodes:
                self.outline_nodes.append(end_i)

            start_intersection = end_intersection
        return Path(*final_lines)

    def chain_valid(self, chained_line, debug=False):
        # output: 0 - stop, 1 - continue, 2 - end
        if len(chained_line) > 3 and chained_line[0] == chained_line[-1]:
            # the loop is closed, yippee!
            if debug:
                print(f"closing loop {format_complex(chained_line)}")
            return 1
        if len(chained_line) >= 3:
            # check whether it's possible to close the loop now
            loop_index = None
            for point in chained_line:
                index = chained_line.index(point)
                try:
                    loop_index = chained_line.index(point, index + 1)
                    break
                except ValueError as err:
                    pass
            if loop_index:
                if debug:
                    print(
                        f"chained_line {len(self._debug_branches)} loops back on itself at index {loop_index} {format_complex(chained_line)} "
                    )
                return 0
        if debug:
            print(f"chained line {format_complex(chained_line)} is valid, continuing")
        return 2

    def get_branches(self, chained_line, debug=False):
        """
        find the next segment in the chained line
        chained_line: list of locations in the current chain
        returns: bool indicating whether the chain is finshed
        """
        state = self.chain_valid(chained_line, debug)
        if state == 1:
            return chained_line
        elif state == 0:
            return None
        branches = list(self.graph[chained_line[-1]].keys())
        # remove the ones already on the line, but not the first point
        branches = [point for point in branches if point not in chained_line[1:]]
        branches.sort(
            key=lambda x: abs(self.graph_locs[x] - self.graph_locs[chained_line[0]])
        )

        for branch in branches:
            chain_to_add = deepcopy(chained_line) + [branch]
            if debug:
                print(f"adding {chain_to_add}")
            self.last_branches.append(chain_to_add)

        return None

    def chain_graph(self):
        # self.plot_graph(connected=False)
        self.simplify_graph()
        if len(self.graph.keys()) > 500:
            msg = f"there are two many edges to fill. Consider using a shorter length piece or not filling the shape."
            inkex.utils.errormsg(msg)
            raise ValueError(msg)
        # self.plot_graph(color="blue", label="simplified_graph", connected=False)
        # debug_screen(self, "test_graph_simplify")
        self.audit_graph()
        # use the edge to make a loop that extends past the edges of the bbox add that to the stacks
        loops: List[Path] = []
        for edge in self.edges:
            loop = [edge]
            # which edge are the end points nearest?
            start_edge = self.find_closest_edge(edge.start)
            end_edge = self.find_closest_edge(edge.end)
            corners_in_between = start_edge.corners_between(end_edge)
            # add the border lines at the end of the path
            loop.append(Line(start=edge.end, end=end_edge.project(edge.end)))
            for corner_in_between in corners_in_between:
                loop.append(
                    Line(
                        start=loop[-1].end,
                        end=corner_in_between.project(self.container),
                    )
                )
            loop.append(Line(start=loop[-1].end, end=start_edge.project(edge.start)))
            loop.append(Line(start=start_edge.project(edge.start), end=edge.start))
            loops.append(loop)

        return loops

    def find_closest_edge(self, point: float) -> NearestEdge:
        bbox = self.container.bbox()
        # left, top, right, bottom
        _distance = [
            point.real - bbox.left(),
            point.imag - bbox.top(),
            point.real - bbox.right(),
            point.imag - bbox.bottom(),
        ]
        _distance = [abs(_d) for _d in _distance]
        _min_distance = min(_distance)
        index_of_min = _distance.index(_min_distance)
        return NearestEdge(index_of_min)

    def audit_overlap(self, chained_line, tail_end=[], curr_point=None):
        # confirm that the chained line does not double back on itself
        for i in range(0, len(chained_line) - 1):
            for j in range(0, len(chained_line) - 1):
                if i == j:
                    continue
                if chained_line[i] != chained_line[j]:
                    continue
                return False
        return True

    def reset_shape(self):
        # remove all markers etc that were added for debugging
        parent = self.get_parent(self.current_shape)
        parent.remove_all()
        parent.insert(-1, self.current_shape)

    def bool_op_shape(self):
        try:
            from pylivarot import intersection, py2geom
        except ImportError as e:
            inkex.utils.errormsg("Fill does not work without pylivarot installed")
            sys.exit(0)
        chained_lines = self.chain_graph()
        start_time = time()
        chained_lines_pv = py2geom.PathVector()
        for chained_line in chained_lines:
            pb = py2geom.PathBuilder()
            start_point = chained_line[0]
            pb.moveTo(py2geom.Point(start_point.real, start_point.imag))
            for i in range(1, len(chained_line)):
                pb.lineTo(py2geom.Point(chained_line[i].real, chained_line[i].imag))
            pb.closePath()
            chained_lines_pv.push_back(pb.flush())
        container_pv = py2geom.parse_svg_path(self.container.d())
        print(f"path building took {time()-start_time}")
        start_time = time()
        intersection_pv = intersection(container_pv, chained_lines_pv)
        print(f"bool op took {time()-start_time}")
        start_time = time()
        output_chained_lines = []
        for piece in intersection_pv:
            piece_d = py2geom.write_svg_path(piece)
            output_chained_lines.append(Path(piece_d))
        print(f"decomp took {time()-start_time}")
        return output_chained_lines

    def snap_nodes(self, node):
        for i, ex_node in enumerate(self.graph_locs):
            diff = abs(ex_node - node)
            if diff < TOLERANCE * self.options.length:
                return i
        self.graph_locs.append(node)
        return len(self.graph_locs) - 1

    def simplify_graph(self):
        """merge any nodes that have only two outputs into a path between the two.
        Keep doing this until there are no more nodes to evaluate"""
        all_nodes = deepcopy(list(self.graph.keys()))
        print(f"before simplification the graph had {len(all_nodes)} nodes")
        while all_nodes:
            to_evaluate = all_nodes.pop()
            branches = list(self.graph[to_evaluate].keys())
            if len(branches) > 2:
                continue

            if 0 < len(branches) < 2:
                # nodes that don't have an input and an output can never be part of a cycle, so delete them
                # del self.graph[to_evaluate]
                continue

            start_i = branches[0]
            end_i = branches[1]
            if (
                start_i == to_evaluate
                or end_i == to_evaluate
                or start_i == end_i
                or end_i in self.graph[start_i]
                or start_i in self.graph[end_i]
            ):
                # skip over reducing this one
                continue
            if not isinstance(self.graph[start_i][to_evaluate], Path):
                self.graph[start_i][to_evaluate] = Path(
                    self.graph[start_i][to_evaluate]
                )
            if not isinstance(self.graph[to_evaluate][end_i], Path):
                self.graph[to_evaluate][end_i] = Path(self.graph[to_evaluate][end_i])
            segments = [*self.graph[start_i][to_evaluate]] + [
                *self.graph[to_evaluate][end_i]
            ]
            segment = Path(*segments)
            self.edges.append(segment)
            self.graph[start_i][end_i] = segment
            self.graph[end_i][start_i] = segment.reversed()
            del self.graph[to_evaluate]
            del self.graph[start_i][to_evaluate]
            del self.graph[end_i][to_evaluate]
            assert self.graph[start_i][end_i]
            assert self.graph[end_i][start_i]

        print(f"after simplification the graph has {len(self.graph.keys())} nodes")

    def audit_graph(self):
        # check whether there are points that are very close together
        nodes = list(self.graph.keys())
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i == j:
                    continue
                diff = abs(self.graph_locs[node_i] - self.graph_locs[node_j])
                if diff < TOLERANCE * self.options.length:
                    print(f"nodes {node_j} and {node_i} are only {diff} apart")

        def out_of_bounds(node_i):
            node = self.graph_locs[node_i]
            return (
                node.real > self.xmax + TOLERANCE
                or node.real < self.xmin - TOLERANCE
                or node.imag > self.ymax + TOLERANCE
                or node.imag < self.ymin - TOLERANCE
            )

        # remove any points that are outside of the bbox from the graph
        for node in nodes:
            if not out_of_bounds(node):
                continue
            branches = list(self.graph[node].keys())
            for branch in branches:
                del self.graph[branch][node]
            del self.graph[node]

        for start_i in self.graph.keys():
            for end_i in self.graph[start_i].keys():
                if not isinstance(self.graph[start_i][end_i], Path):
                    self.graph[start_i][end_i] = Path(self.graph[start_i][end_i])

    def hitomezashi_fill(self, node):
        # greedy algorithm: make a Hitomezashi fill that covers the entire bounding box of the shape,
        # then go through each segment and figure out if it is inside, outside, or intersecting the shape

        self.options.fill = False if self.options.fill == "false" else True
        self.container = parse_path(pattern_vector_to_d(node))

        self.container.approximate_arcs_with_quads()
        self.xmin, self.xmax, self.ymin, self.ymax = self.container.bbox()
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin
        # generate vertical lines
        lines = []
        num_x = int(self.width / self.options.length)
        if self.width / self.options.length % 1:
            num_x += 1
        num_y = int(self.height / self.options.length)
        if self.height / self.options.length % 1:
            num_y += 1
        for x_i in range(num_x):
            x_coord = x_i * self.options.length + self.xmin
            if x_coord == self.xmin:
                # make the line just inside the box
                x_coord += TOLERANCE / 2.0
            if not self.options.gradient:
                odd_even_y = random() > self.options.weight_x
            else:
                odd_even_y = random() > x_i / num_x
            for y_i in range(num_y):
                if y_i % 2 == odd_even_y:
                    continue
                # make the first and last segments a little longer
                if y_i == 0:
                    y_i = -TOLERANCE
                    diff = TOLERANCE * self.options.length
                elif y_i == num_y - 1:
                    diff = TOLERANCE * self.options.length
                else:
                    diff = 0

                y_coord = y_i * self.options.length + self.ymin

                start = x_coord + y_coord * 1j
                end = x_coord + (y_coord + self.options.length + diff) * 1j

                lines.append(Line(start, end))
                assert start != end
        # generate horizontal lines
        for y_i in range(num_y):
            y_coord = y_i * self.options.length + self.ymin
            if not self.options.gradient:
                odd_even_y = random() > self.options.weight_y
            else:
                odd_even_y = random() > y_i / num_y
            for x_i in range(num_x):
                if x_i % 2 == odd_even_y:
                    continue
                # make the first and last segments a little longer
                if x_i == 0:
                    x_i = -TOLERANCE
                    diff = TOLERANCE * self.options.length
                elif x_i == num_x - 1:
                    diff = TOLERANCE * self.options.length
                else:
                    diff = 0
                    # continue # TODO: remove me
                x_coord = x_i * self.options.length + self.xmin
                start = x_coord + y_coord * 1j
                end = (x_coord + self.options.length + diff) + y_coord * 1j
                lines.append(Line(start, end))
                assert start != end
        labels = []
        if not self.options.fill:
            lines = [self.chop_shape(lines)]
            lines = [line.d() for line in lines]
            labels = [i for i in range(len(lines))]
        else:
            _ = self.chop_shape(lines)

            lines = self.chain_graph()
            print(f"graph {self.graph}")
            # next: we need to stack and cut the paths out of each other
            dump(lines, open("tests/data/lines.pickle", "wb"))
            print(f"number of lines to stack {len(lines)}")
            combined_lines, labels = stack_lines(lines)
            # combined_lines = lines
            color_fills = self.color_pattern(combined_lines)
            lines = [line.d() for line in combined_lines]
            print(f"labels {len(labels)} {labels}")
            print(f"combined lines {len(combined_lines)}")
        for i, chained_line in enumerate(lines):
            if chained_line == "":
                raise ValueError(f"got empty chained_path! {i} {chained_line}")
            label = labels[i] if len(labels) > i else i
            pattern_id = (
                "hitomezashi-"
                + node.get("id", f"unknown-{self.curr_path_num}")
                + "-"
                + str(label)
            )
            pattern_style = node.get("style")
            if self.options.fill:
                color = "white" if color_fills[i] else "red"
                pattern_style = pattern_style.replace("fill:none", f"fill:{color}")
                if "fill" not in pattern_style:
                    pattern_style += f";fill:{color}"
            self.add_path_node(chained_line, pattern_style, pattern_id)

    def color_pattern(self, combined_lines):
        problem = Problem()
        _polygons = []
        for i in range(len(combined_lines)):
            problem.addVariable(i, [0, 1])
            _polygons.append(set(segment.start for segment in combined_lines[i]))
        connections = defaultdict(list)
        for i in range(len(combined_lines)):
            for j in range(len(combined_lines)):
                if i == j:
                    continue
                if len(_polygons[i].intersection(_polygons[j])) >= 2:
                    connections[i].append(j)
                    problem.addConstraint(lambda x, y: x != y, (i, j))
        solution = problem.getSolution()

        if not solution:
            N = len(combined_lines)
            HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
            RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
            for i in range(len(combined_lines)):
                r, g, b = RGB_tuples[i]
                _style = f"fill:#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                self.add_path_node(combined_lines[i].d(), style=_style, id=f"shape-{i}")
                # print(i, connections[i])
                # for j in connections[i]:
                #    self.add_path_node(combined_lines[j].d(), style=_style, id=f"neighbor-{i}-{j}")
            debug_screen(self, "two_color_failure")
            raise ValueError("there is no solution!")
        else:
            return solution

    @lru_cache(maxsize=None)
    def is_inside(self, point, debug=False):
        return is_inside(self.container, point, debug, TOLERANCE)


if __name__ == "__main__":
    HitomezashiFill().run()
