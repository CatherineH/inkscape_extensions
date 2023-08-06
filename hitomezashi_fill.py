#!/usr/bin/env python3
import logging
import math
from collections import defaultdict
from typing import List

from constraint import Problem
from functools import lru_cache
import pickle

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
    is_inside,
    intersect_over_all,
    is_segment_diagonal,
    FOLDERNAME,
)
from enum import Enum
from random import random
from svgpathtools import Line, Path, parse_path
import sys
from copy import deepcopy
import colorsys

TOLERANCE = 0.2


class Corner(Enum):
    top_left = 0
    top_right = 1
    bottom_right = 2
    bottom_left = 3

    def project(self, container: Path) -> float:
        # get the bbox corner coordinates
        left, right, top, bottom = container.bbox()
        if self == self.top_left:
            return left + top * 1j
        if self == self.top_right:
            return right + top * 1j
        if self == self.bottom_left:
            return left + bottom * 1j
        if self == self.bottom_right:
            return right + bottom * 1j
        raise ValueError("not sure what corner type this is!")


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
        if self == self.bottom and other == other.top:
            return [Corner(Corner.bottom_right), Corner(Corner.top_right)]
        if self == self.top and other == other.bottom:
            return [Corner(Corner.top_left), Corner(Corner.bottom_left)]
        if self == self.left and other == other.right:
            return [Corner(Corner.bottom_left), Corner(Corner.bottom_right)]
        if self == self.right and other == other.left:
            return [Corner(Corner.top_right), Corner(Corner.top_left)]

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

    def project(self, point: complex, container: inkex.paths.Path) -> float:
        # project the point onto the bbox based on the nearest edge
        left, right, top, bottom = container.bbox()
        if self == self.left:
            return point.imag * 1j + left
        if self == self.right:
            return point.imag * 1j + right
        if self == self.top:
            return point.real + top * 1j
        if self == self.bottom:
            return point.real + bottom * 1j
        raise ValueError(f"not sure what edge this is {self=}")

    def marker(self, container):
        left, right, top, bottom = container.bbox()
        if self == self.left:
            return (top - bottom) * 0.5j + left
        if self == self.right:
            return (top - bottom) * 0.5j + right
        if self == self.top:
            return (right - left) * 0.5j + top * 1j
        if self == self.bottom:
            return (right - left) * 0.5j + bottom * 1j
        raise ValueError(f"not sure what edge this is {self=}")


class GraphType(object):
    def __init__(self, length):
        self._internal = defaultdict(dict)
        self.graph_locs = []
        self.length = length
        self.history = defaultdict(list)

    def snap_nodes(self, node):
        diffs = {}
        for i, ex_node in enumerate(self.graph_locs):
            diff = abs(ex_node - node)
            diffs[i] = diff
            if diff < TOLERANCE * self.length:
                return i
        # for node in diffs:
        #    if diffs[node] < TOLERANCE * self.options.length:
        #        return node
        self.graph_locs.append(node)
        return len(self.graph_locs) - 1

    def set_graph(self, start: float, end: float):
        line_i = self.snap_nodes(start)
        line_j = self.snap_nodes(end)

        self.set(line_i, line_j, Line(deepcopy(start), deepcopy(end)))
        is_segment_diagonal(self.get(line_i, line_j))
        is_segment_diagonal(self.get(line_j, line_i))
        return line_i, line_j

    def set(self, i, j, values):
        if isinstance(values, List):
            values = Path(*values)
        if isinstance(values, Line):
            values = Path(values)
        if not isinstance(values, Path):
            raise ValueError(f"not sure how to convert: {type(values)}")
        self.history[i].append((j, values))
        self.history[j].append((i, values.reversed()))
        self._internal[i][j] = values
        self._internal[j][i] = values.reversed()

    def delete(self, i, j):
        del self._internal[i][j]
        del self._internal[j][i]

    def delete_node(self, i):
        del self._internal[i]

    def get(self, i, j):
        assert isinstance(self._internal[i][j], Path)
        return self._internal[i][j]

    def confirm(self, i, j):
        assert isinstance(self._internal[i][j], Path)
        assert isinstance(self._internal[j][i], Path)

    def get_segments(self):
        return [path for entry in self._internal.values() for path in entry.values()]

    def keys(self):
        return list(self._internal.keys())

    def branches(self, graph_loc):
        return list(self._internal[graph_loc].keys())

    def audit_graph(self):
        # check whether there are points that are very close together
        nodes = self._internal.keys()
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i == j:
                    continue
                diff = abs(self.graph_locs[node_i] - self.graph_locs[node_j])
                if diff < TOLERANCE * self.length:
                    logging.debug(f"nodes {node_j} and {node_i} are only {diff} apart")

        for start_i in self._internal.keys():
            for end_i in self.branches(start_i):
                self.confirm(start_i, end_i)

    def total_length(self):
        return sum(
            path.length()
            for entry in self._internal.values()
            for path in entry.values()
        )


class HitomezashiFill(BaseFillExtension):
    def __init__(self):
        BaseFillExtension.__init__(self, self.hitomezashi_fill)
        self.xmax = None
        self.xmin = None
        self.ymin = None
        self.ymax = None
        self.container = None
        self.saved_solution = []
        self.outline_intersections = []
        self.outline_nodes = []
        # build a graph of which edge points connect where
        self.graph = None
        self.edges = []
        self.edges_to_visit = []
        self.visited_edges = []
        self.chained_lines = []
        self.last_branches = []  # keep track of the last possible branch
        self._debug_branches = []
        self._evaluated = []
        self.x_sequence = []
        self.y_sequence = []
        self.interactive_screen = True

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
        pars.add_argument(
            "--triangle", type=str, default="false", help="fill the stitch shapes"
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
                logging.debug(f"outline nodes are {self.outline_nodes}")
                logging.debug(
                    f"got key error on: {chained_line[i]} not in {self.graph.branches(chained_line[i-1])} {self.graph.branches(chained_line[i])}"
                )
        stroke_length = self.options.length / 10
        self.add_path_node(
            combine_segments(segments).d(),
            f"stroke:{color};stroke-width:{stroke_length};fill:none",
            label,
        )

    def plot_graph(self, color="gray", label="graph", connected=True):
        # dump the graph
        # connected does not plot the graph correctly
        connected = False
        if connected:
            all_graph_segments = self.graph.get_segments()
            self.add_path_node(
                combine_segments(all_graph_segments).d(),
                f"stroke:{color};stroke-width:2;fill:none",
                label,
            )
        else:
            for start_i in self.graph.keys():
                for end_i in self.graph.branches(start_i):
                    try:
                        self.add_path_node(
                            self.graph.get(start_i, end_i).d(),
                            f"stroke:{color};stroke-width:2;fill:none",
                            f"{label}-{start_i}-{end_i}",
                        )
                    except AttributeError as e:
                        logging.debug(f"skipping {self.graph[start_i][end_i]}")
                        pass

    def chop_shape(self, lines: List[Path]) -> Path:
        final_lines = []
        # chop the unconnected lines if they run outside the boundary shop
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

        # make the graph of rectlinear node locations for these chopped down lines
        for line in final_lines:
            if (
                line.length() < TOLERANCE * self.options.length
            ):  # skip this one because it's too short
                logging.debug(f"skipping: {line}")
                continue
            self.graph.set_graph(line.start, line.end)

        # handle how traversal around the shape edge goes
        self.outline_intersections = list(set(self.outline_intersections))
        self.outline_intersections.sort(key=lambda x: -x[0] - x[1])
        intersections_copy = self.outline_intersections
        start_intersection = intersections_copy.pop()
        intersections_copy.insert(
            0, start_intersection
        )  # add the end back onto the front so that we'll close the loop
        while intersections_copy:
            end_intersection = intersections_copy.pop()

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
                logging.debug(f"skipping: {start_intersection=} {end_intersection=}")
                continue
            start_i, end_i = self.graph.set_graph(segment.start, segment.end)
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
                logging.debug(f"closing loop {format_complex(chained_line)}")
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
                    logging.debug(
                        f"chained_line {len(self._debug_branches)} loops back on itself at index {loop_index} {format_complex(chained_line)} "
                    )
                return 0
        if debug:
            logging.debug(
                f"chained line {format_complex(chained_line)} is valid, continuing"
            )
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
        branches = self.graph.branches(chained_line[-1])
        # remove the ones already on the line, but not the first point
        branches = [point for point in branches if point not in chained_line[1:]]
        branches.sort(
            key=lambda x: abs(
                self.graph.graph_locs[x] - self.graph.graph_locs[chained_line[0]]
            )
        )

        for branch in branches:
            chain_to_add = deepcopy(chained_line) + [branch]
            if debug:
                logging.debug(f"adding {chain_to_add}")
            self.last_branches.append(chain_to_add)

        return None

    def chain_graph(self) -> List[Path]:
        self.simplify_graph()
        if len(self.graph.keys()) > 500:
            msg = f"there are two many edges to fill. Consider using a shorter length piece or not filling the shape."
            inkex.utils.errormsg(msg)
            raise ValueError(msg)
        self.audit_graph()
        # use the edge to make a loop that extends past the edges of the bbox add that to the stacks
        loops: List[Path] = []  # self.edges

        for edge_i, edge in enumerate(self.edges):
            print(f"looped {edge_i}/{len(self.edges)}")
            loop = Path(*edge)
            if loop.start == loop.end:
                loops.append(loop)
                continue
            # which edge are the end points nearest?
            start_edge = self.find_closest_edge(edge.start)
            end_edge = self.find_closest_edge(edge.end)
            corners_in_between = end_edge.corners_between(start_edge)
            # add the border lines at the end of the path
            _projected_edge = Line(
                start=edge.end, end=end_edge.project(edge.end, self.container)
            )
            loop.append(_projected_edge)

            for corner_in_between in corners_in_between:
                _edge_in_between = Line(
                    start=loop[-1].end,
                    end=corner_in_between.project(self.container),
                )
                try:
                    is_segment_diagonal(_edge_in_between)
                except AssertionError as err:
                    self.add_path_node(
                        edge.d(), style="fill:none;stroke:blue", id="edge"
                    )
                    self.add_path_node(
                        loop.d(), style="fill:none;stroke:black", id="edge"
                    )
                    self.add_path_node(
                        Path(_edge_in_between).d(),
                        style="fill:none;stroke:green",
                        id="new_edge",
                    )
                    self.add_path_node(
                        self.container.d(),
                        style="fill:none;stroke:gray",
                        id="container",
                    )
                    self.add_marker(edge.end)
                    self.add_marker(end_edge.marker(self.container), label="end_edge")
                    self.add_marker(edge.start, color="purple")
                    self.add_marker(
                        start_edge.marker(self.container),
                        color="purple",
                        label="start_edge",
                    )
                    msg = f"{err}, edges were {start_edge} {end_edge} corner in between was {corner_in_between} {corners_in_between}"
                    print(msg)
                    debug_screen(self, "corner_failure")
                    raise AssertionError(msg)
                loop.append(_edge_in_between)
            loop.append(
                Line(
                    start=loop[-1].end,
                    end=start_edge.project(edge.start, self.container),
                )
            )
            loop.append(Line(start=loop[-1].end, end=edge.start))
            if isinstance(loop, list):
                loop = Path(*loop)
            assert loop.start == loop.end
            for i, segment in enumerate(loop):
                if loop[i - 1].end == segment.start:
                    assert (
                        abs(loop[i - 1].end - segment.start)
                        < TOLERANCE * self.options.length
                    ), f"loop is not continuous! on segment {i} {loop[i-1].end} {segment.start} {loop.d()}"
                    loop[i - 1].end = segment.start
                is_segment_diagonal(segment)

            loops.append(loop)

        """
        _dump = {"container": self.container, "edges": self.edges}
        with open(FOLDERNAME+"/chain_graph.pkl", "wb") as fh:
            pickle.dump(_dump, fh)
        """
        return loops

    def find_closest_edge(self, point: float) -> NearestEdge:
        left, right, top, bottom = self.container.bbox()
        # left, top, right, bottom
        _distance = [
            point.real - left,
            point.imag - top,
            point.real - right,
            point.imag - bottom,
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
        logging.debug(f"path building took {time()-start_time}")
        start_time = time()
        intersection_pv = intersection(container_pv, chained_lines_pv)
        logging.debug(f"bool op took {time()-start_time}")
        start_time = time()
        output_chained_lines = []
        for piece in intersection_pv:
            piece_d = py2geom.write_svg_path(piece)
            output_chained_lines.append(Path(piece_d))
        logging.debug(f"decomp took {time()-start_time}")
        return output_chained_lines

    def simplify_graph(self):
        """merge any nodes that have only two outputs into a path between the two.
        Keep doing this until there are no more nodes to evaluate"""
        self.plot_graph(color="blue")
        original_length = self.graph.total_length()
        _dump = {
            "container": self.container,
            "graph": self.graph,
            "graph_locs": self.graph.graph_locs,
        }
        with open(FOLDERNAME + "/simplify_graph.pkl", "wb") as fh:
            pickle.dump(_dump, fh)
        all_nodes = deepcopy(self.graph.keys())
        logging.debug(f"before simplification the graph had {len(all_nodes)} nodes")
        while all_nodes:
            to_evaluate = all_nodes.pop()
            branches = self.graph.branches(to_evaluate)
            outline_branches = [
                branch for branch in branches if branch in self.outline_nodes
            ]
            inside_branches = [
                branch for branch in branches if branch not in self.outline_nodes
            ]
            if (
                len(inside_branches) != 2
            ):  # segments must have only one input and output
                continue
            if outline_branches:
                branches = inside_branches
                # collapse down the outside branches
                start_outside = outline_branches.pop()
                end_outside = outline_branches.pop()
                outside_edge = [*self.graph.get(start_outside, to_evaluate)] + [
                    *self.graph.get(to_evaluate, end_outside)
                ]
                if not isinstance(outside_edge, Path):
                    outside_edge = Path(*outside_edge)
                # self.edges.append(segment)
                self.graph.set(start_outside, end_outside, outside_edge)
                self.graph.set(end_outside, start_outside, outside_edge.reversed())

            start_i = branches[0]
            end_i = branches[1]
            if (
                start_i == to_evaluate or end_i == to_evaluate
            ):  # the edge is a loop, skip over reducing
                continue

            segments = Path(
                *self.graph.get(start_i, to_evaluate),
                *self.graph.get(to_evaluate, end_i),
            )
            if start_i in self.graph.branches(end_i):
                print(f"closing loop for {start_i} {end_i}")
                # the bridging section between start and end also needs to be added in order to close the loop
                segments += self.graph.get(end_i, start_i)
            print(
                f"start {start_i} to {to_evaluate} point: {self.graph.get(start_i,to_evaluate)}"
            )
            print(
                f"end {end_i} to {to_evaluate} point: {self.graph.get(to_evaluate,end_i)}"
            )
            # confirm that the segments are contiguous
            for i, segment in enumerate(segments):
                if i == 0:
                    continue
                if segments[i - 1].end != segment.start:
                    if (
                        abs(segments[i - 1].end - segment.start)
                        < TOLERANCE * self.options.length
                    ):
                        segments[i - 1].end = segment.start
                    else:
                        segment_names = (
                            ".".join(f"{int(x)}" for x in self.x_sequence)
                            + "_"
                            + ".".join(f"{int(x)}" for x in self.y_sequence)
                        )

                        self.plot_graph(connected=False)
                        self.add_marker(segments[i - 1].end, color="red")
                        self.add_marker(segment.start, color="blue")
                        self.add_marker(
                            self.graph.graph_locs[to_evaluate], color="orange"
                        )
                        debug_screen(
                            self, f"non_contiguous_segments_{segment_names}", show=False
                        )
                        history = "\n".join(
                            str(entry) for entry in self.graph.history[to_evaluate]
                        )
                        raise ValueError(
                            f"chained together non-contiguous segments: {segments[i-1].end} {segment.start}"
                            f" {segments} while evaluating {to_evaluate}: {self.graph.graph_locs[to_evaluate]} {history}"
                        )
            segment = Path(*segments)
            # self.edges.append(segment)
            self.graph.set(start_i, end_i, segment)
            self.graph.set(end_i, start_i, segment.reversed())
            self.graph.delete(to_evaluate, start_i)
            self.graph.delete(to_evaluate, end_i)
            self.graph.confirm(start_i, end_i)

        # second round - join edges
        all_nodes = deepcopy(self.graph.keys())
        logging.debug(f"before simplification the graph had {len(all_nodes)} nodes")
        while all_nodes:
            to_evaluate = all_nodes.pop()

            branches = self.graph.branches(to_evaluate)
            if len(branches) == 0:
                self.graph.delete_node(to_evaluate)

            if len(branches) <= 1:  # skip over closed loops
                continue
            outline_branches = [
                branch for branch in branches if branch in self.outline_nodes
            ]
            inside_branches = [
                branch for branch in branches if branch not in self.outline_nodes
            ]
            if len(inside_branches) == 0:
                start_i = outline_branches.pop()
            else:
                start_i = inside_branches.pop()
            if len(outline_branches) == 0:
                end_i = inside_branches.pop()
            else:
                end_i = outline_branches.pop()

            remaining_branches = inside_branches + outline_branches
            if len(remaining_branches) >= 2:
                # collapse down the outside branches
                start_outside = remaining_branches.pop()
                end_outside = remaining_branches.pop()
                outside_edge = [*self.graph[start_outside][to_evaluate]] + [
                    *self.graph[to_evaluate][end_outside]
                ]
                outside_edge = Path(*outside_edge)
                # self.edges.append(segment)
                self.graph[start_outside][end_outside] = outside_edge
                self.graph[end_outside][start_outside] = outside_edge.reversed()
            for remaining_branch in remaining_branches:
                print(f"removing {remaining_branch} -> {to_evaluate}")
                self.graph.delete(remaining_branch, to_evaluate)

            segments = [*self.graph.get(start_i, to_evaluate)] + [
                *self.graph.get(to_evaluate, end_i)
            ]

            print(
                f"start {start_i} to {to_evaluate} point: {self.graph.get(start_i,to_evaluate)}"
            )
            print(
                f"end {end_i} to {to_evaluate} point: {self.graph.get(to_evaluate,end_i)}"
            )
            # confirm that the segments are contiguous
            for i, segment in enumerate(segments):
                if i == 0:
                    continue
                if segments[i - 1].end != segment.start:
                    if (
                        abs(segments[i - 1].end - segment.start)
                        < TOLERANCE * self.options.length
                    ):
                        segments[i - 1].end = segment.start
                    else:
                        raise ValueError(
                            f"chained together non-contiguous segments: {segments[i-1].end} {segment.start}"
                            f" {segments} while evaluating {to_evaluate}: {self.graph.graph_locs[to_evaluate]}"
                        )
            # self.edges.append(segment)
            self.graph.set(start_i, end_i, segments)
            self.graph.delete(start_i, to_evaluate)
            self.graph.delete(end_i, to_evaluate)
            self.graph.confirm(start_i, end_i)
            # confirm that we haven't destroyed the graph

            self.graph.audit_graph()
            # the resulting graph after simplification should not be too much shorter than the original length
            simplified_length = sum(
                [path.length() for path in self.graph.get_segments()]
            )
            if simplified_length <= original_length * 0.9:
                segment_names = (
                    ".".join(f"{int(x)}" for x in self.x_sequence)
                    + "_"
                    + ".".join(f"{int(x)}" for x in self.y_sequence)
                )
                self.plot_graph(label="oversimp", color="orange")
                debug_screen(self, f"over_simplification_{segment_names}", show=False)
                raise ValueError(
                    f"simplified length {simplified_length} is too short compared to original length {original_length}"
                )

        evaluated_edges = []
        for start_i in self.graph.keys():
            for end_i in self.graph.branches(start_i):
                if {start_i, end_i} not in evaluated_edges:
                    _path = self.graph.get(start_i, end_i)
                    if not isinstance(_path, Path):
                        _path = Path(_path)
                    self.edges.append(_path)
                    evaluated_edges.append({start_i, end_i})
        logging.debug(
            f"after simplification the graph has {len(self.graph.keys())} nodes"
        )

    def audit_graph(self):
        # check whether there are points that are very close together
        nodes = list(self.graph.keys())
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i == j:
                    continue
                diff = abs(
                    self.graph.graph_locs[node_i] - self.graph.graph_locs[node_j]
                )
                if diff < TOLERANCE * self.options.length:
                    logging.debug(f"nodes {node_j} and {node_i} are only {diff} apart")

        def out_of_bounds(node_i):
            node = self.graph.graph_locs[node_i]
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
            branches = self.graph.branches(node)
            for branch in branches:
                self.graph.delete(branch, node)
            self.graph.delete_node(node)

    def triangular_lines(self):
        # you need to go over three times
        # 30 and 60 degrees left to right
        # then one last time horizontally down to up

        triangle_height = math.cos(math.pi / 3) * self.options.length

        print(f"triangle_height {triangle_height}")
        lines = []
        num_x = int(self.width / self.options.length)
        if self.width / self.options.length % 1:
            num_x += 1
        # the height
        num_y = int(self.height / triangle_height)
        if self.height / self.options.length % 1:
            num_y += 1
        # generate horizontal lines
        for y_i in range(num_y):
            y_coord = y_i * triangle_height + self.ymin
            if self.y_sequence:
                odd_even_y = self.y_sequence[y_i % len(self.y_sequence)]
            elif not self.options.gradient:
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

                x_coord = x_i * self.options.length + self.xmin
                if y_i % 2 == 0:
                    x_coord += 0.5 * self.options.length
                start = x_coord + y_coord * 1j
                end = (x_coord + self.options.length + diff) + y_coord * 1j
                lines.append(Line(start, end))
                assert start != end
        num_before_angles = float(len(lines))
        thirty_degree = defaultdict(bool)

        for y_i in range(num_y):
            y_coord = y_i * triangle_height + self.ymin
            for x_i in range(num_x):
                grid_key = math.ceil(y_i / 2) - x_i
                if grid_key not in thirty_degree:
                    thirty_degree[grid_key] = random() > 0.5
                thirty_degree[grid_key] = not thirty_degree[grid_key]
                if thirty_degree[grid_key]:
                    continue
                x_coord = x_i * self.options.length + self.xmin
                if y_i % 2 == 0:
                    x_coord += 0.5 * self.options.length
                start = y_coord * 1j + x_coord
                end = (
                    (y_coord + triangle_height) * 1j
                    + x_coord
                    + 0.5 * self.options.length
                )
                lines.append(Line(start, end))
                assert start != end

        sixty_degree = defaultdict(bool)

        for y_i in range(num_y):
            y_coord = y_i * triangle_height + self.ymin
            for x_i in range(num_x):
                grid_key = math.ceil((num_x - y_i) / 2) - x_i
                if grid_key not in sixty_degree:
                    sixty_degree[grid_key] = random() > 0.5
                sixty_degree[grid_key] = not sixty_degree[grid_key]
                if sixty_degree[grid_key]:
                    continue
                # if not grid_key == 0:
                #    continue
                x_coord = x_i * self.options.length + self.xmin
                if y_i % 2 == 0:
                    x_coord += 0.5 * self.options.length
                start = y_coord * 1j + x_coord
                end = (
                    (y_coord - triangle_height) * 1j
                    + x_coord
                    + 0.5 * self.options.length
                )
                lines.append(Line(start, end))
                assert start != end
        assert num_before_angles < len(lines), "no new lines added!"
        return lines

    def rectilinear_lines(self):
        # generate vertical lines
        lines = []
        num_x = int(self.width / self.options.length)
        if self.width / self.options.length % 1:
            num_x += 1
        num_y = int(self.height / self.options.length)
        if self.height / self.options.length % 1:
            num_y += 1
        true_x_sequence = []
        for x_i in range(num_x):
            x_coord = x_i * self.options.length + self.xmin
            if x_coord == self.xmin:
                # make the line just inside the box
                x_coord += TOLERANCE / 2.0
            if self.x_sequence:
                odd_even_y = self.x_sequence[x_i % len(self.x_sequence)]
            elif not self.options.gradient:
                odd_even_y = random() > self.options.weight_x
            else:
                odd_even_y = random() > x_i / num_x
            true_x_sequence.append(odd_even_y)
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
        true_y_sequence = []
        for y_i in range(num_y):
            y_coord = y_i * self.options.length + self.ymin
            if self.y_sequence:
                odd_even_y = self.y_sequence[y_i % len(self.y_sequence)]
            elif not self.options.gradient:
                odd_even_y = random() > self.options.weight_y
            else:
                odd_even_y = random() > y_i / num_y
            true_y_sequence.append(odd_even_y)

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
        return lines

    def hitomezashi_fill(self, node):
        # greedy algorithm: make a Hitomezashi fill that covers the entire bounding box of the shape,
        # then go through each segment and figure out if it is inside, outside, or intersecting the shape
        self.container = parse_path(pattern_vector_to_d(node))
        self.container.approximate_arcs_with_quads()
        self.graph = GraphType(self.options.length)
        self.options.fill = False if self.options.fill == "false" else True
        self.options.gradient = False if self.options.gradient == "false" else True
        self.options.triangle = False if self.options.triangle == "false" else True

        self.container = parse_path(pattern_vector_to_d(node))

        self.container.approximate_arcs_with_quads()
        self.xmin, self.xmax, self.ymin, self.ymax = self.container.bbox()
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin
        if self.options.triangle:
            lines = self.triangular_lines()
        else:
            lines = self.rectilinear_lines()
        labels = []
        if not self.options.fill:
            _ = [self.chop_shape(lines)]
            self.audit_graph()
            self.simplify_graph()
            # self.saved_solution = [Path(line) for line in lines] #self.edges
            self.saved_solution = self.chain_graph()
            self.saved_solution = [line.d() for line in lines]
            labels = [i for i in range(len(self.saved_solution))]
        else:
            lines = []
            curr_color = True
            start_rows = []
            # combine the lines?
            # combined_lines = inkex.paths.Path().from_svgpathtools(lines.pop())
            unmergable = []
            while lines:
                to_combine_line1 = lines.pop()
                if not lines:
                    combined_lines = to_combine_line1
                to_combine_line2 = lines.pop()
                if not isinstance(to_combine_line1, inkex.paths.Path):
                    to_combine_line1 = inkex.paths.Path().from_svgpathtools(
                        to_combine_line1
                    )
                if not isinstance(to_combine_line2, inkex.paths.Path):
                    to_combine_line2 = inkex.paths.Path().from_svgpathtools(
                        to_combine_line2
                    )

                try:
                    combined_lines = to_combine_line1.union(to_combine_line2)
                    lines.insert(0, combined_lines)
                    self.saved_solution = [combined_lines.to_svgpathtools()]

                    print("combined lines", combined_lines)
                except Exception as err:
                    self.add_path_node(
                        to_combine_line1.d(), style="fill:red", id="to-merge"
                    )
                    self.add_path_node(
                        to_combine_line2.d(), style="fill:blue", id="merged-path"
                    )
                    debug_screen(self, name="merge_failure", show=False)
                    continue

            color_fills = ["red" for _ in self.saved_solution]
            """ chaining graph solution
            lines = self.chop_shape(lines)
            self.plot_graph(connected=False)
            self.saved_solution = self.chain_graph()
            # next: we need to stack and cut the paths out of each other

            combined_lines, labels = stack_lines(self.saved_solution)
            color_fills = self.color_pattern(combined_lines)
            self.saved_solution = combined_lines
            logging.debug(f"labels {len(labels)} {labels}")
            logging.debug(f"combined lines {len(combined_lines)}")
            """
        for i, chained_line in enumerate(self.saved_solution):
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
                color = "red" if color_fills[i] else "white"
                pattern_style = pattern_style.replace("fill:none", f"fill:{color}")
                if "fill" not in pattern_style:
                    pattern_style += f";fill:{color}"
            self.add_path_node(chained_line.d(), pattern_style, pattern_id)

    def color_pattern(self, combined_lines: List[Path]):
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
                # logging.debug(i, connections[i])
                # for j in connections[i]:
                #    self.add_path_node(combined_lines[j].d(), style=_style, id=f"neighbor-{i}-{j}")
            self.plot_graph(connected=False)
            for i, line in enumerate(self.saved_solution):
                self.add_path_node(
                    Path(*line).d(), style=f"fill:none;stroke:blue", id=f"loop{i}"
                )
            segment_names = (
                ".".join(f"{int(x)}" for x in self.x_sequence)
                + "_"
                + ".".join(f"{int(x)}" for x in self.y_sequence)
            )
            debug_screen(
                self, f"two_color_failure_{segment_names}", show=self.interactive_screen
            )
            raise ValueError("there is no solution!")
        else:
            return solution

    @lru_cache(maxsize=None)
    def is_inside(self, point, debug=False):
        return is_inside(self.container, point, debug, TOLERANCE)


if __name__ == "__main__":
    HitomezashiFill().run()
