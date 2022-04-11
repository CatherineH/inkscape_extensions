#!/usr/bin/env python3
from collections import defaultdict
import inkex
from time import time
from common_utils import (
    pattern_vector_to_d,
    BaseFillExtension,
    debug_screen,
    combine_segments,
    format_complex,
    make_stack_tree,
    is_inside,
    intersect_over_all,
find_orientation
)
from random import random
from svgpathtools import Line, Path, parse_path
from functools import lru_cache
import sys
from copy import deepcopy


TOLERANCE = 0.2


class HitomezashiFill(BaseFillExtension):
    def __init__(self):
        inkex.EffectExtension.__init__(self)
        self.xmax = None
        self.xmin = None
        self.ymin = None
        self.ymax = None
        self.container = None
        self.curr_path_num = 0
        self.current_shape = None
        self.outline_intersections = []
        self.outline_nodes = []
        # build a graph of which edge points connect where
        self.graph = defaultdict(dict)
        self.graph_locs = []
        self.visited_points = []
        self.chained_lines = []
        self.last_branches = [] # keep track of the last possible branch
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

    def effect(self):
        self.options.gradient = self.options.gradient == "true"
        self.options.fill = self.options.fill == "true"
        if self.svg.selected:
            for i, shape in self.svg.selected.items():
                self.curr_path_num = i
                self.current_shape = shape
                self.hitomezashi_fill(shape)

    def add_node(self, d_string, style, id):
        parent = self.get_parent(self.current_shape)
        _node = inkex.elements.PathElement()
        _node.set_path(d_string)
        _node.set("style", style)
        _node.set("id", id)
        if self.current_shape.get("transform"):
            _node.set("transform", self.current_shape.get("transform"))
        parent.insert(-1, _node)

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
        self.add_node(
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
        self.add_node(Path(*marker).d(), f"fill:{color};stroke:none", label)

    def plot_graph(self, color="gray", label="graph", connected=True):
        # dump the graph
        if connected:
            all_graph_segments = [
                segment for branch in self.graph.values() for segment in branch.values()
            ]
            self.add_node(
                combine_segments(all_graph_segments).d(),
                f"stroke:{color};stroke-width:2;fill:none",
                label,
            )
        else:
            for start_i in self.graph.keys():
                for end_i in self.graph[start_i].keys():
                    try:
                        self.add_node(self.graph[start_i][end_i].d(), f"stroke:{color};stroke-width:2;fill:none",
                                      f"{label}-{start_i}-{end_i}")
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
                    loop_index = chained_line.index(point, index+1)
                    break
                except ValueError as err:
                    pass
            if loop_index:
                if debug:
                    print(f"chained_line {len(self._debug_branches)} loops back on itself at index {loop_index} {format_complex(chained_line)} ")
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
        branches = [
            point
            for point in branches
            if point not in chained_line[1:]
        ]
        branches.sort(key=lambda x: abs(self.graph_locs[x] - self.graph_locs[chained_line[0]]))

        for branch in branches:
            chain_to_add = deepcopy(chained_line) + [branch]
            if debug:
                print(f"adding {chain_to_add}")
            self.last_branches.append(chain_to_add)

        return None

    def chain_graph(self):
        #self.plot_graph(connected=False)
        self.simplify_graph()
        #self.plot_graph(color="blue", label="simplified_graph", connected=False)
        #debug_screen(self, "test_graph_simplify")
        self.audit_graph()

        # algorithm design
        # dump the keys in the graph into a unique list of points
        self.points_to_visit = list(set(self.graph.keys()))

        self.visited_points = []
        start_time = time()
        print(f"num points to visit: {len(self.points_to_visit)}")
        while self.points_to_visit:
            chain_start_time = time()
            curr_point = self.points_to_visit.pop()

            if curr_point in self.visited_points:
                continue
            chained_line = [curr_point]
            self.get_branches(chained_line)
            assert self.last_branches
            num_iterations = 0

            while self.last_branches:
                chained_line = self.last_branches.pop(0)
                # re-sort the branches every additional 100
                self._debug_branches.append(chained_line)

                is_complete = self.get_branches(chained_line, False)
                if is_complete:
                    break
                num_iterations += 1

            if len(chained_line) < 4 or abs(chained_line[0] - chained_line[-1]) > TOLERANCE or not self.audit_overlap(chained_line):
                self.plot_graph()
                for loc in self._evaluated:
                    self.add_marker(loc, color="green")
                self.add_chained_line(chained_line)
                self.add_marker(chained_line[0])

                debug_screen(self, "test_failed_connect")
                raise ValueError(f"failed on line {format_complex(chained_line)}, aborting ")

            self.last_branches = []
            self._debug_branches = []
            self._evaluated = []
            self.chained_lines.append(chained_line)
            for point in chained_line:
                if point not in self.visited_points:
                    self.visited_points.append(point)

        print(f"chaining took {time()-start_time} num lines {len(self.chained_lines)}")
        # convert to segments
        paths = []
        for chained_line in self.chained_lines:
            segments = []
            for i in range(1, len(chained_line)):
                segments.append(self.graph[chained_line[i - 1]][chained_line[i]])
            paths.append(combine_segments(segments))
        return paths

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
        """ merge any nodes that have only two outputs into a path between the two.
        Keep doing this until there are no more nodes to evaluate """
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
            if start_i == to_evaluate or end_i == to_evaluate or start_i == end_i or end_i in self.graph[start_i] \
                    or start_i in self.graph[end_i]:
                # skip over reducing this one
                continue
            if not isinstance(self.graph[start_i][to_evaluate], Path):
                self.graph[start_i][to_evaluate] = Path(self.graph[start_i][to_evaluate])
            if not isinstance(self.graph[to_evaluate][end_i], Path):
                self.graph[to_evaluate][end_i] = Path(self.graph[to_evaluate][end_i])
            segments = [*self.graph[start_i][to_evaluate]] + [*self.graph[to_evaluate][end_i]]
            segment = Path(*segments)

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

        if not self.options.fill:
            pass
            lines = [self.chop_shape(lines)]
            lines = [line.d() for line in lines]
        else:
            _ = self.chop_shape(lines)
            lines = self.chain_graph()
            # next: we need to stack and cut the paths out of each other
            stack_tree, root_nodes = make_stack_tree(lines)
            line_d_strings = []
            while root_nodes:
                current_node = root_nodes.pop()
                child_nodes = stack_tree[current_node]
                parent_orientation = find_orientation(lines[current_node])
                basic_d_string = f"{lines[current_node].d()} Z"
                for child_node in child_nodes:
                    child_orientation = find_orientation(lines[child_node])
                    if child_orientation == parent_orientation:
                        lines[child_node] = 
                    basic_d_string = f"{basic_d_string} {lines[child_node].d()} Z"
                    if child_node in stack_tree:
                        root_nodes += stack_tree[child_node]
                line_d_strings.append(basic_d_string)
            lines = line_d_strings

        for i, chained_line in enumerate(lines):
            if chained_line == "":
                raise ValueError(f"got empty chained_path! {i} {chained_line}")
            pattern_id = (
                "hitomezashi-"
                + node.get("id", f"unknown-{self.curr_path_num}")
                + "-"
                + str(i)
            )
            pattern_style = node.get("style")
            if self.options.fill:
                pattern_style = pattern_style.replace("fill:none", "fill:red")
                if "fill" not in pattern_style:
                    pattern_style += ";fill:'red'"
            self.add_node(chained_line, pattern_style, pattern_id)

    @lru_cache(maxsize=None)
    def is_inside(self, point, debug=False):
        return is_inside(self.container, point, debug, TOLERANCE)


if __name__ == "__main__":
    HitomezashiFill().run()
