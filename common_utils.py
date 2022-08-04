import logging
from typing import KeysView
import inkex

try:
    from svgpathtools.svg_to_paths import rect2pathd, ellipse2pathd
    from svgpathtools import Path, Line
except ImportError:
    import sys

    inkex.utils.errormsg(f"svgpathtools is not available on {sys.executable}")
from numpy import matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from math import atan
import subprocess
from typing import List
from functools import lru_cache
from os.path import dirname, abspath, join
import asyncio
import signal
import functools
import os
import errno
import datetime


class TimeoutError(Exception):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


FOLDERNAME = join(dirname(abspath(__file__)), "output")
TOLERANCE = 0.2


def append_verify(
    path: List[inkex.paths.PathCommand], in_segment: inkex.paths.PathCommand
):
    path.append(in_segment)
    for i, path_segment in enumerate(path.to_svgpathtools()):
        assert (
            path_segment.start != path_segment.end
        ), f"degenerate path was added! {path_segment=}"


class BaseFillExtension(inkex.EffectExtension):
    def __init__(self, effect_handle, init_handle=None):
        inkex.EffectExtension.__init__(self)

        self.curr_path_num = 0
        self.current_shape = None
        self.effect_handle = effect_handle
        self.init_handle = init_handle

    def get_parent(self, node):
        parent = node.getparent()
        if parent is None:
            parent = self.document.getroot()
        return parent

    def add_path_node(self, d_string, style, id):
        parent = self.get_parent(self.current_shape)
        _node = inkex.elements.PathElement()
        _node.set_path(d_string)
        _node.set("style", style)
        _node.set("id", id)
        if self.current_shape.get("transform"):
            _node.set("transform", self.current_shape.get("transform"))
        parent.insert(-1, _node)

    def add_marker(self, point, label="marker", color="red"):
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

    def effect(self):
        if self.init_handle:
            self.init_handle()
        if self.svg.selected:
            for i, shape in self.svg.selected.items():
                self.curr_path_num = i
                self.current_shape = shape
                self.effect_handle(shape)


def transform_path_vector(pv, affine_t):
    for _path in pv:
        for _curve in _path:
            _curve.transform(affine_t)
    return pv


def find_orientation(in_path):
    points = []
    for segment in in_path:
        points.append(segment.end)
    # https://en.wikipedia.org/wiki/Curve_orientation#Orientation_of_a_simple_polygon
    orientations = 0

    for i in range(len(points)):
        _piece = []
        for j in range(3):
            _piece.append(points[(i + j) % len(points)])
        # x2*y3+x1*y2+y1*x3 - (y1*x2+y2*x3+x1*y3)
        orientations += (
            _piece[1].real * _piece[2].imag
            + _piece[0].real * _piece[1].imag
            + _piece[0].imag * _piece[2].real
            - (
                _piece[0].imag * _piece[1].real
                + _piece[1].imag * _piece[2].real
                + _piece[0].real * _piece[2].imag
            )
        )
    return orientations > 0


def pattern_vector_to_d(pattern_vector):
    if "d" in pattern_vector.attrib:
        return pattern_vector.attrib.get("d")
    elif pattern_vector.TAG == "rect":
        return rect2pathd(pattern_vector.attrib)
    elif pattern_vector.TAG in ["circle", "ellipse"]:
        return ellipse2pathd(pattern_vector.attrib)
    else:
        inkex.utils.errormsg(
            "Shape %s (%s) not yet supported, try Object to path first"
            % (pattern_vector.TAG, pattern_vector.get("id"))
        )


def get_fill_id(node):
    style = node.attrib.get("style")
    inkex_style = dict(inkex.styles.Style().parse_str(style))
    fill = node.attrib.get("fill")

    inkex_style["fill"] = fill or inkex_style.get("fill")
    if not inkex_style["fill"]:
        inkex.utils.errormsg("no 'fill' in inkex style")
        return
    if inkex_style["fill"].find("url(") != 0:
        inkex.utils.errormsg(
            f"for shape {node.attrib.get('id')} fill does not contain a pattern reference"
        )
        return
    if inkex_style["fill"] == "none":
        return
    return inkex_style["fill"].replace("url(#", "")[:-1]


def debug_screen(effect, name=None):
    name = name or str(effect)

    filename = join(FOLDERNAME, f"debug_{name}.svg")
    effect.save(open(filename, "wb"))
    subprocess.run(["inkview", filename])


def make_stack_tree(lines, debug=False):
    """

    Parameters
    ----------
    lines: a list of svgpathtool Paths

    Returns
    -------
    a tuple - first element is a graph, second element is a list of root notes
    """
    assert lines

    @lru_cache(None)
    def pairwise_comparison(i, j, debug=False):
        if i == j:
            return 0
        reverse = _matrix[j][i]
        if reverse:
            if debug:
                print(f"already evaluated {j}, {i}")
            return 0
        path1 = lines[i]
        path2 = lines[j]
        if isinstance(path1, list):
            path1 = Path(*path1)
        if isinstance(path2, list):
            path2 = Path(*path2)
        # returns True if path1 is inside path2
        xmin1, xmax1, ymin1, ymax1 = path1.bbox()
        xmin2, xmax2, ymin2, ymax2 = path2.bbox()
        bbox_inside = (
            xmin1 <= xmin2 + TOLERANCE <= xmax2 - TOLERANCE <= xmax1
            and ymin1 <= ymin2 + TOLERANCE <= ymax2 - TOLERANCE <= ymax1
        )
        if not bbox_inside:
            if debug:
                print(
                    f"bboxes for {j} inside {i} do not overlap: {xmin1}, {xmin2 + TOLERANCE}, {xmax2-TOLERANCE}, {xmax1} & {ymin1}, {ymin2+TOLERANCE}, {ymax2-TOLERANCE}, {ymax1} "
                )
            return 0
        else:
            if debug:
                print(f"{j} bbox is inside {i}")
        # if the bboxes overlap, do something more complex
        # if any of the points in path2 is inside path1, the line must be inside
        segments_inside = []
        for segment in path2:
            print(f"lengths of paths to intersection {len(segment)=} {len(path1)=}")
            loop = asyncio.ProactorEventLoop()
            asyncio.set_event_loop(loop)

            intersections = intersect_over_all(segment, path1, exit_early=True)

            if len(intersections) > 0:
                segments_inside.append(1)
            else:
                print(
                    f"line does not intersect {Path(segment).d()} container {path1.d()}"
                )
                segments_inside.append(
                    is_inside(path1, segment.start, debug=debug, tolerance=0.01)
                )
        if debug:
            print(f"segments for {j} inside {i}: {segments_inside}")
        return all(segments_inside)

    if debug:
        print(f"make_stack_tree number of lines {len(lines)}")
    raw_stack_tree = defaultdict(list)
    _matrix = [[0 for i in range(len(lines))] for j in range(len(lines))]
    for i in range(len(lines)):
        for j in range(len(lines)):
            debug = True if i == 1 and j == 2 else False
            _matrix[i][j] = pairwise_comparison(i, j, debug)
            if _matrix[i][j]:
                raw_stack_tree[i].append(j)
        row_total = sum(_matrix[i])
        for j in range(len(lines)):
            _matrix[i][j] *= row_total
    stack_matrix = matrix(_matrix)
    # convert to a minimum spanning tree such that each line is just in one parent
    assert stack_matrix
    stack_matrix = minimum_spanning_tree(stack_matrix, overwrite=True).toarray()
    stack_tree = defaultdict(list)
    root_nodes = [i for i in range(len(lines))]
    for i, row in enumerate(stack_matrix):
        for j, cell in enumerate(row):
            if not cell:
                continue
            stack_tree[i].append(j)
            if (
                i in root_nodes and j in root_nodes
            ):  # remove j from the root nodes, as it obviously had a parent
                root_nodes.remove(j)
    return stack_tree, root_nodes


def stack_lines(lines):
    stack_tree, root_nodes = make_stack_tree(lines)
    print(f"stack tree {stack_tree} ")
    combined_lines = []
    labels = []
    print(f"root_nodes {root_nodes}")
    while root_nodes:
        current_node = root_nodes.pop()
        if current_node in labels:
            logging.warning(
                f"went over the same node more than once! {current_node}, {labels}"
            )
            continue
        child_nodes = stack_tree[current_node]
        to_inspect = None  # 47
        parent_orientation = find_orientation(lines[current_node])
        combined_line = lines[current_node]
        for child_node in child_nodes:
            child_orientation = find_orientation(lines[child_node])

            if child_orientation == parent_orientation:
                lines[child_node] = lines[child_node].reversed()
            for segment in lines[child_node]:
                combined_line.append(segment)
            root_nodes.append(child_node)
        combined_lines.append(combined_line)

        labels.append(current_node)
    return combined_lines, labels


def paths_are_degenerate(path1, path2):
    if abs(path1.point(0.5) - path2.point(0.5)) > TOLERANCE:
        return False
    if (
        abs(path1.point(0) - path1.point(0)) < TOLERANCE
        and abs(path1.point(1) - path1.point(1)) < TOLERANCE
    ):
        return True
    if (
        abs(path1.point(0) - path1.point(1)) < TOLERANCE
        and abs(path1.point(1) - path1.point(0)) < TOLERANCE
    ):
        return True
    return False


def intersect_over_all(line, path, exit_early=False):
    all_intersections = []
    for i, segment in enumerate(path):
        try:
            if paths_are_degenerate(line, segment):
                continue
            current_intersections = line.intersect(segment)
            if current_intersections and exit_early:
                return current_intersections
            all_intersections += [(t1, t2, i) for (t1, t2) in current_intersections]
        except AssertionError as e:
            # TODO: figure out whether the line being identical counts as an intersection
            all_intersections += [(0, 0, i), (1, 1, i)]
    return all_intersections


def is_inside(container, point, debug=False, tolerance=0.2):
    xmin, xmax, ymin, ymax = container.bbox()
    # if the point is on the edge of the bbox, assume it's outside
    diffs = [
        abs(point.real - xmin),
        abs(point.real - xmax),
        abs(point.imag - ymin),
        abs(point.imag - ymax),
    ]
    diffs = [diff < tolerance for diff in diffs]
    if any(diffs):
        if debug:
            print("point is on bbox")
        return False
    if point.real + tolerance < xmin:
        if debug:
            print(f"to the left of the bbox {point.real} and bbox xmin {xmin}")
        return False
    if point.real - tolerance > xmax:
        if debug:
            print("to the right of the bbox")
        return False
    if point.imag + tolerance < ymin:
        if debug:
            print("below the bbox")
        return False
    if point.imag - tolerance > ymax:
        if debug:
            print("above the bbox")
        return False

    # make sure the lines are actually out of the bbox by adding a shrinking/enlarging factor
    span_line_upper = Line(0.9 * (xmin + ymin * 1j), point)
    span_line_lower = Line(point, 1.1 * (xmax + ymax * 1j))
    upper_intersections = intersect_over_all(span_line_upper, container)
    lower_intersections = intersect_over_all(span_line_lower, container)
    if debug:
        print(f"is_inside debug {upper_intersections} {lower_intersections}")
    return len(upper_intersections) % 2 or len(lower_intersections) % 2


def combine_segments(segments):
    # svgpathtools does not dump "Path" segments when generating d strings
    output_path = Path()
    previous_end = None
    for segment in segments:
        if not isinstance(segment, Path):
            if previous_end:
                segment.start = previous_end
            output_path.insert(len(output_path._segments), segment)
            previous_end = segment.end
        else:
            for path_segment in segment._segments:
                if previous_end:
                    path_segment.start = previous_end
                output_path.insert(len(output_path._segments), path_segment)
                previous_end = path_segment.end
    return output_path


def svgpath_to_shapely_polygon(in_path):
    from shapely.geometry import Polygon

    ring_coords = []
    _coords = []
    last_point = None
    for segment in in_path:
        if last_point:
            if segment.start == last_point:
                _coords.append((segment.end.real, segment.end.imag))
            else:  # assume it's an inner segment
                ring_coords.append(_coords)
                _coords = [
                    [segment.start.real, segment.start.imag],
                    [segment.end.real, segment.end.imag],
                ]
        last_point = segment.end
    ring_coords.append(_coords)
    if len(ring_coords) == 1:
        return Polygon(ring_coords[0])
    elif len(ring_coords) >= 2:
        try:
            return Polygon(ring_coords[0], ring_coords[1:])
        except ValueError as err:
            raise ValueError(f"{err} {ring_coords}")
    else:
        raise ValueError(f"unable to convert {in_path.d()} to Polygon")


def format_complex(input_object):

    if input_object is None:
        return None
    elif (
        isinstance(input_object, list)
        or isinstance(input_object, KeysView)
        or isinstance(input_object, set)
    ):
        return ",".join(
            [f"{num}" if isinstance(num, int) else f"{num:.1f}" for num in input_object]
        )
    elif isinstance(input_object, int):
        return f"{input_object}"
    else:
        return f"{input_object:.1f}"


def limit_atan(imag, real):
    if real == 0:
        return 3.14159 / 2.0
    return atan(imag / real)


def get_clockwise(last_point, curr_point, branches, counter=False):
    # if the previous point was on the inside, pick the clockwise outside location
    unit_root = last_point - curr_point
    units = [branch - curr_point for branch in branches]
    angle_root = limit_atan(unit_root.imag, unit_root.real)
    angles = [
        (limit_atan(unit.imag, unit.real) - angle_root) % 2 * 3.14159 for unit in units
    ]
    compare_func = min if counter else max
    return (
        angles.index(compare_func(angles)),
        branches[angles.index(compare_func(angles))],
    )
