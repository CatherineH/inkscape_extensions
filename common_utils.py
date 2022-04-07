from typing import KeysView
import inkex
from svgpathtools.svg_to_paths import rect2pathd, ellipse2pathd
from svgpathtools import Path, Line
from numpy import matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from math import atan
import subprocess


class BaseFillExtension(inkex.EffectExtension):
    def __init__(self):
        inkex.EffectExtension.__init__(self)

    def get_parent(self, node):
        parent = node.getparent()
        if parent is None:
            parent = self.document.getroot()
        return parent


def bounds_rect(pv):
    from pylivarot import py2geom

    bbox = pv.boundsExact()
    return py2geom.Rect(bbox[py2geom.Dim2.X], bbox[py2geom.Dim2.Y])


def transform_path_vector(pv, affine_t):
    for _path in pv:
        for _curve in _path:
            _curve.transform(affine_t)
    return pv


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
    filename = f"output/debug_{name}.svg"
    effect.save(open(filename, "wb"))
    subprocess.run(["inkview", filename])


def make_stack_tree(lines):
    """

    Parameters
    ----------
    lines: a list of svgpathtool Paths

    Returns
    -------
    a tuple - first element is a graph, second element is a list of root notes
    """
    def pairwise_comparison(path1, path2):
        # returns True if path1 is inside path2
        xmin1, xmax1, ymin1, ymax1 = path1.bbox()
        xmin2, xmax2, ymin2, ymax2 = path2.bbox()
        bbox_inside = xmin1 <= xmin2 <= xmax2 <= xmax1 and ymin1 <= ymin2 <= ymax2 <= ymax1
        if not bbox_inside:
            return False
        # if the bboxes overlap, do something more complex
        # if any of the points in path2 is inside path1, the line must be inside
        for segment in path2:
            segment_inside = is_inside(path1, segment.start)
            if segment_inside:
                return True
        return False

    stack_matrix = matrix([[pairwise_comparison(lines[i], lines[j]) for i in range(len(lines))] for j in range(len(lines))])
    # convert to a minimum spanning tree such that each line is just in one parent
    stack_matrix = minimum_spanning_tree(stack_matrix, overwrite=True).toarray()
    stack_tree = defaultdict(list)
    root_nodes = [i for i in range(len(lines))]
    for i, row in enumerate(stack_matrix):
        for j, cell in enumerate(row):
            if not cell:
                continue
            stack_tree[i].append(j)
            if j in root_nodes:
                root_nodes.remove(j)
    return stack_tree, root_nodes


def intersect_over_all(line, path):
    all_intersections = []
    for i, segment in enumerate(path):
        current_intersections = line.intersect(segment)
        all_intersections += [(t1, t2, i) for (t1, t2) in current_intersections]
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
    if point.real < xmin:
        if debug:
            print("to the left of the bbox")
        return False
    if point.real > xmax:
        if debug:
            print("to the right of the bbox")
        return False
    if point.imag < ymin:
        if debug:
            print("below the bbox")
        return False
    if point.imag > ymax:
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
    for segment in segments:
        if not isinstance(segment, Path):
            output_path.insert(len(output_path._segments), segment)
        else:
            for path_segment in segment._segments:
                output_path.insert(len(output_path._segments), path_segment)
    return output_path


def format_complex(input_object):

    if input_object is None:
        return None
    elif (
        isinstance(input_object, list)
        or isinstance(input_object, KeysView)
        or isinstance(input_object, set)
    ):
        return ",".join([f"{num}" if isinstance(num, int) else f"{num:.1f}" for num in input_object ])
    elif isinstance(input_object, int):
        return f"{input_object}"
    else:
        return f"{input_object:.1f}"


def limit_atan(imag, real):
    if real == 0:
        return 3.14159 / 2.0
    return atan(imag / real)


def get_clockwise(last_point, curr_point, branches):
    # if the previous point was on the inside, pick the clockwise outside location
    unit_root = last_point - curr_point
    units = [branch - curr_point for branch in branches]
    angle_root = limit_atan(unit_root.imag, unit_root.real)
    angles = [
        (limit_atan(unit.imag, unit.real) - angle_root) % 2 * 3.14159 for unit in units
    ]
    return branches[angles.index(max(angles))]
