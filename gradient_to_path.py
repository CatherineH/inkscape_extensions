#!/usr/bin/env python3
import random
from copy import deepcopy
from enum import Enum

import inkex
from collections import defaultdict
from svgpathtools.path import Path

from common_utils import (
    pattern_vector_to_d,
    get_fill_id,
    BaseFillExtension,
)

MAX_RETRIES = 10


class FillShape(Enum):
    lines = "Lines"
    circles = "Circles"


class GradientToPath(BaseFillExtension):
    def __init__(self):
        BaseFillExtension.__init__(self, self.recursive_replace_gradient)
        self._gradients = {}
        self.spacing = 1
        self._style_sheets = {}
        self.circles = False
        self.stops_dict = {}
        self.offsets = []

    def add_arguments(self, pars):
        pars.add_argument("--debug", type=str, default="false", help="debug shapes")
        pars.add_argument(
            "--circles", type=str, default="false", help="use circles instead of lines"
        )
        pars.add_argument(
            "--spacing", type=float, default=1, help="spacing between circles"
        )

    def get_all_gradients(self, node):
        if node.tag.find("Gradient") >= 0:
            _id = node.attrib.get("id")
            # TODO: find if there's some sort of auto parsing here
            self._gradients[_id] = inkex.LinearGradient(
                attrib={
                    "x1": node.get("x1", "0"),
                    "y1": node.get("y1", "0"),
                    "x2": node.get("x2", "1"),
                    "y2": node.get("y2", "1"),
                    "gradientUnits": node.get("gradientUnits", "objectBoundingBox"),
                }
            )
            if node.get("gradientTransform"):
                self._gradients[_id].gradientTransform = inkex.transforms.Transform(
                    node.get("gradientTransform")
                )
            for stop in node.stops:
                _style = inkex.Style(
                    {
                        "stop-color": inkex.Color(stop.attrib.get("stop-color")),
                        "stop-opacity": float(stop.attrib.get("stop-opacity", 1.0)),
                    }
                )
                self._gradients[_id].add(
                    inkex.Stop().update(
                        offset=inkex.utils.parse_percent(
                            stop.attrib.get("offset", "0")
                        ),
                        style=_style,
                    )
                )
        elif node.tag.find("style") >= 0:
            for _stylesheet in node.stylesheet():
                for elem in self.svg.xpath(_stylesheet.to_xpath()):
                    if elem.get("id"):
                        self._style_sheets[elem.get("id")] = _stylesheet
                    elif elem.TAG == "stop":
                        for _key in _stylesheet.keys():
                            elem.set(_key, _stylesheet[_key])
        else:
            for child in node.getchildren():
                self.get_all_gradients(child)

    def recursive_replace_gradient(self, node):
        self.get_all_gradients(self.document.getroot())
        if self.options.circles == "true":
            self.circles = True
        self.spacing = self.options.spacing
        if node.tag in [inkex.addNS("rect", "svg"), inkex.addNS("path", "svg")]:
            node_id = node.get("id")
            pattern_id = None
            if node_id in self._style_sheets:
                css_style_sheet = self._style_sheets[node_id]
                if "fill" in css_style_sheet:
                    pattern_id = css_style_sheet["fill"].replace("url(#", "")[:-1]
            if not pattern_id:
                pattern_id = get_fill_id(node)

            if "d" in node.attrib:
                self.container_path = node.attrib.get("d")
            else:
                self.container_path = pattern_vector_to_d(node)
                if not self.container_path:
                    return
            # needed elements:
            if pattern_id not in self._gradients:
                inkex.utils.errormsg(
                    f"gradient {pattern_id} not found in {self._gradients.keys()}"
                )
                return
            # is the gradient linear?
            if self._gradients[pattern_id].tag != inkex.addNS("linearGradient", "svg"):
                inkex.utils.errormsg(
                    f"gradient type {node.TAG} {node.get('id')} not yet supported"
                )
                return
            # vector direction
            self.gradient = self._gradients[pattern_id]

            # line_colors = self.interleave_block(offsets, stops_dict, num_lines)
            _fill_type = FillShape.circles if self.circles else FillShape.lines
            self.generate(node, _fill_type)
        elif node.tag in [
            inkex.addNS("text", "svg"),
            inkex.addNS("image", "svg"),
            inkex.addNS("use", "svg"),
        ]:
            # this was copied from apply transforms - TODO: autoconvert to path using pylivarot
            inkex.utils.errormsg(
                "Shape %s (%s) not yet supported, try Object to path first"
                % (node.TAG, node.get("id"))
            )
        else:
            for child in node.getchildren():
                self.recursive_pattern_to_path(child)

    def interleave_blocks(self, num_lines):
        line_colors = []
        start_offset = self.offsets.pop(0)
        if start_offset > 0:
            for _ in range(0, num_lines * start_offset):
                line_colors.append(self.stops_dict[start_offset])
        end_offset = None
        bin_size = None
        for i in range(int(num_lines ** 0.5), 1, -1):
            if num_lines % i == 0:
                bin_size = i
                break
        num_bins = int(num_lines / bin_size)
        if not bin_size:
            inkex.utils.errormsg(f"{num_lines} is prime!")
        while self.offsets:
            end_offset = self.offsets.pop(0)

            field_fraction = end_offset - start_offset
            field_bins = num_bins * field_fraction
            print(
                f"evaluating offset {start_offset} {end_offset} {field_bins} {field_fraction} {self.stops_dict[start_offset]} {self.stops_dict[end_offset]}"
            )

            for i in range(int(field_bins)):
                slope = -bin_size / field_bins
                intercept = bin_size
                y_value = slope * i + intercept
                for j in range(bin_size):
                    if j > y_value:
                        line_colors.append(self.stops_dict[end_offset])
                    else:
                        line_colors.append(self.stops_dict[start_offset])
            start_offset = end_offset
        if end_offset < 1:
            for _ in range(0, num_lines - end_offset * num_lines):
                line_colors.append(self.stops_dict[end_offset])
        assert (
            len(line_colors) == num_lines
        ), f"{len(line_colors)} is not {num_lines}, bin_size {bin_size} num_bins {num_bins}"
        return line_colors

    def generate(self, node, fill_type=FillShape.circles):
        bbox = node.bounding_box()
        container_path = Path(pattern_vector_to_d(node))
        print(bbox, container_path.bbox(), container_path.d())
        c_bbox = container_path.bbox()
        bbox = inkex.transforms.BoundingBox(
            x=(c_bbox[0], c_bbox[1]), y=(c_bbox[2], c_bbox[3])
        )
        current_retry = 0
        transf = inkex.transforms.Transform(node.get("transform", None))

        _locations = []
        paths = defaultdict(str)
        curr_i = 0
        stops_used = []
        while current_retry < MAX_RETRIES:
            curr_i += 1
            if curr_i > 50000:
                raise ValueError("something is going wrong, unable to fill shape")
            _x = random.random() * (bbox.right - bbox.left) + bbox.left
            _y = random.random() * (bbox.bottom - bbox.top) + bbox.top
            # first confirm that the _x, _y is inside
            is_inside = inkex.boolean_operations.segment_is_inside(
                container_path, _x + _y * 1j
            )

            if not is_inside:
                continue
            too_close = False
            for _location in _locations:
                if fill_type == FillShape.circles:
                    distance = (
                        (_location[0] - _x) ** 2 + (_location[1] - _y) ** 2
                    ) ** 0.5
                elif fill_type == FillShape.lines:
                    distance = (_location[0] - _x) ** 2
                if (
                    distance < self.spacing
                ):  # the current circle is too close to an existing location
                    current_retry += 1
                    too_close = True
                    break
            if too_close:
                continue
            current_retry = 0
            _color = deepcopy(
                self.sample_color(
                    bbox,
                    inkex.transforms.Vector2d(_x, _y),
                    debug=self.options.debug == "true",
                )
            )
            _locations.append((_x, _y, _color))
            if self.options.debug == "true":
                print(f"adding circle {_x} {_y} {_color}")

            if fill_type == FillShape.circles:
                _node = inkex.elements.Circle.new(
                    center=inkex.transforms.Vector2d(_x, _y), radius=self.spacing / 2
                )
            else:
                _node = inkex.elements.Line.new(
                    start=inkex.transforms.Vector2d(_x, bbox.top),
                    end=inkex.transforms.Vector2d(_x, bbox.bottom),
                )
            if _color not in stops_used:
                stops_used.append(_color)
            stop_i = stops_used.index(_color)
            paths[stop_i] += " " + str(_node.get_path())

        for i, stop_i in enumerate(list(paths.keys())):
            _color = stops_used[stop_i]
            print(i, _color, paths[stop_i])
            _node = inkex.elements.PathElement()
            _node.set("d", paths[stop_i])
            _node.set(
                "style",
                f"fill:none;stroke:{_color.get('stop-color')};stroke-opacity:{_color.get('stop-opacity', '1')};stroke-width:{self.spacing / 2}",
            )
            if transf:
                _node.set("transform", node.get("transform"))
            _node.set("id", f"{node.get('id')}-{stop_i}")
            self.get_parent(node).insert(-1, _node)

    def sample_color(self, bbox, point, debug=False):
        stops = self.gradient.sample_color(bbox, point, debug=debug)
        _random = random.random()
        if debug:
            print(stops, _random)

        if len(stops) == 1:
            return stops[0][1].style
        if _random >= stops[0][0]:
            return stops[0][1].style
        else:
            return stops[1][1].style


if __name__ == "__main__":
    GradientToPath().run()
