#!/usr/bin/env python3
import random
from copy import deepcopy

import inkex
from math import atan2, sin, cos
from collections import defaultdict
from svgpathtools.path import Path

from common_utils import (
    bounds_rect,
    transform_path_vector,
    pattern_vector_to_d,
    get_fill_id,
    BaseFillExtension,
)


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
        pars.add_argument("--circles", type=str, default="false", help="use circles instead of lines")
        pars.add_argument("--spacing", type=float, default=1, help="spacing between circles")

    def get_all_gradients(self, node):
        if node.tag.find("Gradient") >= 0:
            _id = node.attrib.get("id")
            # TODO: find if there's some sort of auto parsing here
            self._gradients[_id] = inkex.LinearGradient(attrib={"x1": node.get("x1", "0"), "y1": node.get("y1", "0"),
                                                                "x2": node.get("x2", "1"), "y2": node.get("y2", "1"),
                                                                "gradientUnits": node.get("gradientUnits", "objectBoundingBox")})
            if node.get("gradientTransform"):
                self._gradients[_id].gradientTransform = inkex.transforms.Transform(node.get("gradientTransform"))
            for stop in node.stops:
                _style = inkex.Style({"stop-color": inkex.Color(stop.attrib.get("stop-color")), "stop-opacity": float(stop.attrib.get("stop-opacity", 1.0))})
                self._gradients[_id].add(inkex.Stop().update(offset=stop.attrib.get("offset", "0"), style=_style))
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

            self.offsets = sorted(self.stops_dict.keys())
            # line_colors = self.interleave_block(offsets, stops_dict, num_lines)
            if not self.circles:
                self.generate_lines(node)
            else:
                self.generate_circles(node)
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

    def random_interpolate(self, offsets, num_lines):
        line_colors = []
        start_offset = offsets.pop(0)
        if start_offset > 0:
            for _ in range(0, num_lines * start_offset):
                line_colors.append(self.stops_dict[start_offset])
        while self.offsets:
            end_offset = self.offsets.pop(0)
            field_fraction = end_offset - start_offset
            num_lines_in_fraction = field_fraction * num_lines
            print(f"num_lines_in_fraction {num_lines_in_fraction}")
            for i in range(int(num_lines_in_fraction)):
                threshold = i / num_lines_in_fraction
                if random.random() < threshold:
                    line_colors.append(self.stops_dict[end_offset])
                else:
                    line_colors.append(self.stops_dict[start_offset])
            start_offset = end_offset

        if end_offset < 1:
            for _ in range(0, num_lines - end_offset * num_lines):
                line_colors.append(self.stops_dict[end_offset])
        assert len(line_colors) == num_lines, f"{len(line_colors)} is not {num_lines}"
        return line_colors

    def interleave_blocks(self, num_lines):
        line_colors = []
        start_offset = self.offsets.pop(0)
        if start_offset > 0:
            for _ in range(0, num_lines * start_offset):
                line_colors.append(self.stops_dict[start_offset])
        end_offset = None
        bin_size = None
        for i in range(int(num_lines**0.5), 1, -1):
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

    def generate_lines(self, node):
        container_pv = py2geom.parse_svg_path(self.container_path)
        _affine = py2geom.Affine()
        _affine *= py2geom.Rotate(_angle)
        container_pv = transform_path_vector(container_pv, _affine)
        container_bbox = bounds_rect(container_pv)
        num_lines = container_bbox.width() / self.spacing

        line_colors = self.random_interpolate(num_lines)
        path_builders = {
            f"{line_color[0]}_{line_color[1]}": py2geom.PathBuilder()
            for line_color in self.stops_dict.values()
        }
        for i in range(int(num_lines)):
            line_color = line_colors.pop(0)
            if line_color[1] == 0:  # don't bother with 0 opacity values
                continue
            path_builders[f"{line_color[0]}_{line_color[1]}"].moveTo(
                py2geom.Point(
                    (i - 0.5) * self.spacing + container_bbox.left(),
                    container_bbox.top(),
                )
            )
            path_builders[f"{line_color[0]}_{line_color[1]}"].lineTo(
                py2geom.Point(
                    (i - 0.5) * self.spacing + container_bbox.left(),
                    container_bbox.bottom(),
                )
            )
            path_builders[f"{line_color[0]}_{line_color[1]}"].lineTo(
                py2geom.Point(
                    (i + 0.5) * self.spacing + container_bbox.left(),
                    container_bbox.bottom(),
                )
            )
            path_builders[f"{line_color[0]}_{line_color[1]}"].lineTo(
                py2geom.Point(
                    (i + 0.5) * self.spacing + container_bbox.left(),
                    container_bbox.top(),
                )
            )
            path_builders[f"{line_color[0]}_{line_color[1]}"].lineTo(
                py2geom.Point(
                    (i - 0.5) * self.spacing + container_bbox.left(),
                    container_bbox.top(),
                )
            )
            path_builders[f"{line_color[0]}_{line_color[1]}"].closePath()
        if not path_builders:
            inkex.utils.errormsg("path_builders is empty")
            return

        for line_color in path_builders:

            color = line_color.split("_")[0]
            opacity = line_color.split("_")[1]
            stroke_opacity = ""
            if opacity:
                stroke_opacity = f"stroke-opacity:{opacity};"
            path_builders[line_color].flush()
            all_gradient_path = path_builders[line_color].peek()
            # gradient_path = get_outline_offset(gradient_path, self.spacing/2.0)
            result = py2geom.PathVector()
            for gradient_path in all_gradient_path:
                gradient_path_pv = py2geom.PathVector()
                gradient_path_pv.push_back(gradient_path)
                result_piece = sp_pathvector_boolop(
                    gradient_path_pv,
                    container_pv,
                    bool_op.bool_op_inters,
                    FillRule.fill_oddEven,
                    FillRule.fill_oddEven,
                    skip_conversion=True,
                )
                for result_piece_path in result_piece:
                    result.push_back(result_piece_path)
            _affine = py2geom.Affine()
            _affine *= py2geom.Rotate(-_angle)
            result = transform_path_vector(result, _affine)
            all_gradient_path = transform_path_vector(all_gradient_path, _affine)
            gradient_path_id = f"gradient-{line_color}"
            gradient_path_style = (
                f"fill:none;stroke:{color};{stroke_opacity}stroke-width:{self.spacing}"
            )
            if self.options.debug == "true":
                gradient_debug_path_d = py2geom.write_svg_path(all_gradient_path)
                gradient_debug_node = inkex.elements.PathElement()
                gradient_debug_node.set_path(gradient_debug_path_d)
                gradient_debug_node.set("id", f"{gradient_path_id}-debug")
                gradient_debug_node.set("style", gradient_path_style)
                self.get_parent(node).insert(0, gradient_debug_node)

            gradient_color_d = py2geom.write_svg_path(result)
            gradient_node = inkex.elements.PathElement()
            gradient_node.set_path(gradient_color_d)

            gradient_node.set("style", gradient_path_style)
            gradient_node.set("id", gradient_path_id)
            self.get_parent(node).insert(0, gradient_node)
        if self.options.debug == "true":
            container_path_d = py2geom.write_svg_path(container_pv)
            container_node = inkex.elements.PathElement()
            container_node.set_path(container_path_d)
            container_node.set("id", "container_path")
            container_node.set(
                "style", f"fill:none;stroke:green;stroke-width:{self.spacing}"
            )
            self.get_parent(node).insert(0, container_node)

    def generate_circles(self, node):
        bbox = node.bounding_box()
        container_path = Path(pattern_vector_to_d(node))
        print(bbox, container_path.bbox(), container_path.d())
        c_bbox = container_path.bbox()
        bbox = inkex.transforms.BoundingBox(x=(c_bbox[0], c_bbox[1]), y=(c_bbox[2], c_bbox[3]))
        MAX_RETRIES = 10
        current_retry = 0
        transf = inkex.transforms.Transform(node.get("transform", None))

        circle_locations = []
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
            is_inside = inkex.boolean_operations.segment_is_inside(container_path, _x+_y*1j)
            if not is_inside:
                continue
            too_close = False
            for circle_location in circle_locations:
                distance = (
                    (circle_location[0] - _x) ** 2 + (circle_location[1] - _y) ** 2
                ) ** 0.5
                if (
                    distance < self.spacing
                ):  # the current circle is too close to an existing location
                    current_retry += 1
                    too_close = True
                    break
            if too_close:
                continue
            current_retry = 0
            _color = deepcopy(self.sample_color(bbox, inkex.transforms.Vector2d(_x, _y) , debug=self.options.debug == "true"))
            circle_locations.append((_x, _y, _color))
            if self.options.debug == "true":
                print(f"adding circle {_x} {_y} {_color}")

            _node = inkex.elements.Circle.new(
                center=inkex.transforms.Vector2d(_x, _y), radius=self.spacing/2
            )
            if _color not in stops_used:
                stops_used.append(_color)
            stop_i = stops_used.index(_color)
            paths[stop_i] += " "+_node.get_path()

        for i, stop_i in enumerate(list(paths.keys())):
            _color = stops_used[stop_i]
            print(i, _color, paths[stop_i])
            _node = inkex.elements.PathElement()
            _node.set("d", paths[stop_i])
            _node.set("style",
                f"fill:none;stroke:{_color.get('stop-color')};stroke-opacity:{_color.get('stop-opacity', '1')};stroke-width:{self.spacing/2}",
            )
            if transf:
                _node.set("transform", node.get("transform"))
            _node.set("id", f"points{i}")
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
