#!/usr/bin/env python3
import inkex

from common_utils import (
    transform_path_vector,
    pattern_vector_to_d,
    get_fill_id,
    BaseFillExtension,
)


class PatternToPath(BaseFillExtension):
    def __init__(self):
        BaseFillExtension.__init__(self, self.recursive_pattern_to_path)
        self._patterns = {}
        self.current_path = 1
        self.current_id = ""
        self.current_pattern_part = 1
        self._debug_pattern_list = []
        self.wrapping_bboxes = inkex.Path()

    def add_arguments(self, pars):
        pars.add_argument(
            "--remove", type=str, default="false", help="Remove the existing path"
        )
        pars.add_argument(
            "--boundaries",
            type=str,
            default="false",
            help="show the wrapping paper boundaries",
        )

    def effect(self):
        self.get_all_patterns(self.document.getroot())
        if self.svg.selected:
            for _, shape in self.svg.selected.items():
                self.recursive_pattern_to_path(shape)
                self.current_path += 1
        else:
            self.recursive_pattern_to_path(self.document.getroot())

    def get_all_patterns(self, node):
        if node.tag.find("pattern") >= 0:
            _id = node.attrib.get("id")
            self._patterns[_id] = node
        else:
            for child in node.getchildren():
                self.get_all_patterns(child)

    def generate_wrapping_paper(self, container_pv, repeating_bbox, repeating_pattern):
        # pattern repeats is a path that takes the repeating_pattern and copies it up and down over
        # the entire container path, like wrapping paper
        pattern_repeats = inkex.Path()

        for i, _path in enumerate(container_pv):

            _path_pv = inkex.Path()
            _path_pv.append(_path)
            container_bbox = _path_pv.bounding_box()
            assert container_bbox, f"container is empty: {_path_pv=} {_path=}"
            # print("path ", i, py2geom.write_svg_path(_path_pv))
            # need to get the bounding box from the bounding box
            pattern_x = container_bbox.left()
            pattern_y = container_bbox.top()

            # for debug only
            num_repeats = 0
            self.wrapping_bboxes.append(
                inkex.paths.move(container_bbox.left(), container_bbox.top())
            )
            self.wrapping_bboxes.append(
                inkex.paths.line(container_bbox.right(), container_bbox.top())
            )
            self.wrapping_bboxes.append(
                inkex.paths.line(container_bbox.right(), container_bbox.bottom())
            )
            self.wrapping_bboxes.append(
                inkex.paths.line(container_bbox.left(), container_bbox.bottom())
            )
            self.wrapping_bboxes.append(
                inkex.paths.line(container_bbox.left(), container_bbox.top())
            )
            num_x_translations = 0
            # set to the initial x/y
            loc = repeating_pattern.bounding_box()

            start_x = pattern_x - loc.left()
            start_y = pattern_y - loc.bottom()
            _affine = inkex.transforms.Transform()
            _affine.add_translate((start_x, start_y))
            repeating_pattern.transform(_affine, inplace=True)
            loc2 = repeating_pattern.bounding_box()
            assert (
                loc.left() == 0 and pattern_x == 0
            ) or loc.left() != loc2.left(), (
                f"pattern does not seem to have moved: {loc.left()} {loc2.left()}"
            )
            print(
                f"boundaries {start_x} {start_y} {pattern_x} {pattern_y} {loc.left()} {loc.top()} {loc.bottom()} {loc.right()}"
            )
            while pattern_x <= container_bbox.left() + container_bbox.width():
                pattern_y = container_bbox.top()
                num_y_translations = 0

                while pattern_y <= container_bbox.top() + container_bbox.height():
                    _affine = inkex.transforms.Transform()
                    _affine.add_translate((0, repeating_bbox.height()))
                    if len(pattern_repeats) == 0:
                        pattern_repeats = repeating_pattern

                    # TODO: analyze whether skipping conversion speeds this up when there are no arches... or convert it first
                    # for some reason, bezier curves with unions merge together in weird ways, but arcs don't, maybe it's the zone close?
                    # _tmp_pv = sp_pathvector_boolop(pathv_to_linear_and_cubic_beziers(repeating_pattern), pathv_to_linear_and_cubic_beziers(pattern_repeats), bool_op.bool_op_union, FillRule.fill_oddEven, FillRule.fill_oddEven, skip_conversion=True)
                    """
                    converted_repeating_pattern = pathv_to_linear_and_cubic_beziers(repeating_pattern)
                    
                    if not converted_repeating_pattern == repeating_pattern:
                        for i,path in enumerate(converted_repeating_pattern):
                            if path!=repeating_pattern[i]:
                                if path.closed() != repeating_pattern[i].closed():
                                    print(f"curves aren't both closed {path.closed()} {repeating_pattern[i].closed()}")
                                else:
                                    for j,curve in enumerate(path):
                                        print(f"same? {i} {curve==repeating_pattern[i][j]}")
                    """
                    _tmp_pv = repeating_pattern.intersect(pattern_repeats)
                    # the new path should always be more complex that the existing pattern
                    if len(_tmp_pv) < len(pattern_repeats):
                        inkex.utils.errormsg(
                            f"{self.current_id} curve counts {len(_tmp_pv)} {len(pattern_repeats)} \
                        inputs were: repeating_pattern: '{repeating_pattern.d()}' pattern_repeats: '{pattern_repeats.d()}'"
                        )
                    pattern_repeats = _tmp_pv
                    num_y_translations += 1
                    num_repeats += 1
                    # for debug only
                    # if num_repeats > 1:
                    #    return pattern_repeats
                    repeating_pattern.transform(_affine, inplace=True)
                    pattern_y += repeating_bbox.height()
                _affine = inkex.transforms.Transform()
                _affine.add_translate(
                    repeating_bbox.width(),
                    -num_y_translations * repeating_bbox.height(),
                )
                num_x_translations += 1
                repeating_pattern.apply_transform(_affine, inplace=True)
                pattern_x += repeating_bbox.width()
        return pattern_repeats

    def recursive_find_pattern(self, pattern_id, transf=None):
        if pattern_id not in self._patterns:
            inkex.errormsg(f"failed to find {pattern_id}, {self._debug_pattern_list}")
        self._debug_pattern_list.append(pattern_id)
        pattern = self._patterns[pattern_id]
        # SVG seems to only respect the first patternTransform
        if not transf:
            transf = inkex.transforms.Transform(pattern.get("patternTransform", None))
        href_tag = "{http://www.w3.org/1999/xlink}href"
        if href_tag in pattern.attrib:
            return self.recursive_find_pattern(pattern.attrib[href_tag][1:], transf)
        if "width" not in pattern.attrib or "height" not in pattern.attrib:
            raise KeyError("missing attributes in ", pattern.attrib)
        return pattern, transf

    def generate_pattern_path(
        self, node, repeating_box, repeating_pattern, pattern_style
    ):
        # repeating_pattern should always have something
        if len(repeating_pattern) == 0:
            inkex.utils.errormsg(
                f"{node.get('id')} pattern piece {self.current_path} is empty"
            )
            return

        pattern_repeats = self.generate_wrapping_paper(
            node, repeating_box, repeating_pattern
        )
        container_intersection = inkex.Path(pattern_vector_to_d(node)).intersect(
            pattern_repeats
        )
        parent = self.get_parent(node)
        unknown_name = f"unknown-{self.current_path}"
        pattern_id = (
            f'pattern-path-{node.get("id", unknown_name)}{self.current_pattern_part}'
        )
        if self.options.boundaries == "true":
            node_wrapping_paper = inkex.elements.PathElement()
            node_wrapping_paper.set_path(pattern_repeats.d())
            node_wrapping_paper.set("style", "fill:none;stroke:blue;stroke-width:2")
            node_wrapping_paper.set("id", f"container-{pattern_id}")
            parent.insert(0, node_wrapping_paper)
        node_pattern = inkex.elements.PathElement()
        node_pattern.set_path(container_intersection.d())
        node_pattern.set("style", pattern_style)
        node_pattern.set("id", pattern_id)
        if self.options.remove == "true":
            node.delete()
        parent.insert(0, node_pattern)

    def recursive_pattern_to_path(self, node):
        if node.tag in [inkex.addNS("rect", "svg"), inkex.addNS("path", "svg")]:
            self.current_id = node.attrib.get("id")
            pattern_id = get_fill_id(node)
            if not pattern_id:
                return
            container_path = node
            # reset the debug information
            self._debug_pattern_list = []
            pattern, pattern_transform = self.recursive_find_pattern(pattern_id)
            pattern_width = float(pattern.attrib["width"])
            pattern_height = float(pattern.attrib["height"])
            d = f"M 0,0 L {pattern_width},0 L {pattern_width},{pattern_height} L 0,{pattern_height} L 0,0 z"

            repeating_box = inkex.Path(d)
            repeating_box.transform(pattern_transform, inplace=True)
            repeating_patterns = []
            pattern_styles = []
            for pattern_vector in pattern.getchildren():
                if "d" not in pattern_vector.attrib:
                    # TODO: autoconvert using pylivarot
                    d_string = pattern_vector_to_d(pattern_vector)
                else:
                    d_string = pattern_vector.attrib["d"]

                pattern_vector_pv = inkex.Path(d_string)
                pattern_vector_pv.transform(pattern_transform, inplace=True)
                pattern_intersection = repeating_box.intersection(pattern_vector_pv)
                if len(pattern_intersection) == 0:
                    inkex.utils.errormsg(
                        f"pattern_intersection is empty, skipping, repeating_box is {repeating_box}, \
                    pattern vector {pattern_vector.attrib['id']} is {d_string}"
                    )
                    continue
                # pattern_intersection = transform_path_vector(pattern_intersection, affine_pattern_transform)
                repeating_patterns.append(pattern_intersection)
                # repeating_patterns.append(pattern_intersection)
                pattern_style = dict(
                    inkex.styles.Style().parse_str(pattern_vector.attrib.get("style"))
                )
                if "stroke-width" in pattern_style:
                    # TODO: scale by the transform
                    pattern_style["stroke-width"] = float(pattern_style["stroke-width"])
                style_attribs = ["fill", "fill-opacity", "stroke", "stroke-width"]
                for attrib in style_attribs:
                    if attrib in pattern_vector.attrib:
                        pattern_style[attrib] = pattern_vector.attrib[attrib]
                pattern_styles.append(pattern_style)
            for i in range(len(pattern_styles) - 1, -1, -1):
                self.current_pattern_part = i + 1
                self.generate_pattern_path(
                    node,
                    repeating_box,
                    repeating_patterns[i],
                    pattern_styles[i],
                )
            # for debug
            if self.options.boundaries == "true":
                node_container = inkex.elements.PathElement()
                node_container.set_path(self.wrapping_bboxes.d())
                node_container.set("style", "fill:none;stroke:black;stroke-width:2")
                node_container.set("id", "container-path")
                node.getparent().insert(0, node_container)

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


if __name__ == "__main__":
    PatternToPath().run()
