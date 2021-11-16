#!/usr/bin/env python3
import inkex
from pylivarot import sp_pathvector_boolop, bool_op, FillRule, py2geom, pathv_to_linear_and_cubic_beziers
from svgpathtools.svg_to_paths import rect2pathd, ellipse2pathd

def transform_path_vector(pv, affine_t):
    for _path in pv:
        for _curve in _path:
            _curve.transform(affine_t)
    return pv

def pattern_vector_to_d(pattern_vector):
    if pattern_vector.TAG == "rect":
        return rect2pathd(pattern_vector.attrib)
    elif pattern_vector.TAG in ["circle", "ellipse"]:
        return ellipse2pathd(pattern_vector.attrib)
    else:
        inkex.utils.errormsg(
            "Shape %s (%s) not yet supported, try Object to path first"
            % (pattern_vector.TAG, pattern_vector.get("id"))
        )


class PatternToPath(inkex.EffectExtension):
    def __init__(self):
        inkex.EffectExtension.__init__(self)
        self._patterns = {}
        self.current_path = 1
        self.current_id = ""
        self.current_pattern_part = 1
        self._debug_pattern_list = []

    def add_arguments(self, pars):
        pars.add_argument("--remove", type=str, default='false',
                          help="Remove the existing path")

    def effect(self):
        self.get_all_patterns(self.document.getroot())
        if self.svg.selected:
            for _, shape in self.svg.selected.items():
                self.recursive_pattern_to_path(shape)
                self.current_path += 1
        else:
            self.recursive_pattern_to_path(self.document.getroot())

    def get_all_patterns(self, node):
        if node.tag.find('pattern') >= 0:
            _id = node.attrib.get('id')
            self._patterns[_id] = node
        else:
            for child in node.getchildren():
                self.get_all_patterns(child)

    def generate_wrapping_paper(self, container_pv, repeating_bbox, repeating_pattern):
        # pattern repeats is a path that takes the repeating_pattern and copies it up and down over 
        # the entire container path, like wrapping paper
        pattern_repeats = py2geom.PathVector()
        
        for i, _path in enumerate(container_pv):
            
            container_bbox = _path.boundsFast()
            _path_pv = py2geom.PathVector()
            _path_pv.push_back(_path)
            print("path ", i, py2geom.write_svg_path(_path_pv))
            container_bbox = py2geom.Rect(container_bbox[py2geom.Dim2.X], container_bbox[py2geom.Dim2.Y])   
            # need to get the bounding box from the bounding box
            pattern_x = container_bbox.left()
            pattern_y = container_bbox.top()
            # for debug only
            num_repeats = 0
            print(f"boundaries {container_bbox.left()} {container_bbox.top()} {container_bbox.width()} {container_bbox.height()}")
            while pattern_x <= container_bbox.left() + container_bbox.width():
                pattern_y = container_bbox.top()
                num_y_translations = 0
                
                while pattern_y <= container_bbox.top() + container_bbox.height():
                    _affine = py2geom.Affine()
                    _affine *= py2geom.Translate(0, repeating_bbox.height())
                    if pattern_repeats.curveCount() == 0:
                        pattern_repeats = repeating_pattern

                    # TODO: analyze whether skipping conversion speeds this up when there are no arches... or convert it first
                    # for some reason, bezier curves with unions merge together in weird ways, but arcs don't, maybe it's the zone close?   
                    #_tmp_pv = sp_pathvector_boolop(pathv_to_linear_and_cubic_beziers(repeating_pattern), pathv_to_linear_and_cubic_beziers(pattern_repeats), bool_op.bool_op_union, FillRule.fill_oddEven, FillRule.fill_oddEven, skip_conversion=True)
                    print(f"before convert {[curve for path in repeating_pattern for curve in path]}")
                    print(f"after {[curve for path in pathv_to_linear_and_cubic_beziers(repeating_pattern) for curve in path]}")
                    _tmp_pv = sp_pathvector_boolop(repeating_pattern, pattern_repeats, bool_op.bool_op_union, FillRule.fill_oddEven, FillRule.fill_oddEven, skip_conversion=True)
                    # the new path should always be more complex that the existing pattern
                    if _tmp_pv.curveCount() < pattern_repeats.curveCount():
                        inkex.utils.errormsg(f"{self.current_id} curve counts {_tmp_pv.curveCount()} {pattern_repeats.curveCount()} \
                        inputs were: repeating_pattern: '{py2geom.write_svg_path(repeating_pattern)}' pattern_repeats: '{py2geom.write_svg_path(pattern_repeats)}'")
                    pattern_repeats = _tmp_pv
                    num_y_translations += 1
                    num_repeats += 1
                    # for debug only
                    if num_repeats > 1:
                        return pattern_repeats
                    repeating_pattern = transform_path_vector(repeating_pattern, _affine)
                    pattern_y += repeating_bbox.height()
                _affine = py2geom.Affine()
                _affine *= py2geom.Translate(repeating_bbox.width(), -num_y_translations*repeating_bbox.height())
                repeating_pattern = transform_path_vector(repeating_pattern, _affine)
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
        href_tag = '{http://www.w3.org/1999/xlink}href'
        if href_tag in pattern.attrib:
            return self.recursive_find_pattern(pattern.attrib[href_tag][1:], transf)
        if "width" not in pattern.attrib or "height" not in pattern.attrib:
            raise KeyError("missing attributes in ", pattern.attrib)
        return pattern, transf

    def generate_pattern_path(self, node, container_path, repeating_box, repeating_pattern, pattern_style):
        # repeating_pattern should always have something
        if repeating_pattern.curveCount() == 0:
            inkex.utils.errormsg(f"{node.get('id')} pattern piece {self.current_path} is empty")
            return
            
        repeating_bbox = repeating_box.boundsFast()
        repeating_bbox = py2geom.Rect(repeating_bbox[py2geom.Dim2.X], repeating_bbox[py2geom.Dim2.Y])
        container_pv = py2geom.parse_svg_path(container_path) 

        pattern_repeats = self.generate_wrapping_paper(container_pv, repeating_bbox, repeating_pattern)        
        container_intersection = sp_pathvector_boolop(container_pv, pattern_repeats, 
                    bool_op.bool_op_inters, FillRule.fill_oddEven, FillRule.fill_oddEven, skip_conversion=True)
        parent = node.getparent()
        node_pattern = inkex.elements.PathElement()
        node_pattern.set_path(py2geom.write_svg_path(container_intersection))
        node_pattern.set('style', pattern_style)
        unknown_name = f"unknown-{self.current_path}"
        node_pattern.set('id', f'pattern-path-{node.get("id", unknown_name)}{self.current_pattern_part}')
        if self.options.remove == 'true':
            node.delete()
        if parent is not None:
            parent.insert(0, node_pattern)
        else:
            self.document.getroot().insert(0, node_pattern)

    def recursive_pattern_to_path(self, node):
        if node.tag in [inkex.addNS('rect', 'svg'), inkex.addNS('path', 'svg')]:
            style = node.attrib.get('style')
            inkex_style = dict(inkex.styles.Style().parse_str(style))
            fill = node.attrib.get('fill')
            self.current_id = node.attrib.get('id')
            inkex_style['fill'] = fill or inkex_style.get('fill')
            if not inkex_style['fill']:
                inkex.utils.errormsg("no 'fill' in inkex style")
                return
            if inkex_style['fill'].find("url(") != 0:
                inkex.utils.errormsg(f"for shape {node.attrib.get('id')} fill does not contain a pattern reference")
                return
            pattern_id = inkex_style['fill'].replace("url(#", "")[:-1]
            if 'd' in node.attrib:
                container_path = node.attrib.get('d')
            else:
                container_path = pattern_vector_to_d(node)
            # reset the debug information
            self._debug_pattern_list = []
            pattern, pattern_transform = self.recursive_find_pattern(pattern_id)
            pattern_width = float(pattern.attrib["width"])
            pattern_height = float(pattern.attrib["height"])
            d = f'M 0,0 L {pattern_width},0 L {pattern_width},{pattern_height} L 0,{pattern_height} L 0,0 z'

            repeating_box = py2geom.parse_svg_path(d) 
            affine_pattern_transform = py2geom.Affine(pattern_transform.a, pattern_transform.b, pattern_transform.c, pattern_transform.d, pattern_transform.e, pattern_transform.f)
            repeating_box = transform_path_vector(repeating_box, affine_pattern_transform)
            repeating_patterns = []
            pattern_styles = []
            for pattern_vector in pattern.getchildren():
                if 'd' not in pattern_vector.attrib:
                    # TODO: autoconvert using pylivarot
                    d_string = pattern_vector_to_d(pattern_vector)
                else:
                    d_string = pattern_vector.attrib['d']
                
                pattern_vector_pv = transform_path_vector(py2geom.parse_svg_path(d_string), affine_pattern_transform)
                pattern_intersection = sp_pathvector_boolop(repeating_box, pattern_vector_pv, bool_op.bool_op_inters, 
                FillRule.fill_oddEven, FillRule.fill_oddEven, skip_conversion=True)
                if pattern_intersection.curveCount() == 0:
                    inkex.utils.errormsg(f"pattern_intersection is empty, skipping, repeating_box is {py2geom.write_svg_path(repeating_box)}, \
                    pattern vector {pattern_vector.attrib['id']} is {d_string}")
                    continue
                #pattern_intersection = transform_path_vector(pattern_intersection, affine_pattern_transform)
                print(f"before arc removal: {py2geom.write_svg_path(pattern_intersection)}")
                print(f" after {py2geom.write_svg_path(pathv_to_linear_and_cubic_beziers(pattern_intersection))}")
                repeating_patterns.append(pathv_to_linear_and_cubic_beziers(pattern_intersection))
                #repeating_patterns.append(pattern_intersection)
                pattern_style = dict(inkex.styles.Style().parse_str(pattern_vector.attrib.get("style")))
                if 'stroke-width' in pattern_style:
                    pattern_style['stroke-width'] = float(pattern_style['stroke-width'])*affine_pattern_transform.expansionX()
                style_attribs = ["fill", "fill-opacity", "stroke", "stroke-width"]
                for attrib in style_attribs:
                    if attrib in pattern_vector.attrib:
                        pattern_style[attrib] = pattern_vector.attrib[attrib]
                pattern_styles.append(pattern_style)
            for i in range(len(pattern_styles)-1, -1, -1):
                self.current_pattern_part = i+1
                self.generate_pattern_path(node, container_path, repeating_box, repeating_patterns[i], pattern_styles[i])
            # for debug
            '''
            node_container = inkex.elements.PathElement()
            node_container.set_path(container_path)
            node_container.set('style', "fill:none;stroke:black;stroke-width:2")
            node_container.set('id', 'container-path')            
            node.getparent().insert(0, node_container)
            '''
                   
        elif node.tag in [inkex.addNS('text', 'svg'),
                          inkex.addNS('image', 'svg'),
                          inkex.addNS('use', 'svg')]:
            # this was copied from apply transforms - TODO: autoconvert to path using pylivarot
            inkex.utils.errormsg(
                "Shape %s (%s) not yet supported, try Object to path first"
                % (node.TAG, node.get("id"))
            )
        else:
            for child in node.getchildren():
                self.recursive_pattern_to_path(child)


if __name__ == '__main__':
    PatternToPath().run()
