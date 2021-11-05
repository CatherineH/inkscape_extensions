#!/usr/bin/env python3
import inkex
from pylivarot import sp_pathvector_boolop, bool_op, FillRule, py2geom
from svgpathtools.svg_to_paths import rect2pathd

def transform_path_vector(pv, affine_t):
    for _path in pv:
        for _curve in _path:
            _curve.transform(affine_t)
    return pv

def pattern_vector_to_d(pattern_vector):
    if pattern_vector.TAG == "rect":
        return rect2pathd(pattern_vector.attrib)
    else:
        inkex.utils.errormsg(
            "Shape %s (%s) not yet supported, try Object to path first"
            % (pattern_vector.TAG, pattern_vector.get("id"))
        )


class PatternToPath(inkex.EffectExtension):
    def __init__(self):
        inkex.EffectExtension.__init__(self)
        self._patterns = {}

    def effect(self):
        self.get_all_patterns(self.document.getroot())
        if self.svg.selected:
            for _, shape in self.svg.selected.items():
                self.recursive_pattern_to_path(shape)
        else:
            self.recursive_pattern_to_path(self.document.getroot())

    def get_all_patterns(self, node):
        if node.tag.find('pattern') >= 0:
            _id = node.attrib.get('id')
            self._patterns[_id] = node
        else:
            for child in node.getchildren():
                self.get_all_patterns(child)

    def recursive_find_pattern(self, pattern_id, transf=None):
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

    def recursive_pattern_to_path(self, node):
        if node.tag in [inkex.addNS('rect', 'svg'), inkex.addNS('path', 'svg')]:
            style = node.attrib.get('style')
            inkex_style = dict(inkex.styles.Style().parse_str(style))
            if 'fill' not in inkex_style:
                inkex.utils.errormsg("no 'fill' in inkex style")
            if inkex_style['fill'].find("url(") != 0:
                inkex.utils.errormsg("fill does not contain a pattern reference")
            pattern_id = inkex_style['fill'].replace("url(#", "")[:-1]
            if 'd' in node.attrib:
                container_path = node.attrib.get('d')
            else:
                container_path = pattern_vector_to_d(node)
            container_bbox = inkex.Path(container_path).bounding_box()
            pattern, pattern_transform = self.recursive_find_pattern(pattern_id)
            pattern_width = float(pattern.attrib["width"])
            pattern_height = float(pattern.attrib["height"])
            d = f'M 0,0 L {pattern_width},0 L {pattern_width},{pattern_height} L 0,{pattern_height} L 0,0'

            repeating_box = py2geom.parse_svg_path(d) 
            affine_pattern_transform = py2geom.Affine(pattern_transform.a, pattern_transform.b, pattern_transform.c, pattern_transform.d, pattern_transform.e, pattern_transform.f)
            repeating_box = transform_path_vector(repeating_box, affine_pattern_transform)
            repeating_bbox = repeating_box.boundsFast()
            repeating_bbox = py2geom.Rect(repeating_bbox[py2geom.Dim2.X], repeating_bbox[py2geom.Dim2.Y])
            repeating_patterns = []
            pattern_styles = []
            for pattern_vector in pattern.getchildren():
                if 'd' not in pattern_vector.attrib:
                    # TODO: autoconvert using pylivarot
                    d_string = pattern_vector_to_d(pattern_vector)
                else:
                    d_string = pattern_vector.attrib['d']
                pattern_intersection = sp_pathvector_boolop(repeating_box, py2geom.parse_svg_path(d_string), bool_op.bool_op_inters, FillRule.fill_oddEven, FillRule.fill_oddEven)
                pattern_intersection = transform_path_vector(pattern_intersection, affine_pattern_transform)
                # TODO: apply styles to individual components of the pattern
                #repeating_patterns.append(pattern_intersection)
                repeating_patterns = pattern_intersection
                pattern_style = dict(inkex.styles.Style().parse_str(pattern_vector.attrib.get("style")))
                pattern_style['stroke-width'] = float(pattern_style['stroke-width'])*affine_pattern_transform.expansionX()
                pattern_styles.append(pattern_style)

            # need to get the bounding box from the bounding box
            pattern_x = container_bbox.left
            pattern_y = container_bbox.top
            pattern_repeats = py2geom.PathVector()
            
            while pattern_x <= container_bbox.left + container_bbox.width:
                pattern_y = container_bbox.top
                num_y_translations = 0
                while pattern_y <= container_bbox.top + container_bbox.height:
                    _affine = py2geom.Affine()
                    _affine *= py2geom.Translate(0, repeating_bbox.height())
                    repeating_patterns = transform_path_vector(repeating_patterns, _affine)
                    if pattern_repeats.curveCount() == 0:
                        pattern_repeats = repeating_patterns
                    pattern_repeats = sp_pathvector_boolop(repeating_patterns, pattern_repeats, bool_op.bool_op_union, FillRule.fill_oddEven, FillRule.fill_oddEven)
                    assert pattern_repeats.curveCount() >= repeating_box.curveCount(), f"curve counts {pattern_repeats.curveCount()} {repeating_box.curveCount()}"
                    num_y_translations += 1
                    pattern_y += repeating_bbox.height()
                _affine = py2geom.Affine()
                _affine *= py2geom.Translate(repeating_bbox.width(), -num_y_translations*repeating_bbox.height())
                repeating_patterns = transform_path_vector(repeating_patterns, _affine)
                pattern_x += repeating_bbox.width() 
            
            container_intersection = sp_pathvector_boolop(py2geom.parse_svg_path(container_path), pattern_repeats, 
                    bool_op.bool_op_inters, FillRule.fill_oddEven, FillRule.fill_oddEven)
            node.set_path(py2geom.write_svg_path(container_intersection))
            node.set('style', pattern_styles[0])
            del inkex_style['fill']
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
