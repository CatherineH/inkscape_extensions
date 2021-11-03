#!/usr/bin/env python3
import inkex
from pylivarot import sp_pathvector_boolop, bool_op, FillRule, py2geom
from svgpathtools.svg_to_paths import rect2pathd


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

    def recursive_find_pattern(self, pattern_id, transf=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]):
        pattern = self._patterns[pattern_id]
        # need to combine transforms
        transf = inkex.transforms.Transform(transf) * inkex.transforms.Transform(pattern.get("transform", None))
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
            pattern_width = pattern.attrib["width"]
            pattern_height = pattern.attrib["height"]
            
            d = f'M 0,0 L {pattern_width},0 L {pattern_width},{pattern_height} L 0,{pattern_height} L 0,0'

            repeating_box = py2geom.parse_svg_path(d) 
            repeating_patterns = ""
            for pattern_vector in pattern.getchildren():
                if 'd' not in pattern_vector.attrib:
                    # TODO: autoconvert using pylivarot
                    d_string = pattern_vector_to_d(pattern_vector)
                else:
                    d_string = pattern_vector.attrib['d']
                pattern_intersection = sp_pathvector_boolop(repeating_box, py2geom.parse_svg_path(d_string), bool_op.bool_op_inters, FillRule.fill_oddEven, FillRule.fill_oddEven)
                repeating_patterns += py2geom.write_svg_path(pattern_intersection)
            pattern_path = inkex.Path(repeating_patterns).transform(pattern_transform)
            pattern_bounding_box = pattern_path.bounding_box()
            pattern_x = container_bbox.left
            pattern_y = container_bbox.top
            
            pattern_repeats = inkex.paths.Path()
            while pattern_x < container_bbox.left + container_bbox.width:
                pattern_y = container_bbox.top
                num_y_translations = 0
                while pattern_y < container_bbox.top + container_bbox.height:
                    pattern_path.translate(0, pattern_bounding_box.height, inplace=True)
                    pattern_repeats += pattern_path.to_absolute()
                    #print(pattern_x, pattern_y, pattern_path)
                    num_y_translations += 1
                    pattern_y += pattern_bounding_box.height
                pattern_path.translate(pattern_bounding_box.width, -num_y_translations*pattern_bounding_box.height, inplace=True)
                
                pattern_x += pattern_bounding_box.width 
            pattern_repeats = inkex.paths.CubicSuperPath(pattern_repeats).to_path(curves_only=False)
            _path_builder = py2geom.PathBuilder()
            control_points = [py2geom.Point(_point.x, _point.y) for _point in pattern_repeats.control_points]
            print("number of segments: ", len(pattern_repeats))
            
            for _segment in pattern_repeats:
                if isinstance(_segment, inkex.paths.Move):
                    _path_builder.moveTo(control_points.pop(0))
                elif isinstance(_segment, inkex.paths.Line):
                    _path_builder.lineTo(control_points.pop(0))
                elif isinstance(_segment, inkex.paths.Quadratic):
                    _path_builder.quadTo(control_points.pop(0), control_points.pop(0))
                elif isinstance(_segment, inkex.paths.Curve):
                    _path_builder.curveTo(control_points.pop(0), control_points.pop(0), control_points.pop(0))
                elif isinstance(_segment, inkex.paths.ZoneClose):
                    _path_builder.flush()
                else:
                    raise ValueError("not sure what to do with: ", _segment)
            _path_builder.flush()
            pattern_repeats_pv = _path_builder.peek()

            container_intersection = sp_pathvector_boolop(py2geom.parse_svg_path(container_path), pattern_repeats_pv, 
                    bool_op.bool_op_inters, FillRule.fill_oddEven, FillRule.fill_oddEven)
            node.path = inkex.paths.CubicSuperPath(py2geom.write_svg_path(container_intersection)).to_path(curves_only=False)
            del inkex_style['fill']
            node.set('style', str(inkex_style))
            node_pattern_repeats = inkex.elements.PathElement()
            node_pattern_repeats.set_path(pattern_repeats)
            node_pattern_repeats.set('style', str(inkex_style))
            node_pattern_repeats.set('id', 'node-pattern-repeats')
            node_container_path = inkex.elements.PathElement()
            node_container_path.set_path(container_path)
            node_container_path.set('style', str(inkex_style))
            node_container_path.set('id', 'node-container-paths')
            
            node.getparent().insert(0, node_pattern_repeats)
            node.getparent().insert(0, node_container_path)
            print(type(self.document))
            # bounding_box = container_path.bounding_box()
                      
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
