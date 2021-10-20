#!/usr/bin/env python3
import inkex
from pylivarot import sp_pathvector_boolop, bool_op, FillRule, py2geom



class PatternToPath(inkex.EffectExtension):
    def __init__(self):
        inkex.EffectExtension.__init__(self)
        self._patterns = {}

    def effect(self):
        self.get_all_patterns(self.document.getroot())
        if self.svg.selected:
            for _id, shape in self.svg.selected.items():
                self.recursive_pattern_to_path(shape)
        else:
            self.recursive_pattern_to_path(self.document.getroot())

    def get_all_patterns(self, node):
        if node.tag.find('pattern') >= 0:
            _id = node.attrib.get('id')
            width = node.attrib.get('width', None)
            height = node.attrib.get('height', None)
            pattern_transform = node.attrib.get('patternTransform', None)
            paths = node.getchildren()
            self._patterns[_id] = node
            print(width, height, pattern_transform, paths)
        else:
            for child in node.getchildren():
                self.get_all_patterns(child)

    def recursive_pattern_to_path(self, node):
        print("in recursive pattern to path", node)
        if node.tag.find('path') >= 0:
            style = node.attrib.get('style')
            style_attributes = {part.split(":")[0]: part.split(":")[1] for part in style.split(';')}
            if 'fill' not in style_attributes:
                return
            if style_attributes['fill'].find("url(") != 0:
                return
            pattern_id = style_attributes['fill'].replace("url(#", "")[:-1]
            container_path = node.attrib.get('d')
            # TODO: figure out how to take the intersection between paths
            print(pattern_id, self._patterns[pattern_id])
            pattern = self._patterns[pattern_id]
            if "width" not in pattern.attrib or "height" not in pattern.attrib:
                raise KeyError("missing attributes in ", pattern.attrib)
            pattern_width = pattern.attrib["width"]
            pattern_height = pattern.attrib["height"]
            
            d = f'M 0,0 L {pattern_width},0 L {pattern_width},{pattern_height} L 0,{pattern_height} L 0,0'

            repeating_box = py2geom.parse_svg_path(d) 
            repeating_patterns = []
            for pattern_vector in pattern.getchildren():
                if 'd' not in pattern_vector.attrib:
                    inkex.utils.errormsg(
                    "Shape %s (%s) not yet supported, try Object to path first"
                    % (pattern_vector.TAG, pattern_vector.get("id"))
                )
                pattern_intersection = sp_pathvector_boolop(repeating_box, py2geom.parse_svg_path(pattern_vector.attrib['d']), bool_op.bool_op_inters, FillRule.fill_oddEven, FillRule.fill_oddEven)
                repeating_patterns.append(pattern_intersection)
            bounding_box = container_path.bounding_box()
                      
        elif node.tag in [inkex.addNS('rect', 'svg'),
                          inkex.addNS('text', 'svg'),
                          inkex.addNS('image', 'svg'),
                          inkex.addNS('use', 'svg')]:
            # this was copied from apply transforms
            inkex.utils.errormsg(
                "Shape %s (%s) not yet supported, try Object to path first"
                % (node.TAG, node.get("id"))
            )
        else:
            for child in node.getchildren():
                self.recursive_pattern_to_path(child)


if __name__ == '__main__':
    PatternToPath().run()
