#!/usr/bin/env python3
import inkex
from pylivarot import sp_pathvector_boolop, bool_op, FillRule, py2geom, get_outline
from common_utils import bounds_rect, transform_path_vector, pattern_vector_to_d, get_fill_id, BaseFillExtension


class GradientToPath(BaseFillExtension):
    def __init__(self):
        BaseFillExtension.__init__(self)
        self._gradients = {}
        self.spacing = 1
        self._style_sheets = {}

    def add_arguments(self, pars):
        pars.add_argument("--debug", type=str, default='false', help="debug shapes")

    def effect(self):
        self.get_all_gradients(self.document.getroot())
        if self.svg.selected:
            for _, shape in self.svg.selected.items():
                self.recursive_replace_gradient(shape)

    def get_all_gradients(self, node):
        if node.tag.find('Gradient') >= 0:
            _id = node.attrib.get('id')
            self._gradients[_id] = node
        elif node.tag.find('style') >=0:
            for _stylesheet in node.stylesheet():
                for elem in self.svg.xpath(_stylesheet.to_xpath()):
                    if elem.get('id'):
                        self._style_sheets[elem.get('id')] = _stylesheet
                    elif elem.TAG == "stop":
                        for _key in _stylesheet.keys():
                            elem.set(_key, _stylesheet[_key])
        else:
            for child in node.getchildren():
                self.get_all_gradients(child)

    def recursive_replace_gradient(self, node):
        if node.tag in [inkex.addNS('rect', 'svg'), inkex.addNS('path', 'svg')]:
            node_id = node.get('id')
            pattern_id = None
            if node_id in self._style_sheets:
                css_style_sheet = self._style_sheets[node_id]
                if "fill" in css_style_sheet:
                    pattern_id = css_style_sheet["fill"].replace("url(#", "")[:-1]
            if not pattern_id:
                pattern_id = get_fill_id(node)
            
            if 'd' in node.attrib:
                container_path = node.attrib.get('d')
            else:
                container_path = pattern_vector_to_d(node)
            # needed elements:
            if pattern_id not in self._gradients:
                inkex.utils.errormsg(f"gradient {pattern_id} not found in {self._gradients.keys()}")
                return
            # is the gradient linear?
            if self._gradients[pattern_id].tag != inkex.addNS('linearGradient', 'svg'):
                inkex.utils.errormsg(f"gradient type {node.TAG} {node.get('id')} not yet supported")
                return
            # vector direction
            gradient = self._gradients[pattern_id]
            x1 = float(gradient.attrib.get('x1', "0"))
            x2 = float(gradient.attrib.get('x2', "1"))
            y1 = float(gradient.attrib.get('y1', "0"))
            y2 = float(gradient.attrib.get('y2', "0"))
            # if the angle isn't horizontal or vertical, throw an error
            _angle = py2geom.atan2(py2geom.Point(x2-x1, y2-y1))
            stops = gradient.stops
            stops_dict = {}
            if not stops:
                inkex.utils.errormsg(f"no stops for {pattern_id}")
                return
            for stop in stops:
                offset = stop.offset
                offset = stop.attrib.get('offset', '0')
                if '%' in offset:
                    offset = float(offset.replace('%', ''))/100.0
                else:
                    offset = float(offset)
                stops_dict[offset] = (stop.attrib['stop-color'], stop.attrib.get('stop-opacity', "1"))
            if not stops_dict:
                inkex.utils.errormsg("no stops_dict")
                return
            container_pv = py2geom.parse_svg_path(container_path)
            _affine = py2geom.Affine()
            _affine *= py2geom.Rotate(_angle)
            container_pv = transform_path_vector(container_pv, _affine)
            container_bbox = bounds_rect(container_pv)
            num_lines = container_bbox.width()/self.spacing
            line_colors = []
            offsets = sorted(stops_dict.keys())
            start_offset = offsets.pop(0)
            if start_offset > 0:
                for _ in range(0, num_lines*start_offset):
                    line_colors.append(stops_dict[start_offset])
            end_offset = None
            bin_size = None
            for i in range(int(num_lines**0.5), 1, -1):
                if num_lines % i == 0:
                    bin_size = i
                    break
            num_bins = int(num_lines/bin_size)
            if not bin_size:
                inkex.utils.errormsg(f"{num_lines} is prime!")
            while offsets:
                end_offset = offsets.pop(0)
                
                field_fraction = end_offset-start_offset
                field_bins = num_bins*field_fraction
                print(f"evaluating offset {start_offset} {end_offset} {field_bins} {field_fraction} {stops_dict[start_offset]} {stops_dict[end_offset]}")
                
                for i in range(int(field_bins)):
                    slope = -bin_size/field_bins
                    intercept = bin_size
                    y_value = slope*i + intercept
                    for j in range(bin_size):
                        if j > y_value:
                            line_colors.append(stops_dict[end_offset])
                        else:
                            line_colors.append(stops_dict[start_offset])
                start_offset = end_offset
            if end_offset < 1:
                for _ in range(0, num_lines-end_offset*num_lines):
                    line_colors.append(stops_dict[end_offset])
            assert len(line_colors) == num_lines, f"{len(line_colors)} is not {num_lines}, bin_size {bin_size} num_bins {num_bins}"
            path_builders = {f"{line_color[0]}_{line_color[1]}": py2geom.PathBuilder() for line_color in stops_dict.values()}
            for i in range(int(num_lines)):
                line_color = line_colors.pop()
                if f"{line_color[0]}_{line_color[1]}" == "blue_1":
                    print(f"adding blue line at {i} {self.spacing} {i*self.spacing+container_bbox.left()}")
                path_builders[f"{line_color[0]}_{line_color[1]}"].moveTo(py2geom.Point(i*self.spacing+container_bbox.left(), container_bbox.top()))
                path_builders[f"{line_color[0]}_{line_color[1]}"].lineTo(py2geom.Point(i*self.spacing+container_bbox.left(), container_bbox.bottom()))
            if not path_builders:
                inkex.utils.errormsg("path_builders is empty")
                return

            for line_color in path_builders:
                path_builders[line_color].flush()
                result = path_builders[line_color].peek()
                #if f"{line_color[0]}_{line_color[1]}" == "blue_1":
                #    print(result)
                result = get_outline(result, self.spacing)
                result = sp_pathvector_boolop(result, container_pv, bool_op.bool_op_inters, 
                    FillRule.fill_oddEven, FillRule.fill_oddEven, skip_conversion=True)
                _affine = py2geom.Affine()
                _affine *= py2geom.Rotate(-_angle)
                result = transform_path_vector(result, _affine)
                
                gradient_color_d = py2geom.write_svg_path(result)
                gradient_node = inkex.elements.PathElement()
                gradient_node.set_path(gradient_color_d)
                color = line_color.split("_")[0]
                opacity = line_color.split("_")[1]
                stroke_opacity = ""
                if opacity:
                    stroke_opacity = f"stroke-opacity:{opacity};"

                gradient_node.set('style', f"fill:none;stroke:{color};{stroke_opacity}stroke-width:{self.spacing}")
                gradient_node.set('id', f'gradient-{line_color}')          
                self.get_parent(node).insert(0, gradient_node)
            if self.options.debug == "true":
                container_path_d = py2geom.write_svg_path(container_pv)
                container_node = inkex.elements.PathElement()
                container_node.set_path(container_path_d)
                container_node.set('id', 'container_path')
                container_node.set('style', f'fill:none;stroke:green;stroke-width:{self.spacing}')
            self.get_parent(node).insert(0, container_node)

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
    GradientToPath().run()