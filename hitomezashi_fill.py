#!/usr/bin/env python3
import inkex
from common_utils import pattern_vector_to_d, BaseFillExtension
from random import random
from svgpathtools import Line, Path, parse_path
from math import acos


def intersect_over_all(line, path):
    all_intersections = []
    for segment in path:
        all_intersections += line.intersect(segment)
    return all_intersections


class HitomezashiFill(BaseFillExtension):
    def __init__(self):
        inkex.EffectExtension.__init__(self)
        self.xmax = None
        self.xmin = None
        self.ymin = None
        self.ymax = None
        self.container = None
        self.curr_path_num = 0

    def add_arguments(self, pars):
        pars.add_argument("--length", type=float, default=1, help="Length of segments")
        pars.add_argument("--weight_x", type=float, default=0.5, help="The probability of getting a 1 along the x axis")
        pars.add_argument("--weight_y", type=float, default=0.5, help="The probability of getting a 1 along the y axis")

    def effect(self):
        if self.svg.selected:
            for i, shape in self.svg.selected.items():
                self.curr_path_num = i
                #ApplyTransform().recursiveFuseTransform(shape)
                self.hitomezashi_fill(shape)
    
    def get_transform(self, node):
        full_transform = None
        while node is not None:
            transf = inkex.transforms.Transform(node.get("transform", None))
            if not full_transform:
                full_transform = transf
            else:
                full_transform *= transf
            node = node.getparent()
        return full_transform

    def hitomezashi_fill(self, node):
        # greedy algorithm: make a Hitomezashi fill that covers the entire bounding box of the shape, 
        # then go through each segment and figure out if it is inside, outside, or intersecting the shape
        
        self.container = parse_path(pattern_vector_to_d(node))
        self.xmin, self.xmax, self.ymin, self.ymax = self.container.bbox()
        width = self.xmax - self.xmin
        height = self.ymax - self.ymin
        # generate vertical lines
        lines = []
        # hitomezashi_fill = py2geom.PathBuilder()
        for x_i in range(int(width/self.options.length)):
            x_coord = x_i*self.options.length + self.xmin
            odd_even_y = random() > self.options.weight_x
            for y_i in range(int(height/self.options.length)):
                if y_i % 2 == odd_even_y:
                    continue
                y_coord = y_i*self.options.length + self.ymin
                lines.append(Line(x_coord + y_coord*1j, x_coord + (y_coord+ self.options.length)*1j))
        # generate horizontal lines
        for y_i in range(int(height/self.options.length)):
            y_coord = y_i*self.options.length + self.ymin
            odd_even_y = random() > self.options.weight_y
            for x_i in range(int(width/self.options.length)):
                if x_i % 2 == odd_even_y:
                    continue
                x_coord = x_i*self.options.length + self.xmin
                lines.append(Line(x_coord + y_coord*1j, (x_coord + self.options.length) + y_coord*1j))
        
        final_lines = []
        for i, line in enumerate(lines):
            # determine whether each point on the line is inside or outside the shape
            start_inside = self.is_inside(line.start)
            end_inside = self.is_inside(line.end)
            intersections = intersect_over_all(line, self.container)
            if not start_inside and not end_inside and not len(intersections): # skip this line, it's not inside the pattern
                continue
            print(i, line, start_inside, end_inside, len(intersections))
            if start_inside and end_inside and not intersections: # add this line and then continue
                final_lines.append(line)

            # if it has intersections, it's trickier
            curr_start = line.start
            for (t1, t2) in intersections:
                if start_inside:
                    final_lines.append(Line(curr_start, line.point(t1)))
                start_inside = not start_inside
                curr_start = line.point(t1)
            if start_inside:
                final_lines.append(Line(curr_start, line.end))
        # extra credit = chain the paths together
        lines = final_lines
        
        chained_lines = [lines.pop()]

        while lines:
            removed_index = None
            for i, line in enumerate(lines):
                if line.start == chained_lines[0].start:
                    removed_index = i
                    chained_lines.insert(0, line.reversed())
                    break
                if line.end == chained_lines[0].start:
                    removed_index = i
                    chained_lines.insert(0, line)
                    break
                if line.start == chained_lines[-1].end:
                    removed_index = i
                    chained_lines.append(line)
                    break
                if line.end == chained_lines[-1].end:
                    removed_index = i
                    chained_lines.append(line.reversed())
                    break
            if removed_index is None:
                removed_index = 0
                chained_lines.append(lines[removed_index])
            del lines[removed_index]
        
        # chained_lines = lines
        full_path = Path(*chained_lines)
        pattern_id = "hitomezashi-"+node.get("id", f"unknown-{self.curr_path_num}")
        pattern_style = node.get("style")
        node_pattern = inkex.elements.PathElement()
        node_pattern.set_path(full_path.d())
        node_pattern.set('style', pattern_style)
        node_pattern.set('id', pattern_id)
        if node.get("transform"):
            node_pattern.set("transform", node.get("transform"))
        parent = self.get_parent(node)    
        parent.insert(0, node_pattern)

    def is_inside(self, point):
        if point == self.xmin + self.ymin*1j:
            return False
        if point == self.xmax + self.ymax*1j:
            return False
        span_line_upper = Line(self.xmin+ self.ymin*1j, point)
        span_line_lower = Line(point, self.xmax + self.ymax*1j)
        upper_intersections = intersect_over_all(span_line_upper, self.container)
        lower_intersections = intersect_over_all(span_line_lower, self.container)
        return len(upper_intersections) % 2 or len(lower_intersections) % 2


if __name__ == "__main__":
    HitomezashiFill().run()