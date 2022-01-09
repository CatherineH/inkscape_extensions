#!/usr/bin/env python3
from collections import defaultdict
import inkex
from common_utils import pattern_vector_to_d, BaseFillExtension
from random import random
from svgpathtools import Line, Path, parse_path
from functools import lru_cache
import sys

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
        pars.add_argument("--fill", type=bool, default=False, help="fill the stitch shapes")

    def effect(self):
        if self.svg.selected:
            for i, shape in self.svg.selected.items():
                self.curr_path_num = i
                self.hitomezashi_fill(shape)

    def chop_shape(self, lines):
        final_lines = []
        for i, line in enumerate(lines):
            # determine whether each point on the line is inside or outside the shape
            start_inside = self.is_inside(line.start)
            end_inside = self.is_inside(line.end)
            intersections = intersect_over_all(line, self.container)
            if not start_inside and not end_inside and not len(intersections): # skip this line, it's not inside the pattern
                continue
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
        return Path(*final_lines)

    def bool_op_shape(self, graph):
        try:
            from pylivarot import intersection, py2geom
        except ImportError as e:
            inkex.utils.errormsg("Fill does not work without pylivarot installed")
            sys.exit(0)
        # algorithm design
        # dump the keys in the graph into a unique list of points
        points_to_visit = list(set(graph.keys()))
        chained_line = []
        visited_points = []
        chained_lines = []
        curr_point = None
        while points_to_visit:
            if not curr_point:
                curr_point = points_to_visit.pop()
            if curr_point in visited_points:
                curr_point = None
                continue
            visited_points.append(curr_point)
            branches = graph[curr_point]
            
            if len(chained_line) > 0 and chained_line[0] == curr_point:
                # the loop is closed, yippee!
                chained_lines.append(chained_line)
                curr_point = None
                continue
            elif len(branches) == 1:
                chained_line += [curr_point] + branches
            elif len(branches) == 2:

                # if we got to a node that has two branches and we're not at the start of a chained line, don't add the one before
                if len(chained_line) >= 2:
                    if chained_line[-1] == branches[0]:
                        chained_line = [curr_point, branches[1]]
                    elif chained_line[-1] == branches[1]:
                        chained_line = [curr_point, branches[0]]
                    else:
                        inkex.utils.errormsg(f"got to bad state! {graph} {curr_point} {chained_line}")
                        sys.exit(0)
                else:
                    chained_line = [branches[0], curr_point, branches[1]]
            elif len(branches) == 0:                
                # crawl the outside of the shape
                # if you're on the left edge, try to go up first
                if curr_point.real == self.xmin:
                    down = curr_point - self.options.length*1j
                    if down in points_to_visit:
                        chained_line += [curr_point, down]
                        curr_point = down
                        continue
                # if you're on the right edge, try to go down first
                if curr_point.real == self.xmax:
                    down = curr_point - self.options.length*1j
                    if down in points_to_visit:
                        chained_line += [curr_point, down]
                        curr_point = down
                        continue
                    up = curr_point + self.options.length*1j
                    if up in points_to_visit:
                        chained_line += [curr_point, up]
                        curr_point = up
                        continue
                # if you're on the bottom edge, try to go left first
                if curr_point.imag == self.ymax:
                    left = curr_point - self.options.length
                    if left in points_to_visit:
                        chained_line += [curr_point, left]
                        curr_point = left
                        continue
                    right = curr_point + self.options.length
                    if right in points_to_visit:
                        chained_line += [curr_point, right]
                        curr_point = right
                        continue
                # if you're on the top edge, try to go right first    
                if curr_point.imag == self.ymin:
                    right = curr_point + self.options.length
                    if right in points_to_visit:
                        chained_line += [curr_point, right]
                        curr_point = right
                        continue
                    left = curr_point - self.options.length
                    if left in points_to_visit:
                        chained_line += [curr_point, left]
                        curr_point = left
                        continue
            else:
                inkex.utils.errormsg("got to bad state! ", graph, curr_point, chained_line)
                sys.exit(0)
            visited_points += branches
        chained_lines_pv = py2geom.PathVector()    
        for chained_line in chained_lines:
            pb = py2geom.PathBuilder()
            start_point = chained_line[0]
            pb.moveTo(py2geom.Point(start_point.real, start_point.imag))
            for i in range(1, len(chained_line)):
                pb.lineTo(py2geom.Point(chained_line[i].real, chained_line[i].imag))
            pb.closePath()
            chained_lines_pv.push_back(pb.flush())
        container_pv = py2geom.parse_svg_path(self.container.d())
        intersection_pv = intersection(container_pv, chained_lines_pv)
        output_chained_lines = []
        for piece in intersection_pv:
            piece_d =  py2geom.write_svg_path(piece)
            output_chained_lines.append(Path(piece_d))
        return chained_lines

    def hitomezashi_fill(self, node):
        # greedy algorithm: make a Hitomezashi fill that covers the entire bounding box of the shape, 
        # then go through each segment and figure out if it is inside, outside, or intersecting the shape
        
        self.container = parse_path(pattern_vector_to_d(node))
        self.xmin, self.xmax, self.ymin, self.ymax = self.container.bbox()
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin
        # generate vertical lines
        lines = []
        # build a graph of which edge points connect where
        graph = defaultdict(list)
        for x_i in range(int(self.width/self.options.length)):
            x_coord = x_i*self.options.length + self.xmin
            odd_even_y = random() > self.options.weight_x
            for y_i in range(int(self.height/self.options.length)):
                if y_i % 2 == odd_even_y:
                    continue
                y_coord = y_i*self.options.length + self.ymin
                start = x_coord + y_coord*1j
                end = x_coord + (y_coord+ self.options.length)*1j  
                lines.append(Line(start, end))
                graph[end].append(start)
                graph[start].append(end)
        # generate horizontal lines
        for y_i in range(int(self.height/self.options.length)):
            y_coord = y_i*self.options.length + self.ymin
            odd_even_y = random() > self.options.weight_y
            for x_i in range(int(self.width/self.options.length)):
                if x_i % 2 == odd_even_y:
                    continue
                x_coord = x_i*self.options.length + self.xmin
                start = x_coord + y_coord*1j
                end = (x_coord + self.options.length) + y_coord*1j
                lines.append(Line(start, end))
                graph[end].append(start)
                graph[start].append(end)
        
        if not self.options.fill:
            lines = [self.chop_shape(lines)]
        else:
            lines = self.bool_op_shape(graph)
        parent = self.get_parent(node)
            
        for i, chained_line in enumerate(lines):
            
            pattern_id = "hitomezashi-"+node.get("id", f"unknown-{self.curr_path_num}")+"-"+str(i)
            pattern_style = node.get("style")
            node_pattern = inkex.elements.PathElement()
            node_pattern.set_path(chained_line.d())
            node_pattern.set('style', pattern_style)
            node_pattern.set('id', pattern_id)
            if node.get("transform"):
                node_pattern.set("transform", node.get("transform"))
            parent.insert(0, node_pattern)

    @lru_cache(maxsize=None)
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