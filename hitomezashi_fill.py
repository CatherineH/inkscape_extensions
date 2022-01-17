#!/usr/bin/env python3
from collections import defaultdict
import inkex
from time import time
from math import atan
from common_utils import pattern_vector_to_d, BaseFillExtension, debug_screen
from random import random
from svgpathtools import Line, Path, parse_path
from functools import lru_cache
import sys

def intersect_over_all(line, path):
    all_intersections = []
    for i,segment in enumerate(path):
        current_intersections = line.intersect(segment)
        all_intersections += [(t1, t2, i) for (t1, t2) in current_intersections]
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
        self.current_shape = None
        self.outline_intersections = []
        # build a graph of which edge points connect where
        self.graph = defaultdict(dict)

    def add_arguments(self, pars):
        pars.add_argument("--length", type=float, default=1, help="Length of segments")
        pars.add_argument("--weight_x", type=float, default=0.5, help="The probability of getting a 1 along the x axis")
        pars.add_argument("--weight_y", type=float, default=0.5, help="The probability of getting a 1 along the y axis")
        pars.add_argument("--fill", type=bool, default=False, help="fill the stitch shapes")

    def effect(self):
        if self.svg.selected:
            for i, shape in self.svg.selected.items():
                self.curr_path_num = i
                self.current_shape = shape
                self.hitomezashi_fill(shape)

    def add_node(self, d_string, style, id):
        parent = self.get_parent(self.current_shape)
        _node = inkex.elements.PathElement()
        _node.set_path(d_string)
        _node.set('style', style)
        _node.set('id', id)
        if self.current_shape.get("transform"):
            _node.set("transform", self.current_shape.get("transform"))
        parent.insert(-1, _node)
    
    def add_marker(self, point, label="marker"):
        marker_size = self.options.length/10
                
        marker = [Line(point+marker_size+marker_size*1j, point-marker_size+marker_size*1j), 
                Line(point-marker_size+marker_size*1j, point-marker_size-marker_size*1j), 
                Line(point-marker_size-marker_size*1j, point+marker_size-marker_size*1j), 
                Line(point+marker_size-marker_size*1j, point+marker_size+marker_size*1j)]
        self.add_node(Path(*marker).d(), "fill:red;stroke:none", label)

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
            for (t1, t2, seg_i) in intersections:
                self.outline_intersections.append((t2, seg_i))
                if start_inside:
                    final_lines.append(Line(curr_start, line.point(t1)))
                start_inside = not start_inside
                curr_start = line.point(t1)
            if start_inside:
                final_lines.append(Line(curr_start, line.end))
        for line in final_lines:
            self.graph[line.start][line.end] = Line(line.start, line.end)
            self.graph[line.end][line.start] = Line(line.end, line.start)
        self.outline_intersections = list(set(self.outline_intersections))
        self.outline_intersections.sort(key= lambda x: -x[0]-x[1])
        intersections_copy = self.outline_intersections
        start_intersection = intersections_copy.pop()
        intersections_copy.insert(0, start_intersection) # add the end back onto the front so that we'll close the loop 
        while intersections_copy:
            end_intersection = intersections_copy.pop()
            start = self.container[start_intersection[1]].point(start_intersection[0])
            end = self.container[end_intersection[1]].point(end_intersection[0])
            
            if start_intersection[1] == end_intersection[1]:
                segment = self.container[start_intersection[1]].cropped(start_intersection[0], end_intersection[0])
            else:
                segments = []
                if start_intersection[0] != 1:
                    segments = [self.container[start_intersection[1]].cropped(start_intersection[0],1)]
                index_i = start_intersection[1] + 1
                while index_i < end_intersection[1]:
                    segments.append(self.container[index_i])
                    index_i += 1
                if end_intersection[0] != 0:
                    segments.append(self.container[end_intersection[1]].cropped(0, end_intersection[0]))
                segment = Path(*segments)
            if segment.length() == 0: # skip this one because it's too short
                print("skipping: ",start_intersection, end_intersection)
                continue   
            self.graph[start][end] = segment
            self.graph[end][start] = segment.reversed()
             
            start_intersection = end_intersection
        return Path(*final_lines)

    def chain_graph(self):
        # algorithm design
        # dump the keys in the graph into a unique list of points
        points_to_visit = list(set(self.graph.keys()))
        chained_line = []
        visited_points = []
        chained_lines = []
        curr_point = None
        start_time = time()
        while points_to_visit:
            total_entries = sum( 1 for branch in self.graph.values() for point in branch)
            print(f"total entries {total_entries}")
            if len(chained_line) > 0:
                curr_point = chained_line[-1]
            if not curr_point:
                curr_point = points_to_visit.pop()
            
            if len(chained_line) > 0 and chained_line[0] == curr_point:
                # the loop is closed, yippee!
                chained_lines.append(chained_line)
                chained_line = []
                curr_point = None
                continue
            if curr_point in visited_points:
                curr_point = None
                chained_lines.append(chained_line)
                chained_line = []
                continue
            visited_points.append(curr_point)
            branches = [point for point in self.graph[curr_point] if (len(chained_line) >= 2 and point != chained_line[-2]) or len(chained_line) < 2]

            if len(branches) == 1:
                chained_line += [curr_point] + branches
            elif len(branches) == 2:
                # if the previous point was on the outside, pick the point on the inside
                if len(chained_line) <2 or chained_line[-2] in self.outline_intersections:
                    if branches[0] in self.outline_intersections:
                        chained_line.append(branches[1])
                    elif branches[1] in self.outline_intersections:
                        chained_line.append(branches[0])
                    else:
                        inkex.utils.errormsg(f"got to bad state! - last point was on outside and only two options are also outside {branches} ")
                        sys.exit(0)
                else: # if the previous point was on the inside, pick the clockwise outside location
                    angle1 = atan(abs(chained_line[-2]-curr_point)/abs(branches[0]-curr_point))
                    angle2 = atan(abs(chained_line[-2]-curr_point)/abs(branches[0]-curr_point))
                    if angle1 > angle2: # clockwise means bigger angle?
                        chained_line.append(branches[0])
                    else:
                        chained_line.append(branches[1])
            elif len(branches) == 3: # we're probably on the outside
                if len(chained_line) > 0:
                    inkex.utils.errormsg(f"got to bad state! {branches} {curr_point[:-2]} ")
                    sys.exit(0)
                else:
                    chained_line = [curr_point, branches[0]]
            else:
                inkex.utils.errormsg(f"got to bad state! no branches- {branches} {curr_point} {self.graph[curr_point]}")
                # dump the graph 
                all_graph_segments = [segment for branch in self.graph.values() for segment in branch.values()]
                self.add_node(Path(*all_graph_segments).d(), "stroke:gray;stroke-width:2;fill:none", "graph")
                self.add_marker(curr_point)
                segments = []
                for i in range(1, len(chained_line)):
                    try:
                        segments.append(self.graph[chained_line[i-1]][chained_line[i]])
                    except KeyError as e:                        
                        print(f"got key error on: {chained_line[i]} not in {self.graph.get(chained_line[i-1], {}).keys()} {self.graph[chained_line[i]].keys()}")
                self.add_node(Path(*segments).d(), "stroke:red;stroke-width:2;fill:none", "chained_path")
                parent = self.get_parent(self.current_shape)
                parent.remove(self.current_shape)
                debug_screen(self, "test_graph")
                sys.exit(0)
        print(f"chaining took {time()-start_time} num lines {len(chained_lines)}")
        # convert to segments
        paths = []
        for chained_line in chained_lines:
            segments = []
            for i in range(1, len(chained_line)):
                segments.append(self.graph[chained_line[i-1]][chained_line[i]])
            paths.append(Path(*segments))
        return paths
        

    def bool_op_shape(self):
        try:
            from pylivarot import intersection, py2geom
            # TODO: it shoud be possible to without doing the boolean operation by cropping the bezier curves on the outside and adding them to the graph
        except ImportError as e:
            inkex.utils.errormsg("Fill does not work without pylivarot installed")
            sys.exit(0)
        chained_lines = self.chain_graph()
        start_time = time()
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
        print(f"path building took {time()-start_time}")
        start_time = time()
        intersection_pv = intersection(container_pv, chained_lines_pv)
        print(f"bool op took {time()-start_time}")
        start_time = time()
        output_chained_lines = []
        for piece in intersection_pv:
            piece_d =  py2geom.write_svg_path(piece)
            output_chained_lines.append(Path(piece_d))
        print(f"decomp took {time()-start_time}")
        return output_chained_lines

    def hitomezashi_fill(self, node):
        # greedy algorithm: make a Hitomezashi fill that covers the entire bounding box of the shape, 
        # then go through each segment and figure out if it is inside, outside, or intersecting the shape
        
        self.container = parse_path(pattern_vector_to_d(node))
        self.container.approximate_arcs_with_quads()
        self.xmin, self.xmax, self.ymin, self.ymax = self.container.bbox()
        # todo: remove me
        self.xmin = int(self.xmin)
        self.ymin = int(self.ymin)
        self.xmax = int(self.xmax)
        self.ymax = int(self.ymax)
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin
        # generate vertical lines
        lines = []

        for x_i in range(int(self.width/self.options.length)):
            x_coord = x_i*self.options.length + self.xmin
            odd_even_y = random() > self.options.weight_x
            for y_i in range(int(self.height/self.options.length)):
                if y_i % 2 == odd_even_y:
                    continue
                y_coord = y_i*self.options.length + self.ymin
                y_coord = int(y_coord) # TODO: remove me
                start = x_coord + y_coord*1j
                end = x_coord + (y_coord+ self.options.length)*1j  
                lines.append(Line(start, end))
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

        if not self.options.fill:
            lines = [self.chop_shape(lines)]
        else:
            _ = self.chop_shape(lines)
            lines = self.chain_graph()
            #lines = self.bool_op_shape(self.graph)
            print(f"chained_lines level {len(lines)}")

        for i, chained_line in enumerate(lines):            
            pattern_id = "hitomezashi-"+node.get("id", f"unknown-{self.curr_path_num}")+"-"+str(i)
            pattern_style = node.get("style")
            self.add_node(chained_line.d(), pattern_style, pattern_id)

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