#!/usr/bin/env python3
from collections import defaultdict
import inkex
from time import time
from common_utils import get_clockwise, pattern_vector_to_d, BaseFillExtension, debug_screen, combine_segments, format_complex, get_clockwise
from random import random
from svgpathtools import Line, Path, parse_path
from functools import lru_cache
import sys

def intersect_over_all(line, path):
    all_intersections = []
    for i,segment in enumerate(path):
        current_intersections = line.intersect(segment)
        all_intersections += [(t1, t2, i) for (t1, t2) in current_intersections]
        line_end_t = segment.point_to_t(line.end)
        line_start_t = segment.point_to_t(line.start)
        '''
        if line_end_t and (1.0, line_end_t, i) not in all_intersections:
            all_intersections += [(1.0, line_end_t, i)]
        if line_start_t and (0.0, line_start_t, i) not in all_intersections:
            all_intersections += [(0.0, line_start_t, i)]
        '''        
    return all_intersections

TOLERANCE = 0.2

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
        self.outline_nodes = []
        # build a graph of which edge points connect where
        self.graph = defaultdict(dict)
        self.visited_points = []
        self.chained_lines = []

    def add_arguments(self, pars):
        pars.add_argument("--length", type=float, default=1, help="Length of segments")
        pars.add_argument("--weight_x", type=float, default=0.5, help="The probability of getting a 1 along the x axis")
        pars.add_argument("--weight_y", type=float, default=0.5, help="The probability of getting a 1 along the y axis")
        pars.add_argument("--gradient", type=str, default='false', help="fill the stitch shapes")
        pars.add_argument("--fill", type=str, default='false', help="fill the stitch shapes")

    def effect(self):
        self.options.gradient = self.options.gradient == 'true'
        self.options.fill = self.options.fill == 'true'
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

    def add_chained_line(self, chained_line, label="chained-line", color="red"):
        segments = []
        for i in range(1, len(chained_line)):
            try:
                segments.append(self.graph[chained_line[i-1]][chained_line[i]])
            except KeyError as e:
                self.add_marker(chained_line[i], label=f"missing-segment-{i}", color="blue")
                self.add_marker(chained_line[i-1], label=f"missing-segment-{i}", color="green")                           
                print(f"got key error on: {chained_line[i]} not in {self.graph.get(chained_line[i-1], {}).keys()} {self.graph[chained_line[i]].keys()}")
        stroke_length = self.options.length/10
        self.add_node(combine_segments(segments).d(), f"stroke:{color};stroke-width:{stroke_length};fill:none", label)
    
    def add_marker(self, point, label="marker", color="red"):
        marker_size = self.options.length/10
                
        marker = [Line(point+marker_size+marker_size*1j, point-marker_size+marker_size*1j), 
                Line(point-marker_size+marker_size*1j, point-marker_size-marker_size*1j), 
                Line(point-marker_size-marker_size*1j, point+marker_size-marker_size*1j), 
                Line(point+marker_size-marker_size*1j, point+marker_size+marker_size*1j)]
        self.add_node(Path(*marker).d(), f"fill:{color};stroke:none", label)
    
    def plot_graph(self):
        # dump the graph 
        all_graph_segments = [segment for branch in self.graph.values() for segment in branch.values()]
        self.add_node(combine_segments(all_graph_segments).d(), "stroke:gray;stroke-width:2;fill:none", "graph")

    def chop_shape(self, lines):
        final_lines = []
        for i, line in enumerate(lines):
            # determine whether each point on the line is inside or outside the shape
            start_inside = self.is_inside(line.start)
            end_inside = self.is_inside(line.end)
            intersections = intersect_over_all(line, self.container)
            if not start_inside and not end_inside and not len(intersections): # skip this line, it's not inside the pattern
                #if not start_inside and self.xmin < line.start.real < self.xmax and self.ymin < line.start.imag < self.ymax:
                #    raise ValueError(f"something went wrong here, start is definitely inside {self.xmin} {line.start} {self.xmax}")
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
            if line.length() < TOLERANCE*self.options.length: # skip this one because it's too short
                print("skipping: ", line)
                continue 
            line.start = self.snap_nodes(line.start)
            line.end = self.snap_nodes(line.end)

            self.graph[line.start][line.end] = Line(line.start, line.end)
            self.graph[line.end][line.start] = Line(line.end, line.start)

        self.outline_intersections = list(set(self.outline_intersections))
        self.outline_intersections.sort(key= lambda x: -x[0]-x[1])
        intersections_copy = self.outline_intersections
        start_intersection = intersections_copy.pop()
        intersections_copy.insert(0, start_intersection) # add the end back onto the front so that we'll close the loop 
        while intersections_copy:
            end_intersection = intersections_copy.pop()
            start = self.snap_nodes(self.container[start_intersection[1]].point(start_intersection[0]))
            end = self.snap_nodes(self.container[end_intersection[1]].point(end_intersection[0]))
            
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
            if segment.length() < TOLERANCE*self.options.length: # skip this one because it's too short
                print("skipping: ",start_intersection, end_intersection)
                continue   
            self.graph[start][end] = segment
            self.graph[end][start] = segment.reversed()
            if start not in self.outline_nodes:
                self.outline_nodes.append(start)
            if end not in self.outline_nodes:
                self.outline_nodes.append(end)
             
            start_intersection = end_intersection
        return Path(*final_lines)

    def find_next(self, chained_line, debug=False):
        """
        find the next segment in the chained line
        chained_line: list of locations in the current chain
        returns: bool indicating whether the chain is finshed
        """
        curr_point = chained_line[-1]
        # if the start of the chain is a possible place you can visit, close the chained line
        if len(chained_line) > 2 and chained_line[0] in self.graph[curr_point]:
            # the loop is closed, yippee!
            if debug:
                print("closing loop")
            chained_line.append(chained_line[0])
            return True
        if len(chained_line)>=3:
            # check whether it's possible to close the loop now
            tail_end = chained_line[:-3]
            in_common = set(tail_end).intersection(set(self.graph[curr_point])-set(self.visited_points))
            if debug:
                print(f"chain end {format_complex(chained_line[-1])} tail end {format_complex(set(tail_end))} avail points {format_complex(set(self.graph[curr_point])-set(self.visited_points))} in_common {format_complex(in_common)}")
            if in_common:
                chained_line.append(in_common.pop())
                # remove all values up until the chained piece from the curr visited points
                
                for point in tail_end:
                    if point == chained_line[-1]:
                        break
                    chained_line.remove(point)
                    self.points_to_visit.insert(0, point)
                if debug:
                    print("closed loop due to visited points")
                return True
            branches = [point for point in self.graph[curr_point] if point not in chained_line[1:]+self.visited_points]

            if not branches: # if there were no branches, still allow the graph to travel along the outside if possible
                branches = list(set(self.graph[curr_point]).intersection(set(self.outline_nodes)))
                
        elif 0<len(chained_line)<3:
            branches = [point for point in self.graph[curr_point] if point not in chained_line]
        else:
            branches = list(self.graph[curr_point].keys())
        if debug:
            print(f"branches are {format_complex(branches)} {format_complex(self.graph[curr_point].keys())}")
        if len(branches) == 1:
            chained_line += branches
        elif len(branches) >= 2:
            last_point = chained_line[-2] if len(chained_line) >= 2 else 0
            # if the previous point was on the outside, pick the point on the inside
            if len(chained_line) >= 2 and chained_line[-2] in self.outline_nodes:
                inside_branches = [branch for branch in branches if branch not in self.outline_nodes]
                
                if len(inside_branches) == 0:
                    chained_line.append(branches[0])
                    return False
                    """
                    inkex.utils.errormsg(f"got to bad state! - last point was on outside and only two options are also outside {branches} ")
                    self.plot_graph()
                    for i,_chained_line in enumerate(self.chained_lines):
                        self.add_chained_line(_chained_line, label=f"chained_line_{i}", color="blue")
                    self.add_marker(branches[0], "branch0", "green")
                    self.add_marker(branches[1], "branch1", "green")
                    self.add_chained_line(chained_line, color="red", label="current_line")
                    if len(chained_line) > 1:
                        self.add_marker(chained_line[-2], "last_point")
                    self.add_marker(curr_point, "curr_point", "red")                   
            
                    debug_screen(self, "test_outside")
                    sys.exit(0)
                    """
                chained_line.append(get_clockwise(last_point, curr_point, inside_branches))
            else:                    
                """                    
                if len(chained_line) == chain_piece_to_inspect and len(chained_lines) == chain_to_inspect:
                    self.plot_graph()
                    self.add_marker(branches[0], color="red", label="branch0")
                    self.add_marker(branches[1], label="branch1", color="black")
                    self.add_marker(last_point, label="last_point", color="blue")
                    self.add_marker(curr_point, label="curr_point", color="green")
                    print(f"current chain {format_complex(chained_line)}")
                    
                    
            
                    debug_screen(self, "test_rotation")
                """
                chained_line.append(get_clockwise(last_point, curr_point, branches))

        else:
            inkex.utils.errormsg(f"got to bad state! chained_line {len(self.chained_lines)} chained_piece {len(chained_line)} no branches- {format_complex(branches)} \
            graph is: {format_complex(self.graph[curr_point].keys())} curr_point: {format_complex(curr_point)} chained_line: {format_complex(chained_line)}")
            
            self.plot_graph()
            for i,_chained_line in enumerate(self.chained_lines):
                self.add_chained_line(_chained_line, label=f"chained_line_{i}", color="blue")
            self.add_marker(curr_point)
            self.add_chained_line(chained_line, color="red", label="current_line")
            
            parent = self.get_parent(self.current_shape)
            parent.remove(self.current_shape)
            
            debug_screen(self, "test_graph")
            self.reset_shape()
            return True
        return False           


    def chain_graph(self):
        self.audit_graph()
        # algorithm design
        # dump the keys in the graph into a unique list of points
        self.points_to_visit = list(set(self.graph.keys()))
        self.points_to_visit = [point for point in self.points_to_visit if point not in self.outline_nodes]
        self.visited_points = []
        curr_point = None
        start_time = time()
        chain_to_inspect = 24
        chain_piece_to_inspect = 3
        while self.points_to_visit:
            curr_point = self.points_to_visit.pop()
            if curr_point in self.visited_points:
                continue
            chained_line = [curr_point]
            
            while not self.find_next(chained_line, True): # chain_to_inspect==len(self.chained_lines) and chain_piece_to_inspect==len(chained_line)
                print(f"{len(chained_line)}")
                pass
            assert len(chained_line) >= 4, f"chained_line is too short! {format_complex(chained_line)}"
            assert chained_line[0] != chained_line[-1], f"chained line is not a loop! {format_complex(chained_line)}"

            self.chained_lines.append(chained_line)
            self.visited_points += chained_line
        print(f"chaining took {time()-start_time} num lines {len(self.chained_lines)}")
        # convert to segments
        paths = []
        for chained_line in self.chained_lines:
            segments = []
            for i in range(1, len(chained_line)):
                segments.append(self.graph[chained_line[i-1]][chained_line[i]])
            paths.append(combine_segments(segments))
        return paths
        
    def reset_shape(self):
        # remove all markers etc that were added for debugging
        parent = self.get_parent(self.current_shape)
        parent.remove_all()
        parent.insert(-1, self.current_shape)

    def bool_op_shape(self):
        try:
            from pylivarot import intersection, py2geom
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

    def snap_nodes(self, node):
        ex_nodes = self.graph.keys()
        for ex_node in ex_nodes:
            diff = abs(ex_node-node)
            if diff < TOLERANCE*self.options.length:
                return ex_node
        return node

    def audit_graph(self):
        # check whether there are points that are very close together
        nodes = list(self.graph.keys())
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i == j:
                    continue
                diff = abs(node_i-node_j)
                if diff < TOLERANCE*self.options.length:
                    print(f"nodes {node_j} and {node_i} are only {diff} apart")
        def out_of_bounds(node):
            return node.real > self.xmax + TOLERANCE or node.real < self.xmin-TOLERANCE or node.imag > self.ymax+TOLERANCE or node.imag< self.ymin -TOLERANCE
        # remove any points that are outside of the bbox from the graph
        for node in nodes:
            if not out_of_bounds(node):
                continue
            for branch in self.graph[node]:
                del self.graph[branch][node]
            del self.graph[node] 
        # TODO: combine segments that have only one input/output with the next segment ? 

    def hitomezashi_fill(self, node):
        # greedy algorithm: make a Hitomezashi fill that covers the entire bounding box of the shape, 
        # then go through each segment and figure out if it is inside, outside, or intersecting the shape
        
        self.container = parse_path(pattern_vector_to_d(node))
        self.container.approximate_arcs_with_quads()
        self.xmin, self.xmax, self.ymin, self.ymax = self.container.bbox()
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin
        # generate vertical lines
        lines = []
        num_x = int(self.width/self.options.length)
        if self.width/self.options.length % 1:
            num_x += 1
        num_y = int(self.height/self.options.length)
        if self.height/self.options.length % 1:
            num_y += 1
        for x_i in range(num_x):
            x_coord = x_i*self.options.length + self.xmin
            if x_coord == self.xmin:
                # make the line just inside the box
                x_coord += TOLERANCE/2.0
            if not self.options.gradient:
                odd_even_y = random() > self.options.weight_x
                assert odd_even_y
            else:
                odd_even_y = random() > x_i/num_x
            for y_i in range(num_y):
                if y_i % 2 == odd_even_y:
                    continue
                # make the first and last segments a little longer
                if y_i == 0:
                    y_i = -TOLERANCE
                    diff = TOLERANCE*self.options.length
                elif y_i == num_y-1:
                    diff = TOLERANCE*self.options.length
                else:
                    diff = 0
                
                y_coord = y_i*self.options.length + self.ymin
                
                start = x_coord + y_coord*1j
                end = x_coord + (y_coord+ self.options.length+ diff)*1j  
                    
                lines.append(Line(start, end))
                assert start != end
        # generate horizontal lines
        for y_i in range(num_y):
            y_coord = y_i*self.options.length + self.ymin
            if not self.options.gradient:
                odd_even_y = random() > self.options.weight_y
                assert odd_even_y
            else:
                odd_even_y = random() > y_i/num_y
            for x_i in range(num_x):
                if x_i % 2 == odd_even_y:
                    continue
                # make the first and last segments a little longer
                if x_i == 0:
                    x_i = -TOLERANCE
                    diff = TOLERANCE*self.options.length
                elif x_i == num_x-1:
                    diff = TOLERANCE*self.options.length
                else:
                    diff = 0
                    # continue # TODO: remove me
                x_coord = x_i*self.options.length + self.xmin
                start = x_coord + y_coord*1j
                end = (x_coord + self.options.length+diff) + y_coord*1j
                lines.append(Line(start, end))
                assert start != end

        if not self.options.fill:
            pass
            lines = [self.chop_shape(lines)]
            #lines = [combine_segments(lines)]
        else:
            _ = self.chop_shape(lines)
            lines = self.chain_graph()
            #lines = self.bool_op_shape(self.graph)

        for i, chained_line in enumerate(lines):         
            if chained_line.d() == "":
                raise ValueError(f"got empty chained_path! {i} {chained_line}")
            if i == 131:
                raise ValueError("I should have been skipped!")
            pattern_id = "hitomezashi-"+node.get("id", f"unknown-{self.curr_path_num}")+"-"+str(i)
            pattern_style = node.get("style")
            self.add_node(chained_line.d(), pattern_style, pattern_id)

    @lru_cache(maxsize=None)
    def is_inside(self, point, debug=False):
        # if the point is on the edge of the bbox, assume it's outside
        diffs = [abs(point.real - self.xmin), abs(point.real - self.xmax), abs(point.imag - self.ymin), abs(point.imag - self.ymax)]
        diffs = [diff < TOLERANCE for diff in diffs]
        if any(diffs):
            if debug:
                print("point is on bbox")
            return False
        if point.real < self.xmin:
            if debug:
                print("to the left of the bbox")
            return False
        if point.real > self.xmax:
            if debug:
                print("to the right of the bbox")
            return False
        if point.imag < self.ymin:
            if debug:
                print("below the bbox")
            return False
        if point.imag > self.ymax:
            if debug:
                print("above the bbox")
            return False

        # make sure the lines are actually out of the bbox by adding a shrinking/enlarging factor
        span_line_upper = Line(0.9*(self.xmin+ self.ymin*1j), point)
        span_line_lower = Line(point, 1.1*(self.xmax + self.ymax*1j))
        upper_intersections = intersect_over_all(span_line_upper, self.container)
        lower_intersections = intersect_over_all(span_line_lower, self.container)
        if debug:
            print(f"is_inside debug {upper_intersections} {lower_intersections}")
        return len(upper_intersections) % 2 or len(lower_intersections) % 2


if __name__ == "__main__":
    HitomezashiFill().run()
