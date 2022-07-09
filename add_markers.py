from copy import deepcopy


import inkex

from common_utils import (
    pattern_vector_to_d,
    BaseFillExtension,
    get_clockwise
)

from math import sin, cos

from collections import defaultdict
import random
from svgpathtools.path import Path, Line


class AddMarkers(BaseFillExtension):
    def __init__(self):
        BaseFillExtension.__init__(self, self.add_marker, self.generate_markers)
        self.all_paths = []
        self.bbox = inkex.transforms.BoundingBox(x=0, y=0)
        assert self.effect_handle == self.add_marker
        # marker path
        self.marker_path = inkex.Path()

    def generate_markers(self):
        top, left, right, bottom = None, None, None, None
        for i, shape in self.svg.selected.items():
            _bbox = shape.bounding_box()
            if not top:
                top = _bbox.top
                left = _bbox.left
                right = _bbox.right
                bottom = _bbox.bottom
            if top > _bbox.top:
                top = _bbox.top
            if left > _bbox.left:
                left = _bbox.left
            if right < _bbox.right:
                right = _bbox.right
            if bottom < _bbox.bottom:
                bottom = _bbox.bottom

        if not left or not right or not top or not bottom:
            raise ValueError(f"invalid overall bounding box on {self.svg.selected.items()}")
        self.bbox = inkex.transforms.BoundingBox(x=(left, right), y=(top, bottom))
        self.add_cross(x=self.bbox.left + self.bbox.width/2, y=self.bbox.top + self.bbox.height/2)
        self.add_cross(x=self.bbox.left, y=self.bbox.top + self.bbox.height / 2)
        self.add_cross(x=self.bbox.right, y=self.bbox.top + self.bbox.height / 2)
        self.add_cross(x=self.bbox.left + self.bbox.width / 2, y=self.bbox.top)
        self.add_cross(x=self.bbox.left + self.bbox.width / 2, y=self.bbox.bottom)

    def add_marker_path(self, segment):
        self.marker_path.append(segment)
        for i,path in enumerate(self.marker_path.to_svgpathtools()):
            assert path.start != path.end, f"degenerate path was added! {path=}"

    def add_cross(self, x, y):
        self.add_marker_path(inkex.paths.Move(x=x+self.options.width/2, y=y+self.options.width/2))
        self.add_marker_path(inkex.paths.line(dx=self.options.length, dy=0))
        self.add_marker_path(inkex.paths.line(dx=0, dy=-self.options.width))
        self.add_marker_path(inkex.paths.line(dx=-self.options.length, dy=0))
        self.add_marker_path(inkex.paths.line(dx=0, dy=-self.options.length))
        self.add_marker_path(inkex.paths.line(dx=-self.options.width, dy=0))
        self.add_marker_path(inkex.paths.line(dx=0, dy=self.options.length))
        self.add_marker_path(inkex.paths.line(dx=-self.options.length, dy=0))
        self.add_marker_path(inkex.paths.line(dx=0, dy=self.options.width))
        self.add_marker_path(inkex.paths.line(dx=self.options.length, dy=0))
        self.add_marker_path(inkex.paths.line(dx=0, dy=self.options.length))
        self.add_marker_path(inkex.paths.line(dx=self.options.width, dy=0))
        self.add_marker_path(inkex.paths.line(dx=0, dy=-self.options.length))
        #self.add_marker_path(inkex.paths.Line(x=x+self.options.width/2, y=y+self.options.width/2))

    def add_arguments(self, pars):
        pars.add_argument("--width", type=float, default=2, help="The width of the marker")
        pars.add_argument(
            "--length",
            type=float,
            default=10,
            help="The length of the marker",
        )

    def add_marker(self, node):
        node.path = node.path.union(self.marker_path)


if __name__ == "__main__":
    AddMarkers().run()
