import inkex
from typing import List
from common_utils import append_verify, BaseFillExtension


class AddMarkers(BaseFillExtension):
    def __init__(self):
        BaseFillExtension.__init__(self, self.add_marker, self.generate_markers)
        self.all_paths = []
        self.bbox = inkex.transforms.BoundingBox(x=0, y=0)
        assert self.effect_handle == self.add_marker
        # marker path
        self.marker_paths: List[inkex.Path()] = []
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

        if left is None or right is None or top is None or bottom is None:
            raise ValueError(
                f"invalid overall bounding box on {left=}  {right=} {top=} {bottom=} {self.svg.selected.items()}"
            )
        self.bbox = inkex.transforms.BoundingBox(x=(left, right), y=(top, bottom))
        self.add_cross(
            x=self.bbox.left + self.bbox.width / 2,
            y=self.bbox.top + self.bbox.height / 2,
        )
        self.add_cross(x=self.bbox.left, y=self.bbox.top + self.bbox.height / 2)
        self.add_cross(x=self.bbox.right, y=self.bbox.top + self.bbox.height / 2)
        self.add_cross(x=self.bbox.left + self.bbox.width / 2, y=self.bbox.top)
        self.add_cross(x=self.bbox.left + self.bbox.width / 2, y=self.bbox.bottom)

    def add_marker_path(self, segment):
        append_verify(self.marker_path, segment)

    def add_cross(self, x, y):
        self.add_marker_path(
            inkex.paths.Move(x=x + self.options.width / 2, y=y + self.options.width / 2)
        )
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
        # self.add_marker_path(inkex.paths.Line(x=x+self.options.width/2, y=y+self.options.width/2))
        self.marker_paths.append(self.marker_path)
        self.marker_path = inkex.Path()

    def add_arguments(self, pars):
        pars.add_argument(
            "--width", type=float, default=2, help="The width of the marker"
        )
        pars.add_argument(
            "--length",
            type=float,
            default=10,
            help="The length of the marker",
        )
        pars.add_argument(
            "--union",
            type=str,
            default="false",
            help="merge the markers with the design",
        )

    def add_marker(self, node):
        for i, _marker in enumerate(self.marker_paths):
            # if the marker intersects with the target shape, don't add it
            if node.path.intersection(_marker):
                continue
            if self.options.union == "false":
                node.path = node.path.union(_marker)
            else:
                self.add_path_node(
                    str(_marker),
                    node.get("style"),
                    node.get("id") + "_marker_" + str(i),
                )


if __name__ == "__main__":
    AddMarkers().run()
