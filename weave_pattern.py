
import inkex

from common_utils import BaseFillExtension
from inspect import getfile

class WeaveFill(BaseFillExtension):
    def __init__(self):
        BaseFillExtension.__init__(self, self.weave_fill)
        self.all_paths = []
        assert self.effect_handle == self.weave_fill

    def add_arguments(self, pars):
        pars.add_argument("--length", type=float, default=3, help="Length of segments")
        pars.add_argument(
            "--thickness",
            type=float,
            default=0.5,
            help="the thickness of the weave",
        )

    def weave_fill(self, shape):
        bbox = shape.bounding_box()
        print(getfile(type(shape)))
        last_vert = False
        length = self.options.length
        thickness = self.options.thickness
        # horizontal lines
        x = bbox.left-length/2.0
        print(f"length {length} thickness {thickness} bbox {bbox}")
        while x < bbox.right:
            if len(self.all_paths) > 1000:
                raise ValueError(f"too many paths! x span: {bbox.width} y span: {bbox.height} length {length}")
            y = bbox.top - length/2.0
            if last_vert:
                y += length/2.0
            y += thickness
            assert y < bbox.bottom
            while y < bbox.bottom:
                path = inkex.Path()
                path.append(inkex.paths.Move(x, y-thickness))
                path.append(inkex.paths.Line(x+length-2*thickness, y-thickness))
                path.append(inkex.paths.Line(x + length - 2*thickness, y))
                path.append(inkex.paths.Line(x, y))
                path.append(inkex.paths.ZoneClose())
                self.all_paths.append(path)
                y += length
            x += length/2.0
            last_vert = not last_vert
        last_vert = False
        # vertical lines
        x = bbox.left - length / 2.0
        while x < bbox.right:
            y = bbox.top - length/2.0
            if last_vert:
                y += length / 2.0
            assert y < bbox.bottom
            while y < bbox.bottom:
                path = inkex.Path()
                path.append(inkex.paths.Move(x, y))
                path.append(inkex.paths.Line(x + thickness, y))
                path.append(inkex.paths.Line(x + thickness, y + length-2*thickness))
                path.append(inkex.paths.Line(x, y + length-2*thickness))
                path.append(inkex.paths.ZoneClose())
                self.all_paths.append(path)
                y += length
            x += length/2.0
            last_vert = not last_vert
        # chop off the pieces that intersect the shape
        container_path = inkex.Path(shape.get_path())
        for i, path in enumerate(self.all_paths):
            self.add_path_node(str(path), "stroke:none;fill:green", f"segment_before{i}")
            path = path.intersection(container_path)
            self.add_path_node(str(path), "stroke:none;fill:pink", f"segment{i}")
