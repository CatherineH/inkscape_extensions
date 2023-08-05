#!/usr/bin/env python3
import inkex
from svgpathtools.path import Path


class RemoveShortPaths(inkex.EffectExtension):
    def __init__(self):
        inkex.EffectExtension.__init__(self)

    def add_arguments(self, pars):
        pars.add_argument("--length", type=float, default=1, help="Length thresholds")

    def effect(self):
        if self.svg.selected:
            for _, shape in self.svg.selected.items():
                self.remove_short_paths(shape)

    def remove_short_paths(self, shape):
        #  continuous_subpaths
        shape_d_string = shape.attrib.get("d")
        surviving_path = Path()

        subpaths = inkex.Path(shape_d_string).to_svgpathtools().continuous_subpaths()
        for subpath in subpaths:
            if subpath.length() < self.options.length:
                continue
            surviving_path += subpath

        node_container = inkex.elements.PathElement()
        node_container.set_path(surviving_path.d())
        node_container.set("style", shape.attrib.get("style"))
        node_container.set("id", shape.attrib.get("id") + "-cutshortpath")
        shape.getparent().insert(0, node_container)


if __name__ == "__main__":
    RemoveShortPaths().run()
