import sys
import os
from os.path import dirname, abspath, join

ROOT_DIR = dirname(dirname(abspath(__file__)))

sys.path.append(ROOT_DIR)

from four_color import FourColorFill
from inkex.tester import TestCase
from xml.etree import ElementTree


class TestFourColor(TestCase):
    effect_class = FourColorFill

    def test_basic(self):
        _file = "triangle_hitomezashi.svg"
        input_xml = ElementTree.parse(self.data_file(_file))
        root = input_xml.getroot()
        target_tag = None
        _nodes = []
        for child in root:
            if child.tag[-1] == "g":
                _nodes += list(child)
        assert _nodes
        targets = [f"--id={_node.get('id')}" for _node in _nodes]
        assert targets
        args = targets + [self.data_file(_file)]
        effect = self.effect_class()
        effect.run(args)
        effect.save(
            open(join(ROOT_DIR, "output/four_color_triangle_hitomezashi.svg"), "wb")
        )


if __name__ == "__main__":
    TestFourColor().test_basic()
