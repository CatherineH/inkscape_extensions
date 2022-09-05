# add the root level extensions folder to the PYTHONPATH
import sys
from os.path import dirname, abspath, join
ROOT_DIR = dirname(dirname(abspath(__file__)))

sys.path.append(ROOT_DIR)

from cut_offset import CutOffset
from inkex.tester import TestCase
import inkex
import os


class TestCutOffset(TestCase):
    effect_class = CutOffset

    def test_basic(self):
        target = "circle1"
        _file = "stack.svg"
        args = [f"--id={target}", "--offset=10", "--target=target", self.data_file(_file)]
        effect = self.effect_class()

        effect.run(args)
        print(ROOT_DIR)
        effect.save(open(os.path.join(ROOT_DIR, "output/cut_offset_target.svg"), "wb"))
        new_path = effect.svg.getElementById(target).path

        assert len(new_path) > 1