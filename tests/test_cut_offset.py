# add the root level extensions folder to the PYTHONPATH
import sys
from os.path import dirname, abspath, join

ROOT_DIR = dirname(dirname(abspath(__file__)))

sys.path.append(ROOT_DIR)

from cut_offset import CutOffset
from inkex.tester import TestCase
import os


class TestCutOffset(TestCase):
    effect_class = CutOffset

    def test_basic(self):
        target = "circle1"
        _file = "stack.svg"
        args = [
            f"--id={target}",
            "--offset=10",
            "--target=target",
            self.data_file(_file),
        ]
        effect = self.effect_class()

        effect.run(args)
        print(ROOT_DIR)
        effect.save(open(os.path.join(ROOT_DIR, "output/cut_offset_target.svg"), "wb"))
        new_path = effect.svg.getElementById(target + "-cut").path
        assert len(new_path) > 1

    def test_basic2(self):
        target = "circle2"
        _file = "stack.svg"
        args = [
            f"--id={target}",
            "--offset=10",
            "--target=target",
            self.data_file(_file),
        ]
        effect = self.effect_class()

        effect.run(args)
        print(ROOT_DIR)
        effect.save(open(os.path.join(ROOT_DIR, "output/cut_offset_target2.svg"), "wb"))
        new_path = effect.svg.getElementById(target + "-cut").path
        assert len(new_path) > 1

    def test_basic3(self):
        target = "circle3"
        _file = "stack.svg"
        args = [
            f"--id={target}",
            "--offset=10",
            "--target=target",
            self.data_file(_file),
        ]
        effect = self.effect_class()

        effect.run(args)
        print(ROOT_DIR)
        effect.save(open(os.path.join(ROOT_DIR, "output/cut_offset_target3.svg"), "wb"))
        new_path = effect.svg.getElementById(target + "-cut").path
        assert len(new_path) > 1

    def test_m_path(self):
        target = "target"
        _file = "m_path.svg"
        args = [
            f"--id={target}",
            "--offset=2",
            "--target=m_path",
            self.data_file(_file),
        ]
        effect = self.effect_class()

        effect.run(args)
        print(ROOT_DIR)
        effect.save(open(os.path.join(ROOT_DIR, "output/cut_offset_m_path.svg"), "wb"))
        new_path = effect.svg.getElementById(target + "-cut").path
        assert len(new_path) > 1
