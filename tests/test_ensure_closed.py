import sys
import os
from os.path import dirname, abspath, join

ROOT_DIR = dirname(dirname(abspath(__file__)))

sys.path.append(ROOT_DIR)

from ensure_closed import EnsureClosed
from inkex.tester import TestCase


class TestEnsureClosed(TestCase):
    effect_class = EnsureClosed

    def test_basic(self):
        target = "path_to_close"
        _file = "path_to_close.svg"
        args = [f"--id={target}", self.data_file(_file)]
        effect = self.effect_class()

        effect.run(args)
        print(ROOT_DIR)
        effect.save(open(os.path.join(ROOT_DIR, f"output/{target}.svg"), "wb"))
        new_path = effect.svg.getElementById(target).path
        assert len(new_path) > 1

    def test_crossed(self):
        target = "path_to_close_crossed"
        _file = "path_to_close_crossed.svg"
        args = [f"--id={target}", self.data_file(_file)]
        effect = self.effect_class()

        effect.run(args)
        print(ROOT_DIR)
        effect.save(open(os.path.join(ROOT_DIR, f"output/{target}.svg"), "wb"))
        new_path = effect.svg.getElementById(target).path
        assert len(new_path) > 1

    def test_crossed_multiple(self):
        target = "path_to_close_crossed_multiple"
        _file = "path_to_close_crossed_multiple.svg"
        args = [f"--id={target}", self.data_file(_file)]
        effect = self.effect_class()

        effect.run(args)
        print(ROOT_DIR)
        effect.save(open(os.path.join(ROOT_DIR, f"output/{target}.svg"), "wb"))
        new_path = effect.svg.getElementById(target).path
        assert len(new_path) > 1
