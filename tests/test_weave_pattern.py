import sys
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))

from weave_pattern import WeaveFill
from inkex.tester import TestCase
from inspect import getfile
FOLDERNAME = join(dirname(dirname(abspath(__file__))), "output")


class TestWeaveFill(TestCase):
    effect_class = WeaveFill

    def test_basic(self):
        target = "rect1"
        _file = "no_fill.svg"
        args = [f"--id={target}", "--length=10", self.data_file(_file)]
        effect = self.effect_class()
        effect.run(args)
        assert effect.svg.selected
        effect.save(open(join(FOLDERNAME, f"{target}_WeaveFill.svg"), "wb"))
        assert effect.all_paths
        old_path = effect.svg.getElementById(target).path
        new_path = effect.svg.getElementById(f"segment0").path

