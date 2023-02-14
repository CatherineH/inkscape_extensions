# add the root level extensions folder to the PYTHONPATH
import sys
from os.path import dirname, abspath, join

ROOT_DIR = dirname(dirname(abspath(__file__)))

sys.path.append(ROOT_DIR)

from add_markers import AddMarkers
from inkex.tester import TestCase
import inkex
import os


class TestAddMarkers(TestCase):
    effect_class = AddMarkers

    def test_basic(self):
        target = "heart"
        _file = "heart.svg"
        args = [f"--id={target}", "--union=true", self.data_file(_file)]
        effect = self.effect_class()
        effect.run(args)
        print(ROOT_DIR)
        effect.save(open(os.path.join(ROOT_DIR, "output/add_markers_heart.svg"), "wb"))
        old_path = effect.svg.getElementById(target).path

    def test_merge(self):
        target = "heart"
        _file = "heart.svg"
        args = [f"--id={target}", self.data_file(_file)]
        effect = self.effect_class()
        effect.run(args)
        effect.save(
            open(os.path.join(ROOT_DIR, "output/add_markers_heart_merge.svg"), "wb")
        )
        old_path = effect.svg.getElementById(target).path
