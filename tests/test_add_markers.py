# add the root level extensions folder to the PYTHONPATH
import sys
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))

from add_markers import AddMarkers
from inkex.tester import TestCase
import inkex


class TestAddMarkers(TestCase):
    effect_class = AddMarkers

    def test_basic(self):
        target = "heart"
        _file = "heart.svg"
        args = [f"--id={target}", self.data_file(_file)]
        effect = self.effect_class()
        effect.run(args)
        effect.save(open("output/add_markers_heart.svg", "wb"))
        old_path = effect.svg.getElementById(target).path
