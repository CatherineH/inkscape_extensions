import sys
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))

from hitomezashi_fill import HitomezashiFill

from common_utils import debug_screen
from inkex.tester import TestCase
import inkex


class TestHitomezashi(TestCase):
    effect_class = HitomezashiFill

    def test_basic(self):
        target = "rect1"
        _file = "no_fill.svg"
        args = [f"--id={target}", self.data_file(_file)]
        effect = self.effect_class()
        effect.run(args)
        effect.save(open(f"output/{target}_hitomezashi.svg", "wb"))
        old_path = effect.svg.getElementById(target).path
        new_path = effect.svg.getElementById(f"hitomezashi-{target}-0").path
        assert len(new_path) > len(old_path)

    def test_heart(self):
        target = "heart"
        _file = "heart.svg"
        args = [
            f"--id={target}",
            "--length=30",
            "--fill=true",
            "--weight_x=0",
            "--weight_y=0",
            self.data_file(_file),
        ]
        effect = self.effect_class()
        effect.run(args)
        """
        for i, point in enumerate(effect.visited_points):
            effect.add_marker(point, label=f"visited_point_{i}", color="red")
        effect.plot_graph()
        debug_screen(effect, "test_heart")
        #effect.save(open(f"output/{target}_hitomezashi.svg","wb"))
        """
        old_path = effect.svg.getElementById(target).path
        new_path = effect.svg.getElementById(f"hitomezashi-{target}-0").path

        assert new_path

    def test_large(self):
        target = "rect31"
        _file = "laptop_cover.svg"
        args = [
            f"--id={target}",
            "--length=10",
            "--gradient=true",
            "--fill=true",
            "--weight_x=0",
            "--weight_y=0",
            self.data_file(_file),
        ]
        effect = self.effect_class()
        effect.run(args)
        old_path = effect.svg.getElementById(target).path
        new_path = effect.svg.getElementById(f"hitomezashi-{target}-0").path
        debug_screen(effect, "test_large")
        assert new_path


if __name__ == "__main__":
    print("test_basic")
    # TestHitomezashi().test_basic()
    print("test_heart")
    # TestHitomezashi().test_heart()
    TestHitomezashi().test_large()
