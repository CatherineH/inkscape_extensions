# add the root level extensions folder to the PYTHONPATH
import sys
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))

from gradient_to_path import GradientToPath
from inkex.tester import TestCase
import inkex


class TestGradientToPath(TestCase):
    effect_class = GradientToPath

    def test_basic(self):
        target = "rect2"
        _file = "w3_linear_gradient.svg"
        args = [f"--id={target}", "--debug", "true", self.data_file(_file)]
        effect = self.effect_class()
        effect.run(args)
        effect.save(open("output/w3_linear_gradient_rect2.svg", "wb"))
        old_path = effect.svg.getElementById(target).path
        new_path = effect.svg.getElementById(f"gradient-red_1").path
        assert len(new_path) > 40, f"len(new_path) {len(new_path)}"
        assert len(new_path) > len(old_path)

    def test_css_trct(self):
        target = "rect1"
        _file = "w3_linear_gradient.svg"
        args = [f"--id={target}", "--debug", "true", self.data_file(_file)]
        effect = self.effect_class()
        effect.run(args)
        effect.save(open("output/w3_linear_gradient_rect1.svg", "wb"))
        old_path = effect.svg.getElementById(target).path
        new_path = effect.svg.getElementById(f"gradient-red_1").path
        assert len(new_path) > 40, f"len(new_path) {len(new_path)}"
        assert len(new_path) > len(old_path)

    def test_rainbow(self):
        target = "target"
        _file = "rainbow_saturated.svg"
        args = [
            f"--id={target}",
            "--debug",
            "true",
            "--circles",
            "true",
            self.data_file(_file),
        ]
        effect = self.effect_class()
        effect.run(args)
        effect.save(open("output/rainbow_saturated.svg", "wb"))
        old_path = effect.svg.getElementById(target).path
        for i in range(0, 5):
            print(f"paths{i}")
            new_path = effect.svg.getElementById(f"points{i}").path
            assert new_path
            stroke = effect.svg.getElementById(f"points{i}").style.get("stroke")
            assert stroke != "None"

    def test_gradient_sampling(self):
        effect = self.effect_class()
        target = "target"
        _file = "rainbow_saturated.svg"
        args = [
            f"--id={target}",
            "--debug",
            "true",
            "--circles",
            "true",
            "--spacing",
            "50",
            self.data_file(_file),
        ]
        effect.run(args)
        bbox = inkex.transforms.BoundingBox(x=(0, 100), y=(0, 100))
        red_color = effect.sample_color(
            bbox, inkex.transforms.Vector2d(0, 0), debug=True
        )
        assert str(red_color) == "stop-color:red;stop-opacity:1.0"
        blue_color = effect.sample_color(
            bbox, inkex.transforms.Vector2d(99.9, 0), debug=True
        )
        assert str(blue_color) == "stop-color:#ff00ff;stop-opacity:1.0"

    def test_gradient_sampling_user_space(self):
        effect = self.effect_class()
        target = "target"
        _file = "rainbow_saturated_user_space.svg"
        args = [
            f"--id={target}",
            "--debug",
            "true",
            "--circles",
            "true",
            "--spacing",
            "50",
            self.data_file(_file),
        ]
        effect.run(args)
        bbox = inkex.transforms.BoundingBox(x=(0, 100), y=(0, 100))
        red_i = effect.sample_color(bbox, inkex.transforms.Vector2d(0, 0), debug=True)
        assert str(red_i) == "stop-color:red;stop-opacity:1.0"
        blue_i = effect.sample_color(
            bbox, inkex.transforms.Vector2d(99.9, 0), debug=True
        )
        assert str(blue_i) == "stop-color:#ff00ff;stop-opacity:1.0"

    def test_rainbow_rotated(self):
        target = "target"
        _file = "rainbow_saturated_rotated.svg"
        args = [
            f"--id={target}",
            "--debug",
            "true",
            "--circles",
            "true",
            self.data_file(_file),
        ]
        effect = self.effect_class()
        effect.run(args)
        effect.save(open("output/rainbow_saturated_rotated.svg", "wb"))
        old_path = effect.svg.getElementById(target).path
        for i in range(0, 5):
            print(f"paths{i}")
            new_path = effect.svg.getElementById(f"points{i}").path
            assert new_path
            stroke = effect.svg.getElementById(f"points{i}").style.get("stroke")
            assert stroke != "None"

    def test_rainbow_odd(self):
        target = "target"
        _file = "rainbow_saturated_odd.svg"
        args = [
            f"--id={target}",
            "--debug",
            "true",
            "--circles",
            "true",
            self.data_file(_file),
        ]
        effect = self.effect_class()
        effect.run(args)
        effect.save(open("output/rainbow_saturated_odd.svg", "wb"))
        old_path = effect.svg.getElementById(target).path
        for i in range(0, 5):
            print(f"paths{i}")
            new_path = effect.svg.getElementById(f"points{i}").path
            assert new_path
            stroke = effect.svg.getElementById(f"points{i}").style.get("stroke")
            assert stroke != "None"


if __name__ == "__main__":
    TestGradientToPath().test_basic()
    TestGradientToPath().test_css_trct()
