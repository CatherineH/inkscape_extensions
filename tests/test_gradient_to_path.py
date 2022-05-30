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
        args = [f"--id={target}", "--debug", "true", "--circles", "true", self.data_file(_file)]
        effect = self.effect_class()
        effect.run(args)
        effect.save(open("output/rainbow_saturated.svg", "wb"))
        old_path = effect.svg.getElementById(target).path
        assert len(new_path) > 40, f"len(new_path) {len(new_path)}"
        assert len(new_path) > len(old_path)

if __name__ == "__main__":
    TestGradientToPath().test_basic()
    TestGradientToPath().test_css_trct()
