import sys
from os.path import dirname, abspath, join
sys.path.append(dirname(dirname(abspath(__file__))))

from hitomezashi_fill import HitomezashiFill
from inkex.tester import TestCase
import inkex

class TestHitomezashi(TestCase):
    effect_class = HitomezashiFill
    def test_basic(self):
        target = 'rect1'
        _file = 'no_fill.svg'
        args = [f'--id={target}', self.data_file(_file)]
        effect = self.effect_class()
        effect.run(args)
        effect.save(open(f"output/{target}_hitomezashi.svg","wb"))
        old_path = effect.svg.getElementById(target).path
        new_path = effect.svg.getElementById(f'hitomezashi-{target}').path
        assert len(new_path) > len(old_path)

    def test_heart(self):
        target = 'heart'
        _file = 'heart.svg'
        args = [f'--id={target}', '--length=30', self.data_file(_file)]
        effect = self.effect_class()
        effect.run(args)
        effect.save(open(f"output/{target}_hitomezashi.svg","wb"))
        old_path = effect.svg.getElementById(target).path
        new_path = effect.svg.getElementById(f'hitomezashi-{target}').path
        # real width: 89.299, height: 105.939
        print(type(old_path))
        print(old_path.bounding_box().width, old_path.bounding_box().height)
        print(new_path.bounding_box().width, new_path.bounding_box().height)
        assert False
        assert len(new_path) > len(old_path)


if __name__ == "__main__":
    TestHitomezashi().test_basic()