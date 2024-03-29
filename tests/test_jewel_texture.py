# add the root level extensions folder to the PYTHONPATH
import sys
from os.path import dirname, abspath, join

ROOT_DIR = dirname(dirname(abspath(__file__)))

sys.path.append(ROOT_DIR)

from jewel_texture import JewelTexture
from inkex.tester import TestCase
import inkex


class TestJewelTexture(TestCase):
    effect_class = JewelTexture

    def test_basic(self):
        target = "rect31"
        _file = "laptop_cover.svg"
        args = [f"--id={target}", "--minimum=20", self.data_file(_file)]
        effect = self.effect_class()
        effect.run(args)
        effect.save(open(join(ROOT_DIR, "output/jewel_texture_output.svg"), "wb"))

        old_path = effect.svg.getElementById(target).path
        for i in range(4):
            print(i)
            new_path = effect.svg.getElementById(f"shapes{i}").path
            assert len(new_path)


if __name__ == "__main__":
    TestJewelTexture().test_basic()
