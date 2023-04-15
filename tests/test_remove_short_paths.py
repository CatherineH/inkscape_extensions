# add the root level extensions folder to the PYTHONPATH
import sys
from os.path import dirname, abspath, join

ROOT_DIR = dirname(dirname(abspath(__file__)))

sys.path.append(ROOT_DIR)

from remove_short_paths import RemoveShortPaths
from inkex.tester import TestCase
import inkex


class TestRemoveShortPaths(TestCase):
    effect_class = RemoveShortPaths

    def test_basic(self):
        target = "iron"
        args = [
            f"--id={target}",
            "--length=10.0",
            self.data_file("short_paths.svg"),
        ]
        effect = self.effect_class()
        effect.run(args)
        old_path = effect.svg.getElementById(target).path
        new_path = effect.svg.getElementById(f"{target}-cutshortpath").path
        effect.save(open(join(ROOT_DIR, "output/test_remove_short_paths.svg"), "wb"))
        subpaths = new_path.to_svgpathtools()
        assert len(subpaths) > 0
        assert len(new_path) > 0
        assert len(old_path) > len(new_path)


if __name__ == "__main__":
    TestRemoveShortPaths().test_basic()
