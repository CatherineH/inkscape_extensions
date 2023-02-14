import sys
from os.path import dirname, abspath, join
import pytest
sys.path.append(dirname(dirname(abspath(__file__))))

from hitomezashi_fill import HitomezashiFill

from common_utils import debug_screen
from inkex.tester import TestCase
import inkex
import pickle
import xml.etree

FOLDERNAME = join(dirname(dirname(abspath(__file__))), "output")


class TestHitomezashi(TestCase):
    effect_class = HitomezashiFill

    def test_basic(self):
        target = "rect1"
        _file = "no_fill.svg"
        args = [f"--id={target}", self.data_file(_file)]
        effect = self.effect_class()
        effect.run(args)
        effect.save(open(join(FOLDERNAME, f"{target}_hitomezashi.svg"), "wb"))
        old_path = effect.svg.getElementById(target).path
        new_path = effect.svg.getElementById(f"hitomezashi-{target}-0").path
        assert len(new_path) > len(old_path)

    @pytest.mark.skip
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

    def test_basic_fill(self):
        target = "rect1"
        _file = "no_fill.svg"
        args = [f"--id={target}", self.data_file(_file)]
        effect = self.effect_class()
        effect.run(args)
        effect.save(open(join(FOLDERNAME, f"{target}_hitomezashi.svg"), "wb"))
        old_path = effect.svg.getElementById(target).path
        new_path = effect.svg.getElementById(f"hitomezashi-{target}-0").path
        assert len(new_path) > len(old_path)

    def test_large(self):
        target = "rect31"
        _file = "laptop_cover.svg"
        args = [
            f"--id={target}",
            "--length=10",
            "--fill=true",
            "--weight_x=0",
            "--weight_y=0",
            self.data_file(_file),
        ]
        effect = self.effect_class()
        effect.run(args)
        old_path = effect.svg.getElementById(target).path
        new_path = effect.svg.getElementById(f"hitomezashi-{target}-0").path
        effect.save(open(join(FOLDERNAME, f"test_large_hitomezashi.svg"), "wb"))
        assert new_path

    def test_large_fill(self):
        target = "rect31"
        _file = "laptop_cover.svg"
        args = [
            f"--id={target}",
            "--length=60",
            "--fill=true",
            "--weight_x=-1",
            "--weight_y=-1",
            self.data_file(_file),
        ]
        effect = self.effect_class()
        effect.run(args)
        assert effect.options.weight_x == -1
        assert effect.options.weight_y == -1
        old_path = effect.svg.getElementById(target).path
        layer1 = effect.svg.getElementById("layer1")
        seen_d_strings = []
        # go over the chained lines and confirm that they don't overlap,
        # also make sure it is not moving after it already moved
        for element in layer1.iterchildren():
            path = element.path
            """
            for seen_d_string in seen_d_strings:
                assert path[0] != seen_d_string.path[0], f"strings {element.get_id()} {seen_d_string.get_id()}:  {path} and {seen_d_string} start at the same location!"

            seen_d_strings.append(element)
            """
            """
            previous_letter = None
            for path_piece in path:
                if path_piece.letter == "M" and previous_letter == "L":
                    assert False, f"path {path} has a move command after a line"
                previous_letter = path_piece.letter
            """

        # new_path = effect.svg.getElementById(f"hitomezashi-{target}-0").path
        # print("calling debug screen now")
        # debug_screen(effect, "test_large_gradient")
        effect.save(open(join(FOLDERNAME, f"test_large_fill.svg"), "wb"))
        # assert new_path

    def test_large_gradient(self):
        target = "rect31"
        _file = "laptop_cover.svg"
        args = [
            f"--id={target}",
            "--length=30",
            "--gradient=true",
            "--fill=true",
            "--weight_x=0",
            "--weight_y=0",
            self.data_file(_file),
        ]
        effect = self.effect_class()

        effect.run(args)
        old_path = effect.svg.getElementById(target).path
        layer1 = effect.svg.getElementById("layer1")
        seen_d_strings = []
        # go over the chained lines and confirm that they don't overlap,
        # also make sure it is not moving after it already moved
        for element in layer1.iterchildren():
            path = element.path
            """
            for seen_d_string in seen_d_strings:
                assert path[0] != seen_d_string.path[0], f"strings {element.get_id()} {seen_d_string.get_id()}:  {path} and {seen_d_string} start at the same location!"

            seen_d_strings.append(element)
            """
            """
            previous_letter = None
            for path_piece in path:
                if path_piece.letter == "M" and previous_letter == "L":
                    assert False, f"path {path} has a move command after a line"
                previous_letter = path_piece.letter
            """

        # new_path = effect.svg.getElementById(f"hitomezashi-{target}-0").path
        # print("calling debug screen now")
        # debug_screen(effect, "test_large_gradient")
        effect.save(open(join(FOLDERNAME, f"test_large_gradient.svg"), "wb"))
        # assert new_path

    def test_chain_graph(self):
        effect = self.effect_class()
        effect.options.length = 30
        effect.document = xml.etree.ElementTree.ElementTree()
        _doc = inkex.elements._svg.SvgDocumentElement()
        _doc.set("viewBox", "0 0 300.26459 210.26459")
        effect.document._setroot(_doc)
        with open(self.data_file("chain_graph.pkl"), "rb") as fh:
            chain_graph = pickle.load(fh)
            effect.container = chain_graph["container"]
            effect.edges = chain_graph["edges"]
        loops = effect.chain_graph()
        # confirm two things with all loops
        # 1. the loop is closed
        # 2. the loop does not contain lines at an angle
        for i,loop in enumerate(loops):
            assert loop.start == loop.end
            for segment in loop:
                xmin, xmax, ymin, ymax = segment.bbox()
                diff_x = xmin-xmax
                diff_y = ymin-ymax
                assert diff_x == 0 or diff_y == 0, f"edge: {effect.edges[i].d()} loop: {loop.d()} - bbox {diff_x} {diff_y}"


if __name__ == "__main__":
    TestHitomezashi().test_large_fill()
    TestHitomezashi().test_large_gradient()
    TestHitomezashi().test_large()
    #TestHitomezashi().test_chain_graph()
