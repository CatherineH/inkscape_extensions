# add the root level extensions folder to the PYTHONPATH

import sys
from os.path import dirname, abspath, join
sys.path.append(dirname(dirname(abspath(__file__))))

from pattern_to_path import PatternToPath
from inkex.tester import TestCase
import inkex


class TestPatternToPath(TestCase):
    effect_class = PatternToPath
    def test_basic(self):
        target = 'rect10'
        args = [f'--id={target}',
                self.data_file('pattern_test.svg')]
        effect = self.effect_class()
        effect.run(args)
        old_path = effect.svg.getElementById(target).path
        new_path = effect.svg.getElementById(f'pattern-path-{target}1').path
        effect.save(open("output/pattern_test_output.svg","wb"))
        assert len(new_path) > len(old_path)

    def test_remove(self):
        target = 'w3rect'
        args = [f'--id={target}', '--remove', 'true', self.data_file('w3_example.svg')]
        effect = self.effect_class()
        effect.run(args)
        effect.save(open("output/w3_example_output_remove.svg","wb"))
        new_path = effect.svg.getElementById(f'pattern-path-{target}1').path
        assert new_path
        old_path = effect.svg.getElementById(target)
        assert old_path is None        

    def test_w3_basic(self):
        args = ['--id=w3rect', '--remove', 'true',
                self.data_file('w3_example.svg')]
        effect = self.effect_class()
        effect.run(args)
        pattern_object1 = effect.svg.getElementById('pattern-path-w3rect1')
        pattern_object2 = effect.svg.getElementById('pattern-path-w3rect2')
        pattern_object3 = effect.svg.getElementById('pattern-path-w3rect3')
        effect.save(open("output/w3_example_output.svg","wb"))
        
        assert pattern_object1 is not None
        style_dict = dict(inkex.styles.Style().parse_str(pattern_object1.attrib.get("style")))
        assert style_dict.get('fill') == "skyblue"
        segments = [segment for segment in pattern_object1.path if isinstance(segment, inkex.paths.zoneClose)]
        assert len(segments) == 1

        assert pattern_object2 is not None
        style_dict = dict(inkex.styles.Style().parse_str(pattern_object2.attrib.get("style")))
        assert style_dict.get('fill') == "url(#Gradient2)"
        segments = [segment for segment in pattern_object2.path if isinstance(segment, inkex.paths.zoneClose)]
        assert len(segments) == 16

        assert pattern_object3 is not None
        style_dict = dict(inkex.styles.Style().parse_str(pattern_object3.attrib.get("style")))
        assert style_dict.get('fill') == "url(#Gradient1)"
        segments = [segment for segment in pattern_object3.path if isinstance(segment, inkex.paths.zoneClose)]
        assert len(segments) == 16
        
    def test_no_fill(self):
        args = ['--id=rect1', '--remove', 'true',
                self.data_file('no_fill.svg')]
        effect = self.effect_class()
        effect.run(args)
        output = effect.svg.getElementById('rect1')
        assert output is not None

    def test_no_pattern_fill(self):
        args = ['--id=rect2', '--remove', 'true',
                self.data_file('no_fill.svg')]
        effect = self.effect_class()
        effect.run(args)
        output = effect.svg.getElementById('rect2')
        assert output is not None

    def test_complicated(self):
        full_filename = join(dirname(abspath(__file__)), "data", "blockcc855_pattern.svg")
        print(full_filename)
        pattern = "blackblocks"
        args = [f'--id={pattern}',full_filename]
        print(args)
        effect = self.effect_class()
        effect.run(args)
        effect.save(open("output/blockcc855_pattern_output.svg","wb"))
        pattern_object1 = effect.svg.getElementById(f'pattern-path-{pattern}1')
        assert len(pattern_object1.path) > 0
    

if __name__ == "__main__":
    TestPatternToPath().test_complicated()