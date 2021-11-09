# add the root level extensions folder to the PYTHONPATH

import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from pattern_to_path import PatternToPath
from inkex.tester import TestCase
import inkex


class TestPatternToPath(TestCase):
    effect_class = PatternToPath
    def test_basic(self):
        args = ['--id=rect10',
                self.data_file('pattern_test.svg')]
        effect = self.effect_class()
        effect.run(args)
        old_path = effect.original_document.getroot().getElement('//svg:path').path
        new_path = effect.svg.getElementById('pattern-path-rect101').path
        effect.save(open("output/pattern_test_output.svg","wb"))
        assert len(new_path) > len(old_path)

    def test_w3_basic(self):
        args = ['--id=w3rect',
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
        

if __name__ == "__main__":
    TestPatternToPath().test_w3_basic()