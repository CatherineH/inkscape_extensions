# add the root level extensions folder to the PYTHONPATH

import sys
from os.path import dirname, abspath
from inkex.utils import filename_arg
sys.path.append(dirname(dirname(abspath(__file__))))

from pattern_to_path import PatternToPath
from inkex.tester import TestCase


class TestPatternToPath(TestCase):
    effect_class = PatternToPath
    def test_basic(self):
        args = ['--id=rect10',
                self.data_file('pattern_test.svg')]
        effect = self.effect_class()
        effect.run(args)
        old_path = effect.original_document.getroot().getElement('//svg:path').path
        new_path = effect.svg.getElement('//svg:path').path
        assert len(new_path) > len(old_path)

