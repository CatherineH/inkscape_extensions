import inkex
from typing import List
from common_utils import append_verify, BaseFillExtension


class CutOffset(BaseFillExtension):
    def __init__(self):
        BaseFillExtension.__init__(self, self.cut_target, init_handle=self.find_offset_target)
        self.offset_target = inkex.Path()

    def add_arguments(self, pars):
        pars.add_argument(
            "--target-id", type=str, default="target", help="The ID of the target"
        )
        pars.add_argument(
            "--offset",
            type=float,
            default=10,
            help="Spacing offset between the target and the selected pieces",
        )

    def find_offset_target(self):
        _path = self.svg.getElementById(self.options.target_id)
        self.offset_target = _path.path.offset(self.options.offset)

    def cut_target(self, node):
        node.path = node.path.difference(self.offset_target)


if __name__ == "__main__":
    CutOffset().run()
