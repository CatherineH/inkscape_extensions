import inkex
from typing import List
from common_utils import append_verify, BaseFillExtension


class EnsureClosed(BaseFillExtension):
    def __init__(self):
        BaseFillExtension.__init__(self, self.close_paths)

    def close_paths(self, node):
        _svgpath = node.path.to_svgpathtools()
        if not _svgpath.isclosed():
            # check whether the path crosses over itself?
            intersections = []
            for i1, segment1 in enumerate(_svgpath):
                for i2, segment2 in enumerate(_svgpath):
                    if i1 >= i2:
                        continue
                    _seg_intersections = segment1.intersect(segment2)
                    if _seg_intersections:
                        intersections += [[i1, i2, _seg_intersections]]

            if not intersections:
                # if the path doesn't cross itself, move the last point to the first point and then close the path
                _svgpath[-1].end = _svgpath[0].start
            else:
                # if there are intersections, crop the path to that location
                i1, i2, ts = intersections.pop()
                ts = ts.pop()
                # the last intersection added is going to be the last
                start_T = _svgpath.t2T(i1, ts[0])
                end_T = _svgpath.t2T(i2, ts[1])
                _svgpath = _svgpath.cropped(start_T, end_T)

            node.path = node.path.from_svgpathtools(_svgpath)


if __name__ == "__main__":
    EnsureClosed().run()
