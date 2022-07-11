import sys
from os.path import dirname, abspath, join
FOLDERNAME = dirname(abspath(__file__))

sys.path.append(dirname(FOLDERNAME))

from pickle import load
from common_utils import make_stack_tree, stack_lines
import svgwrite



def test_stack_tree():

    lines = load(open(join(FOLDERNAME, "data/test_lines.pickle"), "rb"))

    dwg = svgwrite.Drawing(join(FOLDERNAME, '../output/test_stack_tree.svg'), profile='full')
    for i, line in enumerate(lines):
        dwg.add(dwg.path(d=line.d(), stroke=svgwrite.rgb(10, 10, 16, '%'), id=f"path-{i}"))
    dwg.save()
    stack_tree, root_nodes = make_stack_tree(lines, True)
    print(stack_tree)
    assert 0 in stack_tree[1]
    assert 2 in stack_tree[1]
    assert 3 in stack_tree[1]
    assert 4 in stack_tree[1]
    assert 5 in stack_tree[1]
    assert 6 in stack_tree[1]
    assert 7 in stack_tree[1]
    assert 8 in stack_tree[1]
    assert 9 in stack_tree[1]
    assert 10 in stack_tree[1]
    assert 11 in stack_tree[1]
    assert 12 in stack_tree[1]
    assert 13 in stack_tree[1]
    assert 14 not in stack_tree[1]


def test_stack_tree_edge_pieces():
    lines = load(open(join(FOLDERNAME, "data/test_lines2.pickle"), "rb"))

    dwg = svgwrite.Drawing(join(FOLDERNAME, '../output/test_stack_tree2.svg'), profile='full')
    for i, line in enumerate(lines):
        dwg.add(dwg.path(d=line.d(), stroke=svgwrite.rgb(10, 10, 16, '%'), id=f"path-{i}"))
    dwg.save()
    stack_tree, root_nodes = make_stack_tree(lines, True)
    print(stack_tree)
    assert 11 in stack_tree[3]
    assert 5 in stack_tree[1]
    assert 6 in stack_tree[1]
    assert 7 in stack_tree[1]
    assert 8 in stack_tree[1]
    assert 9 in stack_tree[1]
    assert 10 in stack_tree[1]


def test_stack_lines():
    lines = load(open(join(FOLDERNAME, "data/test_lines.pickle"), "rb"))
    dwg = svgwrite.Drawing(join(FOLDERNAME, '../output/test_stack_lines.svg'), profile='full')
    for i, line in enumerate(lines):
        dwg.add(dwg.path(d=line.d(), stroke=svgwrite.rgb(0, 0, 0, '%'), fill=svgwrite.rgb(255, 0, 0, '%'), id=f"path-before-{i}"))
    lines_after, labels = stack_lines(lines)

    for i, line in enumerate(lines_after):
        dwg.add(dwg.path(d=line.d(), stroke=svgwrite.rgb(0, 0, 0, '%'), fill=svgwrite.rgb(0, 255, 0, '%'),
                         id=f"path-after-{labels[i]}"))
    dwg.save()
    print(labels)
    assert 1 in labels


if __name__ == "__main__":
    test_stack_tree()
    #test_stack_tree_edge_pieces()
    #test_stack_lines()