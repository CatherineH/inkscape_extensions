import inkex
from svgpathtools.svg_to_paths import rect2pathd, ellipse2pathd
import subprocess

class BaseFillExtension(inkex.EffectExtension):
    def __init__(self):
        inkex.EffectExtension.__init__(self)
    
    def get_parent(self, node):
        parent = node.getparent()
        if parent is None:
            parent = self.document.getroot()
        return parent

def bounds_rect(pv):
    from pylivarot import py2geom
    bbox = pv.boundsExact()
    return py2geom.Rect(bbox[py2geom.Dim2.X], bbox[py2geom.Dim2.Y])


def transform_path_vector(pv, affine_t):
    for _path in pv:
        for _curve in _path:
            _curve.transform(affine_t)
    return pv


def pattern_vector_to_d(pattern_vector):
    if 'd' in pattern_vector.attrib:
        return pattern_vector.attrib.get('d')
    elif pattern_vector.TAG == "rect":
        """         if "rx" in pattern_vector.attrib or "ry" in pattern_vector.attrib:
                    inkex.utils.errormsg(
                        "Rect %s has rounded edges, this not yet supported, try Object to path first"
                        % (pattern_vector.get("id"))
                    )
                    return """
        return rect2pathd(pattern_vector.attrib)
    elif pattern_vector.TAG in ["circle", "ellipse"]:
        return ellipse2pathd(pattern_vector.attrib)
    else:
        inkex.utils.errormsg(
            "Shape %s (%s) not yet supported, try Object to path first"
            % (pattern_vector.TAG, pattern_vector.get("id"))
        )


def get_fill_id(node):
    style = node.attrib.get('style')
    inkex_style = dict(inkex.styles.Style().parse_str(style))
    fill = node.attrib.get('fill')
    
    inkex_style['fill'] = fill or inkex_style.get('fill')
    if not inkex_style['fill']:
        inkex.utils.errormsg("no 'fill' in inkex style")
        return
    if inkex_style['fill'].find("url(") != 0:
        inkex.utils.errormsg(f"for shape {node.attrib.get('id')} fill does not contain a pattern reference")
        return
    if inkex_style['fill'] == "none":
        return
    return inkex_style['fill'].replace("url(#", "")[:-1]

def debug_screen(effect, name=None):
    name = name or str(effect)
    filename = f"output/debug_{name}.svg"
    effect.save(open(filename,"wb"))
    subprocess.run(["inkview", filename])
