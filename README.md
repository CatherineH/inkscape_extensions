Catherine's Inkscape Extensions
-------------------------------

My inkscape extensions. Designed to work with python3/inkscape > 1.0.

Getting this to work with Inkscape
==================================

Find the location of the "user extensions" folder in Edit -> Preferences -> System, then create symlinks from the folder
where you've checked out these files to. See the "link_extensions.bat" file.

This library is heavily dependent on the library `svgpathtools`, which is not compatible with the python that ships
with the windows version of inkscape, to fix this, delete the versions of python in the inkscape/bin folder.


Pattern to Path
===============

If a path's fill is a &langle;pattern&rangle; element, remove the fill and create the path so that it looks the same for svg viewers that don't render patterns fills.

Requires my python bindings to livarot (https://github.com/CatherineH/livarot_pybind) in order to work.

For example, for the [Mozilla developer pattern fill example](https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Patterns)

![mozilla developer pattern fill example](https://raw.githubusercontent.com/CatherineH/inkscape_extensions/main/tests/data/w3_example.svg)

Here's how it renders in Silhouette Studio:

![silhouette studio render](https://raw.githubusercontent.com/CatherineH/inkscape_extensions/main/doc/silhouette_pattern_fill.png)

and here's how it renders the file after running this extension:
![silhouette studio render](https://raw.githubusercontent.com/CatherineH/inkscape_extensions/main/doc/silhouette_path_fill.png)


and here's how it renders in Cricut Design Studio before this extension:

![silhouette studio render](https://raw.githubusercontent.com/CatherineH/inkscape_extensions/main/doc/cricut_pattern_fill.png)

and here's how it renders after:

![silhouette studio render](https://raw.githubusercontent.com/CatherineH/inkscape_extensions/main/doc/cricut_path_fill.png)

Hitozemashi Stitch Patterns
====================

inspired by [Numberphile's video on hitozemashi stitch patterns](https://www.youtube.com/watch?v=JbfhzlMk2eY)
