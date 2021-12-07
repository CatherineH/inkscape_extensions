#!/usr/bin/env python3
import inkex

class RemoveShortPaths(inkex.EffectExtension):
    def __init__(self):
        inkex.EffectExtension.__init__(self)
    
    def add_arguments(self, pars):
        pars.add_argument("--length", type=float, default=1, help="Length thresholds")

    def effect(self):
        if self.svg.selected:
            for _, shape in self.svg.selected.items():
                self.remove_short_paths(shape)
    
    def remove_short_paths(self):
        pass

if __name__ == "__main__":
    RemoveShortPaths().run()